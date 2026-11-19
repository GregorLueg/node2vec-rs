//! The two entry points and the training tail they share.

use std::sync::Arc;

use node2vec_rs::cpu::train::{
    create_negative_table, create_negative_table_per_type, train_node2vec_cpu, CpuTrainArgs,
    NegativeTable,
};
use node2vec_rs::cpu::NEGATIVE_TABLE_SIZE;
use node2vec_rs::prelude::{HetGraph, Metapath, Node2VecError as CoreError, Node2VecGraph};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Node2VecError;

/// How often, in tokens, each hogwild thread re-derives its decayed learning
/// rate. The CLI value; not tuned for the bindings.
const LR_UPDATE_RATE: usize = 10_000;

////////////
// Params //
////////////

/// Walk and training parameters shared by both entry points.
///
/// Extracted from a Python dict so the two signatures do not each restate a
/// dozen keyword arguments. `node2vec_rs._api` builds the dict and owns the
/// defaults.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct Params {
    /// Walks started from each node (of the start type, for metapath2vec).
    walks_per_node: usize,
    /// Nodes per walk.
    walk_length: usize,
    /// Embedding dimension.
    dim: usize,
    /// Maximum skip-gram window.
    window: usize,
    /// Passes over the walks.
    epochs: usize,
    /// Negative samples per positive pair.
    neg: usize,
    /// Initial SGD learning rate, decayed linearly to near zero.
    lr: f32,
    /// Subsampling threshold for frequent nodes; zero disables it.
    sample: f32,
    /// L2-normalise every output row.
    normalise: bool,
    /// Seed for walks, initialisation and the negative tables.
    seed: usize,
    /// Threads for this call, or `None` for rayon's default.
    n_threads: Option<usize>,
    /// Progress bars and training log on the Rust side.
    verbose: bool,
}

/////////////
// Helpers //
/////////////

/// Convert a core error into the Python exception.
///
/// ### Params
///
/// * `e` - The core error
///
/// ### Returns
///
/// A `Node2VecError` carrying the core's message.
fn to_py(e: CoreError) -> PyErr {
    Node2VecError::new_err(e.to_string())
}

/// Run `f` on a pool of `n_threads`, or on rayon's global pool.
///
/// A fresh pool per call rather than `build_global`, which succeeds only once
/// per process and would make a second call from a notebook fail.
///
/// ### Params
///
/// * `n_threads` - Thread count, `None` for rayon's default
/// * `f` - The work. Call this inside `Python::detach`, never around it.
///
/// ### Returns
///
/// Whatever `f` returns, or the pool build failure.
fn in_pool<T: Send>(n_threads: Option<usize>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    match n_threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(|pool| pool.install(f))
            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        None => Ok(f()),
    }
}

/// Zip the edge arrays into the core's `(from, to, weight)` triplets.
///
/// ### Params
///
/// * `src` - Source node index per edge
/// * `dst` - Target node index per edge
/// * `weight` - Weight per edge, or `None` for 1.0 throughout
///
/// ### Returns
///
/// The triplets. Copies, since the arrays cannot cross into `detach`.
fn edges(
    src: &PyReadonlyArray1<u32>,
    dst: &PyReadonlyArray1<u32>,
    weight: Option<&PyReadonlyArray1<f32>>,
) -> PyResult<Vec<(u32, u32, f32)>> {
    let src = src.as_slice()?;
    let dst = dst.as_slice()?;
    Ok(match weight {
        Some(w) => {
            let w = w.as_slice()?;
            (0..src.len()).map(|i| (src[i], dst[i], w[i])).collect()
        }
        None => (0..src.len()).map(|i| (src[i], dst[i], 1.0)).collect(),
    })
}

/// Train skip-gram on finished walks and return the input embedding.
///
/// Must run inside the pool, since the hogwild trainer splits the walks into
/// one chunk per thread of whatever pool it is called from.
///
/// ### Params
///
/// * `walks` - The walks
/// * `vocab_size` - Number of nodes, i.e. output rows
/// * `neg_table` - Negative sampling table
/// * `params` - Training parameters
///
/// ### Returns
///
/// Row-major `vocab_size * dim` embedding.
fn fit(
    walks: Vec<Vec<u32>>,
    vocab_size: usize,
    neg_table: NegativeTable,
    params: &Params,
) -> Vec<f32> {
    let args = CpuTrainArgs {
        dim: params.dim,
        lr: params.lr,
        epochs: params.epochs,
        neg: params.neg,
        window: params.window,
        lr_update_rate: LR_UPDATE_RATE,
        n_threads: rayon::current_num_threads(),
        verbose: params.verbose,
        sample: params.sample,
    };
    let (mut input, _) = train_node2vec_cpu(walks, vocab_size, args, neg_table, params.seed);
    if params.normalise {
        input.norm_self();
    }
    input.into_vec()
}

/// Wrap a flat row-major buffer as an `(n, dim)` array.
///
/// ### Params
///
/// * `py` - GIL token
/// * `data` - Row-major buffer
/// * `dim` - Columns
///
/// ### Returns
///
/// The 2-D float32 array.
fn to_array<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    dim: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = data.len() / dim;
    data.into_pyarray(py).reshape([n, dim])
}

/////////////////
// Entrypoints //
/////////////////

/// node2vec on a homogeneous graph.
///
/// ### Params
///
/// * `src` - Source node index per edge, dense in `0..n_nodes`
/// * `dst` - Target node index per edge
/// * `weight` - Weight per edge, or `None`
/// * `n_nodes` - Number of nodes, i.e. output rows
/// * `directed` - Keep edges one-way
/// * `p` - Return parameter
/// * `q` - In-out parameter
/// * `params` - Shared walk and training parameters, as a dict
///
/// ### Returns
///
/// `(n_nodes, dim)` float32 embedding, row `i` for node index `i`.
#[pyfunction]
#[pyo3(signature = (src, dst, weight, n_nodes, *, directed, p, q, params))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_node2vec<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weight: Option<PyReadonlyArray1<f32>>,
    n_nodes: usize,
    directed: bool,
    p: f32,
    q: f32,
    params: Params,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let edges = edges(&src, &dst, weight.as_ref())?;

    let data = py.detach(|| {
        in_pool(params.n_threads, || {
            let graph = Node2VecGraph::from_edges(&edges, directed, p, q);
            let walks = graph.generate_walks(
                params.walks_per_node,
                params.walk_length,
                params.seed,
                params.verbose,
            );
            let neg_table = NegativeTable::Global(create_negative_table(
                n_nodes,
                &walks,
                NEGATIVE_TABLE_SIZE,
                params.seed,
            ));
            fit(walks, n_nodes, neg_table, &params)
        })
    })?;

    to_array(py, data, params.dim)
}

/// metapath2vec, or metapath2vec++, on a heterogeneous graph.
///
/// ### Params
///
/// * `src` - Source node index per edge, into `node_ids`
/// * `dst` - Target node index per edge
/// * `weight` - Weight per edge, or `None` for an unweighted graph
/// * `node_ids` - Node id per node index, as strings
/// * `node_types` - Node type per node index
/// * `metapath` - Schema such as `"gene-pathway-gene"`
/// * `plus` - Per-type negative sampling, i.e. metapath2vec++
/// * `directed` - Keep edges one-way
/// * `params` - Shared walk and training parameters, as a dict
///
/// ### Returns
///
/// `(embedding, node_ids, walk_stats)`. Rows follow the core's dense order,
/// contiguous by type, which is **not** the input order; `node_ids` names each
/// row.
#[pyfunction]
#[pyo3(signature = (src, dst, weight, node_ids, node_types, metapath, *, plus, directed, params))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(crate) fn run_metapath2vec<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<u32>,
    dst: PyReadonlyArray1<u32>,
    weight: Option<PyReadonlyArray1<f32>>,
    node_ids: Vec<String>,
    node_types: Vec<String>,
    metapath: &str,
    plus: bool,
    directed: bool,
    params: Params,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Vec<String>, Bound<'py, PyDict>)> {
    let weighted = weight.is_some();
    let edges = edges(&src, &dst, weight.as_ref())?;
    let nodes: Vec<(String, String)> = node_ids.into_iter().zip(node_types).collect();

    let (data, names, stats) = py
        .detach(|| {
            in_pool(params.n_threads, || -> Result<_, CoreError> {
                let graph = HetGraph::from_edges(nodes, edges, weighted, directed)?;
                let schema = Metapath::parse(metapath, graph.type_names())?;
                let (walks, stats) = graph.generate_walks(
                    &schema,
                    params.walks_per_node,
                    params.walk_length,
                    params.seed,
                    params.verbose,
                )?;
                let neg_table = if plus {
                    NegativeTable::PerType {
                        tables: create_negative_table_per_type(
                            graph.node_types(),
                            graph.n_types(),
                            &walks,
                            NEGATIVE_TABLE_SIZE,
                            params.seed,
                        ),
                        node_type: Arc::new(graph.node_types().to_vec()),
                    }
                } else {
                    NegativeTable::Global(create_negative_table(
                        graph.n_nodes(),
                        &walks,
                        NEGATIVE_TABLE_SIZE,
                        params.seed,
                    ))
                };
                let data = fit(walks, graph.n_nodes(), neg_table, &params);
                Ok((data, graph.node_names().to_vec(), stats))
            })
        })?
        .map_err(to_py)?;

    let out = PyDict::new(py);
    out.set_item("start_nodes", stats.start_nodes)?;
    out.set_item("attempted", stats.attempted)?;
    out.set_item("truncated", stats.truncated)?;
    out.set_item("dropped", stats.dropped)?;
    out.set_item("mean_length", stats.mean_length)?;
    out.set_item("walk_length", stats.walk_length)?;

    Ok((to_array(py, data, params.dim)?, names, out))
}
