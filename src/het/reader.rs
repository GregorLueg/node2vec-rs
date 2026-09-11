//! Reading a heterogeneous graph from a node table and an edge table.
//!
//! Heterogeneous node ids are strings namespaced per type, `GO:0006915` next
//! to `ENSG00000141510`, so unlike [`crate::reader::read_graph`] this interns
//! them rather than parsing them as `u32`. Interned ids are sorted contiguous
//! by type, which earns its keep twice: per-type negative tables become a
//! range rather than a filter, and "every node of type `t`" is a range check.

use csv::Reader;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::File;

use crate::errors::Node2VecError;
use crate::het::HetGraph;

/// Node types are interned into a `u8`.
const MAX_TYPES: usize = u8::MAX as usize + 1;

//////////
// Main //
//////////

/// Read a heterogeneous graph from a node table and an edge table.
///
/// The node table is `id,type`; the edge table is `from,to[,weight]`. Both are
/// read positionally with a header row, matching the convention in
/// [`crate::reader::read_graph`], so the header names are documentation.
/// A missing weight column makes the graph unweighted and collapses neighbour
/// sampling to a single uniform draw.
///
/// ### Params
///
/// * `node_path` - Path to the node table
/// * `edge_path` - Path to the edge table
/// * `directed` - When false, every edge is inserted in both directions
///
/// ### Returns
///
/// The typed CSR, with dense ids assigned contiguously by node type.
pub fn read_het_graph(
    node_path: &str,
    edge_path: &str,
    directed: bool,
) -> Result<HetGraph, Node2VecError> {
    let nodes = read_node_table(node_path)?;
    let index: FxHashMap<&str, u32> = nodes
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id.as_str(), i as u32))
        .collect();

    let mut rdr = Reader::from_reader(File::open(edge_path)?);
    let mut edges = Vec::new();
    let mut weighted = false;

    for (row, result) in rdr.records().enumerate() {
        let record = result?;
        // Header is row zero in the file, so the first record is line two.
        let line = row + 2;

        let resolve = |field: &str| {
            index
                .get(field)
                .copied()
                .ok_or_else(|| Node2VecError::UnknownNode {
                    id: field.to_string(),
                    line,
                })
        };
        let from = resolve(&record[0])?;
        let to = resolve(&record[1])?;

        let weight = match record.get(2).map(str::parse::<f32>) {
            Some(Ok(w)) => {
                weighted = true;
                w
            }
            _ => 1.0,
        };

        edges.push((from, to, weight));
        if !directed {
            edges.push((to, from, weight));
        }
    }

    build_het_graph(nodes, edges, weighted)
}

impl HetGraph {
    /// Build a heterogeneous graph from in-memory node and edge lists.
    ///
    /// The counterpart to [`read_het_graph`] for callers that already hold the
    /// tables, e.g. an FFI layer.
    ///
    /// ### Params
    ///
    /// * `nodes` - `(id, type)` pairs; their order defines the node indices
    /// * `edges` - `(from, to, weight)` with zero-based indices into `nodes`
    /// * `weighted` - Whether `weight` carries information
    /// * `directed` - When false, every edge is inserted in both directions
    ///
    /// ### Returns
    ///
    /// The typed CSR, with dense ids assigned contiguously by node type, so
    /// the row order of any embedding follows [`HetGraph::node_names`], not
    /// the input order.
    pub fn from_edges(
        nodes: Vec<(String, String)>,
        edges: Vec<(u32, u32, f32)>,
        weighted: bool,
        directed: bool,
    ) -> Result<Self, Node2VecError> {
        if nodes.is_empty() {
            return Err(Node2VecError::EmptyNodeTable {
                path: "<memory>".to_string(),
            });
        }

        let n_nodes = nodes.len();
        if let Some(&index) = edges
            .iter()
            .flat_map(|(from, to, _)| [from, to])
            .find(|&&i| i as usize >= n_nodes)
        {
            return Err(Node2VecError::EdgeOutOfRange { index, n_nodes });
        }

        let edges = if directed {
            edges
        } else {
            edges
                .into_iter()
                .flat_map(|(from, to, w)| [(from, to, w), (to, from, w)])
                .collect()
        };

        build_het_graph(nodes, edges, weighted)
    }
}

/// Read the node table into `(id, type)` pairs in file order.
///
/// ### Params
///
/// * `path` - Path to the node table
///
/// ### Returns
///
/// The rows in file order; interning and sorting happen in
/// [`build_het_graph`].
fn read_node_table(path: &str) -> Result<Vec<(String, String)>, Node2VecError> {
    let mut rdr = Reader::from_reader(File::open(path)?);
    let mut nodes = Vec::new();

    for result in rdr.records() {
        let record = result?;
        nodes.push((record[0].to_string(), record[1].to_string()));
    }

    if nodes.is_empty() {
        return Err(Node2VecError::EmptyNodeTable {
            path: path.to_string(),
        });
    }

    Ok(nodes)
}

/// Build the typed CSR from resolved node and edge lists.
///
/// Split out from [`read_het_graph`] so the CSR construction is testable
/// without touching the filesystem.
///
/// ### Params
///
/// * `nodes` - `(id, type)` pairs in file order
/// * `edges` - `(from, to, weight)` with endpoints as **file-order** node
///   indices; both directions must already be present for an undirected graph
/// * `weighted` - Whether `weight` carries information
///
/// ### Returns
///
/// The graph with dense ids reassigned contiguously by node type.
pub(crate) fn build_het_graph(
    nodes: Vec<(String, String)>,
    edges: Vec<(u32, u32, f32)>,
    weighted: bool,
) -> Result<HetGraph, Node2VecError> {
    let n_nodes = nodes.len();

    // Intern type strings in order of first appearance.
    let mut type_names: Vec<String> = Vec::new();
    let mut type_ids: FxHashMap<&str, u8> = FxHashMap::default();
    let mut file_order_type = Vec::with_capacity(n_nodes);
    for (_, type_name) in &nodes {
        let id = match type_ids.get(type_name.as_str()) {
            Some(&id) => id,
            None => {
                if type_names.len() == MAX_TYPES {
                    return Err(Node2VecError::TooManyTypes {
                        found: type_names.len() + 1,
                    });
                }
                let id = type_names.len() as u8;
                type_names.push(type_name.clone());
                type_ids.insert(type_name.as_str(), id);
                id
            }
        };
        file_order_type.push(id);
    }
    let n_types = type_names.len();

    // Sort node ids contiguous by type, stable so file order survives within a
    // type. `remap` takes a file-order index to its dense id.
    let mut order: Vec<u32> = (0..n_nodes as u32).collect();
    order.sort_by_key(|&i| file_order_type[i as usize]);
    let mut remap = vec![0u32; n_nodes];
    for (dense, &old) in order.iter().enumerate() {
        remap[old as usize] = dense as u32;
    }

    let node_type: Vec<u8> = order.iter().map(|&i| file_order_type[i as usize]).collect();
    let node_names: Vec<String> = order.iter().map(|&i| nodes[i as usize].0.clone()).collect();

    // Sorting the edge list by (source, target type) once means the CSR fill
    // is a single pass and every row comes out already grouped by type.
    let mut edges: Vec<(u32, u32, f32)> = edges
        .into_iter()
        .map(|(from, to, w)| (remap[from as usize], remap[to as usize], w))
        .collect();
    // The third key makes this a total order, so the CSR layout, and with it
    // walk reproducibility, does not depend on how rayon split the sort.
    edges.par_sort_unstable_by_key(|&(from, to, _)| (from, node_type[to as usize], to));

    let mut offsets = vec![0u32; n_nodes + 1];
    for &(from, _, _) in &edges {
        offsets[from as usize + 1] += 1;
    }
    for i in 0..n_nodes {
        offsets[i + 1] += offsets[i];
    }

    let targets: Vec<u32> = edges.iter().map(|&(_, to, _)| to).collect();

    let mut type_starts = vec![0u32; n_nodes * n_types + 1];
    type_starts[n_nodes * n_types] = targets.len() as u32;
    type_starts[..n_nodes * n_types]
        .par_chunks_mut(n_types)
        .enumerate()
        .for_each(|(node, row_starts)| {
            let end = offsets[node + 1] as usize;
            let mut cursor = offsets[node] as usize;
            for (t, start) in row_starts.iter_mut().enumerate() {
                *start = cursor as u32;
                while cursor < end && node_type[targets[cursor] as usize] as usize == t {
                    cursor += 1;
                }
            }
        });

    let cum_weights = if weighted {
        let raw: Vec<f32> = edges.iter().map(|&(_, _, w)| w).collect();
        cumulative_by_run(&raw, &type_starts, n_nodes, n_types)
    } else {
        Vec::new()
    };

    Ok(HetGraph {
        node_type,
        targets,
        cum_weights,
        type_starts,
        n_types,
        weighted,
        node_names,
        type_names,
    })
}

/// Turn raw edge weights into a cumulative distribution per `(node, type)` run.
///
/// Each run is normalised to end at 1.0 so sampling is a binary search against
/// a single `[0, 1)` draw, mirroring
/// [`crate::graph::Node2VecGraph::sample_from_cumulative`].
///
/// ### Params
///
/// * `raw` - Edge weights in CSR order
/// * `type_starts` - Run boundaries
/// * `n_nodes` - Node count
/// * `n_types` - Type count
///
/// ### Returns
///
/// Cumulative weights in CSR order.
fn cumulative_by_run(raw: &[f32], type_starts: &[u32], n_nodes: usize, n_types: usize) -> Vec<f32> {
    let mut cumulative = raw.to_vec();
    for node in 0..n_nodes {
        for t in 0..n_types {
            let base = node * n_types + t;
            let run = type_starts[base] as usize..type_starts[base + 1] as usize;
            if run.is_empty() {
                continue;
            }
            let total: f32 = raw[run.clone()].iter().sum();
            let mut acc = 0.0;
            for slot in &mut cumulative[run] {
                acc += *slot / total;
                *slot = acc;
            }
        }
    }
    cumulative
}

///////////
// Tests //
///////////

#[cfg(test)]
mod het_reader_tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    /// Write a node and an edge table, returning both paths.
    fn fixture(node_body: &str, edge_body: &str) -> (tempfile::TempDir, String, String) {
        let dir = tempdir().unwrap();
        let node_path = dir.path().join("nodes.csv");
        let edge_path = dir.path().join("edges.csv");
        write!(File::create(&node_path).unwrap(), "{node_body}").unwrap();
        write!(File::create(&edge_path).unwrap(), "{edge_body}").unwrap();
        let n = node_path.to_str().unwrap().to_string();
        let e = edge_path.to_str().unwrap().to_string();
        (dir, n, e)
    }

    #[test]
    fn test_string_ids_resolve_and_sort_by_type() {
        // Deliberately interleaved in the file so the sort has work to do.
        let (_d, n, e) = fixture(
            "id,type\nENSG1,gene\nGO:1,term\nENSG2,gene\nGO:2,term\n",
            "from,to\nENSG1,GO:1\nENSG2,GO:2\n",
        );
        let g = read_het_graph(&n, &e, false).unwrap();

        assert_eq!(g.n_nodes(), 4);
        assert_eq!(g.n_types(), 2);
        assert_eq!(g.node_types(), &[0, 0, 1, 1]);
        assert_eq!(g.node_names(), &["ENSG1", "ENSG2", "GO:1", "GO:2"]);
        assert_eq!(g.type_names(), &["gene", "term"]);
    }

    #[test]
    fn test_undirected_doubles_the_edges() {
        let (_d, n, e) = fixture("id,type\na1,a\nb1,b\n", "from,to\na1,b1\n");
        assert_eq!(read_het_graph(&n, &e, false).unwrap().n_edges(), 2);
        assert_eq!(read_het_graph(&n, &e, true).unwrap().n_edges(), 1);
    }

    #[test]
    fn test_unknown_endpoint_names_the_id_and_line() {
        let (_d, n, e) = fixture("id,type\na1,a\nb1,b\n", "from,to\na1,b1\na1,nope\n");
        match read_het_graph(&n, &e, false) {
            Err(Node2VecError::UnknownNode { id, line }) => {
                assert_eq!(id, "nope");
                assert_eq!(line, 3);
            }
            other => panic!("expected UnknownNode, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_weight_column_is_unweighted() {
        let (_d, n, e) = fixture("id,type\na1,a\nb1,b\n", "from,to\na1,b1\n");
        let g = read_het_graph(&n, &e, false).unwrap();
        assert!(!g.weighted);
        assert!(g.cum_weights.is_empty());

        let (_d, n, e) = fixture("id,type\na1,a\nb1,b\n", "from,to,weight\na1,b1,2.5\n");
        let g = read_het_graph(&n, &e, false).unwrap();
        assert!(g.weighted);
        assert_eq!(g.cum_weights.len(), g.n_edges());
    }

    #[test]
    fn test_empty_node_table_errors() {
        let (_d, n, e) = fixture("id,type\n", "from,to\n");
        assert!(matches!(
            read_het_graph(&n, &e, false),
            Err(Node2VecError::EmptyNodeTable { .. })
        ));
    }

    #[test]
    fn test_type_runs_bracket_the_neighbours() {
        // a1 links to one a and two b nodes; the runs must split them.
        let (_d, n, e) = fixture(
            "id,type\na1,a\na2,a\nb1,b\nb2,b\n",
            "from,to\na1,a2\na1,b1\na1,b2\n",
        );
        let g = read_het_graph(&n, &e, true).unwrap();

        let of_a = g.neighbours_of_type(0, 0);
        let of_b = g.neighbours_of_type(0, 1);
        assert_eq!(of_a.len(), 1);
        assert_eq!(of_b.len(), 2);
        assert_eq!(g.targets[of_a][0], 1);
        let mut bs: Vec<u32> = g.targets[of_b].to_vec();
        bs.sort_unstable();
        assert_eq!(bs, vec![2, 3]);

        // A node with no neighbours of a type yields an empty run, not a panic.
        assert!(g.neighbours_of_type(1, 0).is_empty());
    }

    fn mem_nodes() -> Vec<(String, String)> {
        [("a1", "a"), ("b1", "b"), ("a2", "a")]
            .iter()
            .map(|(i, t)| (i.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn test_from_edges_matches_reader_layout() {
        let g = HetGraph::from_edges(mem_nodes(), vec![(0, 1, 1.0), (2, 1, 1.0)], false, false)
            .unwrap();
        assert_eq!(g.n_edges(), 4);
        assert_eq!(g.node_types(), &[0, 0, 1]);
        assert_eq!(g.node_names(), &["a1", "a2", "b1"]);

        let g = HetGraph::from_edges(mem_nodes(), vec![(0, 1, 1.0)], false, true).unwrap();
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn test_from_edges_rejects_out_of_range() {
        match HetGraph::from_edges(mem_nodes(), vec![(0, 3, 1.0)], false, false) {
            Err(Node2VecError::EdgeOutOfRange { index, n_nodes }) => {
                assert_eq!(index, 3);
                assert_eq!(n_nodes, 3);
            }
            other => panic!("expected EdgeOutOfRange, got {other:?}"),
        }
        assert!(matches!(
            HetGraph::from_edges(Vec::new(), Vec::new(), false, false),
            Err(Node2VecError::EmptyNodeTable { .. })
        ));
    }

    #[test]
    fn test_cumulative_runs_end_at_one() {
        let (_d, n, e) = fixture(
            "id,type\na1,a\nb1,b\nb2,b\n",
            "from,to,weight\na1,b1,3.0\na1,b2,1.0\n",
        );
        let g = read_het_graph(&n, &e, true).unwrap();
        let run = g.neighbours_of_type(0, 1);
        assert_eq!(run.len(), 2);
        assert!((g.cum_weights[run.end - 1] - 1.0).abs() < 1e-6);
        assert!((g.cum_weights[run.start] - 0.75).abs() < 1e-6);
    }
}
