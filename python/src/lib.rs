//! Python bindings for `node2vec-rs`.
//!
//! Deliberately thin: two one-shot functions over the hogwild CPU trainer that
//! return numpy arrays. Id interning, defaults and validation live in
//! `node2vec_rs/`; this side only builds the graph, walks and trains.
//!
//! Everything expensive runs inside `Python::detach`, so the GIL is dropped
//! for the whole rayon fan-out.

#![warn(missing_docs)]

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod train;

create_exception!(
    _node2vec_rs,
    Node2VecError,
    PyValueError,
    "Raised when the node2vec-rs core rejects its input, e.g. a bad metapath."
);

/// Assemble the compiled module.
#[pymodule]
fn _node2vec_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("Node2VecError", m.py().get_type::<Node2VecError>())?;
    m.add_function(wrap_pyfunction!(train::run_node2vec, m)?)?;
    m.add_function(wrap_pyfunction!(train::run_metapath2vec, m)?)?;
    Ok(())
}
