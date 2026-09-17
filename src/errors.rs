//! Crate-level error type.

use thiserror::Error;

/// Errors returned by `node2vec-rs`.
#[derive(Debug, Error)]
pub enum Node2VecError {
    /// Underlying IO failure while reading a table or writing embeddings.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Malformed CSV.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// A node id in the edge table has no entry in the node table.
    #[error("Unknown node id '{id}' in the edge table on line {line}.")]
    UnknownNode {
        /// The string id that could not be resolved
        id: String,
        /// One-based line number in the edge table, header included
        line: usize,
    },

    /// A homogeneous edge table held something that is not a `u32`.
    #[error("Cannot cast '{value}' to u32 on line {line}.")]
    ParseNodeId {
        /// The offending field
        value: String,
        /// One-based line number, header included
        line: usize,
    },

    /// The metapath names a type that is absent from the node table.
    #[error("Unknown node type '{name}' in the metapath. Known types: {known}.")]
    UnknownType {
        /// The type name that could not be resolved
        name: String,
        /// Comma-separated list of the types the node table did define
        known: String,
    },

    /// Node types are interned into a `u8`, which caps them at 256.
    #[error("The node table holds {found} distinct types; at most 256 are supported.")]
    TooManyTypes {
        /// Number of distinct type strings found
        found: usize,
    },

    /// The metapath is not a valid cyclic schema.
    #[error("Invalid metapath: {reason}")]
    InvalidMetapath {
        /// Why the schema was rejected
        reason: String,
    },

    /// No node carries the metapath's starting type, so no walk can begin.
    ///
    /// Distinct from a graph that merely dead-ends: here there is nothing to
    /// walk from at all, which is almost always a typo in the schema.
    #[error("No node has the metapath's starting type '{type_name}'.")]
    EmptyStartSet {
        /// The starting type that matched no node
        type_name: String,
    },

    /// An in-memory edge references a node index outside the node table.
    #[error("Edge endpoint {index} is out of range for {n_nodes} nodes.")]
    EdgeOutOfRange {
        /// The offending zero-based node index
        index: u32,
        /// Number of nodes in the node table
        n_nodes: usize,
    },

    /// The node table was empty.
    #[error("The node table '{path}' contained no rows.")]
    EmptyNodeTable {
        /// Path that was read
        path: String,
    },
}
