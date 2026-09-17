//! Useful stuff for across the crate.

pub use crate::errors::Node2VecError;
pub use crate::graph::{compute_transition_prob, Node2VecGraph};
pub use crate::het::reader::read_het_graph;
pub use crate::het::{HetGraph, Metapath, WalkStats};
pub use crate::reader::read_graph;
pub use crate::Args;

#[cfg(feature = "burn")]
pub use crate::burn::batch::SkipGramBatcher;
#[cfg(feature = "burn")]
pub use crate::burn::dataset::WalkDataset;
#[cfg(feature = "burn")]
pub use crate::burn::model::SkipGramConfig;
#[cfg(feature = "burn")]
pub use crate::burn::train::{sample_negatives, train, TrainingConfig};

#[cfg(feature = "cpu")]
pub use crate::cpu::train::{
    create_negative_table, create_negative_table_per_type, train_node2vec_cpu, CpuTrainArgs,
    NegativeTable,
};
#[cfg(feature = "cpu")]
pub use crate::cpu::NEGATIVE_TABLE_SIZE;
