//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)] // I want these loops!
#![warn(missing_docs)]

pub mod graph;
pub mod reader;

#[cfg(feature = "burn")]
pub mod burn;

#[cfg(feature = "cpu")]
pub mod cpu;
pub mod prelude;

use clap::Parser;

///////////////
// Arguments //
///////////////

/// CLI arguments for `node2vec-rs`
#[derive(Parser)]
#[command(name = "node2vec")]
#[command(about = "Node2Vec implementation using Burn", long_about = None)]
pub struct Args {
    /// The input CSV. Needs to be provided.
    #[arg(short, long)]
    pub input: String,

    /// Where to store the outputs. Defaults to `"/tmp/node2vec"`.
    #[arg(short, long, default_value = "/tmp/node2vec")]
    pub output: String,

    /// Backend to use. One of `"cpu"` or `"burn"`
    #[arg(long, default_value = "cpu")]
    pub backend: String,

    /// Shall the graph be treated as a directed graph. Defaults to `false`.
    #[arg(short, long, default_value_t = false)]
    pub directed: bool,

    /// Size of the embedding to create. Defaults to `16`.
    #[arg(short, long, default_value_t = 16)]
    pub embedding_dim: usize,

    /// How much of the data should be in the trainings data vs. validation
    /// data. Defaults to `0.9`.
    #[arg(short, long, default_value_t = 0.9)]
    pub split: f32,

    /// Number of random walks to do per node. Defaults to `20`.
    #[arg(long, default_value_t = 20)]
    pub walks_per_node: usize,

    /// Length of the random walks. Defaults to `20`.
    #[arg(long, default_value_t = 20)]
    pub walk_length: usize,

    /// Window size parameter for the skipgram model. Defaults to `2`.
    #[arg(long, default_value_t = 2)]
    pub window_size: usize,

    /// Batch size during training. Defaults to `256`.
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    /// Number of workers to use during the generation of the batches. Defaults
    /// to `4`.
    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,

    /// Number of epochs to train the model for. Defaults to `5`.
    #[arg(long, default_value_t = 5)]
    pub num_epochs: usize,

    /// Number of negative examples to sample. Defaults to `5`.
    #[arg(long, default_value_t = 5)]
    pub num_negatives: usize,

    /// Seed for reproducibility. Defaults to `42`.
    #[arg(long, default_value_t = 42)]
    pub seed: usize,

    /// Learning rate for the Adam optimiser. Defaults to `1-e3`.
    #[arg(long, default_value_t = 1.0e-3)]
    pub learning_rate: f64,

    /// p parameter for the node2vec random walks and controls the probability
    /// to return to origin node. Defaults to `1.0`.
    #[arg(long, default_value_t = 1.0)]
    pub p: f32,

    /// q parameter for node2vec random walks and controls the probability
    /// to venture on a different node from the origin node. Defaults to `1.0`.
    #[arg(long, default_value_t = 1.0)]
    pub q: f32,

    /// Subsampling threshold; nodes with frequency above this are randomly
    /// dropped during training.
    #[arg(long, default_value_t = 1.0e-3)]
    pub sample: f32,
}
