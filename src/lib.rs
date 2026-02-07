#![allow(clippy::needless_range_loop)]

pub mod burn;
pub mod cpu;
pub mod prelude;

use clap::Parser;

///////////////
// Arguments //
///////////////

/// CLI arguments
///
/// ### Fields
///
/// * `input` - The input CSV. Needs to be provided.
/// * `output` - Where to store the outputs. Defaults to `"/tmp/node2vec"`.
/// * `directed` - Shall the graph be treated as a directed graph. Defaults
///   to `false`.
/// * `embedding_dim` - Size of the embedding to create. Defaults to `16`.
/// * `split` - How much of the data should be in the trainings data vs.
///   validation data. Defaults to `0.9`.
/// * `walks_per_node` - Number of random walks to do per node. Defaults to
///   `20`.
/// * `walk_length` - Length of the random walks. Defaults to `20`.
/// * `window_size` - Window size parameter for the skipgram model. Defaults to
///   `2`.
/// * `batch_size` - Batch size during training. Defaults to `256`.
/// * `num_workers` - Number of workers to use during the generation of the
///   batches. Defaults to `4`.
/// * `num_epochs` - Number of epochs to train the model for. Defaults to `5`.
/// * `num_negatives` - Number of negative examples to sample. Defaults to `5`.
/// * `seed` - Seed for reproducibility. Defaults to `42`.
/// * `learning_rate` - Learning rate for the Adam optimiser. Defaults to
///   `1-e3`.
/// * `p` - p parameter for the node2vec random walks and controls the
///   probability to return to origin node. Defaults to `1.0`.
/// * `q` - q parameter for node2vec random walks and controls the probability
///   to venture on a different node from the origin node. Defaults to `1.0`.
#[derive(Parser)]
#[command(name = "node2vec")]
#[command(about = "Node2Vec implementation using Burn", long_about = None)]
pub struct Args {
    #[arg(short, long)]
    pub input: String,

    #[arg(short, long, default_value = "/tmp/node2vec")]
    pub output: String,

    #[arg(short, long, default_value_t = false)]
    pub directed: bool,

    #[arg(short, long, default_value_t = 16)]
    pub embedding_dim: usize,

    #[arg(short, long, default_value_t = 0.9)]
    pub split: f32,

    #[arg(long, default_value_t = 20)]
    pub walks_per_node: usize,

    #[arg(long, default_value_t = 20)]
    pub walk_length: usize,

    #[arg(long, default_value_t = 2)]
    pub window_size: usize,

    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,

    #[arg(long, default_value_t = 5)]
    pub num_epochs: usize,

    #[arg(long, default_value_t = 5)]
    pub num_negatives: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 1.0e-3)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 1.0)]
    pub p: f32,

    #[arg(long, default_value_t = 1.0)]
    pub q: f32,
}
