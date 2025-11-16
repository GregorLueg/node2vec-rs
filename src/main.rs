mod batch;
mod dataset;
mod graph;
mod model;
mod reader;
mod train;

use burn::backend::{Autodiff, LibTorch};
use clap::Parser;
use model::SkipGramConfig;
use reader::read_graph;
use train::{Args, TrainingConfig, train};

fn main() {
    let args = Args::parse();

    type MyBackend = LibTorch<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::libtorch::LibTorchDevice::Cpu;

    let training_config = TrainingConfig::from_args(&args);

    let graph = read_graph(
        &args.input,
        args.directed,
        training_config.p,
        training_config.q,
    )
    .expect("Failed to read graph");

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;

    let walks = graph.generate_walks(
        training_config.walks_per_node,
        training_config.walk_length,
        training_config.seed,
    );

    let split_idx = (walks.len() as f32 * args.split) as usize;
    let train_walks = walks[..split_idx].to_vec();
    let valid_walks = walks[split_idx..].to_vec();

    let model_config = SkipGramConfig::new(vocab_size as usize, args.embedding_dim);

    train::<MyAutodiffBackend>(
        &args.output,
        model_config,
        training_config,
        train_walks,
        valid_walks,
        device,
    );
}
