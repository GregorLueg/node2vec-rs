mod batch;
mod dataset;
mod graph;
mod model;
mod reader;
mod train;

use burn::backend::{Autodiff, NdArray, Wgpu};
use model::SkipGramConfig;
use reader::read_graph;
use train::{TrainingConfig, train};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let graph =
        read_graph("test/data/test_graph.csv", false, 1.0, 1.0).expect("Failed to read graph");

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;

    let training_config = TrainingConfig::new();
    let walks = graph.generate_walks(
        training_config.walks_per_node,
        training_config.walk_length,
        training_config.seed,
    );

    let split_idx = (walks.len() as f32 * 0.9) as usize;
    let train_walks = walks[..split_idx].to_vec();
    let valid_walks = walks[split_idx..].to_vec();

    let model_config = SkipGramConfig::new(vocab_size as usize, 16);

    train::<MyAutodiffBackend>(
        "/tmp/node2vec",
        model_config,
        training_config,
        train_walks,
        valid_walks,
        device,
    );
}
