mod batch;
mod dataset;
mod graph;
mod model;
mod reader;
mod train;

use clap::Parser;
use model::SkipGramConfig;
use reader::read_graph;
use train::{train, Args, TrainingConfig};

/// Default version uses torch CPU... It's fast across most platforms
#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use super::{train, SkipGramConfig};
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };

    pub fn run(
        output: &str,
        model_config: SkipGramConfig,
        training_config: super::TrainingConfig,
        train_walks: Vec<Vec<u32>>,
        valid_walks: Vec<Vec<u32>>,
        seed: &u64,
    ) {
        use burn::prelude::Backend;

        let device = LibTorchDevice::Cpu;

        LibTorch::<f32>::seed(&device, *seed);

        let model = train::<Autodiff<LibTorch>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
        );

        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use super::{train, SkipGramConfig};
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };

    pub fn run(
        output: &str,
        model_config: SkipGramConfig,
        training_config: super::TrainingConfig,
        train_walks: Vec<Vec<u32>>,
        valid_walks: Vec<Vec<u32>>,
        seed: &u64,
    ) {
        use burn::prelude::Backend;

        let device = WgpuDevice::default();

        Wgpu::<f32>::seed(&device, *seed);

        let model = train::<Autodiff<Wgpu>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
        );

        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate"
))]
mod ndarray {
    use super::{train, SkipGramConfig};
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };

    pub fn run(
        output: &str,
        model_config: SkipGramConfig,
        training_config: super::TrainingConfig,
        train_walks: Vec<Vec<u32>>,
        valid_walks: Vec<Vec<u32>>,
        seed: &u64,
    ) {
        use burn::prelude::Backend;

        let device = NdArrayDevice::Cpu;

        NdArray::<f32>::seed(&device, *seed);

        let model = train::<Autodiff<NdArray>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
        );

        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

fn main() {
    let args = Args::parse();

    let training_config = TrainingConfig::from_args(&args);

    let graph = read_graph(
        &args.input,
        args.directed,
        training_config.p,
        training_config.q,
    )
    .expect("Failed to read graph");

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;

    let seed = training_config.seed;

    let walks = graph.generate_walks(
        training_config.walks_per_node,
        training_config.walk_length,
        training_config.seed,
    );

    let split_idx = (walks.len() as f32 * args.split) as usize;
    let train_walks = walks[..split_idx].to_vec();
    let valid_walks = walks[split_idx..].to_vec();

    let model_config = SkipGramConfig::new(vocab_size as usize, args.embedding_dim);

    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(
        &args.output,
        model_config,
        training_config,
        train_walks,
        valid_walks,
        &seed,
    );

    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate"
    ))]
    ndarray::run(
        &args.output,
        model_config,
        training_config,
        train_walks,
        valid_walks,
        &seed,
    );

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    wgpu::run(
        &args.output,
        model_config,
        training_config,
        train_walks,
        valid_walks,
        &seed,
    );
}
