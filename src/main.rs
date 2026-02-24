use node2vec_rs::prelude::*;

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };
    use node2vec_rs::prelude::*;

    pub fn run(
        output: &str,
        model_config: &SkipGramConfig,
        training_config: &TrainingConfig,
        train_walks: &[Vec<u32>],
        valid_walks: &[Vec<u32>],
        seed: &usize,
    ) {
        use burn::prelude::Backend;
        let device = LibTorchDevice::Cpu;
        LibTorch::<f32>::seed(&device, *seed as u64);
        let model = train::<Autodiff<LibTorch>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
            *seed,
        );
        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

#[cfg(feature = "tch-mps")]
mod tch_mps {
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };
    use node2vec_rs::prelude::*;

    pub fn run(
        output: &str,
        model_config: &SkipGramConfig,
        training_config: &TrainingConfig,
        train_walks: &[Vec<u32>],
        valid_walks: &[Vec<u32>],
        seed: &usize,
    ) {
        use burn::prelude::Backend;
        let device = LibTorchDevice::Mps;
        LibTorch::<f32>::seed(&device, *seed as u64);
        let model = train::<Autodiff<LibTorch>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
            *seed,
        );
        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };
    use node2vec_rs::prelude::*;

    pub fn run(
        output: &str,
        model_config: &SkipGramConfig,
        training_config: &TrainingConfig,
        train_walks: &[Vec<u32>],
        valid_walks: &[Vec<u32>],
        seed: &usize,
    ) {
        use burn::prelude::Backend;
        let device = WgpuDevice::default();
        Wgpu::<f32>::seed(&device, *seed as u64);
        let model = train::<Autodiff<Wgpu>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
            *seed,
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
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };
    use node2vec_rs::prelude::*;

    pub fn run(
        output: &str,
        model_config: &SkipGramConfig,
        training_config: &TrainingConfig,
        train_walks: &[Vec<u32>],
        valid_walks: &[Vec<u32>],
        seed: &usize,
    ) {
        use burn::prelude::Backend;
        let device = NdArrayDevice::Cpu;
        NdArray::<f32>::seed(&device, *seed as u64);
        let model = train::<Autodiff<NdArray>>(
            output,
            model_config,
            training_config,
            train_walks,
            valid_walks,
            device,
            *seed,
        );
        let embeddings_path = std::path::Path::new(output).join("embeddings.csv");
        model
            .write_embeddings_csv(embeddings_path.to_str().unwrap())
            .expect("Failed to write embeddings");
    }
}

fn main() {
    use clap::Parser;

    let args = Args::parse();

    let graph =
        read_graph(&args.input, args.directed, args.p, args.q).expect("Failed to read graph");

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;
    let seed = args.seed;

    let walks = graph.generate_walks(args.walks_per_node, args.walk_length, seed);

    match args.backend.as_str() {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                use node2vec_rs::cpu::train::{
                    create_negative_table, train_node2vec_cpu, CpuTrainArgs,
                };

                let neg_table = create_negative_table(
                    vocab_size as usize,
                    &walks,
                    node2vec_rs::cpu::NEGATIVE_TABLE_SIZE,
                    seed as usize,
                );

                let cpu_args = CpuTrainArgs {
                    dim: args.embedding_dim,
                    lr: args.learning_rate as f32,
                    epochs: args.num_epochs,
                    neg: args.num_negatives,
                    window: args.window_size,
                    lr_update_rate: 10_000,
                    n_threads: args.num_workers,
                    verbose: true,
                    sample: args.sample,
                };

                let (mut input_mat, _output_mat) = train_node2vec_cpu(
                    walks,
                    vocab_size as usize,
                    cpu_args,
                    neg_table,
                    seed as usize,
                );

                input_mat.norm_self();

                std::fs::create_dir_all(&args.output).ok();
                let embeddings_path = std::path::Path::new(&args.output).join("embeddings.csv");
                input_mat
                    .write_csv(embeddings_path.to_str().unwrap())
                    .expect("Failed to write embeddings");
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("Binary not compiled with the `cpu` feature");
            }
        }

        "burn" => {
            #[cfg(feature = "burn")]
            {
                let training_config = TrainingConfig::from_args(&args);
                let split_idx = (walks.len() as f32 * args.split) as usize;
                let train_walks = walks[..split_idx].to_vec();
                let valid_walks = walks[split_idx..].to_vec();
                let model_config = SkipGramConfig::new(vocab_size as usize, args.embedding_dim);

                #[cfg(feature = "tch-cpu")]
                tch_cpu::run(
                    &args.output,
                    &model_config,
                    &training_config,
                    &train_walks,
                    &valid_walks,
                    &seed,
                );

                #[cfg(feature = "tch-mps")]
                tch_mps::run(
                    &args.output,
                    &model_config,
                    &training_config,
                    &train_walks,
                    &valid_walks,
                    &seed,
                );

                #[cfg(any(
                    feature = "ndarray",
                    feature = "ndarray-blas-openblas",
                    feature = "ndarray-blas-accelerate"
                ))]
                ndarray::run(
                    &args.output,
                    &model_config,
                    &training_config,
                    &train_walks,
                    &valid_walks,
                    &seed,
                );

                #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
                wgpu::run(
                    &args.output,
                    &model_config,
                    &training_config,
                    &train_walks,
                    &valid_walks,
                    &seed,
                );
            }
            #[cfg(not(feature = "burn"))]
            {
                panic!("Binary not compiled with the `burn` feature");
            }
        }

        other => panic!("Unknown backend: {other}. Use 'cpu' or 'burn'."),
    }
}
