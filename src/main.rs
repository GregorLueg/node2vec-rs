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

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::{
        flex::{Flex, FlexDevice},
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
        let device = FlexDevice;
        Flex::<f32>::seed(&device, *seed as u64);
        let model = train::<Autodiff<Flex>>(
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

/////////////
// Helpers //
/////////////

/// Report a failure the way a CLI should and stop.
///
/// The crate's errors carry a usable `Display`, so a panic backtrace would only
/// bury the message that matters.
///
/// ### Params
///
/// * `err` - The failure to report
fn fail(err: impl std::fmt::Display) -> ! {
    eprintln!("error: {err}");
    std::process::exit(1);
}

/// Write the dense id to string id mapping next to the embeddings.
///
/// `Matrix::write_csv` emits one row per node with no id column, so on the
/// heterogeneous path, where ids were interned from strings, the embeddings are
/// unreadable without this.
///
/// ### Params
///
/// * `path` - Where to write the table
/// * `graph` - The graph the ids were interned from
///
/// ### Returns
///
/// Nothing, or the IO failure.
fn write_node_table(path: &str, graph: &HetGraph) -> std::io::Result<()> {
    use std::io::{BufWriter, Write};

    let mut file = BufWriter::new(std::fs::File::create(path)?);
    writeln!(file, "row,id,type")?;
    let type_names = graph.type_names();
    for (row, (id, &node_type)) in graph
        .node_names()
        .iter()
        .zip(graph.node_types())
        .enumerate()
    {
        writeln!(file, "{row},{id},{}", type_names[node_type as usize])?;
    }
    Ok(())
}

//////////
// Main //
//////////

fn main() {
    use clap::Parser;

    let args = Args::parse();
    let seed = args.seed;

    // `--nodes` is the switch: absent, this is byte-for-byte node2vec.
    let het = match (&args.nodes, &args.metapath) {
        (Some(node_path), Some(spec)) => {
            let graph =
                read_het_graph(node_path, &args.input, args.directed).unwrap_or_else(|e| fail(e));
            let schema = Metapath::parse(spec, graph.type_names()).unwrap_or_else(|e| fail(e));
            Some((graph, schema))
        }
        (Some(_), None) => fail("--nodes needs --metapath, e.g. --metapath gene-pathway-gene"),
        (None, Some(_)) => fail("--metapath needs --nodes to supply the node types"),
        (None, None) => None,
    };

    let (walks, vocab_size, het_graph) = match het {
        Some((graph, schema)) => {
            let (walks, stats) = graph
                .generate_walks(&schema, args.walks_per_node, args.walk_length, seed)
                .unwrap_or_else(|e| fail(e));
            println!("Metapath walks: {stats}");
            let vocab_size = graph.n_nodes();
            (walks, vocab_size, Some(graph))
        }
        None => {
            let graph =
                read_graph(&args.input, args.directed, args.p, args.q).unwrap_or_else(|e| fail(e));
            let vocab_size = (graph.adjacency.keys().max().copied().unwrap_or(0) + 1) as usize;
            let walks = graph.generate_walks(args.walks_per_node, args.walk_length, seed);
            (walks, vocab_size, None)
        }
    };

    match args.backend.as_str() {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                use node2vec_rs::cpu::train::{
                    create_negative_table, create_negative_table_per_type, train_node2vec_cpu,
                    CpuTrainArgs, NegativeTable,
                };

                let neg_table = match (&het_graph, args.metapath_plus) {
                    (Some(graph), true) => NegativeTable::PerType {
                        tables: create_negative_table_per_type(
                            graph.node_types(),
                            graph.n_types(),
                            &walks,
                            node2vec_rs::cpu::NEGATIVE_TABLE_SIZE,
                            seed,
                        ),
                        node_type: std::sync::Arc::new(graph.node_types().to_vec()),
                    },
                    _ => NegativeTable::Global(create_negative_table(
                        vocab_size,
                        &walks,
                        node2vec_rs::cpu::NEGATIVE_TABLE_SIZE,
                        seed,
                    )),
                };

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

                let (mut input_mat, _output_mat) =
                    train_node2vec_cpu(walks, vocab_size, cpu_args, neg_table, seed);

                input_mat.norm_self();

                std::fs::create_dir_all(&args.output).ok();
                let out = std::path::Path::new(&args.output);
                input_mat
                    .write_csv(out.join("embeddings.csv").to_str().unwrap())
                    .expect("Failed to write embeddings");

                if let Some(graph) = &het_graph {
                    write_node_table(out.join("nodes.csv").to_str().unwrap(), graph)
                        .expect("Failed to write the node table");
                }
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("Binary not compiled with the `cpu` feature");
            }
        }

        "burn" => {
            if het_graph.is_some() {
                fail("the burn backend does not implement metapath2vec; use --backend cpu");
            }

            #[cfg(feature = "burn")]
            {
                let training_config = TrainingConfig::from_args(&args);
                let split_idx = (walks.len() as f32 * args.split) as usize;
                let train_walks = walks[..split_idx].to_vec();
                let valid_walks = walks[split_idx..].to_vec();
                let model_config = SkipGramConfig::new(vocab_size, args.embedding_dim);

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

                #[cfg(feature = "flex")]
                flex::run(
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

        other => fail(format!("unknown backend '{other}'; use 'cpu' or 'burn'")),
    }
}
