use node2vec_rs::prelude::*;

const KARATE_CSV: &str = "tests/data/karate.csv";

///////////////////
// General tests //
///////////////////

#[test]
fn test_karate_graph_loads() {
    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).expect("Failed to load karate club graph");

    // Karate club has 34 nodes
    assert_eq!(graph.adjacency.len(), 34);

    // Should have edges for node 1 (most connected)
    assert!(graph.adjacency.get(&1).unwrap().len() > 10);
}

#[test]
fn test_karate_walks_generation() {
    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();

    let walks = graph.generate_walks(10, 20, 42);

    // Should generate 10 walks per node
    assert_eq!(walks.len(), 34 * 10);

    // Each walk should have length <= 20
    for walk in &walks {
        assert!(walk.len() <= 20);
        assert!(!walk.is_empty());
    }

    // All nodes in walks should be valid (1-34)
    for walk in &walks {
        for &node in walk {
            assert!((1..=34).contains(&node));
        }
    }
}

#[test]
fn test_karate_directed_vs_undirected() {
    let undirected = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();
    let directed = read_graph(KARATE_CSV, true, 1.0, 1.0).unwrap();

    // Undirected should have roughly 2x the edges
    let undirected_edges: usize = undirected.adjacency.values().map(|v| v.len()).sum();
    let directed_edges: usize = directed.adjacency.values().map(|v| v.len()).sum();

    assert!(undirected_edges > directed_edges);
    assert_eq!(directed_edges, 78); // Original edge count
}

/////////
// CPU //
/////////

#[test]
#[cfg(feature = "cpu")]
fn test_karate_cpu_end_to_end() {
    use node2vec_rs::prelude::*;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();
    let walks = graph.generate_walks(5, 10, 42);

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;
    let neg_table = NegativeTable::Global(create_negative_table(
        vocab_size as usize,
        &walks,
        NEGATIVE_TABLE_SIZE,
        42,
    ));

    let args = CpuTrainArgs {
        dim: 8,
        lr: 0.025,
        epochs: 1,
        neg: 3,
        window: 2,
        lr_update_rate: 10_000,
        n_threads,
        verbose: false,
        sample: 1e-3,
    };

    let (mut input_mat, _output_mat) =
        train_node2vec_cpu(walks, vocab_size as usize, args, neg_table, 42);

    input_mat.norm_self();

    let dir = tempfile::tempdir().unwrap();
    let csv_path = dir.path().join("embeddings.csv");
    input_mat
        .write_csv(csv_path.to_str().unwrap())
        .expect("Failed to write embeddings");

    assert!(csv_path.exists());

    // Verify shape by reading back
    let contents = std::fs::read_to_string(&csv_path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), vocab_size as usize);
    let cols: Vec<&str> = lines[0].split(',').collect();
    assert_eq!(cols.len(), 8);
}

#[test]
#[cfg(feature = "cpu")]
fn test_karate_cpu_community_structure() {
    use node2vec_rs::prelude::*;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();
    let walks = graph.generate_walks(10, 80, 42);

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;
    let neg_table = NegativeTable::Global(create_negative_table(
        vocab_size as usize,
        &walks,
        NEGATIVE_TABLE_SIZE,
        42,
    ));

    let args = CpuTrainArgs {
        dim: 16,
        lr: 0.025,
        epochs: 15,
        neg: 5,
        window: 10,
        lr_update_rate: 10_000,
        n_threads,
        verbose: true,
        sample: 1e-3,
    };

    let (input_mat, output_mat) =
        train_node2vec_cpu(walks, vocab_size as usize, args, neg_table, 42);

    let mut combined = input_mat.average_with(&output_mat);
    combined.norm_self();

    // Extract rows as vecs for similarity computation
    fn get_row(mat: &node2vec_rs::cpu::matrix::Matrix, i: usize) -> Vec<f32> {
        let slice = mat.row_as_slice(i);
        slice.to_vec()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    let faction1: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22];
    let faction2: Vec<usize> = vec![
        9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    ];

    let mut faction1_sim = Vec::new();
    for &i in &faction1 {
        for &j in &faction1 {
            if i < j {
                faction1_sim.push(cosine_similarity(
                    &get_row(&combined, i),
                    &get_row(&combined, j),
                ));
            }
        }
    }

    let mut faction2_sim = Vec::new();
    for &i in &faction2 {
        for &j in &faction2 {
            if i < j {
                faction2_sim.push(cosine_similarity(
                    &get_row(&combined, i),
                    &get_row(&combined, j),
                ));
            }
        }
    }

    let mut between_sim = Vec::new();
    for &i in &faction1 {
        for &j in &faction2 {
            between_sim.push(cosine_similarity(
                &get_row(&combined, i),
                &get_row(&combined, j),
            ));
        }
    }

    let avg_faction1: f32 = faction1_sim.iter().sum::<f32>() / faction1_sim.len() as f32;
    let avg_faction2: f32 = faction2_sim.iter().sum::<f32>() / faction2_sim.len() as f32;
    let avg_between: f32 = between_sim.iter().sum::<f32>() / between_sim.len() as f32;

    eprintln!("CPU - Faction 1 avg similarity: {:.4}", avg_faction1);
    eprintln!("CPU - Faction 2 avg similarity: {:.4}", avg_faction2);
    eprintln!("CPU - Between factions avg similarity: {:.4}", avg_between);

    assert!(
        avg_faction1 > avg_between,
        "Faction 1 internal similarity ({:.4}) should exceed between-faction ({:.4})",
        avg_faction1,
        avg_between
    );
    assert!(
        avg_faction2 > avg_between,
        "Faction 2 internal similarity ({:.4}) should exceed between-faction ({:.4})",
        avg_faction2,
        avg_between
    );

    let node1_to_own: f32 = faction1
        .iter()
        .filter(|&&n| n != 1)
        .map(|&n| cosine_similarity(&get_row(&combined, 1), &get_row(&combined, n)))
        .sum::<f32>()
        / (faction1.len() - 1) as f32;

    let node1_to_other: f32 = faction2
        .iter()
        .map(|&n| cosine_similarity(&get_row(&combined, 1), &get_row(&combined, n)))
        .sum::<f32>()
        / faction2.len() as f32;

    let node34_to_own: f32 = faction2
        .iter()
        .filter(|&&n| n != 34)
        .map(|&n| cosine_similarity(&get_row(&combined, 34), &get_row(&combined, n)))
        .sum::<f32>()
        / (faction2.len() - 1) as f32;

    let node34_to_other: f32 = faction1
        .iter()
        .map(|&n| cosine_similarity(&get_row(&combined, 34), &get_row(&combined, n)))
        .sum::<f32>()
        / faction1.len() as f32;

    eprintln!(
        "CPU - Node 1 to own: {:.4}, to other: {:.4}",
        node1_to_own, node1_to_other
    );
    eprintln!(
        "CPU - Node 34 to own: {:.4}, to other: {:.4}",
        node34_to_own, node34_to_other
    );

    assert!(
        node1_to_own > node1_to_other,
        "Node 1 should be more similar to its own faction"
    );
    assert!(
        node34_to_own > node34_to_other,
        "Node 34 should be more similar to its own faction"
    );
}

//////////
// Burn //
//////////

#[test]
#[cfg(feature = "tch-cpu")]
fn test_karate_end_to_end_training() {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;
    use burn::prelude::Backend;

    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();
    let walks = graph.generate_walks(5, 10, 42);

    let split = (walks.len() as f32 * 0.8) as usize;
    let train_walks = walks[..split].to_vec();
    let valid_walks = walks[split..].to_vec();

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;
    let model_config = SkipGramConfig::new(vocab_size as usize, 8);
    let training_config = TrainingConfig {
        walks_per_node: 5,
        walk_length: 10,
        window_size: 2,
        batch_size: 32,
        num_workers: 1,
        num_epochs: 1,
        num_negatives: 3,
        seed: 42,
        learning_rate: 1e-3,
        p: 1.0,
        q: 1.0,
    };

    let device = LibTorchDevice::Cpu;
    LibTorch::<f32>::seed(&device, 42);

    let model = train::<Autodiff<LibTorch>>(
        "/tmp/karate_test",
        &model_config,
        &training_config,
        &train_walks,
        &valid_walks,
        device,
        42,
    );

    // Verify embeddings are correct shape
    let embeddings = model.embeddings_to_vec();
    assert_eq!(embeddings.len(), vocab_size as usize);
    assert_eq!(embeddings[0].len(), 8);

    // Check model output
    assert!(std::path::Path::new("/tmp/karate_test/model.mpk").exists());
}

#[test]
#[cfg(feature = "tch-cpu")]
fn test_karate_community_structure() {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;
    use burn::prelude::Backend;

    let graph = read_graph(KARATE_CSV, false, 1.0, 1.0).unwrap();
    let walks = graph.generate_walks(10, 80, 42);

    let split = (walks.len() as f32 * 0.8) as usize;
    let train_walks = walks[..split].to_vec();
    let valid_walks = walks[split..].to_vec();

    let vocab_size = graph.adjacency.keys().max().unwrap() + 1;
    let model_config = SkipGramConfig::new(vocab_size as usize, 16);

    let training_config = TrainingConfig {
        walks_per_node: 10,
        walk_length: 80,
        window_size: 3,
        batch_size: 128,
        num_workers: 1,
        num_epochs: 15,
        num_negatives: 5,
        seed: 42,
        learning_rate: 1e-2,
        p: 1.0,
        q: 1.0,
    };

    let device = LibTorchDevice::Cpu;
    LibTorch::<f32>::seed(&device, 42);

    let model = train::<Autodiff<LibTorch>>(
        "/tmp/karate_community_test",
        &model_config,
        &training_config,
        &train_walks,
        &valid_walks,
        device,
        42,
    );

    let embeddings = model.combined_embeddings_to_vec();

    // Known Karate club communities
    let faction1: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22];
    let faction2: Vec<usize> = vec![
        9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    ];

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    // Within-faction similarity
    let mut faction1_sim = Vec::new();
    for &i in &faction1 {
        for &j in &faction1 {
            if i < j {
                faction1_sim.push(cosine_similarity(&embeddings[i], &embeddings[j]));
            }
        }
    }

    let mut faction2_sim = Vec::new();
    for &i in &faction2 {
        for &j in &faction2 {
            if i < j {
                faction2_sim.push(cosine_similarity(&embeddings[i], &embeddings[j]));
            }
        }
    }

    // Between-faction similarity
    let mut between_sim = Vec::new();
    for &i in &faction1 {
        for &j in &faction2 {
            between_sim.push(cosine_similarity(&embeddings[i], &embeddings[j]));
        }
    }

    let avg_faction1: f32 = faction1_sim.iter().sum::<f32>() / faction1_sim.len() as f32;
    let avg_faction2: f32 = faction2_sim.iter().sum::<f32>() / faction2_sim.len() as f32;
    let avg_between: f32 = between_sim.iter().sum::<f32>() / between_sim.len() as f32;

    eprintln!("Faction 1 avg similarity: {:.4}", avg_faction1);
    eprintln!("Faction 2 avg similarity: {:.4}", avg_faction2);
    eprintln!("Between factions avg similarity: {:.4}", avg_between);

    // Community structure should be detectable
    assert!(avg_faction1 > avg_between,
        "Faction 1 internal similarity ({:.4}) should be higher than between-faction similarity ({:.4})",
        avg_faction1, avg_between);
    assert!(avg_faction2 > avg_between,
        "Faction 2 internal similarity ({:.4}) should be higher than between-faction similarity ({:.4})",
        avg_faction2, avg_between);

    // The two leaders should be more similar to their own faction
    let node1_to_faction1: f32 = faction1
        .iter()
        .filter(|&&n| n != 1)
        .map(|&n| cosine_similarity(&embeddings[1], &embeddings[n]))
        .sum::<f32>()
        / (faction1.len() - 1) as f32;

    let node1_to_faction2: f32 = faction2
        .iter()
        .map(|&n| cosine_similarity(&embeddings[1], &embeddings[n]))
        .sum::<f32>()
        / faction2.len() as f32;

    let node34_to_faction2: f32 = faction2
        .iter()
        .filter(|&&n| n != 34)
        .map(|&n| cosine_similarity(&embeddings[34], &embeddings[n]))
        .sum::<f32>()
        / (faction2.len() - 1) as f32;

    let node34_to_faction1: f32 = faction1
        .iter()
        .map(|&n| cosine_similarity(&embeddings[34], &embeddings[n]))
        .sum::<f32>()
        / faction1.len() as f32;

    eprintln!(
        "Node 1 to own faction: {:.4}, to other faction: {:.4}",
        node1_to_faction1, node1_to_faction2
    );
    eprintln!(
        "Node 34 to own faction: {:.4}, to other faction: {:.4}",
        node34_to_faction2, node34_to_faction1
    );

    assert!(
        node1_to_faction1 > node1_to_faction2,
        "Node 1 should be more similar to its own faction"
    );
    assert!(
        node34_to_faction2 > node34_to_faction1,
        "Node 34 should be more similar to its own faction"
    );
}

////////////////////////////
// Heterogeneous fixtures //
////////////////////////////

/// Path to the heterogeneous node table: 4 genes, 2 pathways, 2 diseases.
const HET_NODES: &str = "tests/data/het_nodes.csv";
/// Path to the matching edge table, unweighted.
const HET_EDGES: &str = "tests/data/het_edges.csv";

#[test]
fn test_het_graph_loads_and_sorts_by_type() {
    let graph = read_het_graph(HET_NODES, HET_EDGES, false).unwrap();

    assert_eq!(graph.n_nodes(), 8);
    assert_eq!(graph.n_types(), 3);
    assert_eq!(graph.n_edges(), 20); // 10 edges, both directions
    assert_eq!(graph.type_names(), &["gene", "pathway", "disease"]);

    // Ids must be contiguous by type for the range-based start-node scan.
    let types = graph.node_types();
    assert_eq!(types, &[0, 0, 0, 0, 1, 1, 2, 2]);
    assert_eq!(&graph.node_names()[..4], &["g1", "g2", "g3", "g4"]);
}

#[test]
fn test_het_metapath_walks_respect_the_schema() {
    let graph = read_het_graph(HET_NODES, HET_EDGES, false).unwrap();
    let schema = Metapath::parse("gene-pathway-gene", graph.type_names()).unwrap();
    let (walks, stats) = graph.generate_walks(&schema, 10, 11, 42).unwrap();

    assert_eq!(stats.start_nodes, 4);
    assert_eq!(stats.attempted, 40);
    // Every gene here reaches a pathway, so nothing dead-ends.
    assert_eq!(stats.truncated, 0);
    assert_eq!(stats.dropped, 0);

    for walk in &walks {
        assert_eq!(walk.len(), stats.walk_length);
        for (step, &node) in walk.iter().enumerate() {
            let expected = if step % 2 == 0 { 0 } else { 1 };
            assert_eq!(graph.node_types()[node as usize], expected);
        }
    }
}

#[test]
fn test_het_metapath_rejects_a_schema_that_does_not_close() {
    let graph = read_het_graph(HET_NODES, HET_EDGES, false).unwrap();
    assert!(Metapath::parse("gene-pathway", graph.type_names()).is_err());
    assert!(Metapath::parse("gene-pathway-disease", graph.type_names()).is_err());
    assert!(Metapath::parse("gene-protein-gene", graph.type_names()).is_err());
    assert!(Metapath::parse("gene-pathway-gene-disease-gene", graph.type_names()).is_ok());
}

#[cfg(feature = "cpu")]
#[test]
fn test_het_end_to_end_metapath2vec() {
    use node2vec_rs::cpu::train::{
        create_negative_table_per_type, train_node2vec_cpu, CpuTrainArgs, NegativeTable,
    };
    use std::sync::Arc;

    let graph = read_het_graph(HET_NODES, HET_EDGES, false).unwrap();
    let schema = Metapath::parse("gene-pathway-gene-disease-gene", graph.type_names()).unwrap();
    let (walks, stats) = graph.generate_walks(&schema, 40, 17, 42).unwrap();
    assert!(!walks.is_empty(), "{stats}");

    let vocab_size = graph.n_nodes();

    // metapath2vec++: negatives drawn per node type.
    let neg_table = NegativeTable::PerType {
        tables: create_negative_table_per_type(
            graph.node_types(),
            graph.n_types(),
            &walks,
            NEGATIVE_TABLE_SIZE,
            42,
        ),
        node_type: Arc::new(graph.node_types().to_vec()),
    };

    let args = CpuTrainArgs {
        dim: 8,
        lr: 0.025,
        epochs: 2,
        neg: 5,
        window: 3,
        lr_update_rate: 100,
        n_threads: 2,
        verbose: false,
        sample: 1e-3,
    };

    let (mut input_mat, _) = train_node2vec_cpu(walks, vocab_size, args, neg_table, 42);
    input_mat.norm_self();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embeddings.csv");
    input_mat.write_csv(path.to_str().unwrap()).unwrap();

    // One row per node, and the row index is the dense id the node table maps.
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), vocab_size);
    assert_eq!(lines[0].split(',').count(), 8);
    assert_eq!(graph.node_names().len(), lines.len());

    // Trained vectors must not still be the zero or the initial rows.
    let trained: Vec<f32> = lines[0]
        .split(',')
        .map(|v| v.parse::<f32>().unwrap())
        .collect();
    assert!(trained.iter().any(|v| v.abs() > 1e-6));
    assert!(trained.iter().all(|v| v.is_finite()));
}
