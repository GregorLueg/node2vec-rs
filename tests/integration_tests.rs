use node2vec_rs::model::SkipGramConfig;
use node2vec_rs::reader::read_graph;
use node2vec_rs::train::{train, TrainingConfig};
use std::path::Path;

const KARATE_CSV: &str = "tests/data/karate.csv";

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
        model_config,
        training_config,
        train_walks,
        valid_walks,
        device,
    );

    // Verify embeddings are correct shape
    let embeddings = model.embeddings_to_vec();
    assert_eq!(embeddings.len(), vocab_size as usize);
    assert_eq!(embeddings[0].len(), 8);

    // Check model output
    assert!(Path::new("/tmp/karate_test/model.mpk").exists());
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
