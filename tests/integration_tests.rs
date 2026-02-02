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
        model_config,
        training_config,
        train_walks,
        valid_walks,
        device,
    );

    let embeddings = model.embeddings_to_vec();

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
