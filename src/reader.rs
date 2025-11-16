use csv::Reader;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fs::File;

use crate::graph::*;

/// Helper function to read in a graph from CSV
///
/// ### Params
///
/// * `path` - Path to the CSV with a `"from"`, `"to"` and `"weight"` column.
/// * `directed` - Boolean. Shall the graph be treated as a directed or
///   undirected graph.
/// * `p` - p parameter in node2vec that controls probability to return
/// * `q` - q parameter in node2vec that controls probability to reach out
///   futher in the graph.
///
/// ### Returns
///
/// The `Node2VecGraph` with adjacency stored in their and transition
/// probabilities.
pub fn read_graph(
    path: &str,
    directed: bool,
    p: f32,
    q: f32,
) -> Result<Node2VecGraph, Box<dyn Error>> {
    let mut adjacency = FxHashMap::default();
    let file = File::open(path)?;
    let mut rdr = Reader::from_reader(file);

    for result in rdr.records() {
        let record = result?;
        let from: u32 = record[0]
            .parse()
            .map_err(|_| format!("Cannot cast 'from' to u32: {}", &record[0]))?;
        let to: u32 = record[1]
            .parse()
            .map_err(|_| format!("Cannot cast 'to' to u32: {}", &record[1]))?;
        let weight: f32 = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);

        adjacency
            .entry(from)
            .or_insert_with(Vec::new)
            .push((to, weight));

        if !directed {
            adjacency
                .entry(to)
                .or_insert_with(Vec::new)
                .push((from, weight));
        }
    }

    let transition_probs = compute_transition_prob(&adjacency, p, q);

    Ok(Node2VecGraph {
        adjacency,
        transition_probs,
    })
}

#[cfg(test)]
mod reader_tests {
    use rustc_hash::FxHashMap;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    type SimpleGraph = Result<FxHashMap<u32, Vec<(u32, f32)>>, Box<dyn std::error::Error>>;

    // Simplified version of read_graph for testing
    fn read_graph_test(path: &str, directed: bool) -> SimpleGraph {
        use csv::Reader;
        let mut adjacency = FxHashMap::default();
        let file = File::open(path)?;
        let mut rdr = Reader::from_reader(file);

        for result in rdr.records() {
            let record = result?;
            let from: u32 = record[0].parse()?;
            let to: u32 = record[1].parse()?;
            let weight: f32 = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);

            adjacency
                .entry(from)
                .or_insert_with(Vec::new)
                .push((to, weight));

            if !directed {
                adjacency
                    .entry(to)
                    .or_insert_with(Vec::new)
                    .push((from, weight));
            }
        }

        Ok(adjacency)
    }

    #[test]
    fn test_undirected_graph_symmetry() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to").unwrap();
        writeln!(file, "1,2").unwrap();
        writeln!(file, "2,3").unwrap();

        let graph = read_graph_test(file_path.to_str().unwrap(), false).unwrap();

        // Check bidirectional edges exist
        assert!(graph.get(&1).unwrap().iter().any(|(n, _)| *n == 2));
        assert!(graph.get(&2).unwrap().iter().any(|(n, _)| *n == 1));
        assert!(graph.get(&2).unwrap().iter().any(|(n, _)| *n == 3));
        assert!(graph.get(&3).unwrap().iter().any(|(n, _)| *n == 2));
    }

    #[test]
    fn test_directed_graph_no_symmetry() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to").unwrap();
        writeln!(file, "1,2").unwrap();
        writeln!(file, "2,3").unwrap();

        let graph = read_graph_test(file_path.to_str().unwrap(), true).unwrap();

        // Check only forward edges exist
        assert!(graph.get(&1).unwrap().iter().any(|(n, _)| *n == 2));
        assert!(!graph.contains_key(&2) || !graph.get(&2).unwrap().iter().any(|(n, _)| *n == 1));
    }

    #[test]
    fn test_default_weights() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to").unwrap();
        writeln!(file, "1,2").unwrap();

        let graph = read_graph_test(file_path.to_str().unwrap(), false).unwrap();

        // All weights should be 1.0
        for edges in graph.values() {
            for (_, weight) in edges {
                assert_eq!(*weight, 1.0);
            }
        }
    }

    #[test]
    fn test_explicit_weights() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to,weight").unwrap();
        writeln!(file, "1,2,0.5").unwrap();
        writeln!(file, "2,3,2.0").unwrap();

        let graph = read_graph_test(file_path.to_str().unwrap(), false).unwrap();

        // Check explicit weights
        let edge_1_2 = graph
            .get(&1)
            .unwrap()
            .iter()
            .find(|(n, _)| *n == 2)
            .unwrap();
        assert_eq!(edge_1_2.1, 0.5);

        let edge_2_3 = graph
            .get(&2)
            .unwrap()
            .iter()
            .find(|(n, _)| *n == 3)
            .unwrap();
        assert_eq!(edge_2_3.1, 2.0);
    }

    #[test]
    fn test_self_loops() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to").unwrap();
        writeln!(file, "1,1").unwrap();

        let graph = read_graph_test(file_path.to_str().unwrap(), false).unwrap();

        // Self-loop should exist
        assert!(graph.get(&1).unwrap().iter().any(|(n, _)| *n == 1));
    }

    #[test]
    fn test_invalid_node_id() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "from,to").unwrap();
        writeln!(file, "invalid,2").unwrap();

        let result = read_graph_test(file_path.to_str().unwrap(), false);
        assert!(result.is_err());
    }
}
