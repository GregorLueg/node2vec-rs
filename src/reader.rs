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
