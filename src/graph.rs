use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Structure to store the Node2Vec graph
///
/// ### Fields
///
/// * `adjacency` - The adjacency stored as an FxHashMap.
/// * `transition_probs` - The transition probabilities stored in a FxHashMap.
#[derive(Debug, Clone)]
pub struct Node2VecGraph {
    pub adjacency: FxHashMap<u32, Vec<(u32, f32)>>,
    pub transition_probs: FxHashMap<(u32, u32), Vec<(u32, f32)>>,
}

/// Compute the transition probabilities
///
/// ### Params
///
/// * `adjacency` - The adjacency of the graph stored as a HashMap
/// * `p` - p parameter in node2vec that controls probability to return
/// * `q` - q parameter in node2vec that controls probability to reach out
///   futher in the graph.
///
/// ### Returns
///
/// The transition probabilities as an `FxHashMap`.
pub fn compute_transition_prob(
    adjacency: &FxHashMap<u32, Vec<(u32, f32)>>,
    p: f32,
    q: f32,
) -> FxHashMap<(u32, u32), Vec<(u32, f32)>> {
    let neighbours_set: FxHashMap<u32, FxHashSet<u32>> = adjacency
        .iter()
        .map(|(node, edges)| (*node, edges.iter().map(|(n, _)| *n).collect()))
        .collect();

    adjacency
        .par_iter()
        .flat_map(|(curr_node, curr_edges)| {
            curr_edges.par_iter().filter_map(|(prev_node, _)| {
                adjacency.get(curr_node).map(|next_edges| {
                    let mut probs = Vec::new();
                    let mut total = 0_f32;

                    for (next_node, weight) in next_edges.iter() {
                        let unnorm_prob = if next_node == prev_node {
                            weight / p
                        } else if neighbours_set
                            .get(prev_node)
                            .map(|s| s.contains(next_node))
                            .unwrap_or(false)
                        {
                            *weight
                        } else {
                            weight / q
                        };
                        total += unnorm_prob;
                        probs.push((*next_node, unnorm_prob));
                    }

                    let mut cumulative = 0.0;
                    let normalised: Vec<(u32, f32)> = probs
                        .into_iter()
                        .map(|(node, prob)| {
                            cumulative += prob / total;
                            (node, cumulative)
                        })
                        .collect();

                    ((*prev_node, *curr_node), normalised)
                })
            })
        })
        .collect()
}
