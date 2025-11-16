use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/////////////
// Helpers //
/////////////

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

/////////////////////
// Graph structure //
/////////////////////

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

impl Node2VecGraph {
    /// Generates random walks from the graph
    ///
    /// ### Params
    /// * `walks_per_node` - Number of walks to generate starting from each node
    /// * `walk_length` - Length of each walk
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Vector of walks, where each walk is a sequence of node IDs
    pub fn generate_walks(
        &self,
        walks_per_node: usize,
        walk_length: usize,
        seed: u64,
    ) -> Vec<Vec<u32>> {
        use rayon::prelude::*;

        self.adjacency
            .par_iter()
            .flat_map(|(start_node, _)| {
                (0..walks_per_node).into_par_iter().map(move |walk_idx| {
                    // seed per thread
                    let walk_seed = seed
                        .wrapping_mul(*start_node as u64)
                        .wrapping_add(walk_idx as u64);
                    let mut rng = StdRng::seed_from_u64(walk_seed);
                    self.single_walk(*start_node, walk_length, &mut rng)
                })
            })
            .collect()
    }

    /// Performs a single biased random walk
    ///
    /// ### Params
    ///
    /// * `start_node` - Node to start the walk from
    /// * `walk_length` - Maximum length of the walk
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// The vector of node IDs for this walk
    fn single_walk(&self, start_node: u32, walk_length: usize, rng: &mut StdRng) -> Vec<u32> {
        let mut walk: Vec<u32> = Vec::with_capacity(walk_length);
        walk.push(start_node);

        if walk_length == 1 {
            return walk;
        }

        let mut curr = if let Some(neighbours) = self.adjacency.get(&start_node) {
            self.sample_neighbor(neighbours, rng)
        } else {
            return walk;
        };

        walk.push(curr);

        for _ in 2..walk_length {
            let prev = walk[walk.len() - 2];

            if let Some(probs) = self.transition_probs.get(&(prev, curr)) {
                curr = self.sample_from_cumulative(probs, rng);
                walk.push(curr);
            } else {
                break;
            }
        }

        walk
    }

    /// Samples a neighbour based on edge weights
    ///
    /// ### Params
    ///
    /// * `neighbours` - Slice of (node, weight) tuples
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Node ID based on the neighbours
    fn sample_neighbor(&self, neighbours: &[(u32, f32)], rng: &mut impl Rng) -> u32 {
        let total: f32 = neighbours.iter().map(|(_, w)| w).sum();
        let mut rand_val = rng.random::<f32>() * total;

        for (node, weight) in neighbours {
            rand_val -= weight;
            if rand_val <= 0.0 {
                return *node;
            }
        }

        neighbours[0].0
    }

    /// Samples from a cumulative probability distribution
    ///
    /// ### Params
    ///
    /// * `cumulative` - Cumulative probabilities as (node, cumulative_prob)
    ///   pairs
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// The node ID
    fn sample_from_cumulative(&self, cumulative: &[(u32, f32)], rng: &mut impl Rng) -> u32 {
        let rand_val = rng.random::<f32>();

        match cumulative.binary_search_by(|(_, cum_prob)| cum_prob.partial_cmp(&rand_val).unwrap())
        {
            Ok(idx) => cumulative[idx].0,
            Err(idx) => {
                if idx < cumulative.len() {
                    cumulative[idx].0
                } else {
                    cumulative[cumulative.len() - 1].0
                }
            }
        }
    }
}
