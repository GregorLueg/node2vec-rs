//! Heterogeneous graphs and metapath-constrained random walks.
//!
//! metapath2vec is node2vec over a typed graph with the walk constrained by a
//! metapath schema. The walk is **first-order**: the next hop depends only on
//! the current node and the schema position, never on where the walk came
//! from. That is why nothing here resembles
//! [`crate::graph::compute_transition_prob`] and its `(prev, curr)`-keyed
//! table, whose memory is O(sum of deg^2) and which would be fatal on a
//! knowledge graph.
//!
//! The skip-gram objective is unchanged, so walks feed the existing CPU
//! trainer verbatim.
//!
//! ### References
//!
//! Dong, Chawla and Swami, *metapath2vec: Scalable Representation Learning for
//! Heterogeneous Networks*, KDD 2017.

pub mod reader;

use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::errors::Node2VecError;

/// Separator between type names in a metapath specification.
///
/// Type names may therefore not contain it. Documented rather than escaped,
/// because every realistic type name (`gene`, `pathway`, `GO`) is a bare word.
const METAPATH_SEP: char = '-';

/// Shortest walk that produces a single skip-gram pair.
///
/// Anything below this contributes nothing to training but still counts
/// towards the token total that paces the learning-rate decay in
/// [`crate::cpu::train::train_node2vec_cpu`], so such walks are dropped.
const MIN_WALK_LEN: usize = 2;

//////////////
// Metapath //
//////////////

/// A cyclic metapath schema over node types.
///
/// Stored as the full sequence including the repeated final type, so
/// `"a-b-a"` holds `[a, b, a]` with period 2. Keeping the closing type makes
/// the "is it cyclic" invariant checkable by looking at the value rather than
/// remembering it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metapath {
    /// Type ids in walk order. First and last entries are equal.
    types: Vec<u8>,
}

impl Metapath {
    /// Parse a metapath from a hyphen-separated specification.
    ///
    /// ### Params
    ///
    /// * `spec` - Schema such as `"gene-pathway-gene"`
    /// * `type_names` - Type names indexed by dense type id, from the graph
    ///
    /// ### Returns
    ///
    /// The validated schema, or an error naming the offending type or the
    /// reason the sequence is not cyclic.
    pub fn parse(spec: &str, type_names: &[String]) -> Result<Self, Node2VecError> {
        let parts: Vec<&str> = spec.split(METAPATH_SEP).map(str::trim).collect();

        if parts.len() < 3 {
            return Err(Node2VecError::InvalidMetapath {
                reason: format!(
                    "'{spec}' has {} entries; a schema needs at least three so that it \
                     closes on its starting type",
                    parts.len()
                ),
            });
        }

        let mut types = Vec::with_capacity(parts.len());
        for name in &parts {
            let id = type_names.iter().position(|t| t == name).ok_or_else(|| {
                Node2VecError::UnknownType {
                    name: (*name).to_string(),
                    known: type_names.join(", "),
                }
            })?;
            types.push(id as u8);
        }

        if types[0] != types[types.len() - 1] {
            return Err(Node2VecError::InvalidMetapath {
                reason: format!(
                    "'{spec}' starts on '{}' and ends on '{}'; a schema must close on \
                     its starting type so that it composes",
                    parts[0],
                    parts[parts.len() - 1]
                ),
            });
        }

        Ok(Self { types })
    }

    /// Number of hops in one full cycle of the schema.
    ///
    /// ### Returns
    ///
    /// The period, i.e. the stored length minus the repeated closing type.
    #[inline]
    pub fn period(&self) -> usize {
        self.types.len() - 1
    }

    /// The type a walk must start on.
    ///
    /// ### Returns
    ///
    /// The dense id of the first type in the schema.
    #[inline]
    pub fn start_type(&self) -> u8 {
        self.types[0]
    }

    /// The type required at a given walk position.
    ///
    /// ### Params
    ///
    /// * `step` - Zero-based position in the walk
    ///
    /// ### Returns
    ///
    /// The dense type id, wrapping through the schema.
    #[inline]
    fn type_at(&self, step: usize) -> u8 {
        self.types[step % self.period()]
    }
}

///////////////
// WalkStats //
///////////////

/// What actually came out of walk generation.
///
/// A wrong schema, a sparse type or walks that dead-end on the first hop still
/// produce vectors, and those vectors are garbage. These counts are the only
/// warning, so they are reported unconditionally rather than behind a debug
/// flag.
#[derive(Clone, Copy, Debug)]
pub struct WalkStats {
    /// Number of nodes carrying the schema's starting type.
    pub start_nodes: usize,
    /// Walks attempted, i.e. `start_nodes * walks_per_node`.
    pub attempted: usize,
    /// Walks that hit a dead end before reaching the requested length.
    pub truncated: usize,
    /// Walks discarded for being shorter than two nodes.
    pub dropped: usize,
    /// Mean length of the walks that survived.
    pub mean_length: f32,
    /// Requested walk length after rounding up to close the schema.
    pub walk_length: usize,
}

impl std::fmt::Display for WalkStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pct = |n: usize| {
            if self.attempted == 0 {
                0.0
            } else {
                100.0 * n as f32 / self.attempted as f32
            }
        };
        write!(
            f,
            "{} start nodes, {} walks of length {} requested; {} truncated ({:.1}%), \
             {} dropped ({:.1}%), mean realised length {:.1}",
            self.start_nodes,
            self.attempted,
            self.walk_length,
            self.truncated,
            pct(self.truncated),
            self.dropped,
            pct(self.dropped),
            self.mean_length
        )
    }
}

//////////////
// HetGraph //
//////////////

/// A heterogeneous graph as a typed CSR.
///
/// Neighbours are sorted by target node type within each row, and
/// `type_starts` indexes those runs, so `neighbours_of_type` hands back a
/// slice rather than a filtered iterator. Node ids are dense and **contiguous
/// by type**, which turns "every node of type `t`" into a range: the walker
/// iterates it directly instead of filtering the whole graph.
#[derive(Clone, Debug)]
pub struct HetGraph {
    /// Node type per node id. Length `n_nodes`.
    node_type: Vec<u8>,
    /// Neighbour ids, sorted by target node type within each row. CSR row
    /// starts are not held separately: row `v` begins at
    /// `type_starts[v * n_types]`, which is the same number.
    targets: Vec<u32>,
    /// Cumulative edge weight, restarting and renormalising at every
    /// `(node, type)` run boundary, so a run's last entry is 1.0. Sampling a
    /// neighbour of a given type is then a binary search over the run.
    /// Meaningless, and never read, when `weighted` is false.
    cum_weights: Vec<f32>,
    /// Start index into `targets` for each `(node, type)` pair. The run for
    /// node `v` and type `t` is `type_starts[v * n_types + t]` up to
    /// `type_starts[v * n_types + t + 1]`. Length `n_nodes * n_types + 1`.
    type_starts: Vec<u32>,
    /// Number of distinct node types.
    n_types: usize,
    /// Whether the edge table carried a weight column. When it did not, every
    /// weight is 1.0 and sampling collapses to a single `random_range`.
    weighted: bool,
    /// Original string id per dense node id.
    node_names: Vec<String>,
    /// Original type name per dense type id.
    type_names: Vec<String>,
}

impl HetGraph {
    /// Number of nodes.
    ///
    /// ### Returns
    ///
    /// The node count, which is also the vocabulary size for the trainer.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.node_type.len()
    }

    /// Number of edges held in the CSR.
    ///
    /// ### Returns
    ///
    /// The edge count, counting both directions when the graph was read as
    /// undirected.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.targets.len()
    }

    /// Number of distinct node types.
    ///
    /// ### Returns
    ///
    /// The type count.
    #[inline]
    pub fn n_types(&self) -> usize {
        self.n_types
    }

    /// Node type per dense node id.
    ///
    /// ### Returns
    ///
    /// A slice of length `n_nodes`, needed by the per-type negative sampler.
    #[inline]
    pub fn node_types(&self) -> &[u8] {
        &self.node_type
    }

    /// Original string ids, indexed by dense node id.
    ///
    /// ### Returns
    ///
    /// A slice of length `n_nodes`. Without this the embedding CSV, whose row
    /// index is the node id, cannot be read back.
    #[inline]
    pub fn node_names(&self) -> &[String] {
        &self.node_names
    }

    /// Original type names, indexed by dense type id.
    ///
    /// ### Returns
    ///
    /// A slice of length `n_types`.
    #[inline]
    pub fn type_names(&self) -> &[String] {
        &self.type_names
    }

    /// The half-open dense id range holding every node of a type.
    ///
    /// Relies on the reader sorting nodes contiguously by type.
    ///
    /// ### Params
    ///
    /// * `node_type` - Dense type id
    ///
    /// ### Returns
    ///
    /// The `start..end` range, empty when no node carries the type.
    fn type_range(&self, node_type: u8) -> std::ops::Range<u32> {
        let start = self.node_type.partition_point(|&t| t < node_type);
        let end = self.node_type.partition_point(|&t| t <= node_type);
        start as u32..end as u32
    }

    /// The neighbours of a node that carry a given type.
    ///
    /// ### Params
    ///
    /// * `node` - Dense node id
    /// * `node_type` - Dense type id the neighbour must carry
    ///
    /// ### Returns
    ///
    /// The index range into `targets` and `cum_weights`, empty on a dead end.
    #[inline]
    fn neighbours_of_type(&self, node: usize, node_type: u8) -> std::ops::Range<usize> {
        let base = node * self.n_types + node_type as usize;
        self.type_starts[base] as usize..self.type_starts[base + 1] as usize
    }

    /// Sample one neighbour from a `(node, type)` run.
    ///
    /// ### Params
    ///
    /// * `run` - Index range into `targets`, assumed non-empty
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// The sampled neighbour's dense node id.
    #[inline]
    fn sample_in_run(&self, run: std::ops::Range<usize>, rng: &mut impl Rng) -> u32 {
        if !self.weighted {
            return self.targets[rng.random_range(run)];
        }

        let start = run.start;
        let cumulative = &self.cum_weights[run];
        let draw = rng.random::<f32>();
        // Cumulative is non-decreasing and ends at 1.0, so a hit and a miss
        // both land on the right slot; only the final clamp guards a draw that
        // rounds past the last entry.
        let offset = match cumulative.binary_search_by(|c| c.partial_cmp(&draw).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx.min(cumulative.len() - 1),
        };
        self.targets[start + offset]
    }

    /// Walk one metapath from a start node.
    ///
    /// Truncates on a dead end, matching what the homogeneous
    /// [`crate::graph::Node2VecGraph`] does on a missing transition.
    ///
    /// ### Params
    ///
    /// * `start_node` - Dense id of the node to start from, of the schema's
    ///   starting type
    /// * `schema` - The metapath
    /// * `walk_length` - Maximum number of nodes in the walk
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// The walk, possibly shorter than `walk_length`.
    fn single_walk(
        &self,
        start_node: u32,
        schema: &Metapath,
        walk_length: usize,
        rng: &mut StdRng,
    ) -> Vec<u32> {
        let mut walk = Vec::with_capacity(walk_length);
        walk.push(start_node);

        let mut curr = start_node as usize;
        for step in 1..walk_length {
            let run = self.neighbours_of_type(curr, schema.type_at(step));
            if run.is_empty() {
                break;
            }
            let next = self.sample_in_run(run, rng);
            walk.push(next);
            curr = next as usize;
        }

        walk
    }

    /// Generate metapath-constrained walks over the whole graph.
    ///
    /// Walks start only on nodes of the schema's starting type, so
    /// `walks_per_node` means "per node of that type", not per node of the
    /// graph. `walk_length` is rounded up so the walk closes on a full number
    /// of schema cycles; truncating mid-schema would bias the final window.
    ///
    /// ### Params
    ///
    /// * `schema` - The metapath
    /// * `walks_per_node` - Walks to start from each node of the starting type
    /// * `walk_length` - Requested number of nodes per walk, rounded up
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// The walks and the statistics that say whether the schema was any good.
    ///
    /// ### Errors
    ///
    /// [`Node2VecError::EmptyStartSet`] when no node carries the starting
    /// type, which is almost always a typo in the schema rather than a graph
    /// worth walking.
    pub fn generate_walks(
        &self,
        schema: &Metapath,
        walks_per_node: usize,
        walk_length: usize,
        seed: usize,
    ) -> Result<(Vec<Vec<u32>>, WalkStats), Node2VecError> {
        let range = self.type_range(schema.start_type());
        if range.is_empty() {
            return Err(Node2VecError::EmptyStartSet {
                type_name: self
                    .type_names
                    .get(schema.start_type() as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("<type {}>", schema.start_type())),
            });
        }

        let walk_length = round_to_schema(walk_length, schema.period());
        let start_nodes = range.len();
        let attempted = start_nodes * walks_per_node;

        let progress = ProgressBar::new(attempted as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[Metapath walk gen: {elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut walks: Vec<Vec<u32>> = range
            .into_par_iter()
            .flat_map(|start_node| {
                let progress = progress.clone();
                (0..walks_per_node).into_par_iter().map(move |walk_idx| {
                    let mut rng = StdRng::seed_from_u64(walk_seed(seed, start_node, walk_idx));
                    let walk = self.single_walk(start_node, schema, walk_length, &mut rng);
                    progress.inc(1);
                    walk
                })
            })
            .collect();

        progress.finish_and_clear();

        let truncated = walks.iter().filter(|w| w.len() < walk_length).count();
        walks.retain(|w| w.len() >= MIN_WALK_LEN);
        let dropped = attempted - walks.len();
        let total: usize = walks.iter().map(Vec::len).sum();
        let mean_length = if walks.is_empty() {
            0.0
        } else {
            total as f32 / walks.len() as f32
        };

        Ok((
            walks,
            WalkStats {
                start_nodes,
                attempted,
                truncated,
                dropped,
                mean_length,
                walk_length,
            },
        ))
    }
}

/////////////
// Helpers //
/////////////

/// Derive a per-walk seed.
///
/// Lifted verbatim from [`crate::graph::Node2VecGraph::generate_walks`] so
/// both paths seed identically: a hash mix rather than a shared RNG, which
/// keeps generation lock-free and reproducible regardless of thread count.
///
/// ### Params
///
/// * `seed` - Global seed
/// * `start_node` - Dense id the walk starts from
/// * `walk_idx` - Index of this walk among those starting at `start_node`
///
/// ### Returns
///
/// The seed for this walk's RNG.
#[inline]
fn walk_seed(seed: usize, start_node: u32, walk_idx: usize) -> u64 {
    let mut walk_seed = seed as u64;
    walk_seed ^= (start_node as u64).wrapping_mul(0x517cc1b727220a95);
    walk_seed ^= (walk_idx as u64).wrapping_mul(0xcbf29ce484222325);
    walk_seed.wrapping_mul(0x4f1bbcdcbfa54c43)
}

/// Round a walk length up to close a whole number of schema cycles.
///
/// A walk of `n` nodes takes `n - 1` hops, so the schema closes when
/// `n - 1` is a multiple of the period.
///
/// ### Params
///
/// * `walk_length` - Requested number of nodes
/// * `period` - Hops in one schema cycle
///
/// ### Returns
///
/// The adjusted length, never below `period + 1`.
fn round_to_schema(walk_length: usize, period: usize) -> usize {
    let hops = walk_length.saturating_sub(1).max(period);
    1 + hops.div_ceil(period) * period
}

///////////
// Tests //
///////////

#[cfg(test)]
mod het_tests {
    use super::*;
    use crate::het::reader::build_het_graph;

    /// Toy graph: types a (0,1), b (2,3), c (4).
    ///
    /// Edges wire a-b both ways, plus one dangling c that no `a-b-a` walk can
    /// reach, and node 1 has no `b` neighbour so it dead-ends immediately.
    fn toy() -> HetGraph {
        build_het_graph(
            vec![
                ("n0".into(), "a".into()),
                ("n1".into(), "a".into()),
                ("n2".into(), "b".into()),
                ("n3".into(), "b".into()),
                ("n4".into(), "c".into()),
            ],
            vec![
                (0, 2, 1.0),
                (0, 3, 1.0),
                (2, 0, 1.0),
                (3, 0, 1.0),
                (4, 4, 1.0),
            ],
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_metapath_parse_requires_closure() {
        let names = vec!["a".to_string(), "b".to_string()];
        assert!(Metapath::parse("a-b-a", &names).is_ok());
        assert!(Metapath::parse("a-b", &names).is_err());
        assert!(Metapath::parse("a-b-b", &names).is_err());
        assert!(Metapath::parse("a-z-a", &names).is_err());
    }

    #[test]
    fn test_metapath_period_and_cycle() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mp = Metapath::parse("a-b-a", &names).unwrap();
        assert_eq!(mp.period(), 2);
        assert_eq!(mp.type_at(0), 0);
        assert_eq!(mp.type_at(1), 1);
        assert_eq!(mp.type_at(2), 0);
        assert_eq!(mp.type_at(3), 1);
    }

    #[test]
    fn test_round_to_schema() {
        assert_eq!(round_to_schema(20, 2), 21);
        assert_eq!(round_to_schema(21, 2), 21);
        assert_eq!(round_to_schema(1, 2), 3);
        assert_eq!(round_to_schema(10, 3), 10);
        assert_eq!(round_to_schema(11, 3), 13);
    }

    #[test]
    fn test_ids_contiguous_by_type() {
        let g = toy();
        assert_eq!(g.node_types(), &[0, 0, 1, 1, 2]);
        assert_eq!(g.type_range(0), 0..2);
        assert_eq!(g.type_range(1), 2..4);
        assert_eq!(g.type_range(2), 4..5);
    }

    #[test]
    fn test_walks_follow_the_schema() {
        let g = toy();
        let mp = Metapath::parse("a-b-a", g.type_names()).unwrap();
        let (walks, _) = g.generate_walks(&mp, 5, 7, 42).unwrap();

        assert!(!walks.is_empty());
        for walk in &walks {
            for (step, &node) in walk.iter().enumerate() {
                assert_eq!(
                    g.node_types()[node as usize],
                    mp.type_at(step),
                    "walk {walk:?} broke the schema at step {step}"
                );
            }
        }
    }

    #[test]
    fn test_dead_end_accounting_is_exact() {
        let g = toy();
        let mp = Metapath::parse("a-b-a", g.type_names()).unwrap();
        let walks_per_node = 5;
        let (walks, stats) = g.generate_walks(&mp, walks_per_node, 7, 42).unwrap();

        // Node 0 reaches b and back; node 1 has no b neighbour, so every walk
        // from it is length 1 and gets dropped. Nothing else can start.
        assert_eq!(stats.start_nodes, 2);
        assert_eq!(stats.attempted, 2 * walks_per_node);
        assert_eq!(stats.truncated, walks_per_node);
        assert_eq!(stats.dropped, walks_per_node);
        assert_eq!(walks.len(), walks_per_node);
        assert!(walks.iter().all(|w| w[0] == 0));
    }

    #[test]
    fn test_empty_start_set_errors() {
        let g = toy();
        let mp = Metapath {
            types: vec![3, 0, 3],
        };
        assert!(matches!(
            g.generate_walks(&mp, 1, 3, 42),
            Err(Node2VecError::EmptyStartSet { .. })
        ));
    }

    #[test]
    fn test_walks_are_deterministic() {
        let g = toy();
        let mp = Metapath::parse("a-b-a", g.type_names()).unwrap();
        let (a, _) = g.generate_walks(&mp, 5, 9, 7).unwrap();
        let (b, _) = g.generate_walks(&mp, 5, 9, 7).unwrap();
        assert_eq!(a, b);

        let (c, _) = g.generate_walks(&mp, 5, 9, 8).unwrap();
        assert_ne!(a, c);
    }

    /// The degeneracy check: one type, period-1 schema, so the metapath walk
    /// is a plain weight-proportional first-order walk. Assert the realised
    /// transition frequencies match the weights.
    ///
    /// This is deliberately at the walk-distribution level. Comparing
    /// embeddings against the node2vec path cannot be asserted, because the
    /// trainer is Hogwild and its output is not reproducible across thread
    /// counts.
    #[test]
    fn test_degenerate_walk_matches_edge_weights() {
        let g = build_het_graph(
            vec![
                ("n0".into(), "x".into()),
                ("n1".into(), "x".into()),
                ("n2".into(), "x".into()),
            ],
            // From node 0: weight 3 to node 1, weight 1 to node 2.
            vec![(0, 1, 3.0), (0, 2, 1.0), (1, 0, 1.0), (2, 0, 1.0)],
            true,
        )
        .unwrap();
        let mp = Metapath { types: vec![0, 0] };
        assert_eq!(mp.period(), 1);

        let (walks, _) = g.generate_walks(&mp, 4000, 2, 42).unwrap();
        let from_zero: Vec<u32> = walks.iter().filter(|w| w[0] == 0).map(|w| w[1]).collect();

        let to_one = from_zero.iter().filter(|&&n| n == 1).count() as f32;
        let share = to_one / from_zero.len() as f32;
        assert!(
            (share - 0.75).abs() < 0.02,
            "expected ~0.75 of hops to the weight-3 neighbour, got {share}"
        );
    }

    #[test]
    fn test_unweighted_graph_skips_the_search() {
        let g = toy();
        assert!(!g.weighted);
        let mp = Metapath::parse("a-b-a", g.type_names()).unwrap();
        let (walks, _) = g.generate_walks(&mp, 200, 3, 1).unwrap();
        // Node 0 has two b neighbours, sampled uniformly.
        let seconds: Vec<u32> = walks.iter().map(|w| w[1]).collect();
        let twos = seconds.iter().filter(|&&n| n == 2).count() as f32;
        let share = twos / seconds.len() as f32;
        assert!((share - 0.5).abs() < 0.1, "got {share}");
    }
}
