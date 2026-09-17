//! Train loop for the CPU-based version of node2vec-rs. Uses Hogwild! SGD
//! to fit the model.

use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thousands::*;

use crate::cpu::matrix::{Matrix, MatrixWrapper};
use crate::cpu::word2vec_model::Word2Vec;

////////////////
// Constansts //
////////////////

/// Exponent applied to walk-token counts when building a negative table.
///
/// Mikolov's unigram^0.75, which flattens the distribution enough that rare
/// nodes still turn up as negatives.
const NEG_POW: f64 = 0.75;

/// Floor on the size of a per-type negative table.
///
/// Sizing purely by a type's share of walk tokens starves a rare type down to
/// a handful of entries, at which point the shuffled cursor cycles visibly and
/// the same few negatives come back. 64k is enough to hide that.
const MIN_TYPE_TABLE_SIZE: usize = 64_000;

////////////////////
// NegativeTable //
////////////////////

/// Where negative samples are drawn from.
///
/// Plain metapath2vec, and node2vec, draw from one global unigram^0.75 table.
/// metapath2vec++ normalises the softmax per node type, so a context node of
/// type `t` needs a negative of type `t`; that is the only place in the whole
/// trainer where a node type is visible.
///
/// ### References
///
/// Dong, Chawla and Swami, *metapath2vec*, KDD 2017, section 4.2.
#[derive(Clone, Debug)]
pub enum NegativeTable {
    /// One table over every node, shuffled.
    Global(Arc<Vec<u32>>),
    /// One shuffled table per node type, selected by the context node's type.
    PerType {
        /// Tables indexed by dense type id.
        tables: Vec<Arc<Vec<u32>>>,
        /// Node type per dense node id.
        node_type: Arc<Vec<u8>>,
    },
}

impl NegativeTable {
    /// Number of independent cursors a model needs to walk this table.
    ///
    /// ### Returns
    ///
    /// One for the global table, one per type otherwise.
    #[inline]
    pub fn n_cursors(&self) -> usize {
        match self {
            Self::Global(_) => 1,
            Self::PerType { tables, .. } => tables.len(),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Train arguments for CPU implementation.
#[derive(Clone, Debug)]
pub struct CpuTrainArgs {
    /// Dimension of the embedding vectors.
    pub dim: usize,
    /// Learning rate.
    pub lr: f32,
    /// Number of epochs to train.
    pub epochs: usize,
    /// Number of negative samples to use.
    pub neg: usize,
    /// Window size for context words.
    pub window: usize,
    /// Learning rate update rate.
    pub lr_update_rate: usize,
    /// Number of threads to use.
    pub n_threads: usize,
    /// Whether to print progress.
    pub verbose: bool,
    /// Subsampling threshold; nodes with frequency above this are randomly
    /// dropped during training.
    pub sample: f32,
}

/// Skipgram training on a single walk
///
/// ### Params
///
/// * `model` - The Word2Vec model
/// * `walk` - The walk (sequence of node IDs)
/// * `rng` - Random number generator
/// * `window_dist` - Uniform distribution for sampling window size
fn skipgram(
    model: &mut Word2Vec,
    walk: &[u32],
    rng: &mut StdRng,
    window_dist: &Uniform<usize>,
    keep_probs: &[f32],
) {
    // subsample the walk: drop highly frequent nodes to bring rare nodes closer
    let mut active_walk = Vec::with_capacity(walk.len());
    for &node in walk {
        let prob = keep_probs[node as usize];
        if prob >= 1.0 || rng.random::<f32>() < prob {
            active_walk.push(node);
        }
    }

    // train on the active walk
    let length = active_walk.len();
    for w in 0..length {
        let bound = window_dist.sample(rng);
        let start = w.saturating_sub(bound);
        let end = (w + bound + 1).min(length);

        for c in start..end {
            if c != w {
                model.update(active_walk[w] as usize, active_walk[c] as usize);
            }
        }
    }
}

/// Train the model on a single thread
///
/// ### Params
///
/// * `walks` - The walks (sequences of node IDs)
/// * `input` - The input matrix
/// * `output` - The output matrix
/// * `args` - The training arguments
/// * `neg_table` - The negative sampling table
/// * `keep_probs` - Per-node probability of being kept during subsampling.
/// * `processed_tokens` - The number of processed tokens
/// * `total_tokens` - The total number of tokens
/// * `thread_id` - The thread ID
/// * `seed` - The random seed
/// * `progress` - The progress bar
#[allow(clippy::too_many_arguments)]
fn train_thread(
    walks: &[Vec<u32>],
    input: &Arc<MatrixWrapper>,
    output: &Arc<MatrixWrapper>,
    args: &CpuTrainArgs,
    neg_table: &NegativeTable,
    keep_probs: &Arc<Vec<f32>>,
    processed_tokens: &Arc<AtomicUsize>,
    total_tokens: usize,
    thread_id: usize,
    seed: usize,
    progress: Option<&ProgressBar>,
) {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let window_dist = Uniform::new(1, args.window + 1).unwrap();

    let input_ptr = input.inner.get();
    let output_ptr = output.inner.get();
    let neg_start = seed.wrapping_add(thread_id);

    let mut model = unsafe {
        Word2Vec::new(
            &mut *input_ptr,
            &mut *output_ptr,
            args.dim,
            args.lr,
            args.neg,
            neg_table.clone(),
            neg_start,
        )
    };

    let mut local_token_count = 0;

    for walk in walks {
        skipgram(&mut model, walk, &mut rng, &window_dist, keep_probs);

        // Use original walk length to pace the learning rate decay consistently
        local_token_count += walk.len();

        if local_token_count >= args.lr_update_rate {
            let global_tokens = processed_tokens.fetch_add(local_token_count, Ordering::SeqCst);
            let progress_ratio = global_tokens as f32 / total_tokens as f32;
            let new_lr = args.lr * (1.0 - progress_ratio).max(0.0001);
            model.set_lr(new_lr);

            if thread_id == 0 {
                if let Some(pb) = progress {
                    pb.set_position(global_tokens as u64);
                    pb.set_message(format!("{:.6}", new_lr));
                }
            }

            local_token_count = 0;
        }
    }

    processed_tokens.fetch_add(local_token_count, Ordering::SeqCst);
}

/// Count how often each node appears across the walks
///
/// Frequencies come from the walks rather than from graph degree, because that
/// is the distribution skip-gram actually sees.
///
/// ### Params
///
/// * `vocab_size` - Number of unique nodes
/// * `walks` - The walks to count over
///
/// ### Returns
///
/// Per-node token counts, length `vocab_size`.
fn count_tokens(vocab_size: usize, walks: &[Vec<u32>]) -> Vec<u32> {
    let mut counts = vec![0u32; vocab_size];
    for walk in walks {
        for &node in walk {
            counts[node as usize] += 1;
        }
    }
    counts
}

/// Fill a shuffled unigram^0.75 table from a set of candidate nodes
///
/// ### Params
///
/// * `counts` - Per-node token counts
/// * `candidates` - Node ids eligible for this table
/// * `table_size` - Target number of entries
/// * `seed` - Random seed for the shuffle
///
/// ### Returns
///
/// The shuffled table. Empty when no candidate appeared in any walk.
fn fill_negative_table(
    counts: &[u32],
    candidates: impl Iterator<Item = usize>,
    table_size: usize,
    seed: usize,
) -> Arc<Vec<u32>> {
    let candidates: Vec<usize> = candidates.filter(|&i| counts[i] > 0).collect();
    let z: f64 = candidates
        .iter()
        .map(|&i| (counts[i] as f64).powf(NEG_POW))
        .sum();

    let mut table = Vec::new();
    if z > 0.0 {
        for &idx in &candidates {
            let c = (counts[idx] as f64).powf(NEG_POW);
            let n_samples = (c * table_size as f64 / z) as usize;
            table.extend(std::iter::repeat_n(idx as u32, n_samples));
        }
    }

    let mut rng = StdRng::seed_from_u64(seed as u64);
    table.shuffle(&mut rng);

    Arc::new(table)
}

/// Create one global negative sampling table
///
/// ### Params
///
/// * `vocab_size` - Number of unique nodes
/// * `walks` - The walks to compute node frequencies from
/// * `neg_table_size` - Size of negative sampling table
/// * `seed` - Random seed for shuffling the negative sampling table.
///
/// ### Returns
///
/// The table, ready to wrap in [`NegativeTable::Global`].
pub fn create_negative_table(
    vocab_size: usize,
    walks: &[Vec<u32>],
    neg_table_size: usize,
    seed: usize,
) -> Arc<Vec<u32>> {
    let counts = count_tokens(vocab_size, walks);
    fill_negative_table(&counts, 0..vocab_size, neg_table_size, seed)
}

/// Create one negative sampling table per node type, for metapath2vec++
///
/// The per-type budgets share `neg_table_size` in proportion to each type's
/// share of walk tokens, so the total stays at roughly one global table's worth
/// rather than multiplying it by the type count.
///
/// ### Params
///
/// * `node_type` - Node type per dense node id
/// * `n_types` - Number of distinct node types
/// * `walks` - The walks to compute node frequencies from
/// * `neg_table_size` - Total budget across all types
/// * `seed` - Random seed for shuffling
///
/// ### Returns
///
/// Tables indexed by dense type id. A type absent from the walks gets an empty
/// table, which is unreachable: a node can only be a training target if a walk
/// visited it.
pub fn create_negative_table_per_type(
    node_type: &[u8],
    n_types: usize,
    walks: &[Vec<u32>],
    neg_table_size: usize,
    seed: usize,
) -> Vec<Arc<Vec<u32>>> {
    let counts = count_tokens(node_type.len(), walks);

    let mut per_type_tokens = vec![0u64; n_types];
    for (node, &count) in counts.iter().enumerate() {
        per_type_tokens[node_type[node] as usize] += count as u64;
    }
    let total: u64 = per_type_tokens.iter().sum();

    (0..n_types)
        .map(|t| {
            if per_type_tokens[t] == 0 {
                return Arc::new(Vec::new());
            }
            let share = per_type_tokens[t] as f64 / total as f64;
            let size = ((neg_table_size as f64 * share) as usize).max(MIN_TYPE_TABLE_SIZE);
            let candidates = (0..node_type.len()).filter(|&n| node_type[n] as usize == t);
            fill_negative_table(&counts, candidates, size, seed.wrapping_add(t))
        })
        .collect()
}

//////////
// Main //
//////////

/// Train node2vec model on generated walks
///
/// ### Params
///
/// * `walks` - The generated random walks
/// * `vocab_size` - Size of vocabulary (number of unique nodes)
/// * `args` - Training arguments
/// * `neg_table` - Negative sampling table
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Tuple of (input_matrix, output_matrix) containing learned embeddings
pub fn train_node2vec_cpu(
    mut walks: Vec<Vec<u32>>,
    vocab_size: usize,
    args: CpuTrainArgs,
    neg_table: NegativeTable,
    seed: usize,
) -> (Matrix, Matrix) {
    let mut input_mat = Matrix::new(vocab_size, args.dim);
    let mut output_mat = Matrix::new(vocab_size, args.dim);

    input_mat.uniform(1.0 / args.dim as f32, seed);
    output_mat.zero();

    let input = Arc::new(input_mat.make_send());
    let output = Arc::new(output_mat.make_send());

    let total_tokens: usize = walks.iter().map(|w| w.len()).sum();
    let total_tokens_all_epochs = total_tokens * args.epochs;

    let mut counts = vec![0usize; vocab_size];
    for walk in &walks {
        for &node in walk {
            if (node as usize) < vocab_size {
                counts[node as usize] += 1;
            }
        }
    }

    let mut keep_probs = vec![1.0f32; vocab_size];
    if args.sample > 0.0 {
        let total_f64 = total_tokens as f64;
        for (node, &count) in counts.iter().enumerate() {
            if count > 0 {
                let freq = count as f64 / total_f64;
                // Mikolov's subsampling formula
                let keep_prob =
                    ((freq / args.sample as f64).sqrt() + 1.0) * (args.sample as f64 / freq);
                keep_probs[node] = keep_prob.min(1.0) as f32;
            }
        }
    }
    let keep_probs = Arc::new(keep_probs);

    let processed_tokens = Arc::new(AtomicUsize::new(0));

    if args.verbose {
        println!(
            "Training on {} random walks ({} tokens per epoch, {} total)",
            walks.len().separate_with_underscores(),
            total_tokens.separate_with_underscores(),
            total_tokens_all_epochs.separate_with_underscores()
        );
    }

    let progress = if args.verbose && std::io::stdout().is_terminal() {
        let pb = ProgressBar::new(total_tokens_all_epochs as u64);
        pb.set_style(
                ProgressStyle::default_bar()
                    .template("[Training: {elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) | lr: {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
        Some(pb)
    } else {
        None
    };

    let start_time = Instant::now();

    for epoch in 0..args.epochs {
        if args.verbose {
            if progress.is_some() {
                println!("\nEpoch {}/{}", epoch + 1, args.epochs);
            } else {
                println!("  Epoch {}/{}", epoch + 1, args.epochs);
            }
        }

        let mut epoch_rng = StdRng::seed_from_u64(seed as u64 + epoch as u64);
        walks.shuffle(&mut epoch_rng);

        let walks_per_thread = walks.len().div_ceil(args.n_threads);
        let walk_chunks: Vec<&[Vec<u32>]> = walks.chunks(walks_per_thread).collect();

        walk_chunks
            .par_iter()
            .enumerate()
            .for_each(|(thread_id, thread_walks)| {
                train_thread(
                    thread_walks,
                    &input,
                    &output,
                    &args,
                    &neg_table,
                    &keep_probs, // <-- Pass it here
                    &processed_tokens,
                    total_tokens_all_epochs,
                    thread_id,
                    seed.wrapping_add(epoch * args.n_threads + thread_id),
                    progress.as_ref(),
                );
            });
    }

    if let Some(pb) = progress {
        pb.finish_with_message(format!("lr: {:.6}", 0.0));
    }

    if args.verbose {
        let elapsed = start_time.elapsed();
        let tokens_per_sec = total_tokens_all_epochs as f64 / elapsed.as_secs_f64();
        println!(
            "\nTraining complete in {:.2}s ({:.0} tokens/sec)",
            elapsed.as_secs_f64(),
            tokens_per_sec
        );
    }

    let input = Arc::try_unwrap(input)
        .expect("Failed to unwrap input matrix")
        .inner
        .into_inner();
    let output = Arc::try_unwrap(output)
        .expect("Failed to unwrap output matrix")
        .inner
        .into_inner();

    (input, output)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod train_tests {
    use super::*;

    /// Four nodes: 0 and 1 of type 0, 2 and 3 of type 1.
    const NODE_TYPE: [u8; 4] = [0, 0, 1, 1];

    /// Walks that visit every node, node 2 most often.
    fn walks() -> Vec<Vec<u32>> {
        vec![vec![0, 2, 0, 3], vec![1, 2, 1, 2], vec![0, 2, 1, 3]]
    }

    #[test]
    fn test_global_table_holds_only_visited_nodes() {
        let table = create_negative_table(4, &walks(), 10_000, 42);
        assert!(!table.is_empty());
        assert!(table.iter().all(|&n| n < 4));
    }

    #[test]
    fn test_per_type_tables_are_type_pure() {
        let tables = create_negative_table_per_type(&NODE_TYPE, 2, &walks(), 200_000, 42);
        assert_eq!(tables.len(), 2);
        for (t, table) in tables.iter().enumerate() {
            assert!(!table.is_empty(), "type {t} came out empty");
            assert!(
                table.iter().all(|&n| NODE_TYPE[n as usize] as usize == t),
                "type {t} table leaked another type"
            );
        }
    }

    #[test]
    fn test_per_type_budget_stays_near_the_global_one() {
        // Giving every type a full table would multiply the budget by the type
        // count; the shares must add up to roughly one table instead.
        let budget = 1_000_000;
        let tables = create_negative_table_per_type(&NODE_TYPE, 2, &walks(), budget, 42);
        let total: usize = tables.iter().map(|t| t.len()).sum();
        assert!(
            total <= budget + 2 * MIN_TYPE_TABLE_SIZE,
            "total was {total}"
        );
    }

    #[test]
    fn test_type_absent_from_the_walks_gets_an_empty_table() {
        let node_type = [0u8, 0, 1, 1, 2];
        let tables = create_negative_table_per_type(&node_type, 3, &walks(), 200_000, 42);
        assert!(tables[2].is_empty());
        assert!(!tables[0].is_empty());
    }

    #[test]
    fn test_negative_table_cursor_count() {
        let global = NegativeTable::Global(create_negative_table(4, &walks(), 1000, 1));
        assert_eq!(global.n_cursors(), 1);

        let per_type = NegativeTable::PerType {
            tables: create_negative_table_per_type(&NODE_TYPE, 2, &walks(), 1000, 1),
            node_type: Arc::new(NODE_TYPE.to_vec()),
        };
        assert_eq!(per_type.n_cursors(), 2);
    }
}
