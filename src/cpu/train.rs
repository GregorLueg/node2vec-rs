use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::cpu::matrix::{Matrix, MatrixWrapper};
use crate::cpu::word2vec_model::Word2Vec;

/////////////
// Helpers //
/////////////

/// Train arguments for CPU implementation.
///
/// ### Fields
///
/// * `dim` - Dimension of the embedding vectors.
/// * `lr` - Learning rate.
/// * `epochs` - Number of epochs to train.
/// * `neg` - Number of negative samples to use.
/// * `window` - Window size for context words.
/// * `lr_update_rate` - Learning rate update rate.
/// * `n_threads` - Number of threads to use.
/// * `verbose` - Whether to print progress.
pub struct CpuTrainArgs {
    pub dim: usize,
    pub lr: f32,
    pub epochs: usize,
    pub neg: usize,
    pub window: usize,
    pub lr_update_rate: usize,
    pub n_threads: usize,
    pub verbose: bool,
}

/// Skipgram training on a single walk
///
/// ### Params
///
/// * `model` - The Word2Vec model
/// * `walk` - The walk (sequence of node IDs)
/// * `rng` - Random number generator
/// * `window_dist` - Uniform distribution for sampling window size
fn skipgram(model: &mut Word2Vec, walk: &[u32], rng: &mut StdRng, window_dist: &Uniform<usize>) {
    let length = walk.len();
    for w in 0..length {
        let bound = window_dist.sample(rng);
        let start = w.saturating_sub(bound);
        let end = (w + bound + 1).min(length);

        for c in start..end {
            if c != w {
                model.update(walk[w] as usize, walk[c] as usize);
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
/// * `processed_tokens` - The number of processed tokens
/// * `total_tokens` - The total number of tokens
/// * `thread_id` - The thread ID
/// * `seed` - The random seed
/// * `progress` - The progress bar
fn train_thread(
    walks: &[Vec<u32>],
    input: &Arc<MatrixWrapper>,
    output: &Arc<MatrixWrapper>,
    args: &CpuTrainArgs,
    neg_table: &Arc<Vec<usize>>,
    processed_tokens: &Arc<AtomicUsize>,
    total_tokens: usize,
    thread_id: usize,
    seed: usize,
    progress: Option<&ProgressBar>,
) {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let window_dist = Uniform::new(1, args.window + 1).unwrap();

    // get mutable access to matrices
    let input_ptr = input.inner.get();
    let output_ptr = output.inner.get();

    // seed
    let neg_start = seed.wrapping_add(thread_id) as usize;

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
        skipgram(&mut model, walk, &mut rng, &window_dist);
        local_token_count += walk.len();

        // update learning rate periodically
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

    // final update
    processed_tokens.fetch_add(local_token_count, Ordering::SeqCst);
}

/// Create negative sampling table
///
/// ### Params
///
/// * `vocab_size` - Number of unique nodes
/// * `walks` - The walks to compute node frequencies from
/// * `neg_table_size` - Size of negative sampling table
///
/// ### Returns
///
/// Negative sampling table as Arc<Vec<usize>>
pub fn create_negative_table(
    vocab_size: usize,
    walks: &[Vec<u32>],
    neg_table_size: usize,
    seed: usize,
) -> Arc<Vec<usize>> {
    const NEG_POW: f64 = 0.75;

    // count node frequencies
    let mut counts = vec![0u32; vocab_size];
    for walk in walks {
        for &node in walk {
            counts[node as usize] += 1;
        }
    }

    // build negative sampling table
    let mut negative_table = Vec::new();
    let mut z = 0.0;
    for &count in &counts {
        z += (count as f64).powf(NEG_POW);
    }

    for (idx, &count) in counts.iter().enumerate() {
        if count > 0 {
            let c = (count as f64).powf(NEG_POW);
            let n_samples = (c * neg_table_size as f64 / z) as usize;
            for _ in 0..n_samples {
                negative_table.push(idx);
            }
        }
    }

    // shuffle
    let mut rng = StdRng::seed_from_u64(seed as u64);
    negative_table.shuffle(&mut rng);

    Arc::new(negative_table)
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
    walks: Vec<Vec<u32>>,
    vocab_size: usize,
    args: CpuTrainArgs,
    neg_table: Arc<Vec<usize>>,
    seed: usize,
) -> (Matrix, Matrix) {
    // Initialise matrices
    let mut input_mat = Matrix::new(vocab_size, args.dim);
    let mut output_mat = Matrix::new(vocab_size, args.dim);

    input_mat.uniform(1.0 / args.dim as f32, seed);
    output_mat.zero();

    let input = Arc::new(input_mat.make_send());
    let output = Arc::new(output_mat.make_send());

    // Calculate total tokens for progress tracking
    let total_tokens: usize = walks.iter().map(|w| w.len()).sum();
    let total_tokens_all_epochs = total_tokens * args.epochs;

    let processed_tokens = Arc::new(AtomicUsize::new(0));

    if args.verbose {
        println!(
            "Training on {} walks ({} tokens per epoch, {} total)",
            walks.len(),
            total_tokens,
            total_tokens_all_epochs
        );
    }

    let progress = if args.verbose {
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

    // Split walks across threads
    let walks_per_thread = (walks.len() + args.n_threads - 1) / args.n_threads;
    let walk_chunks: Vec<Vec<Vec<u32>>> = walks
        .chunks(walks_per_thread)
        .map(|chunk| chunk.to_vec())
        .collect();

    // Train for multiple epochs
    for epoch in 0..args.epochs {
        if args.verbose {
            println!("\nEpoch {}/{}", epoch + 1, args.epochs);
        }

        // Parallel training across threads
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

    // Unwrap matrices
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
