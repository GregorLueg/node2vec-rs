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
/// * `sample` - Subsampling threshold; nodes with frequency above this are
///   randomly dropped during training.
#[derive(Clone, Debug)]
pub struct CpuTrainArgs {
    pub dim: usize,
    pub lr: f32,
    pub epochs: usize,
    pub neg: usize,
    pub window: usize,
    pub lr_update_rate: usize,
    pub n_threads: usize,
    pub verbose: bool,
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
    neg_table: &Arc<Vec<usize>>,
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

/// Create negative sampling table
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
    mut walks: Vec<Vec<u32>>,
    vocab_size: usize,
    args: CpuTrainArgs,
    neg_table: Arc<Vec<usize>>,
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
