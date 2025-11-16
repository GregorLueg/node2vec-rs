#![allow(clippy::needless_range_loop)]

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

/// Batch structure for skip-gram training containing context-target pairs
///
/// ### Fields
///
/// * `contexts` - The context tokens
/// * `targets` - The target tokens
#[derive(Clone, Debug)]
pub struct SkipGramBatch<B: Backend> {
    pub contexts: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1, Int>,
}

/// Batcher that converts random walks into skip-gram training pairs
///
/// ### Fields
///
/// * `window_size` - The window size of the centre word.
#[derive(Clone)]
pub struct SkipGramBatcher {
    window_size: usize,
}

impl SkipGramBatcher {
    /// Creates a new batcher with the specified context window size
    ///
    /// ### Params
    ///
    /// * `window_size` - Number of words to consider on each side of the centre
    ///   word
    ///
    /// ### Returns
    ///
    /// Initialised self.
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl<B: Backend> Batcher<B, Vec<u32>, SkipGramBatch<B>> for SkipGramBatcher {
    /// Generate a batch
    ///
    /// ### Params
    ///
    /// * `items` - The random walks for node2vec
    /// * `device` - The device on which to store the tensors
    ///
    /// ### Return
    ///
    /// The `SkipGramBatch` with contexts and targets.
    fn batch(&self, items: Vec<Vec<u32>>, device: &B::Device) -> SkipGramBatch<B> {
        let capacity = items.iter().map(|w| w.len() * self.window_size * 2).sum();
        let mut contexts = Vec::with_capacity(capacity);
        let mut targets = Vec::with_capacity(capacity);

        for walk in items {
            for (center_idx, &center_node) in walk.iter().enumerate() {
                let start = center_idx.saturating_sub(self.window_size);
                let end = (center_idx + self.window_size + 1).min(walk.len());

                for target_idx in start..end {
                    if target_idx != center_idx {
                        contexts.push(center_node as i64);
                        targets.push(walk[target_idx] as i64);
                    }
                }
            }
        }

        let n = contexts.len();
        let contexts_data = TensorData::new(contexts, [n]).convert::<B::IntElem>();
        let targets_data = TensorData::new(targets, [n]).convert::<B::IntElem>();

        SkipGramBatch {
            contexts: Tensor::from_data(contexts_data, device),
            targets: Tensor::from_data(targets_data, device),
        }
    }
}
