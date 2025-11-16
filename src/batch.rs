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

#[cfg(test)]
mod batch_tests {
    fn generate_skipgram_pairs(walk: &[u32], window_size: usize) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();

        for (center_idx, &center_node) in walk.iter().enumerate() {
            let start = center_idx.saturating_sub(window_size);
            let end = (center_idx + window_size + 1).min(walk.len());

            for target_idx in start..end {
                if target_idx != center_idx {
                    pairs.push((center_node, walk[target_idx]));
                }
            }
        }

        pairs
    }

    #[test]
    fn test_skipgram_pairs_window_size_1() {
        let walk = vec![1, 2, 3, 4];
        let pairs = generate_skipgram_pairs(&walk, 1);

        // Expected pairs with window size 1
        let expected = vec![
            (1, 2), // center=1, context=2
            (2, 1), // center=2, context=1
            (2, 3), // center=2, context=3
            (3, 2), // center=3, context=2
            (3, 4), // center=3, context=4
            (4, 3), // center=4, context=3
        ];

        assert_eq!(pairs.len(), expected.len());
        for pair in expected {
            assert!(pairs.contains(&pair));
        }
    }

    #[test]
    fn test_skipgram_pairs_window_size_2() {
        let walk = vec![1, 2, 3, 4, 5];
        let pairs = generate_skipgram_pairs(&walk, 2);

        // Expected pairs with window size 2
        let expected = vec![
            // center=1 (index 0)
            (1, 2),
            (1, 3),
            // center=2 (index 1)
            (2, 1),
            (2, 3),
            (2, 4),
            // center=3 (index 2)
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 5),
            // center=4 (index 3)
            (4, 2),
            (4, 3),
            (4, 5),
            // center=5 (index 4)
            (5, 3),
            (5, 4),
        ];

        assert_eq!(pairs.len(), expected.len());
        for pair in expected {
            assert!(pairs.contains(&pair), "Missing pair: {:?}", pair);
        }
    }

    #[test]
    fn test_no_self_pairs() {
        let walk = vec![1, 2, 3, 4];
        let pairs = generate_skipgram_pairs(&walk, 2);

        // No (x, x) pairs should exist
        for (context, target) in pairs {
            assert_ne!(context, target);
        }
    }

    #[test]
    fn test_single_node_walk() {
        let walk = vec![1];
        let pairs = generate_skipgram_pairs(&walk, 2);

        // Single node should produce no pairs
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_two_node_walk() {
        let walk = vec![1, 2];
        let pairs = generate_skipgram_pairs(&walk, 1);

        let expected = vec![(1, 2), (2, 1)];

        assert_eq!(pairs.len(), expected.len());
        for pair in expected {
            assert!(pairs.contains(&pair));
        }
    }

    #[test]
    fn test_window_size_larger_than_walk() {
        let walk = vec![1, 2, 3];
        let pairs = generate_skipgram_pairs(&walk, 10);

        // Should still only consider actual walk length
        let expected = vec![(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)];

        assert_eq!(pairs.len(), expected.len());
        for pair in expected {
            assert!(pairs.contains(&pair));
        }
    }

    #[test]
    fn test_window_size_zero() {
        let walk = vec![1, 2, 3, 4];
        let pairs = generate_skipgram_pairs(&walk, 0);

        // Window size 0 should produce no pairs
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_repeated_nodes_in_walk() {
        let walk = vec![1, 2, 1, 3];
        let pairs = generate_skipgram_pairs(&walk, 1);

        // Should handle repeated nodes correctly
        assert!(pairs.contains(&(1, 2))); // first 1
        assert!(pairs.contains(&(2, 1))); // second appearance of 1
        assert!(pairs.contains(&(1, 3))); // second 1
    }

    #[test]
    fn test_pair_count_formula() {
        let walk = vec![1, 2, 3, 4, 5, 6];
        let window_size = 2;
        let pairs = generate_skipgram_pairs(&walk, window_size);

        // For interior nodes, should have 2*window_size pairs
        // For edge nodes, should have fewer
        // Total calculation: each node contributes min(2*window_size, viable_context_size) pairs

        // Node at index 0: can see 2 nodes ahead = 2 pairs
        // Node at index 1: can see 1 behind, 2 ahead = 3 pairs
        // Node at index 2: can see 2 behind, 2 ahead = 4 pairs
        // Node at index 3: can see 2 behind, 2 ahead = 4 pairs
        // Node at index 4: can see 2 behind, 1 ahead = 3 pairs
        // Node at index 5: can see 2 behind = 2 pairs
        // Total = 2 + 3 + 4 + 4 + 3 + 2 = 18

        assert_eq!(pairs.len(), 18);
    }

    #[test]
    fn test_batch_multiple_walks() {
        let walks = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let window_size = 1;

        let mut all_pairs = Vec::new();
        for walk in walks {
            let pairs = generate_skipgram_pairs(&walk, window_size);
            all_pairs.extend(pairs);
        }

        // Should have 4 pairs from each walk
        assert_eq!(all_pairs.len(), 8);
    }
}
