use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

#[derive(Clone)]
pub struct SkipGramBatch<B: Backend> {
    pub contexts: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct SkipGramBatcher {
    window_size: usize,
}

impl SkipGramBatcher {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl<B: Backend> Batcher<B, Vec<u32>, SkipGramBatch<B>> for SkipGramBatcher {
    fn batch(&self, items: Vec<Vec<u32>>, device: &B::Device) -> SkipGramBatch<B> {
        let mut contexts = Vec::new();
        let mut targets = Vec::new();

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
