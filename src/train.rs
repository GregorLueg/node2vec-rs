use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn::train::LearnerBuilder;
use burn::train::metric::LossMetric;
use burn::train::{TrainOutput, TrainStep, ValidStep};
use rand::Rng;

use crate::batch::*;
use crate::dataset::*;
use crate::model::*;

// Custom output for skip-gram
pub struct SkipGramOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> SkipGramOutput<B> {
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

// Helper to sample negatives
fn sample_negatives<B: Backend>(
    batch_size: usize,
    vocab_size: usize,
    num_neg: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut rng = rand::rng();
    let data: Vec<i64> = (0..batch_size * num_neg)
        .map(|_| rng.random_range(0..vocab_size) as i64)
        .collect();

    Tensor::from_data(
        TensorData::new(data, [batch_size, num_neg]).convert::<B::IntElem>(),
        device,
    )
}

// Implement TrainStep
impl<B: AutodiffBackend> TrainStep<SkipGramBatch<B>, SkipGramOutput<B>> for SkipGramModel<B> {
    fn step(&self, batch: SkipGramBatch<B>) -> TrainOutput<SkipGramOutput<B>> {
        let batch_size = batch.contexts.dims()[0];
        let negatives = sample_negatives(batch_size, self.vocab_size, 5, &batch.contexts.device());

        let loss = self.forward(batch.contexts, batch.targets, negatives);
        let loss_mean = loss.mean();

        TrainOutput::new(self, loss_mean.backward(), SkipGramOutput::new(loss_mean))
    }
}

// Training function
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: SkipGramConfig,
    train_walks: Vec<Vec<u32>>,
    valid_walks: Vec<Vec<u32>>,
    device: B::Device,
) {
    // Create dataloaders
    let batcher_train = SkipGramBatcher::new(5);
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(64)
        .shuffle(42)
        .num_workers(4)
        .build(WalkDataset::new(train_walks));

    let dataloader_valid = DataLoaderBuilder::new(batcher_train)
        .batch_size(64)
        .build(WalkDataset::new(valid_walks));

    // Create model and optimizer
    let model = config.init(&device);
    let optim = AdamConfig::new().init();

    // Build learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(10)
        .summary()
        .build(model, optim, 1e-3);

    let model = learner.fit(dataloader_train, dataloader_valid);
}
