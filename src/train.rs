use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::batch::*;
use crate::dataset::*;
use crate::model::*;

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

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub walks_per_node: usize,
    #[config(default = 10)]
    pub walk_length: usize,
    #[config(default = 2)]
    pub window_size: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 4)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub num_negatives: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: SkipGramConfig,
    training_config: TrainingConfig,
    train_walks: Vec<Vec<u32>>,
    valid_walks: Vec<Vec<u32>>,
    device: B::Device,
) {
    let mut model = config.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let batcher = SkipGramBatcher::new(training_config.window_size);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(WalkDataset::new(train_walks));

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .num_workers(training_config.num_workers)
        .build(WalkDataset::new(valid_walks));

    for epoch in 1..=training_config.num_epochs {
        // Training
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let batch_size = batch.contexts.dims()[0];
            let negatives = sample_negatives(
                batch_size,
                model.vocab_size,
                training_config.num_negatives,
                &batch.contexts.device(),
            );
            let loss = model
                .forward(batch.contexts, batch.targets, negatives)
                .mean();

            let loss_scalar: f64 = loss.clone().into_scalar().elem();

            if iteration % 10 == 0 {
                println!(
                    "[Train - Epoch {} - Iteration {}] Loss {:.6}",
                    epoch, iteration, loss_scalar
                );
            }

            total_loss += loss_scalar;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(training_config.learning_rate, model, grads);
        }

        println!(
            "[Train - Epoch {}] Average Loss: {:.6}",
            epoch,
            total_loss / num_batches as f64
        );

        // Validation
        let model_valid = model.valid();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in dataloader_valid.iter() {
            let batch_size = batch.contexts.dims()[0];
            let negatives = sample_negatives(
                batch_size,
                model_valid.vocab_size,
                5,
                &batch.contexts.device(),
            );
            let loss = model_valid
                .forward(batch.contexts, batch.targets, negatives)
                .mean();

            let loss_scalar: f64 = loss.into_scalar().elem();

            total_loss += loss_scalar;
            num_batches += 1;
        }

        println!(
            "[Valid - Epoch {}] Average Loss: {:.6}\n",
            epoch,
            total_loss / num_batches as f64
        );
    }

    std::fs::create_dir_all(artifact_dir).ok();
    // model
    //     .save_file(
    //         format!("{artifact_dir}/model"),
    //         &burn::record::CompactRecorder::new(),
    //     )
    //     .expect("Model should save successfully");
}
