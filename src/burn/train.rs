use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{ElementConversion, Module};
use burn::record::CompactRecorder;
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor, TensorData};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::burn::batch::*;
use crate::burn::dataset::*;
use crate::burn::model::*;
use crate::Args;

/// Sample negative examples
///
/// ### Params
///
/// * `batch_size` - Size of the batch
/// * `vocab_size` - Vocabulary size
/// * `num_neg` - Number of negative examples to use
/// * `device` - Device on which to run the tensors
///
/// ### Returns
///
/// Tensor of negative examples
pub fn sample_negatives<B: Backend>(
    batch_size: usize,
    vocab_size: usize,
    num_neg: usize,
    device: &B::Device,
    seed: u64,
) -> Tensor<B, 2, Int> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<i64> = (0..batch_size * num_neg)
        .map(|_| rng.random_range(0..vocab_size) as i64)
        .collect();

    Tensor::from_data(
        TensorData::new(data, [batch_size, num_neg]).convert::<B::IntElem>(),
        device,
    )
}

/// Training configuration
///
/// ### Fields
///
/// * `walks_per_node` - Walks per node. Defaults to `20`.
/// * `walk_length` - Walk length per random walk. Defaults to `20`.
/// * `window_size` - The window size parameter for the SkipGram model. Defaults
///   to `2`.
/// * `batch_size` - Batch size for the training. Defaults to `256`.
/// * `num_workers` - Number of workers to use for the generation of batches.
///   Defaults to `4`.
/// * `num_epochs` - Number of epochs to run the algorithm for. Defaults to `5`.
/// * `num_negatives` - Number of negative samples to generate. Defaults to `5`.
/// * `seed` - Random seed for reproducibility.
/// * `learning_rate` - Learning rate for the Adam optimiser. Defaults to
///   `1e-3`.
/// * `p` - p parameter for the node2vec random walks and controls the
///   probability to return to origin node. Defaults to `1.0`.
/// * `q` - q parameter for node2vec random walks and controls the probability
///   to venture on a different node from the origin node. Defaults to `1.0`.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 20)]
    pub walks_per_node: usize,
    #[config(default = 20)]
    pub walk_length: usize,
    #[config(default = 2)]
    pub window_size: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 5)]
    pub num_negatives: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
    #[config(default = 1_f32)]
    pub q: f32,
    #[config(default = 1_f32)]
    pub p: f32,
}

impl TrainingConfig {
    /// Generate a config from the arguments
    ///
    /// ### Params
    ///
    /// * `args` - The Clap parse arguments
    ///
    /// ### Returns
    ///
    /// Self with all of the parameters provided via the command line.
    pub fn from_args(args: &Args) -> Self {
        Self {
            walks_per_node: args.walks_per_node,
            walk_length: args.walk_length,
            window_size: args.window_size,
            batch_size: args.batch_size,
            num_workers: args.num_workers,
            num_epochs: args.num_epochs,
            num_negatives: args.num_negatives,
            seed: args.seed as u64,
            learning_rate: args.learning_rate,
            p: args.p,
            q: args.q,
        }
    }
}

/// Train function for node2vec
///
/// ### Parameters
///
/// * `artifact_dir` - Where to store the artifacts of the training.
/// * `model_config` - The model configuration. Contains the embedding
///   dimensions and vocabulary size.
/// * `training_config` - Contains the configuration for the training of the
///   model.
/// * `train_walks` - The random walks for training of the model.
/// * `valid_walks` - The random walks for validation of the model.
/// * `device` - The device on which to run the training.
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    model_config: &SkipGramConfig,
    training_config: &TrainingConfig,
    train_walks: &[Vec<u32>],
    valid_walks: &[Vec<u32>],
    device: B::Device,
    seed: usize,
) -> SkipGramModel<B> {
    let mut model = model_config.init::<B>(&device);
    let mut optim = AdamConfig::new().init();
    let batcher = SkipGramBatcher::new(training_config.window_size);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(WalkDataset::new(train_walks.to_vec()));

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .num_workers(training_config.num_workers)
        .build(WalkDataset::new(valid_walks.to_vec()));

    let train_batches = dataloader_train.iter().count();
    let valid_batches = dataloader_valid.iter().count();

    let epoch_bar = ProgressBar::new(training_config.num_epochs as u64);
    epoch_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} epochs")
            .unwrap()
            .progress_chars("##-"),
    );

    for epoch in 1..=training_config.num_epochs {
        // Training
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let train_bar = ProgressBar::new(train_batches as u64);
        train_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[Epoch {msg}] {bar:40.green/yellow} {pos}/{len} batches | Loss: {prefix}",
                )
                .unwrap()
                .progress_chars("=>-"),
        );
        train_bar.set_message(epoch.to_string());

        for (batch_idx, batch) in dataloader_train.iter().enumerate() {
            let neg_seed = seed.wrapping_mul(epoch + 1).wrapping_add(batch_idx);
            let batch_size = batch.centers.dims()[0];
            let negatives = sample_negatives(
                batch_size,
                model.vocab_size,
                training_config.num_negatives,
                &batch.centers.device(),
                neg_seed as u64,
            );

            let loss = model
                .forward(batch.centers, batch.contexts, negatives)
                .mean();

            let loss_scalar: f64 = loss.clone().into_scalar().elem();
            total_loss += loss_scalar;
            num_batches += 1;

            train_bar.set_prefix(format!("{:.6}", loss_scalar));
            train_bar.inc(1);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(training_config.learning_rate, model, grads);
        }

        train_bar.finish_with_message(format!("Avg Loss: {:.6}", total_loss / num_batches as f64));

        // Validation
        let model_valid = model.valid();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let valid_bar = ProgressBar::new(valid_batches as u64);
        valid_bar.set_style(
            ProgressStyle::default_bar()
                .template("[Validation] {bar:40.magenta/blue} {pos}/{len} batches | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        for (batch_idx, batch) in dataloader_valid.iter().enumerate() {
            let neg_seed = seed
                .wrapping_mul(epoch + 1)
                .wrapping_add(batch_idx)
                .wrapping_add(usize::MAX / 2);
            let batch_size = batch.centers.dims()[0];
            let negatives = sample_negatives(
                batch_size,
                model_valid.vocab_size,
                5,
                &batch.centers.device(),
                neg_seed as u64,
            );
            let loss = model_valid
                .forward(batch.centers, batch.contexts, negatives)
                .mean();

            let loss_scalar: f64 = loss.into_scalar().elem();
            total_loss += loss_scalar;
            num_batches += 1;

            valid_bar.inc(1);
        }

        valid_bar.finish_with_message(format!(
            "Avg. Validation Loss: {:.6}",
            total_loss / num_batches as f64
        ));
        println!("--- Epoch done ---");
        epoch_bar.inc(1);
    }

    epoch_bar.finish_with_message("Training complete");

    std::fs::create_dir_all(artifact_dir).ok();
    model
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    model
}
