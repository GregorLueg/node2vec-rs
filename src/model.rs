use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::activation;
use burn::tensor::{Int, Tensor, backend::Backend};

/// SkipGram model for word embeddings.
///
/// Uses negative sampling to train word embeddings by predicting context words
/// from target words.
///
/// ### Fields
///
/// * `target_embd` - The target embedding
/// * `context_embd` - The context embedding
/// * `vocab_size` - The vocabulary size, i.e., number of nodes in node2vec
#[derive(Module, Debug)]
pub struct SkipGramModel<B: Backend> {
    target_embd: Embedding<B>,
    context_embd: Embedding<B>,
    pub vocab_size: usize,
}

/// Config for the SkipGram model
///
/// ### Fields
///
/// * `vocab_size` - Size of the vocabulary
/// * `embedding_dim` - Size of the embedding
#[derive(Config, Debug)]
pub struct SkipGramConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl SkipGramConfig {
    /// Initialise the model
    ///
    /// ### Params
    ///
    /// * `device` - The device on which to run the model
    ///
    /// ### Returns
    ///
    /// Initialised model
    pub fn init<B: Backend>(&self, device: &B::Device) -> SkipGramModel<B> {
        SkipGramModel {
            target_embd: EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device),
            context_embd: EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device),
            vocab_size: self.vocab_size,
        }
    }
}

impl<B: Backend> SkipGramModel<B> {
    /// Forward pass computing the loss for a batch.
    ///
    /// ### Params
    ///
    /// * `targets` - Target word indices [batch_size]
    /// * `contexts` - Context word indices [batch_size]
    /// * `negatives` - Negative sample indices [batch_size, num_negatives]
    ///
    /// ### Returns
    ///
    /// Loss tensor [batch_size]
    pub fn forward(
        &self,
        targets: Tensor<B, 1, Int>,
        contexts: Tensor<B, 1, Int>,
        negatives: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let targets_2d: Tensor<B, 2, Int> = targets.clone().unsqueeze_dim(1);
        let contexts_2d: Tensor<B, 2, Int> = contexts.unsqueeze_dim(1);

        let batch_size = targets.dims()[0];
        let num_neg = negatives.dims()[1];

        // [batch_size, 1, embedding_dim] -> [batch_size, embedding_dim]
        let target_embed_3d = self.target_embd.forward(targets_2d);
        let embedding_dim = target_embed_3d.dims()[2];
        let target_embed: Tensor<B, 2> = target_embed_3d.reshape([batch_size, embedding_dim]);

        let context_embed_3d = self.context_embd.forward(contexts_2d);
        let context_embed: Tensor<B, 2> = context_embed_3d.reshape([batch_size, embedding_dim]);

        // [batch_size, embedding_dim] -> sum_dim(1) -> [batch_size, 1] -> reshape -> [batch_size]
        let pos_dot_2d: Tensor<B, 2> = (target_embed.clone() * context_embed).sum_dim(1);
        let pos_dot: Tensor<B, 1> = pos_dot_2d.reshape([batch_size]);

        let neg_embed: Tensor<B, 3> = self.context_embd.forward(negatives);
        let target_expanded: Tensor<B, 3> = target_embed.unsqueeze_dim(1);

        // [batch_size, num_neg, embedding_dim] -> sum_dim(2) -> [batch_size, num_neg]
        let neg_dot: Tensor<B, 2> = (target_expanded * neg_embed)
            .sum_dim(2)
            .reshape([batch_size, num_neg]);

        // Loss calculation
        let pos_loss: Tensor<B, 1> = activation::log_sigmoid(pos_dot).neg();

        let neg_loss_2d: Tensor<B, 2> = activation::log_sigmoid(neg_dot.neg()).sum_dim(1);
        let neg_loss: Tensor<B, 1> = neg_loss_2d.reshape([batch_size]).neg();

        pos_loss + neg_loss
    }

    /// Returns a reference to the target embeddings.
    ///
    /// ### Returns
    ///
    /// Reference to the internal embedding
    pub fn embeddings(&self) -> &Embedding<B> {
        &self.target_embd
    }
}
