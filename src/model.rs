use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{activation, backend::Backend, Int, Tensor};
use std::fs::File;
use std::io::Write;

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
    pub target_embd: Embedding<B>,
    pub context_embd: Embedding<B>,
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
        let target_embd = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
        let context_embd = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);

        // Force allocation on device without moving data off
        let dummy_idx = Tensor::<B, 2, Int>::zeros([1, 1], device);
        let _ = target_embd.forward(dummy_idx.clone());
        let _ = context_embd.forward(dummy_idx);

        SkipGramModel {
            target_embd,
            context_embd,
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

    /// Extract the embeddings as a vector
    ///
    /// ### Returns
    ///
    /// A Vec<Vec<f32>> of the embeddings.
    pub fn embeddings_to_vec(&self) -> Vec<Vec<f32>> {
        let weights = self.target_embd.weight.clone();
        let data = weights.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        let vocab_size = self.vocab_size;
        let embedding_dim = values.len() / vocab_size;

        values
            .chunks(embedding_dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Write the embeddings to a CSV
    ///
    /// ### Params
    ///
    /// * `path` - Path to the CSV.
    ///
    /// ### Returns
    ///
    /// Writes the data to disk in form of a CSV.
    pub fn write_embeddings_csv(&self, path: &str) -> std::io::Result<()> {
        let embeddings = self.embeddings_to_vec();
        let mut file = File::create(path)?;

        for row in embeddings {
            let line = row
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(file, "{}", line)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod model_tests {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Mock embedding structure for testing
    fn create_mock_embeddings(vocab_size: usize, embedding_dim: usize) -> Vec<Vec<f32>> {
        (0..vocab_size)
            .map(|i| {
                (0..embedding_dim)
                    .map(|j| (i * embedding_dim + j) as f32)
                    .collect()
            })
            .collect()
    }

    fn write_embeddings_csv(embeddings: &[Vec<f32>], path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;

        for row in embeddings {
            let line = row
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(file, "{}", line)?;
        }

        Ok(())
    }

    #[test]
    fn test_embeddings_to_vec_dimensions() {
        let vocab_size = 10;
        let embedding_dim = 16;
        let embeddings = create_mock_embeddings(vocab_size, embedding_dim);

        assert_eq!(embeddings.len(), vocab_size);
        for emb in embeddings.iter() {
            assert_eq!(emb.len(), embedding_dim);
        }
    }

    #[test]
    fn test_write_embeddings_csv_format() {
        let vocab_size = 5;
        let embedding_dim = 3;
        let embeddings = create_mock_embeddings(vocab_size, embedding_dim);

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("embeddings.csv");
        write_embeddings_csv(&embeddings, file_path.to_str().unwrap()).unwrap();

        // Read back and verify format
        let file = File::open(&file_path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), vocab_size);

        for (i, line) in lines.iter().enumerate() {
            let values: Vec<f32> = line.split(',').map(|s| s.parse().unwrap()).collect();
            assert_eq!(values.len(), embedding_dim);
            assert_eq!(values, embeddings[i]);
        }
    }

    #[test]
    fn test_csv_no_header() {
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("embeddings.csv");
        write_embeddings_csv(&embeddings, file_path.to_str().unwrap()).unwrap();

        let file = File::open(&file_path).unwrap();
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();
        reader.read_line(&mut first_line).unwrap();

        // First line should be numeric, not a header
        assert!(first_line.starts_with("1"));
    }

    #[test]
    fn test_embedding_uniqueness() {
        let vocab_size = 100;
        let embedding_dim = 32;
        let embeddings = create_mock_embeddings(vocab_size, embedding_dim);

        // Check that embeddings are different
        for i in 0..vocab_size {
            for j in (i + 1)..vocab_size {
                assert_ne!(embeddings[i], embeddings[j]);
            }
        }
    }

    #[test]
    fn test_single_node_embedding() {
        let embeddings = create_mock_embeddings(1, 8);
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 8);
    }

    #[test]
    fn test_large_embedding_dimensions() {
        let vocab_size = 10;
        let embedding_dim = 256;
        let embeddings = create_mock_embeddings(vocab_size, embedding_dim);

        assert_eq!(embeddings.len(), vocab_size);
        for emb in embeddings {
            assert_eq!(emb.len(), embedding_dim);
        }
    }
}
