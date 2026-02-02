use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{activation, backend::Backend, Int, Tensor, TensorData};
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
/// * `embedding_dim` - The embedding dimension
#[derive(Module, Debug)]
pub struct SkipGramModel<B: Backend> {
    pub target_embd: Embedding<B>,
    pub context_embd: Embedding<B>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
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
        // Scale initialisation to prevent sigmoid saturation
        // With scale 1/sqrt(embedding_dim), expected dot product magnitude is O(1)
        let init_scale = 1.0 / (self.embedding_dim as f64).sqrt();

        let target_embd = EmbeddingConfig::new(self.vocab_size, self.embedding_dim)
            .with_initializer(burn::nn::Initializer::Uniform {
                min: -init_scale,
                max: init_scale,
            })
            .init(device);
        let context_embd = EmbeddingConfig::new(self.vocab_size, self.embedding_dim)
            .with_initializer(burn::nn::Initializer::Uniform {
                min: -init_scale,
                max: init_scale,
            })
            .init(device);

        // Force allocation on device without moving data off
        let dummy_idx = Tensor::<B, 2, Int>::zeros([1, 1], device);
        let _ = target_embd.forward(dummy_idx.clone());
        let _ = context_embd.forward(dummy_idx);

        SkipGramModel {
            target_embd,
            context_embd,
            vocab_size: self.vocab_size,
            embedding_dim: self.embedding_dim,
        }
    }
}

impl<B: Backend> SkipGramModel<B> {
    /// Forward pass computing the loss for a batch.
    ///
    /// ### Params
    ///
    /// * `centers` - Center word indices [batch_size]
    /// * `contexts` - Context word indices [batch_size]
    /// * `negatives` - Negative sample indices [batch_size, num_negatives]
    ///
    /// ### Returns
    ///
    /// Loss tensor [batch_size]
    pub fn forward(
        &self,
        centers: Tensor<B, 1, Int>,
        contexts: Tensor<B, 1, Int>,
        negatives: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let centers_2d: Tensor<B, 2, Int> = centers.clone().unsqueeze_dim(1);
        let contexts_2d: Tensor<B, 2, Int> = contexts.unsqueeze_dim(1);

        let batch_size = centers.dims()[0];
        let num_neg = negatives.dims()[1];

        // [batch_size, 1, embedding_dim] -> [batch_size, embedding_dim]
        let center_embed_3d = self.target_embd.forward(centers_2d);
        let embedding_dim = center_embed_3d.dims()[2];
        let center_embed: Tensor<B, 2> = center_embed_3d.reshape([batch_size, embedding_dim]);

        let context_embed_3d = self.context_embd.forward(contexts_2d);
        let context_embed: Tensor<B, 2> = context_embed_3d.reshape([batch_size, embedding_dim]);

        // [batch_size, embedding_dim] -> sum_dim(1) -> [batch_size, 1] -> reshape -> [batch_size]
        let pos_dot_2d: Tensor<B, 2> = (center_embed.clone() * context_embed).sum_dim(1);
        let pos_dot: Tensor<B, 1> = pos_dot_2d.reshape([batch_size]);

        let neg_embed: Tensor<B, 3> = self.context_embd.forward(negatives);
        let center_expanded: Tensor<B, 3> = center_embed.unsqueeze_dim(1);

        // [batch_size, num_neg, embedding_dim] -> sum_dim(2) -> [batch_size, num_neg]
        let neg_dot: Tensor<B, 2> = (center_expanded * neg_embed)
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
    /// The returned vector is indexed by node ID, so embeddings[n] contains
    /// the embedding for node n. This works for both 0-indexed and 1-indexed
    /// graphs (for 1-indexed graphs, embeddings[0] will be untrained).
    ///
    /// ### Returns
    ///
    /// A Vec<Vec<f32>> of the embeddings, indexed by node ID.
    pub fn embeddings_to_vec(&self) -> Vec<Vec<f32>> {
        self.extract_embeddings(&self.target_embd)
    }

    /// Extract combined embeddings (average of target and context)
    ///
    /// This often gives better results as it uses information from both
    /// embedding matrices.
    ///
    /// ### Returns
    ///
    /// A Vec<Vec<f32>> of the combined embeddings, indexed by node ID.
    pub fn combined_embeddings_to_vec(&self) -> Vec<Vec<f32>> {
        let target = self.extract_embeddings(&self.target_embd);
        let context = self.extract_embeddings(&self.context_embd);

        target
            .into_iter()
            .zip(context)
            .map(|(t, c)| {
                t.into_iter()
                    .zip(c)
                    .map(|(tv, cv)| (tv + cv) / 2.0)
                    .collect()
            })
            .collect()
    }

    /// Internal helper to extract embeddings from an embedding layer
    fn extract_embeddings(&self, embd: &Embedding<B>) -> Vec<Vec<f32>> {
        let device = embd.weight.device();
        let vocab_size = self.vocab_size;
        let embedding_dim = self.embedding_dim;

        // Create indices for all vocabulary items [0, 1, 2, ..., vocab_size-1]
        let indices: Vec<i64> = (0..vocab_size as i64).collect();
        let idx_tensor: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(indices, [vocab_size, 1]).convert::<B::IntElem>(),
            &device,
        );

        // Get all embeddings via forward pass - this guarantees correct ordering
        // Shape: [vocab_size, 1, embedding_dim]
        let all_emb = embd.forward(idx_tensor);

        // Reshape to [vocab_size, embedding_dim]
        let all_emb_2d: Tensor<B, 2> = all_emb.reshape([vocab_size, embedding_dim]);

        // Extract row by row to guarantee correct ordering
        let mut result = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            // Select row i: narrow on dimension 0
            let row = all_emb_2d.clone().narrow(0, i, 1);
            let row_flat: Tensor<B, 1> = row.reshape([embedding_dim]);
            let data = row_flat.to_data();
            let vec: Vec<f32> = data.to_vec().unwrap();
            result.push(vec);
        }

        result
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

    /// Write the embeddings to a CSV with node IDs as first column
    ///
    /// ### Params
    ///
    /// * `path` - Path to the CSV.
    ///
    /// ### Returns
    ///
    /// Writes the data to disk in form of a CSV with node_id as first column.
    pub fn write_embeddings_csv_with_ids(&self, path: &str) -> std::io::Result<()> {
        let embeddings = self.embeddings_to_vec();
        let mut file = File::create(path)?;

        for (node_id, row) in embeddings.iter().enumerate() {
            let line = std::iter::once(node_id.to_string())
                .chain(row.iter().map(|v| v.to_string()))
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
