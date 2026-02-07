pub mod matrix;
pub mod simd;
pub mod train;
pub mod word2vec_model;

pub const SIGMOID_TABLE_SIZE: usize = 512;
pub const MAX_SIGMOID: f32 = 8f32;
pub const NEGATIVE_TABLE_SIZE: usize = 10000000;
pub const LOG_TABLE_SIZE: usize = 512;
