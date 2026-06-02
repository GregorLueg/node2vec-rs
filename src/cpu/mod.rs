//! Base CPU version of `ann-search-rs`. Leverages handrolled SIMD with
//! Hogwild! SGD for the training updates.

pub mod matrix;
pub mod simd;
pub mod train;
pub mod word2vec_model;

/// Number of buckets in the sigmoid lookup table.
pub const SIGMOID_TABLE_SIZE: usize = 512;

/// Clamp range for sigmoid input; values outside `[-MAX_SIGMOID, MAX_SIGMOID]`
/// saturate to 0 or 1.
pub const MAX_SIGMOID: f32 = 8f32;

/// Number of entries in the unigram negative sampling table.
pub const NEGATIVE_TABLE_SIZE: usize = 10000000;

/// Number of buckets in the log lookup table.
pub const LOG_TABLE_SIZE: usize = 512;
