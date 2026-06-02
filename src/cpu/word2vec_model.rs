//! The CPU-based node2vec model.

use std::sync::Arc;

use crate::cpu::matrix::Matrix;
use crate::cpu::simd::saxpy_simd;
use crate::cpu::*;

////////////////
// Constansts //
////////////////

const SIGMOID_TABLE_SIZE_F32: f32 = SIGMOID_TABLE_SIZE as f32;
const LOG_TABLE_SIZE_F32: f32 = LOG_TABLE_SIZE as f32;

/////////////
// Helpers //
/////////////

/// Initialise the sigmoid table
///
/// ### Returns
///
/// A table of sigmoid values
fn init_sigmoid_table() -> [f32; SIGMOID_TABLE_SIZE + 1] {
    let mut sigmoid_table = [0f32; SIGMOID_TABLE_SIZE + 1];
    for i in 0..SIGMOID_TABLE_SIZE + 1 {
        let x = (i as f32 * 2. * MAX_SIGMOID) / SIGMOID_TABLE_SIZE_F32 - MAX_SIGMOID;
        sigmoid_table[i] = 1.0 / (1.0 + (-x).exp());
    }
    sigmoid_table
}

/// Initialise the log table
///
/// ### Returns
///
/// A table of log values
fn init_log_table() -> [f32; LOG_TABLE_SIZE + 1] {
    let mut log_table = [0f32; LOG_TABLE_SIZE + 1];
    for i in 0..LOG_TABLE_SIZE + 1 {
        let x = (i as f32 + 1e-5) / LOG_TABLE_SIZE_F32;
        log_table[i] = x.ln();
    }
    log_table
}

//////////
// Main //
//////////

/// Initialise the word2vec model
pub struct Word2Vec<'a> {
    /// The input matrix
    pub input: &'a mut Matrix,
    /// The output matrix
    output: &'a mut Matrix,
    /// The dimension of the model
    dim: usize,
    /// The learning rate
    lr: f32,
    /// The number of negative samples
    neg: usize,
    /// The gradient vector
    grad: Vec<f32>,
    /// The number of negative samples per positive sample
    neg_pos: usize,
    /// The sigmoid table
    sigmoid_table: [f32; SIGMOID_TABLE_SIZE + 1],
    /// The log table
    log_table: [f32; LOG_TABLE_SIZE + 1],
    /// The negative table
    negative_table: Arc<Vec<usize>>,
    /// The loss
    loss: f64,
    /// The number of samples
    n_samples: u64,
}

impl<'a> Word2Vec<'a> {
    /// Generate a new word2vec model
    ///
    /// ### Params
    ///
    /// * `input` - The input matrix
    /// * `output` - The output matrix
    /// * `dim` - The dimension of the model
    /// * `lr` - The learning rate
    /// * `neg` - The number of negative samples
    /// * `neg_table` - The negative table
    ///
    /// ### Returns
    ///
    /// A new initialised word2vec model
    pub fn new(
        input: &'a mut Matrix,
        output: &'a mut Matrix,
        dim: usize,
        lr: f32,
        neg: usize,
        neg_table: Arc<Vec<usize>>,
        neg_start: usize,
    ) -> Word2Vec<'a> {
        Self {
            input,
            output,
            dim,
            lr,
            neg,
            grad: vec![0f32; dim],
            neg_pos: neg_start % neg_table.len(),
            sigmoid_table: init_sigmoid_table(),
            log_table: init_log_table(),
            negative_table: neg_table,
            loss: 0.,
            n_samples: 0,
        }
    }

    /// Return the loss value
    ///
    /// ### Returns
    ///
    /// The loss value
    #[inline]
    pub fn get_loss(&self) -> f64 {
        self.loss / self.n_samples as f64
    }

    /// Set the learning rate
    ///
    /// ### Params
    ///
    /// * `lr` - The learning rate
    #[inline(always)]
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Get the learning rate
    ///
    /// ### Returns
    ///
    /// The learning rate
    #[inline(always)]
    pub fn get_lr(&self) -> f32 {
        self.lr
    }

    /// Return a negative
    ///
    /// ### Params
    ///
    /// * `target` - The target value
    ///
    /// ### Returns
    ///
    /// A negative example
    fn get_negative(&mut self, target: usize) -> usize {
        loop {
            let negative = self.negative_table[self.neg_pos];
            self.neg_pos = (self.neg_pos + 1) % self.negative_table.len();
            if target != negative {
                return negative;
            }
        }
    }

    /// Update the model
    ///
    /// ### Params
    ///
    /// * `input` - The input value
    /// * `target` - The target value
    #[inline(always)]
    pub fn update(&mut self, input: usize, target: usize) {
        self.loss += self.negative_sampling(input, target);
        self.n_samples += 1;
    }

    /// Negative sampling
    ///
    /// ### Params
    ///
    /// * `input` - The input value
    /// * `target` - The target value
    fn negative_sampling(&mut self, input: usize, target: usize) -> f64 {
        let input_emb = self.input.get_row(input);
        let mut loss = 0f32;
        self.grad_zero();
        for i in 0..self.neg + 1 {
            if i == 0 {
                loss += self.binary_logistic(input_emb, target, 1);
            } else {
                let neg_sample = self.get_negative(target);
                loss += self.binary_logistic(input_emb, neg_sample, 0);
            }
        }
        unsafe { self.input.add_row(self.grad.as_mut_ptr(), input, 1.0) };
        loss as f64
    }

    /// Return the log value
    ///
    /// ### Params
    ///
    /// * `x` - The value to log
    ///
    /// ### Returns
    ///
    /// The log value
    #[inline]
    fn log(&self, x: f32) -> f32 {
        if x > 1.0 {
            x
        } else {
            let i = (x * (LOG_TABLE_SIZE_F32)) as usize;
            unsafe { *self.log_table.get_unchecked(i) }
        }
    }

    /// Set the gradient to zero
    #[inline]
    fn grad_zero(&mut self) {
        self.grad.fill(0.0);
    }

    /// Return the sigmoid value
    ///
    /// ### Params
    ///
    /// * `x` - The value to sigmoid
    ///
    /// ### Returns
    ///
    /// The sigmoid value
    #[inline]
    fn sigmoid(&self, x: f32) -> f32 {
        if x < -MAX_SIGMOID {
            0f32
        } else if x > MAX_SIGMOID {
            1f32
        } else {
            let i = (x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE_F32 / MAX_SIGMOID / 2.;
            unsafe { *self.sigmoid_table.get_unchecked(i as usize) }
        }
    }

    /// Add a multiple of a row to the gradient
    ///
    /// ### Params
    ///
    /// * `other` - The row to add
    /// * `a` - The scalar to multiply the row by
    #[inline(always)]
    fn add_mul_row(&mut self, other: *const f32, a: f32) {
        unsafe {
            let source_slice = std::slice::from_raw_parts(other, self.dim);
            saxpy_simd(&mut self.grad, source_slice, a);
        }
    }

    /// Calculate the binary logistic loss
    ///
    /// ### Params
    ///
    /// * `input_emb` - The input embedding
    /// * `target` - The target word index
    /// * `label` - The label: `1` positive; `-1` = negative
    ///
    /// ### Returns
    ///
    /// The binary logistic loss
    #[inline]
    fn binary_logistic(&mut self, input_emb: *mut f32, target: usize, label: i32) -> f32 {
        let sum = unsafe { self.output.dot_row(input_emb, target) };
        let score = self.sigmoid(sum);
        let alpha = self.lr * (label as f32 - score);
        let tar_emb = self.output.get_row(target);
        self.add_mul_row(tar_emb, alpha);
        unsafe { self.output.add_row(input_emb, target, alpha) };
        if label == 1 {
            -self.log(score)
        } else {
            -self.log(1.0 - score)
        }
    }
}
