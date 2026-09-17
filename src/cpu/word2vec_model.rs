//! The CPU-based node2vec model.

use crate::cpu::matrix::Matrix;
use crate::cpu::simd::saxpy_simd;
use crate::cpu::train::NegativeTable;
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
    /// Cursor into each negative table, one per table. Walking a pre-shuffled
    /// table beats drawing a fresh random index: it is one increment, and the
    /// shuffle already supplies the randomness.
    neg_pos: Vec<usize>,
    /// The sigmoid table
    sigmoid_table: [f32; SIGMOID_TABLE_SIZE + 1],
    /// The log table
    log_table: [f32; LOG_TABLE_SIZE + 1],
    /// Where negatives are drawn from
    negative_table: NegativeTable,
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
        neg_table: NegativeTable,
        neg_start: usize,
    ) -> Word2Vec<'a> {
        // Threads start at different offsets so they do not march through the
        // same stretch of the table in lockstep.
        let neg_pos = vec![neg_start; neg_table.n_cursors()];
        Self {
            input,
            output,
            dim,
            lr,
            neg,
            grad: vec![0f32; dim],
            neg_pos,
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

    /// Return a negative sample for a context node
    ///
    /// Under [`NegativeTable::PerType`] the negative is drawn from the context
    /// node's own type, which is the whole of the metapath2vec++ change.
    ///
    /// ### Params
    ///
    /// * `target` - The context node the negative is being drawn against
    ///
    /// ### Returns
    ///
    /// A node id that is not `target`, unless the table is too degenerate to
    /// offer one, in which case `target` itself comes back rather than
    /// spinning forever.
    fn get_negative(&mut self, target: usize) -> usize {
        let Self {
            negative_table,
            neg_pos,
            ..
        } = self;

        let (table, cursor) = match negative_table {
            NegativeTable::Global(table) => (table.as_slice(), 0),
            NegativeTable::PerType { tables, node_type } => {
                let t = node_type[target] as usize;
                (tables[t].as_slice(), t)
            }
        };

        if table.is_empty() {
            return target;
        }

        // One sweep of the table is enough to prove there is no other node in
        // it; without the bound a single-node type would loop forever.
        for _ in 0..table.len() {
            let negative = table[neg_pos[cursor] % table.len()] as usize;
            neg_pos[cursor] = neg_pos[cursor].wrapping_add(1);
            if target != negative {
                return negative;
            }
        }
        target
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

///////////
// Tests //
///////////

#[cfg(test)]
mod word2vec_tests {
    use super::*;
    use crate::cpu::train::NegativeTable;
    use std::sync::Arc;

    /// Six nodes: 0-2 of type 0, 3-5 of type 1.
    const NODE_TYPE: [u8; 6] = [0, 0, 0, 1, 1, 1];

    /// Scratch matrices; only `get_negative` is under test here.
    fn matrices() -> (Matrix, Matrix) {
        (Matrix::new(6, 4), Matrix::new(6, 4))
    }

    #[test]
    fn test_per_type_negatives_match_the_context_type() {
        let table = NegativeTable::PerType {
            tables: vec![Arc::new(vec![0, 1, 2]), Arc::new(vec![3, 4, 5])],
            node_type: Arc::new(NODE_TYPE.to_vec()),
        };
        let (mut input, mut output) = matrices();
        let mut w2v = Word2Vec::new(&mut input, &mut output, 4, 0.1, 5, table, 0);

        for target in 0..6usize {
            for _ in 0..20 {
                let negative = w2v.get_negative(target);
                assert_eq!(
                    NODE_TYPE[negative], NODE_TYPE[target],
                    "negative {negative} has the wrong type for target {target}"
                );
                assert_ne!(negative, target, "the true target came back as a negative");
            }
        }
    }

    #[test]
    fn test_global_negatives_ignore_type_but_exclude_the_target() {
        let table = NegativeTable::Global(Arc::new(vec![0, 1, 2, 3, 4, 5]));
        let (mut input, mut output) = matrices();
        let mut w2v = Word2Vec::new(&mut input, &mut output, 4, 0.1, 5, table, 0);

        let mut seen_other_type = false;
        for _ in 0..20 {
            let negative = w2v.get_negative(0);
            assert_ne!(negative, 0);
            seen_other_type |= NODE_TYPE[negative] != NODE_TYPE[0];
        }
        assert!(seen_other_type, "the global table should cross types");
    }

    /// A type holding exactly one node has no valid negative for it. The
    /// cursor must give up rather than spin.
    #[test]
    fn test_degenerate_table_does_not_hang() {
        let table = NegativeTable::PerType {
            tables: vec![Arc::new(vec![0]), Arc::new(vec![])],
            node_type: Arc::new(NODE_TYPE.to_vec()),
        };
        let (mut input, mut output) = matrices();
        let mut w2v = Word2Vec::new(&mut input, &mut output, 4, 0.1, 5, table, 0);

        assert_eq!(w2v.get_negative(0), 0);
        assert_eq!(w2v.get_negative(3), 3);
    }
}
