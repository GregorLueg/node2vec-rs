use faer::Mat;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use std::cell::UnsafeCell;

use crate::cpu::simd::*;

/// Dense matrix stored in row-major format
///
/// The matrix data is stored as a flat vector where each row
/// is laid out contiguously in memory for cache efficiency.
/// This layout enables efficient SIMD operations on individual rows.
///
/// ### Fields
///
/// - `n_col` - Number of elements per row (dimension of embeddings)
/// - `data` - Flat storage of matrix data (rows * ncol elements)
#[derive(Debug)]
pub struct Matrix {
    n_col: usize,
    n_row: usize,
    data: Vec<f32>,
}

/// Thread-safe wrapper around Matrix for concurrent training
///
/// Uses `UnsafeCell` to allow interior mutability across threads.
/// Safety is ensured by the training algorithm's access patterns where
/// each thread writes to distinct rows.
///
/// ### Fields
///
/// * `inner` - The inner matrix data (unsafe cell)
#[derive(Debug)]
pub struct MatrixWrapper {
    pub inner: UnsafeCell<Matrix>,
}

/// SAFETY: This is intentionally unsound. Multiple threads will concurrently
/// read and write overlapping rows (e.g. a target in one thread may be a
/// negative sample in another). This mirrors the deliberate data race in
/// Mikolov's original word2vec C implementation and the word2vec-rs crate
/// it was ported from. SGD tolerates stale/torn reads and the resulting
/// embeddings converge in practice. Do not use MatrixWrapper as a general-
/// purpose concurrent container.
unsafe impl Sync for MatrixWrapper {}

impl Matrix {
    /// Creates a new matrix initialised with zeros
    ///
    /// ### Params
    ///
    /// * `rows` - Number of rows in the matrix (typically vocabulary size)
    /// * `n_col` - Number of elements per row (embedding dimension)
    ///
    /// ### Returns
    ///
    /// A new Matrix with all elements set to 0.0
    pub fn new(n_row: usize, n_col: usize) -> Matrix {
        Matrix {
            data: vec![0f32; n_col * n_row],
            n_col,
            n_row,
        }
    }

    /// Normalises all rows in the matrix to unit length
    ///
    /// Each row vector is divided by its L2 norm, making it a unit vector.
    pub fn norm_self(&mut self) {
        let num_rows = self.n_row;
        for i in 0..num_rows {
            let n = self.norm(i);
            if n > 0.0 {
                let start = i * self.n_col;
                let end = start + self.n_col;
                for j in start..end {
                    self.data[j] /= n;
                }
            }
        }
    }

    /// Wraps the matrix in a thread-safe wrapper
    ///
    /// ### Returns
    ///
    /// A `MatrixWrapper` that can be safely shared across threads using Arc
    pub fn make_send(self) -> MatrixWrapper {
        MatrixWrapper {
            inner: UnsafeCell::new(self),
        }
    }

    /// Initialises matrix with uniform random values
    ///
    /// ### Params
    ///
    /// * `bound` - Values will be sampled uniformly from [-bound, bound]
    /// * `seed` - Seed for reproducibility.
    pub fn uniform(&mut self, bound: f32, seed: usize) {
        let between = Uniform::new(-bound, bound).unwrap();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        for v in &mut self.data {
            *v = between.sample(&mut rng);
        }
    }

    /// Computes the L2 norm of a matrix row
    ///
    /// ### Params
    ///
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// The L2 norm (Euclidean length) of the row vector
    pub fn norm(&self, i: usize) -> f32 {
        let start = i * self.n_col;
        let end = start + self.n_col;
        norm_l2_simd(&self.data[start..end])
    }

    /// Sets all matrix elements to zero
    #[inline(always)]
    pub fn zero(&mut self) {
        for v in self.data.iter_mut() {
            *v = 0f32;
        }
    }

    /// Adds a scaled vector to a matrix row (SAXPY operation)
    ///
    /// Performs: `row[i] = row[i] + mul * vec`
    ///
    /// This is used during gradient updates where we add scaled gradients
    /// to embedding vectors.
    ///
    /// ### Params
    ///
    /// * `vec` - Pointer to the vector to add (must have `n_col` elements)
    /// * `i` - Row index
    /// * `mul` - Scaling factor for the vector
    ///
    /// ### Safety
    ///
    /// The caller must ensure `vec` points to at least `n_col` valid f32
    /// elements.
    #[inline(always)]
    pub unsafe fn add_row(&mut self, vec: *const f32, i: usize, mul: f32) {
        let start = i * self.n_col;
        unsafe {
            let row_slice =
                std::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(start), self.n_col);
            let vec_slice = std::slice::from_raw_parts(vec, self.n_col);
            saxpy_simd(row_slice, vec_slice, mul);
        }
    }

    /// Computes dot product between a vector and a matrix row
    ///
    /// ### Params
    ///
    /// * `vec` - Pointer to the vector (must have `row_size` elements)
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// The dot product result
    ///
    /// ### Safety
    ///
    /// The caller must ensure `vec` points to at least `row_size` valid f32
    /// elements
    #[inline(always)]
    pub unsafe fn dot_row(&self, vec: *const f32, i: usize) -> f32 {
        let start = i * self.n_col;
        unsafe {
            let row_slice = std::slice::from_raw_parts(self.data.as_ptr().add(start), self.n_col);
            let vec_slice = std::slice::from_raw_parts(vec, self.n_col);
            dot_simd(row_slice, vec_slice)
        }
    }

    /// Computes dot product between two matrix rows
    ///
    /// ### Params
    ///
    /// * `i` - First row index
    /// * `j` - Second row index
    ///
    /// ### Returns
    ///
    /// The dot product of row i and row j
    #[inline(always)]
    pub fn dot_two_row(&self, i: usize, j: usize) -> f32 {
        let start_i = i * self.n_col;
        let start_j = j * self.n_col;
        unsafe {
            let row_i = std::slice::from_raw_parts(self.data.as_ptr().add(start_i), self.n_col);
            let row_j = std::slice::from_raw_parts(self.data.as_ptr().add(start_j), self.n_col);
            dot_simd(row_i, row_j)
        }
    }

    /// Gets a mutable pointer to a matrix row
    ///
    /// ### Params
    ///
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// Mutable pointer to the start of row i
    ///
    /// ### Safety
    ///
    /// The caller must ensure the pointer is used correctly and doesn't
    /// create aliasing issues. Primarily used for passing to SIMD functions.
    #[inline(always)]
    pub fn get_row(&mut self, i: usize) -> *mut f32 {
        unsafe { self.data.as_mut_ptr().add(i * self.n_col) }
    }

    /// Returns a shared slice of row i
    ///
    /// ### Params
    ///
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// Slice of the row data
    #[inline(always)]
    pub fn row_as_slice(&self, i: usize) -> &[f32] {
        let start = i * self.n_col;
        &self.data[start..start + self.n_col]
    }

    /// Gets a const pointer to a matrix row
    ///
    /// ### Params
    ///
    /// * `i` - Row index
    ///
    /// ### Returns
    ///
    /// Const pointer to the start of row i
    #[inline(always)]
    pub fn get_row_unmod(&self, i: usize) -> *const f32 {
        unsafe { self.data.as_ptr().add(i * self.n_col) }
    }

    /// Returns the number of elements per row
    #[inline(always)]
    pub fn n_col(&self) -> usize {
        self.n_col
    }

    /// Returns the total number of rows
    #[inline(always)]
    pub fn n_rows(&self) -> usize {
        self.n_row
    }

    /// Converts the matrix to a Faer matrix
    ///
    /// ### Returns
    ///
    /// Faer matrix
    pub fn to_faer(&self) -> Mat<f32> {
        Mat::from_fn(self.n_row, self.n_col, |i, j| self.data[i * self.n_col + j])
    }

    /// Write the matrix rows to a CSV file
    ///
    /// ### Params
    ///
    /// * `path` - Path to the output CSV
    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        for i in 0..self.n_row {
            let start = i * self.n_col;
            let end = start + self.n_col;
            let line = self.data[start..end]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(file, "{}", line)?;
        }
        Ok(())
    }

    /// Compute the element-wise average of two matrices
    ///
    /// ### Params
    ///
    /// * `other` - The other matrix (must have same dimensions)
    ///
    /// ### Returns
    ///
    /// A new matrix where each element is (self + other) / 2
    ///
    /// ### Panics
    ///
    /// Panics if dimensions do not match.
    pub fn average_with(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.n_row, other.n_row, "Row count mismatch");
        assert_eq!(self.n_col, other.n_col, "Column count mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a + b) * 0.5)
            .collect();
        Matrix {
            n_row: self.n_row,
            n_col: self.n_col,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new(10, 5);
        assert_eq!(matrix.n_rows(), 10);
        assert_eq!(matrix.n_col(), 5);
        assert_eq!(matrix.data.len(), 50);
    }

    #[test]
    fn test_matrix_zero() {
        let mut matrix = Matrix::new(5, 4);
        matrix.uniform(1.0, 123);
        matrix.zero();
        for v in &matrix.data {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_matrix_uniform() {
        let mut matrix = Matrix::new(10, 10);
        matrix.uniform(1.0, 123);

        // Check values are within bounds
        for v in &matrix.data {
            assert!(v.abs() <= 1.0);
        }

        // Check not all zeros
        assert!(matrix.data.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_dot_two_row() {
        let mut matrix = Matrix::new(3, 4);
        matrix.data = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0];

        let dot = matrix.dot_two_row(0, 1);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;
        assert!((dot - expected).abs() < 1e-5);
    }

    #[test]
    fn test_add_row() {
        let mut matrix = Matrix::new(2, 4);
        matrix.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let vec = [1.0, 1.0, 1.0, 1.0];
        unsafe { matrix.add_row(vec.as_ptr(), 0, 2.0) };

        assert!((matrix.data[0] - 3.0).abs() < 1e-5);
        assert!((matrix.data[1] - 4.0).abs() < 1e-5);
        assert!((matrix.data[2] - 5.0).abs() < 1e-5);
        assert!((matrix.data[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![3.0, 4.0, 0.0, 1.0, 2.0, 2.0];

        let norm0 = matrix.norm(0);
        assert!((norm0 - 5.0).abs() < 1e-5);

        let norm1 = matrix.norm(1);
        assert!((norm1 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_self() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![3.0, 4.0, 0.0, 1.0, 2.0, 2.0];

        matrix.norm_self();

        let norm0 = matrix.norm(0);
        let norm1 = matrix.norm(1);

        assert!((norm0 - 1.0).abs() < 1e-5);
        assert!((norm1 - 1.0).abs() < 1e-5);
    }
}
