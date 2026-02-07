#![allow(dead_code)]

use std::sync::OnceLock;
use wide::{f32x4, f32x8};

/////////////
// Helpers //
/////////////

// Enum for the different architectures and potential SIMD levels
#[derive(Clone, Copy, Debug)]
pub enum SimdLevel {
    /// Scalar version
    Scalar,
    /// 128-bit (also covers NEON which is used by Apple)
    Sse,
    /// 256-bit
    Avx2,
    /// 512-bit
    Avx512,
}

static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Function to detect which SIMD implementation to use
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return SimdLevel::Sse;
            }
            return SimdLevel::Scalar;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            SimdLevel::Sse
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    })
}

/////////////////
// Dot product //
/////////////////

//////////////////////////
// Individual functions //
//////////////////////////

/// Dot product - f32, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Dot product - f32, optimised for 128-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f32, optimised for 256-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f32, optimised for 512-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// 256-bit fallback version for avx512 (dot product)
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_avx2(a, b)
}

///////////////////////
// Dispatch function //
///////////////////////

/// Wrapper for dot product - f32, optimised for SIMD
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline]
pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    match detect_simd_level() {
        SimdLevel::Avx512 => dot_f32_avx512(a, b),
        SimdLevel::Avx2 => dot_f32_avx2(a, b),
        SimdLevel::Sse => dot_f32_sse(a, b),
        SimdLevel::Scalar => dot_f32_scalar(a, b),
    }
}

///////////
// SAXPY //
///////////

/// SAXPY (y = a*x + y) - f32, scalar
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline(always)]
fn saxpy_f32_scalar(dst: &mut [f32], source: &[f32], scale: f32) {
    for i in 0..dst.len() {
        dst[i] += scale * source[i];
    }
}

/// SAXPY - f32, optimised for 128-bit
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline(always)]
fn saxpy_f32_sse(dst: &mut [f32], source: &[f32], scale: f32) {
    let len = dst.len();
    let chunks = len / 4;
    let scale_vec = f32x4::splat(scale);

    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = source.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let vdst = f32x4::from(*(dst_ptr.add(offset) as *const [f32; 4]));
            let vsrc = f32x4::from(*(src_ptr.add(offset) as *const [f32; 4]));
            let result = vdst + scale_vec * vsrc;
            *(dst_ptr.add(offset) as *mut [f32; 4]) = result.into();
        }
    }

    for i in (chunks * 4)..len {
        dst[i] += scale * source[i];
    }
}

/// SAXPY - f32, optimised for 256-bit
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline(always)]
fn saxpy_f32_avx2(dst: &mut [f32], source: &[f32], scale: f32) {
    let len = dst.len();
    let chunks = len / 8;
    let scale_vec = f32x8::splat(scale);

    unsafe {
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = source.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let vdst = f32x8::from(*(dst_ptr.add(offset) as *const [f32; 8]));
            let vsrc = f32x8::from(*(src_ptr.add(offset) as *const [f32; 8]));
            let result = vdst + scale_vec * vsrc;
            *(dst_ptr.add(offset) as *mut [f32; 8]) = result.into();
        }
    }

    for i in (chunks * 8)..len {
        dst[i] += scale * source[i];
    }
}

/// SAXPY - f32, optimised for 512-bit
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn saxpy_f32_avx512(dst: &mut [f32], source: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let chunks = len / 16;

    unsafe {
        let scale_vec = _mm512_set1_ps(scale);
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = source.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let vdst = _mm512_loadu_ps(dst_ptr.add(offset));
            let vsrc = _mm512_loadu_ps(src_ptr.add(offset));
            let result = _mm512_fmadd_ps(scale_vec, vsrc, vdst);
            _mm512_storeu_ps(dst_ptr.add(offset), result);
        }
    }

    for i in (chunks * 16)..len {
        dst[i] += scale * source[i];
    }
}

/// 256-bit fallback version for avx512 (saxpy product)
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn saxpy_f32_avx512(dst: &mut [f32], source: &[f32], scale: f32) {
    saxpy_f32_avx2(dst, source, scale)
}

///////////////////////
// Dispatch function //
///////////////////////

/// Dispatch function for SAXPY
///
/// ### Params
///
/// * `dst` - Destination vector
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline]
pub fn saxpy_simd(dst: &mut [f32], source: &[f32], scale: f32) {
    match detect_simd_level() {
        SimdLevel::Avx512 => saxpy_f32_avx512(dst, source, scale),
        SimdLevel::Avx2 => saxpy_f32_avx2(dst, source, scale),
        SimdLevel::Sse => saxpy_f32_sse(dst, source, scale),
        SimdLevel::Scalar => saxpy_f32_scalar(dst, source, scale),
    }
}

/////////////
// L2 norm //
/////////////

/// L2 norm (Euclidean length) - f32, scalar
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[inline(always)]
fn norm_l2_f32_scalar(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// L2 norm - f32, optimised for 128-bit
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[inline(always)]
fn norm_l2_f32_sse(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            acc += va * va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * a[i];
    }
    sum.sqrt()
}

/// L2 norm - f32, optimised for 256-bit
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[inline(always)]
fn norm_l2_f32_avx2(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            acc += va * va;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * a[i];
    }
    sum.sqrt()
}

/// L2 norm - f32, optimised for 512-bit
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn norm_l2_f32_avx512(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, va, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * a[i];
        }
        sum.sqrt()
    }
}

/// 256-bit fallback version for avx512 (L2 norm)
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn norm_l2_f32_avx512(a: &[f32]) -> f32 {
    norm_l2_f32_avx2(a)
}

///////////////////////
// Dispatch function //
///////////////////////

/// Dispatch function for L2 norm
#[inline]
pub fn norm_l2_simd(a: &[f32]) -> f32 {
    match detect_simd_level() {
        SimdLevel::Avx512 => norm_l2_f32_avx512(a),
        SimdLevel::Avx2 => norm_l2_f32_avx2(a),
        SimdLevel::Sse => norm_l2_f32_sse(a),
        SimdLevel::Scalar => norm_l2_f32_scalar(a),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;

        let result = dot_simd(&a, &b);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];

        let result = dot_simd(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_product_various_sizes() {
        // Test sizes that align and don't align with SIMD widths
        for size in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 100, 128, 256] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

            let expected = dot_f32_scalar(&a, &b);
            let result = dot_simd(&a, &b);

            assert!(
                (result - expected).abs() < 1e-4,
                "Failed for size {}: expected {}, got {}",
                size,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_dot_product_negative() {
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![-1.0, 2.0, -3.0, 4.0];
        let expected = -1.0 + -2.0 * 2.0 + 3.0 * -3.0 + -4.0 * 4.0;

        let result = dot_simd(&a, &b);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_all_implementations() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (i + 1) as f32 * 0.2).collect();

        let scalar_result = dot_f32_scalar(&a, &b);
        let sse_result = dot_f32_sse(&a, &b);
        let avx2_result = dot_f32_avx2(&a, &b);
        let avx512_result = dot_f32_avx512(&a, &b);

        assert!((scalar_result - sse_result).abs() < 1e-2);
        assert!((scalar_result - avx2_result).abs() < 1e-2);
        assert!((scalar_result - avx512_result).abs() < 1e-2);
    }

    #[test]
    fn test_saxpy_basic() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0];
        let source = vec![2.0, 3.0, 4.0, 5.0];
        let scale = 2.0;

        saxpy_simd(&mut dst, &source, scale);

        let expected = [5.0, 8.0, 11.0, 14.0];
        for (d, e) in dst.iter().zip(expected.iter()) {
            assert!((d - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_saxpy_zero_scale() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0];
        let source = vec![2.0, 3.0, 4.0, 5.0];
        let expected = dst.clone();

        saxpy_simd(&mut dst, &source, 0.0);

        assert_eq!(dst, expected);
    }

    #[test]
    fn test_saxpy_negative_scale() {
        let mut dst = vec![10.0, 20.0, 30.0, 40.0];
        let source = vec![1.0, 2.0, 3.0, 4.0];
        let scale = -2.0;

        saxpy_simd(&mut dst, &source, scale);

        let expected = [8.0, 16.0, 24.0, 32.0];
        for (d, e) in dst.iter().zip(expected.iter()) {
            assert!((d - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_saxpy_various_sizes() {
        for size in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 100, 128, 256] {
            let mut dst: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let source: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
            let scale = 0.5;

            let mut expected = dst.clone();
            saxpy_f32_scalar(&mut expected, &source, scale);

            saxpy_simd(&mut dst, &source, scale);

            for (i, (d, e)) in dst.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (d - e).abs() < 1e-4,
                    "Failed at index {} for size {}: expected {}, got {}",
                    i,
                    size,
                    e,
                    d
                );
            }
        }
    }

    #[test]
    fn test_saxpy_all_implementations() {
        let size = 100;
        let source: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let scale = 1.5;

        let mut dst_scalar: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut dst_sse = dst_scalar.clone();
        let mut dst_avx2 = dst_scalar.clone();
        let mut dst_avx512 = dst_scalar.clone();

        saxpy_f32_scalar(&mut dst_scalar, &source, scale);
        saxpy_f32_sse(&mut dst_sse, &source, scale);
        saxpy_f32_avx2(&mut dst_avx2, &source, scale);
        saxpy_f32_avx512(&mut dst_avx512, &source, scale);

        for i in 0..size {
            assert!((dst_scalar[i] - dst_sse[i]).abs() < 1e-4);
            assert!((dst_scalar[i] - dst_avx2[i]).abs() < 1e-4);
            assert!((dst_scalar[i] - dst_avx512[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_saxpy_inplace() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0];
        let source = dst.clone();
        let scale = 1.0;

        saxpy_simd(&mut dst, &source, scale);

        let expected = [2.0, 4.0, 6.0, 8.0];
        for (d, e) in dst.iter().zip(expected.iter()) {
            assert!((d - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_level_detection() {
        let level = detect_simd_level();
        // Just verify it returns something valid
        match level {
            SimdLevel::Scalar | SimdLevel::Sse | SimdLevel::Avx2 | SimdLevel::Avx512 => {}
        }
    }

    #[test]
    fn test_dot_product_large() {
        let size = 10_000;
        let a: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| ((i + 50) % 100) as f32).collect();

        let expected = dot_f32_scalar(&a, &b);
        let result = dot_simd(&a, &b);

        // Relative tolerance for large accumulations
        let rel_error = (result - expected).abs() / expected.abs();
        assert!(
            rel_error < 1e-4,
            "result: {}, expected: {}, rel_error: {}",
            result,
            expected,
            rel_error
        );
    }

    #[test]
    fn test_saxpy_large() {
        let size = 10_000;
        let mut dst: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let source: Vec<f32> = (0..size).map(|i| ((i + 50) % 100) as f32).collect();
        let scale = 0.75;

        let mut expected = dst.clone();
        saxpy_f32_scalar(&mut expected, &source, scale);

        saxpy_simd(&mut dst, &source, scale);

        for (d, e) in dst.iter().zip(expected.iter()) {
            assert!((d - e).abs() < 1e-4);
        }
    }

    #[test]
    fn test_norm_basic() {
        let a = vec![3.0, 4.0];
        let result = norm_l2_simd(&a);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let result = norm_l2_simd(&a);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_norm_various_sizes() {
        for size in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 100] {
            let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

            let expected = norm_l2_f32_scalar(&a);
            let result = norm_l2_simd(&a);

            assert!(
                (result - expected).abs() < 1e-3,
                "Failed for size {}: expected {}, got {}",
                size,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_norm_all_implementations() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();

        let scalar = norm_l2_f32_scalar(&a);
        let sse = norm_l2_f32_sse(&a);
        let avx2 = norm_l2_f32_avx2(&a);
        let avx512 = norm_l2_f32_avx512(&a);

        assert!((scalar - sse).abs() < 1e-3);
        assert!((scalar - avx2).abs() < 1e-3);
        assert!((scalar - avx512).abs() < 1e-3);
    }
}
