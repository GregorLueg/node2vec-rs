//! SIMD kernels behind the Hogwild update.
//!
//! Three operations carry the whole CPU trainer: a dot product between an
//! input row and an output row, a SAXPY accumulating the gradient, and an L2
//! norm used once when the embeddings are written out.
//!
//! The 256- and 512-bit kernels are hand-written against [`std::arch`] and
//! carry a `#[target_feature]` attribute, because that is the only way a
//! runtime-dispatched kernel in a shipped crate actually runs at that width.
//! `wide` picks its vector width from `#[cfg(target_feature = "avx")]`, a
//! **compile-time** cfg, so in a crate built for the x86-64 baseline its
//! `f32x8` is a pair of `f32x4` and every operation on it is two SSE ops. Only
//! [`SimdLevel::Sse`] and the aarch64 path use `wide`, where it is correct.

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;
use wide::f32x4;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

////////////////
// Constants //
////////////////

/// Independent accumulators in the reduction kernels.
///
/// An FMA has roughly four cycles of latency against half a cycle of
/// throughput, so a single accumulator serialises the loop on its own
/// dependency chain. Four is enough to cover the latency on every current
/// part; more only lengthens the tail.
const SIMD_ACCUMULATORS: usize = 4;

/// Environment variable that lowers the dispatch level.
///
/// The width a kernel runs at is otherwise invisible from outside the process,
/// so this is the only way to price one path against another in a benchmark.
/// x86-64 only: elsewhere there is nothing to choose between.
#[cfg(target_arch = "x86_64")]
const SIMD_OVERRIDE_VAR: &str = "NODE2VEC_SIMD";

//////////////////
// Dispatch //
//////////////////

/// The SIMD width a kernel was compiled for.
///
/// A level means only what the kernel it selects was compiled for. The
/// variants are ordered by width, so they compare, and [`detect_simd_level`]
/// uses that to clamp an override to what the CPU can actually run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Scalar version
    Scalar,
    /// 128-bit, which also covers the NEON baseline on aarch64
    Sse,
    /// 256-bit
    Avx2,
    /// 512-bit
    Avx512,
}

#[cfg(target_arch = "x86_64")]
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detect which SIMD implementation to use.
///
/// Only x86-64 needs a runtime probe, and it must keep one: a shipped crate
/// cannot assume AVX2 without an illegal-instruction crash on older hardware.
///
/// FMA is probed alongside AVX2 because the wide kernels are compiled with
/// `#[target_feature(enable = "avx2,fma")]` and use `_mm256_fmadd_ps`. Every
/// part with AVX2 has FMA, but the level has to promise what the kernels it
/// selects actually require.
///
/// ### Returns
///
/// The widest level available on this target, lowered by
/// [`SIMD_OVERRIDE_VAR`] if it asks.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        let detected = if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            SimdLevel::Sse
        } else {
            SimdLevel::Scalar
        };
        cap_simd_level(detected)
    })
}

/// Lower the detected level if [`SIMD_OVERRIDE_VAR`] asks for it.
///
/// Downgrade only. Naming a level the CPU does not have would run an illegal
/// instruction, so a request above `detected` is refused rather than honoured.
/// Read once, inside the `OnceLock`, so it costs nothing per call.
///
/// ### Params
///
/// * `detected` - The widest level this CPU actually supports
///
/// ### Returns
///
/// The requested level when it is no wider than `detected`, else `detected`.
#[cfg(target_arch = "x86_64")]
fn cap_simd_level(detected: SimdLevel) -> SimdLevel {
    let Ok(request) = std::env::var(SIMD_OVERRIDE_VAR) else {
        return detected;
    };
    // Set-but-empty is how a shell passes "no opinion".
    if request.trim().is_empty() {
        return detected;
    }

    let requested = match request.trim().to_lowercase().as_str() {
        "scalar" => SimdLevel::Scalar,
        "sse" => SimdLevel::Sse,
        "avx2" => SimdLevel::Avx2,
        "avx512" => SimdLevel::Avx512,
        other => {
            eprintln!("{SIMD_OVERRIDE_VAR}: unknown level {other:?}, using {detected:?}");
            return detected;
        }
    };

    if requested > detected {
        eprintln!("{SIMD_OVERRIDE_VAR}: this CPU cannot run {requested:?}, using {detected:?}");
        return detected;
    }
    requested
}

/// Detect which SIMD implementation to use.
///
/// NEON is part of the aarch64 baseline, so this is a constant and the whole
/// dispatch folds away at compile time.
///
/// ### Returns
///
/// [`SimdLevel::Sse`], the 128-bit path, which is what NEON provides.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub const fn detect_simd_level() -> SimdLevel {
    SimdLevel::Sse
}

/// Detect which SIMD implementation to use.
///
/// ### Returns
///
/// [`SimdLevel::Scalar`] on targets with no vector path here.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub const fn detect_simd_level() -> SimdLevel {
    SimdLevel::Scalar
}

/////////////////
// Dot product //
/////////////////

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

/// Dot product - f32, 128-bit
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
    let len = a.len().min(b.len());
    let lanes = 4 * SIMD_ACCUMULATORS;
    let blocks = len / lanes;
    let singles = (len - blocks * lanes) / 4;
    let mut acc = [f32x4::ZERO; SIMD_ACCUMULATORS];

    // Counted loops with the trip count known up front, and raw loads rather
    // than `try_into` on a subslice. Both matter: a `while offset + 4 <= len`
    // form and the bounds check each cost more than the arithmetic at these
    // widths. Measured on NEON, not assumed.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..blocks {
            let base = i * lanes;
            for (k, slot) in acc.iter_mut().enumerate() {
                let at = base + k * 4;
                let va = f32x4::from(*(a_ptr.add(at) as *const [f32; 4]));
                let vb = f32x4::from(*(b_ptr.add(at) as *const [f32; 4]));
                *slot += va * vb;
            }
        }

        for j in 0..singles {
            let at = blocks * lanes + j * 4;
            let va = f32x4::from(*(a_ptr.add(at) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(at) as *const [f32; 4]));
            acc[0] += va * vb;
        }
    }

    let mut sum = (acc[0] + acc[1] + acc[2] + acc[3]).reduce_add();
    for i in (blocks * lanes + singles * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Horizontal sum of a 256-bit vector.
///
/// ### Params
///
/// * `v` - The vector to reduce
///
/// ### Returns
///
/// The sum of its eight lanes.
///
/// ### Safety
///
/// Requires AVX on the calling CPU, which the dispatcher has probed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}

/// Dot product - f32, 256-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
///
/// ### Safety
///
/// Requires AVX2 and FMA on the calling CPU, which the dispatcher has probed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let block = 8 * SIMD_ACCUMULATORS;
    let mut offset = 0;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while offset + block <= len {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset)),
            _mm256_loadu_ps(b_ptr.add(offset)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 8)),
            _mm256_loadu_ps(b_ptr.add(offset + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 16)),
            _mm256_loadu_ps(b_ptr.add(offset + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 24)),
            _mm256_loadu_ps(b_ptr.add(offset + 24)),
            acc3,
        );
        offset += block;
    }

    // Whole vectors that did not fill a block.
    while offset + 8 <= len {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(offset)),
            _mm256_loadu_ps(b_ptr.add(offset)),
            acc0,
        );
        offset += 8;
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum = hsum_f32_avx2(acc);

    for i in offset..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f32, 512-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
///
/// ### Safety
///
/// Requires AVX-512F on the calling CPU, which the dispatcher has probed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let block = 16 * SIMD_ACCUMULATORS;
    let mut offset = 0;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    while offset + block <= len {
        acc0 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a_ptr.add(offset)),
            _mm512_loadu_ps(b_ptr.add(offset)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a_ptr.add(offset + 16)),
            _mm512_loadu_ps(b_ptr.add(offset + 16)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a_ptr.add(offset + 32)),
            _mm512_loadu_ps(b_ptr.add(offset + 32)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a_ptr.add(offset + 48)),
            _mm512_loadu_ps(b_ptr.add(offset + 48)),
            acc3,
        );
        offset += block;
    }

    while offset + 16 <= len {
        acc0 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a_ptr.add(offset)),
            _mm512_loadu_ps(b_ptr.add(offset)),
            acc0,
        );
        offset += 16;
    }

    let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut sum = _mm512_reduce_add_ps(acc);

    for i in offset..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dispatch function for the dot product
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
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe { dot_f32_avx512(a, b) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { dot_f32_avx2(a, b) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdLevel::Avx512 | SimdLevel::Avx2 => dot_f32_sse(a, b),
        SimdLevel::Sse => dot_f32_sse(a, b),
        SimdLevel::Scalar => dot_f32_scalar(a, b),
    }
}

///////////
// SAXPY //
///////////

/// SAXPY - f32, scalar
///
/// ### Params
///
/// * `dst` - Destination vector, updated in place
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline(always)]
fn saxpy_f32_scalar(dst: &mut [f32], source: &[f32], scale: f32) {
    for (d, &s) in dst.iter_mut().zip(source.iter()) {
        *d += s * scale;
    }
}

/// SAXPY - f32, 128-bit
///
/// A streaming operation with no reduction, so the unrolling buys load/store
/// overlap rather than hiding an accumulator chain.
///
/// ### Params
///
/// * `dst` - Destination vector, updated in place
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline(always)]
fn saxpy_f32_sse(dst: &mut [f32], source: &[f32], scale: f32) {
    let len = dst.len().min(source.len());
    let chunks = len / 4;
    let scale_vec = f32x4::splat(scale);

    unsafe {
        let d_ptr = dst.as_mut_ptr();
        let s_ptr = source.as_ptr();

        for i in 0..chunks {
            let at = i * 4;
            let vd = f32x4::from(*(d_ptr.add(at) as *const [f32; 4]));
            let vs = f32x4::from(*(s_ptr.add(at) as *const [f32; 4]));
            *(d_ptr.add(at) as *mut [f32; 4]) = (vd + scale_vec * vs).into();
        }
    }

    for i in (chunks * 4)..len {
        dst[i] += scale * source[i];
    }
}

/// SAXPY - f32, 256-bit
///
/// ### Params
///
/// * `dst` - Destination vector, updated in place
/// * `source` - Source vector
/// * `scale` - Scale factor
///
/// ### Safety
///
/// Requires AVX2 and FMA on the calling CPU, which the dispatcher has probed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn saxpy_f32_avx2(dst: &mut [f32], source: &[f32], scale: f32) {
    let len = dst.len().min(source.len());
    let block = 8 * SIMD_ACCUMULATORS;
    let mut offset = 0;
    let scale_vec = _mm256_set1_ps(scale);
    let d_ptr = dst.as_mut_ptr();
    let s_ptr = source.as_ptr();

    while offset + block <= len {
        for k in 0..SIMD_ACCUMULATORS {
            let at = offset + k * 8;
            let out = _mm256_fmadd_ps(
                _mm256_loadu_ps(s_ptr.add(at)),
                scale_vec,
                _mm256_loadu_ps(d_ptr.add(at)),
            );
            _mm256_storeu_ps(d_ptr.add(at), out);
        }
        offset += block;
    }

    while offset + 8 <= len {
        let out = _mm256_fmadd_ps(
            _mm256_loadu_ps(s_ptr.add(offset)),
            scale_vec,
            _mm256_loadu_ps(d_ptr.add(offset)),
        );
        _mm256_storeu_ps(d_ptr.add(offset), out);
        offset += 8;
    }

    for i in offset..len {
        dst[i] += source[i] * scale;
    }
}

/// SAXPY - f32, 512-bit
///
/// ### Params
///
/// * `dst` - Destination vector, updated in place
/// * `source` - Source vector
/// * `scale` - Scale factor
///
/// ### Safety
///
/// Requires AVX-512F on the calling CPU, which the dispatcher has probed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn saxpy_f32_avx512(dst: &mut [f32], source: &[f32], scale: f32) {
    let len = dst.len().min(source.len());
    let block = 16 * SIMD_ACCUMULATORS;
    let mut offset = 0;
    let scale_vec = _mm512_set1_ps(scale);
    let d_ptr = dst.as_mut_ptr();
    let s_ptr = source.as_ptr();

    while offset + block <= len {
        for k in 0..SIMD_ACCUMULATORS {
            let at = offset + k * 16;
            let out = _mm512_fmadd_ps(
                _mm512_loadu_ps(s_ptr.add(at)),
                scale_vec,
                _mm512_loadu_ps(d_ptr.add(at)),
            );
            _mm512_storeu_ps(d_ptr.add(at), out);
        }
        offset += block;
    }

    while offset + 16 <= len {
        let out = _mm512_fmadd_ps(
            _mm512_loadu_ps(s_ptr.add(offset)),
            scale_vec,
            _mm512_loadu_ps(d_ptr.add(offset)),
        );
        _mm512_storeu_ps(d_ptr.add(offset), out);
        offset += 16;
    }

    for i in offset..len {
        dst[i] += source[i] * scale;
    }
}

/// Dispatch function for SAXPY
///
/// ### Params
///
/// * `dst` - Destination vector, updated in place
/// * `source` - Source vector
/// * `scale` - Scale factor
#[inline]
pub fn saxpy_simd(dst: &mut [f32], source: &[f32], scale: f32) {
    match detect_simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe { saxpy_f32_avx512(dst, source, scale) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { saxpy_f32_avx2(dst, source, scale) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdLevel::Avx512 | SimdLevel::Avx2 => saxpy_f32_sse(dst, source, scale),
        SimdLevel::Sse => saxpy_f32_sse(dst, source, scale),
        SimdLevel::Scalar => saxpy_f32_scalar(dst, source, scale),
    }
}

/////////////
// L2 norm //
/////////////

/// Dispatch function for the L2 norm
///
/// A self dot product plus a square root. There is no separate kernel because
/// there is no separate work, and this runs once per row when the embeddings
/// are written rather than inside the training loop.
///
/// ### Params
///
/// * `a` - Input vector
///
/// ### Returns
///
/// The L2 norm
#[inline]
pub fn norm_l2_simd(a: &[f32]) -> f32 {
    dot_simd(a, a).sqrt()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Reassociating a sum across four accumulators moves the last bits, so
    /// equality against the scalar kernel is checked relatively.
    const TOLERANCE: f32 = 1e-4;

    fn vector(len: usize, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.37 + offset).sin() * 2.0)
            .collect()
    }

    /// A dot-product kernel for one dispatch level.
    type DotKernel = fn(&[f32], &[f32]) -> f32;

    /// Every kernel the current CPU can run, paired with its name.
    ///
    /// Checking the wide paths against each other, as the previous tests did,
    /// passes trivially when they are secretly the same code. Scalar is the
    /// only independent reference.
    fn available_dots() -> Vec<(&'static str, DotKernel)> {
        #[allow(unused_mut)]
        let mut kernels: Vec<(&'static str, DotKernel)> = vec![("sse", dot_f32_sse)];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                kernels.push(("avx2", |a, b| unsafe { dot_f32_avx2(a, b) }));
            }
            if is_x86_feature_detected!("avx512f") {
                kernels.push(("avx512", |a, b| unsafe { dot_f32_avx512(a, b) }));
            }
        }
        kernels
    }

    #[test]
    fn test_dot_matches_scalar_across_lengths() {
        // Lengths straddle every block, whole-vector and scalar tail boundary.
        for len in [
            0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 257,
        ] {
            let a = vector(len, 0.0);
            let b = vector(len, 1.0);
            let expect = dot_f32_scalar(&a, &b);

            for (name, kernel) in available_dots() {
                let got = kernel(&a, &b);
                assert!(
                    (got - expect).abs() <= TOLERANCE * expect.abs().max(1.0),
                    "{name} disagreed with scalar at len {len}: {got} vs {expect}"
                );
            }
            assert!((dot_simd(&a, &b) - expect).abs() <= TOLERANCE * expect.abs().max(1.0));
        }
    }

    #[test]
    fn test_dot_known_values() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert!((dot_simd(&a, &b) - 70.0).abs() < 1e-5);
        assert_eq!(dot_simd(&a, &[0.0; 4]), 0.0);

        let neg = vec![-1.0, -2.0, -3.0, -4.0];
        assert!((dot_simd(&a, &neg) + 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_saxpy_matches_scalar_across_lengths() {
        for len in [0, 1, 3, 8, 17, 32, 33, 64, 129, 257] {
            let source = vector(len, 1.0);
            let base = vector(len, 0.0);

            let mut expect = base.clone();
            saxpy_f32_scalar(&mut expect, &source, 0.75);

            let mut got = base.clone();
            saxpy_simd(&mut got, &source, 0.75);

            for i in 0..len {
                assert!(
                    (got[i] - expect[i]).abs() <= TOLERANCE * expect[i].abs().max(1.0),
                    "saxpy disagreed at len {len}, index {i}"
                );
            }
        }
    }

    #[test]
    fn test_saxpy_zero_and_negative_scale() {
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![10.0; 5];
        saxpy_simd(&mut dst, &source, 0.0);
        assert_eq!(dst, vec![10.0; 5]);

        saxpy_simd(&mut dst, &source, -1.0);
        assert_eq!(dst, vec![9.0, 8.0, 7.0, 6.0, 5.0]);
    }

    #[test]
    fn test_norm_l2() {
        assert!((norm_l2_simd(&[3.0, 4.0]) - 5.0).abs() < 1e-5);
        assert_eq!(norm_l2_simd(&[0.0; 16]), 0.0);

        for len in [1, 7, 16, 33, 129] {
            let a = vector(len, 0.0);
            let expect = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm_l2_simd(&a) - expect).abs() <= TOLERANCE * expect.max(1.0));
        }
    }

    #[test]
    fn test_level_ordering() {
        assert!(SimdLevel::Scalar < SimdLevel::Sse);
        assert!(SimdLevel::Sse < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    /// An override may only lower the level. Honouring a request above what
    /// the CPU has would run an illegal instruction.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_override_only_downgrades() {
        assert_eq!(cap_simd_level(SimdLevel::Scalar), SimdLevel::Scalar);

        // Safety: this test is the only reader and writer of the variable in
        // this process, and it restores it before returning.
        unsafe {
            std::env::set_var(SIMD_OVERRIDE_VAR, "avx512");
            assert_eq!(cap_simd_level(SimdLevel::Sse), SimdLevel::Sse);

            std::env::set_var(SIMD_OVERRIDE_VAR, "scalar");
            assert_eq!(cap_simd_level(SimdLevel::Avx2), SimdLevel::Scalar);

            std::env::set_var(SIMD_OVERRIDE_VAR, "nonsense");
            assert_eq!(cap_simd_level(SimdLevel::Avx2), SimdLevel::Avx2);

            std::env::set_var(SIMD_OVERRIDE_VAR, "  ");
            assert_eq!(cap_simd_level(SimdLevel::Avx2), SimdLevel::Avx2);

            std::env::remove_var(SIMD_OVERRIDE_VAR);
        }
    }

    #[test]
    fn test_detected_level_is_runnable() {
        // The dispatcher must never select a kernel this CPU cannot execute;
        // if it did, this call would fault rather than fail.
        let a = vector(64, 0.0);
        assert!(dot_simd(&a, &a).is_finite());
        assert!(matches!(
            detect_simd_level(),
            SimdLevel::Scalar | SimdLevel::Sse | SimdLevel::Avx2 | SimdLevel::Avx512
        ));
    }
}
