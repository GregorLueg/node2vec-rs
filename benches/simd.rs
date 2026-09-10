//! Benchmarks for the SIMD kernels behind the Hogwild update.
//!
//! The dimensions cover what the trainer actually runs at. `--embedding-dim`
//! defaults to 16, which is two AVX2 vectors: too short for multiple
//! accumulators to hide FMA latency, so the wide paths are expected to earn
//! little there and most at 512.
//!
//! To price one dispatch level against another, force it:
//!
//! ```text
//! cargo bench --bench simd
//! NODE2VEC_SIMD=sse cargo bench --bench simd
//! NODE2VEC_SIMD=scalar cargo bench --bench simd
//! ```

use node2vec_rs::cpu::simd::{detect_simd_level, dot_simd, norm_l2_simd, saxpy_simd};

/// Embedding widths to measure. 16 is the CLI default, 512 is where the wide
/// kernels should show their best case.
const DIMS: &[usize] = &[16, 64, 128, 512];

fn main() {
    println!("SIMD level: {:?}", detect_simd_level());
    divan::main();
}

/// Deterministic filler, so runs are comparable.
fn vector(dim: usize, offset: f32) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i as f32) * 0.017 + offset).sin())
        .collect()
}

#[divan::bench(args = DIMS)]
fn dot(bencher: divan::Bencher, dim: usize) {
    let a = vector(dim, 0.0);
    let b = vector(dim, 1.0);
    bencher.bench_local(|| dot_simd(divan::black_box(&a), divan::black_box(&b)));
}

#[divan::bench(args = DIMS)]
fn saxpy(bencher: divan::Bencher, dim: usize) {
    let source = vector(dim, 1.0);
    let mut dst = vector(dim, 0.0);
    bencher.bench_local(|| saxpy_simd(divan::black_box(&mut dst), divan::black_box(&source), 0.5));
}

#[divan::bench(args = DIMS)]
fn norm_l2(bencher: divan::Bencher, dim: usize) {
    let a = vector(dim, 0.0);
    bencher.bench_local(|| norm_l2_simd(divan::black_box(&a)));
}
