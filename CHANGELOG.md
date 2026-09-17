# Changelog

## [0.3.0] - unreleased

### Added

- metapath2vec over a typed CSR. `--nodes` takes an `id,type` node table and
  `--metapath` a schema such as `gene-pathway-gene`. Node ids are interned
  strings, sorted contiguous by type. Without `--nodes` nothing changes.
- metapath2vec++ via `--metapath-plus`, with negatives drawn per node type.
- `nodes.csv` written next to the embeddings on the heterogeneous path.
- Walk stats printed for every metapath run.
- `NODE2VEC_SIMD` env var to lower the SIMD level, for comparing paths.
- divan benchmark for the SIMD kernels.
- docs.rs builds with `burn` enabled; CI fails on rustdoc warnings.

### Changed

- **Breaking:** `train_node2vec_cpu` takes a `NegativeTable`, and the negative
  tables hold `u32`.
- New `Node2VecError` (`thiserror`). The CLI prints the error and exits
  instead of panicking.
- CI calls the shared Rust test workflow.

### Fixed

- SIMD dispatch. AVX2 kernels had no `#[target_feature]`, so ran at 128 bits,
  and AVX-512 was gated on the compile-time cfg rather than the runtime probe.
  256- and 512-bit kernels are now written against `std::arch` with four
  accumulators, and FMA is probed with AVX2. On NEON, `dot` at 512 dims went
  from 85.0 to 36.8 ns. The x86 kernels are unmeasured.
- Broken intra-doc links in the `burn` and `het` modules.

## [0.2.0] - 2026-06-02

### Changed

- Burn 0.20 to 0.21.
- The `ndarray`, `ndarray-blas-openblas` and `ndarray-blas-accelerate` features
  are replaced by `flex`.
- Feature flags documented in the crate docs via `document-features`.
- Release workflow publishes automatically.

## [0.1.6] - 2026-02-25

### Added

- Subsampling of frequent nodes in the CPU trainer (`--sample`).

### Fixed

- Shuffling during CPU training.
- Better seeding of the random walks.

## [0.1.5] - 2026-02-22

### Changed

- Less noisy progress output.

## [0.1.4] - 2026-02-07

### Added

- Gensim-style CPU backend with SIMD dot, SAXPY and L2 norm, and Hogwild! SGD.
  This is the new default.

### Changed

- **Breaking:** Burn is optional behind the `burn` feature. Default is `cpu`
  instead of `tch-cpu`.
- `metal` and `vulkan` go through `burn/wgpu`.
- Release profile with LTO and a single codegen unit.

### Fixed

- Shuffling of the training data.

## [0.1.3] - 2026-02-03

### Changed

- Faster embedding extraction.
- Better embedding initialisation and scaling.
- More in the prelude.

### Fixed

- Forward pass, and node ordering in the extracted embeddings.

## [0.1.2] - 2026-02-02

### Added

- `tch-mps` feature for libtorch on Apple Silicon.

### Changed

- Burn 0.19 to 0.20.

## [0.1.1] - 2025-11-19

### Added

- `ndarray-blas-openblas` and `ndarray-blas-accelerate` features.

### Changed

- More fields and functions public.

## [0.1.0] - 2025-11-16

First release. node2vec with biased walks and skip-gram training on Burn,
with libtorch, ndarray and wgpu backends, a CLI and CI.

[0.3.0]: https://github.com/GregorLueg/node2vec-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GregorLueg/node2vec-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GregorLueg/node2vec-rs/releases/tag/v0.1.0
