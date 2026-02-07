[![CI](https://github.com/GregorLueg/node2vec-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/node2vec-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/node2vec-rs.svg)](https://crates.io/crates/node2vec-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# node2vec-rs

A Rust implementation of [node2vec](https://arxiv.org/abs/1607.00653) with two
training backends:

- **CPU** (default) -- a Gensim-style word2vec implementation with SIMD-accelerated
  linear algebra. Fast, lightweight, no framework dependencies.
- **Burn** -- uses the [Burn](https://github.com/tracel-ai/burn) deep learning
  framework with pluggable backends (libtorch, ndarray, wgpu).

## What is node2vec?

node2vec is an algorithmic framework for learning continuous feature
representations for nodes in networks. It uses biased random walks to generate
node sequences, which are then used to learn embeddings via a Skip-Gram model.

## Quick Start

```bash
# Build with default CPU backend
cargo build --release

# Run
cargo run --release -- --input tests/data/karate.csv
```

## Backends

### CPU (default)

The CPU backend uses a Gensim-style word2vec implementation with
SIMD-accelerated dot products, SAXPY, and L2 norms. It automatically detects
the best available instruction set (AVX-512, AVX2, SSE/NEON, or scalar
fallback).

```bash
cargo run --release -- --backend cpu --input tests/data/karate.csv
```

When using multiple threads, results are not fully reproducible across runs
due to a tolerated race condition in the shared embedding updates (mirroring
the original word2vec C implementation). Use --num-workers 1 for
deterministic results.

### Burn

The Burn backend supports multiple compute backends via feature flags:

```bash
# libtorch CPU (default Burn backend)
cargo run --release --features tch-cpu -- --backend burn --input tests/data/karate.csv

# libtorch MPS (Apple Silicon GPU)
cargo run --release --features tch-mps -- --backend burn --input tests/data/karate.csv

# wgpu
cargo run --release --features wgpu -- --backend burn --input tests/data/karate.csv

# ndarray
cargo run --release --features ndarray -- --backend burn --input tests/data/karate.csv

# ndarray with Apple Accelerate
cargo run --release --features ndarray-blas-accelerate -- --backend burn --input tests/data/karate.csv

# ndarray with OpenBLAS
cargo run --release --features ndarray-blas-openblas -- --backend burn --input tests/data/karate.csv
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | Gensim-style CPU backend |
| `tch-cpu` | Burn + libtorch CPU |
| `tch-mps` | Burn + libtorch MPS (Apple Silicon) |
| `ndarray` | Burn + ndarray |
| `ndarray-blas-openblas` | Burn + ndarray + OpenBLAS |
| `ndarray-blas-accelerate` | Burn + ndarray + Apple Accelerate |
| `wgpu` | Burn + wgpu |
| `metal` | Burn + wgpu (Metal) |
| `vulkan` | Burn + wgpu (Vulkan) |

## Input Format

The input graph should be a CSV file with edges:

```
from,to,weight
1,2,0.5
2,3,0.5
```

The `weight` column is optional and defaults to `1.0` if omitted.

## Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | *required* | Input graph file path |
| `--output` | `-o` | `/tmp/node2vec` | Output directory for model artefacts and embeddings |
| `--backend` | | `cpu` | Backend to use: `cpu` or `burn` |
| `--directed` | `-d` | `false` | Whether the graph is directed |
| `--embedding-dim` | `-e` | `16` | Embedding dimension |
| `--split` | `-s` | `0.9` | Training/validation split ratio (Burn only) |
| `--walks-per-node` | | `20` | Number of walks per node |
| `--walk-length` | | `20` | Length of each walk |
| `--window-size` | | `2` | Skip-Gram window size |
| `--batch-size` | | `256` | Training batch size (Burn only) |
| `--num-workers` | | `4` | Number of threads/workers |
| `--num-epochs` | | `5` | Number of training epochs |
| `--num-negatives` | | `5` | Number of negative samples |
| `--seed` | | `42` | Random seed |
| `--learning-rate` | | `0.001` | Learning rate (Adam for Burn, linear decay for CPU) |
| `--p` | | `1.0` | Return parameter |
| `--q` | | `1.0` | In-out parameter |

## node2vec Parameters

The `p` and `q` parameters control the random walk behaviour:

- **p** -- Controls the likelihood of returning to the previous node. Higher
  values make walks less likely to revisit nodes.
- **q** -- Controls the likelihood of exploring new parts of the graph. Values
  < 1 encourage exploration (BFS-like), values > 1 encourage local search
  (DFS-like).

## Examples

Train on the karate club graph with the CPU backend:

```bash
cargo run --release -- --input tests/data/karate.csv
```

Train on a directed graph with custom node2vec parameters:

```bash
cargo run --release -- --input tests/data/test_graph.csv --directed --p 0.5 --q 2.0
```

Full customisation:

```bash
cargo run --release -- \
  --input data/network.csv \
  --output ./models/my_embeddings \
  --embedding-dim 128 \
  --num-epochs 20 \
  --learning-rate 0.025 \
  --num-workers 8 \
  --p 2.0 \
  --q 0.5
```

Using the Burn backend with libtorch:

```bash
cargo run --release --features tch-cpu -- \
  --backend burn \
  --input tests/data/karate.csv \
  --embedding-dim 64 \
  --num-epochs 10 \
  --batch-size 512 \
  --learning-rate 0.0005
```

## Reproducibility

The CPU backend uses concurrent, lock-free updates to shared embedding matrices
(mirroring the original word2vec C implementation and Gensim). This means
results may vary slightly between runs when using multiple threads. For fully
deterministic results, set `--num-workers 1`.

The Burn backend is deterministic for a given seed and backend configuration.

## Testing

```bash
# Run CPU tests (default)
cargo test

# Run Burn tests
cargo test --features tch-cpu

# Run all tests
cargo test --all-features

# With output
cargo test -- --nocapture
```

## Licence

MIT License

Copyright (c) 2025 Gregor Alexander Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
