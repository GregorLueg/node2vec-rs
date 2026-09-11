[![CI](https://github.com/GregorLueg/node2vec-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/node2vec-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/node2vec-rs.svg)](https://crates.io/crates/node2vec-rs)
[![docs.rs](https://img.shields.io/docsrs/node2vec-rs)](https://docs.rs/node2vec-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# node2vec-rs

A Rust implementation of [node2vec](https://arxiv.org/abs/1607.00653) with two
training backends:

- **CPU** (default) -- a Gensim-style word2vec implementation with SIMD-accelerated
  linear algebra. Fast, lightweight, no framework dependencies.
- **Burn** -- uses the [Burn](https://github.com/tracel-ai/burn) deep learning
  framework with pluggable backends (libtorch, flex, wgpu).

Heterogeneous graphs are covered too: hand it a node table with types and a
metapath and you get [metapath2vec](https://dl.acm.org/doi/10.1145/3097983.3098036)
(or metapath2vec++) on the CPU backend.

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
deterministic results. The core idea here is to use the
[Hogwild! style SGD](https://arxiv.org/abs/1106.5730) for maximum performance.

### Burn

The Burn backend supports multiple compute backends via feature flags:

```bash
# libtorch CPU (default Burn backend)
cargo run --release --features tch-cpu -- --backend burn --input tests/data/karate.csv

# libtorch MPS (Apple Silicon GPU)
cargo run --release --features tch-mps -- --backend burn --input tests/data/karate.csv

# wgpu
cargo run --release --features wgpu -- --backend burn --input tests/data/karate.csv

# flex
cargo run --release --features flex -- --backend burn --input tests/data/karate.csv
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | Gensim-style CPU backend |
| `tch-cpu` | Burn + libtorch CPU |
| `tch-mps` | Burn + libtorch MPS (Apple Silicon) |
| `flex` | Burn + flex CPU framework |
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
| `--sample` | | `0.001` | Subsampling threshold for frequent nodes (CPU only) |
| `--nodes` | | | Node table (`id,type`); switches to metapath2vec |
| `--metapath` | | | Metapath schema, e.g. `gene-pathway-gene`; required with `--nodes` |
| `--metapath-plus` | | `false` | Per-type negative sampling (metapath2vec++) |

## node2vec Parameters

The `p` and `q` parameters control the random walk behaviour:

- **p** -- Controls the likelihood of returning to the previous node. Higher
  values make walks less likely to revisit nodes.
- **q** -- Controls the likelihood of exploring new parts of the graph. Values
  < 1 encourage exploration (BFS-like), values > 1 encourage local search
  (DFS-like).

## metapath2vec

Got a typed graph, say genes, pathways and GO terms? Pass a node table with
`--nodes` and a schema with `--metapath`, and the walks follow the schema
instead of the p/q bias. The skip-gram objective is the same, so it trains on
the same CPU backend.

The node table is `id,type`; the edge table stays `from,to[,weight]`. Ids are
strings here, so `GO:0006915` and `ENSG00000141510` are fine as they are:

```
id,type
g1,gene
p1,pathway
```

```bash
cargo run --release -- \
  --input tests/data/het_edges.csv \
  --nodes tests/data/het_nodes.csv \
  --metapath gene-pathway-gene
```

A few things to know:

- The schema is hyphen separated and has to close on its starting type
  (`gene-pathway-gene`, not `gene-pathway`). Type names can't contain `-`.
- Walks are first-order, so `--p` and `--q` are ignored. This also means no
  `(prev, curr)` transition table, which would blow up on a knowledge graph.
- `--metapath-plus` draws negatives from the context node's own type, i.e.
  metapath2vec++.
- Besides `embeddings.csv`, the run writes `nodes.csv` mapping each embedding
  row back to its string id and type.
- Walk stats (start nodes, truncated and dropped walks) are always printed. A
  wrong schema still produces vectors, just garbage ones, so read them.
- CPU backend only. `--backend burn` with `--nodes` errors out.

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
