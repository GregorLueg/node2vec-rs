# node2vec-rs

A Rust implementation of node2vec using the Burn deep learning framework.

## What is node2vec?

node2vec is an algorithmic framework for learning continuous feature 
representations for nodes in networks. It uses biased random walks to generate 
node sequences, which are then used to learn embeddings via a Skip-Gram model.

## Usage

Build and run:

```bash
cargo build --release
cargo run --release -- --input <PATH TO GRAPH CSV>
```

In the standard version, this implementation uses the libtorch CPU backend. This
works well across most OS, but `ndarray` and `wgpu` are also enabled (more to 
that later).

### Input Format

The input graph should be a CSV file with edges in the format:
```
from,to,weight
1,2,0.5
2,3,0.5
```

You can find a Barabasi-based graph in `tests/data/test_graph.csv`.

### Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | *required* | Input graph file path |
| `--output` | `-o` | `/tmp/node2vec` | Output directory for model artefacts |
| `--directed` | `-d` | `false` | Whether the graph is directed |
| `--embedding-dim` | `-e` | `32` | Embedding dimension |
| `--split` | `-s` | `0.9` | Training split ratio |
| `--walks-per-node` | | `20` | Number of walks per node |
| `--walk-length` | | `20` | Length of each walk |
| `--window-size` | | `2` | Skip-Gram window size |
| `--batch-size` | | `256` | Training batch size |
| `--num-workers` | | `4` | Number of workers for batch generation |
| `--num-epochs` | | `5` | Number of training epochs |
| `--num-negatives` | | `5` | Number of negative samples |
| `--seed` | | `42` | Random seed for reproducibility |
| `--learning-rate` | | `0.001` | Learning rate for Adam optimiser |
| `--p` | | `1.0` | Return parameter (likelihood of returning to previous node) |
| `--q` | | `1.0` | In-out parameter (likelihood of exploring new nodes) |

### Examples

Train with karate data set
```bash
cargo run --release -- --input tests/data/karate.csv
```

If you want to use a different back-end you have these options:
```bash
# runs the code on the WGPU backend
cargo run --release --no-default-features --features wgpu -- --input tests/data/karate.csv
```

```bash
# runs the code on the ndarray backend
cargo run --release --no-default-features --features ndarray -- --input tests/data/karate.csv
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
  --batch-size 512 \
  --learning-rate 0.0005 \
  --p 2.0 \
  --q 0.5
```

## node2vec Parameters

The `p` and `q` parameters control the random walk behaviour:

- **p**: Controls the likelihood of returning to the previous node. Higher 
  values make walks less likely to revisit nodes.
- **q**: Controls the likelihood of exploring new parts of the graph. Values 
  < 1 encourage exploration (BFS-like), values > 1 encourage local search (DFS-like).

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