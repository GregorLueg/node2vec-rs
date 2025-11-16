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

### Input Format

The input graph should be a CSV file with edges in the format:
```
from,to,weight
1,2,0.5
2,3,0.5
```

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

Train with custom embedding size and epochs:
```bash
cargo run --release -- --input data/karate.csv --embedding-dim 64 --num-epochs 10
```

Train on a directed graph with custom node2vec parameters:
```bash
cargo run --release -- --input data/graph.csv --directed --p 0.5 --q 2.0
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

MIT