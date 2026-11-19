# node2vec

node2vec runs second-order biased random walks over a homogeneous graph and
feeds them to skip-gram. Nodes that turn up in the same walks end up close
together.

```python
import itertools

import numpy as np
import node2vec_rs

# Two 8-cliques joined by a single bridge edge.
edges = [
    (f"{side}{i}", f"{side}{j}")
    for side in "ab"
    for i, j in itertools.combinations(range(8), 2)
]
edges.append(("a0", "b0"))
src, dst = zip(*edges)

emb = node2vec_rs.node2vec(src, dst, dim=16, walk_length=20, n_threads=1)

side = np.array([i[0] for i in emb.ids])
sim = emb.vectors @ emb.vectors.T  # cosine, rows are unit length
sim[side[:, None] == side[None, :]].mean(), sim[side[:, None] != side[None, :]].mean()
```

## The graph

`src` and `dst` name the endpoints of each edge, `weight` is optional and must
be positive. Undirected by default, so every edge goes in both ways; pass
`directed=True` to keep them one-way. Every node that appears in an edge gets a
row in the output, sorted by id. Isolated nodes have no edges to appear in, so
they get no row.

## `p` and `q`

The walk is biased by where it just came from.

| | Low | High |
|---|---|---|
| `p` (return) | Walk keeps doubling back, stays local | Walk rarely returns |
| `q` (in-out) | Walk heads outwards, DFS-like, picks up communities | Walk hugs the start, BFS-like, picks up structural roles |

`p = q = 1` is a plain weighted random walk, i.e. DeepWalk.

## Memory

The second-order transition table is keyed on `(previous, current)` node pairs,
so it grows with the sum of squared degrees. A few hubs with tens of thousands
of neighbours each get painful fast. If that bites, metapath2vec walks are
first-order and stay linear in the edge count.

## Defaults

`dim=128`, `walks_per_node=10`, `walk_length=80` and `window=10` follow the
node2vec paper. `lr=0.025`, `neg=5` and `sample=1e-3` are the word2vec
defaults. `epochs=5`. Output rows are L2-normalised unless you pass
`normalise=False`.

## Threads and reproducibility

All cores by default; `n_threads` caps it for that call. Hogwild! SGD means the
threads write to the shared weight matrices without locks, which is the whole
reason it is fast. The flip side: with more than one thread the result changes
from run to run at the same seed.

Need bit-identical vectors? Pass `n_threads=1`. Walk generation itself is
reproducible at any thread count.

`verbose=True` prints progress bars and a training log from Rust straight to the
terminal. Jupyter doesn't show them.
