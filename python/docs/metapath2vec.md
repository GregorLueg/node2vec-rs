# metapath2vec

metapath2vec is node2vec for typed graphs. The walk follows a schema over node
types, say `gene-pathway-gene`, so a gene only ever steps to a pathway and a
pathway only ever steps back to a gene. Two genes end up close when they share
pathways, not because some hub happens to sit next to both.

```python
import node2vec_rs

emb = node2vec_rs.metapath2vec(
    src=["g0", "g1", "g2", "g3", "g4", "g5"],
    dst=["pA", "pA", "pA", "pB", "pB", "pB"],
    node_ids=["g0", "g1", "g2", "g3", "g4", "g5", "pA", "pB"],
    node_types=["gene"] * 6 + ["pathway"] * 2,
    metapath="gene-pathway-gene",
    dim=16,
    walk_length=21,
)
emb.walk_stats
```

## Nodes and edges

Unlike `node2vec()`, the nodes come in as their own table: `node_ids`, unique,
and `node_types`, one type name per id. At most 256 types. Every edge endpoint
has to be in `node_ids`, or you get a `ValueError` naming the first one that
isn't.

Leave `weight` out and the graph is unweighted, so each step is a single uniform
draw over the neighbours of the right type.

## The schema

Type names joined by hyphens, closing on the type it started on:

- `gene-pathway-gene`
- `author-paper-venue-paper-author`
- `drug-target-disease-target-drug`

The closing type makes the schema cycle, and `walk_length` is rounded up to a
whole number of cycles. Walks start only on nodes of the first type, so
`walks_per_node` means walks per node of that type. Type names can't contain a
hyphen.

A typo in a type name raises `Node2VecError` with the types the graph does
have.

## metapath2vec++

`plus=True` draws negatives from the context node's own type rather than from
every node. A pathway context gets pathway negatives. That is the "++" variant
from the paper (Dong, Chawla and Swami, KDD 2017).

## Check `walk_stats`

A wrong schema still produces vectors. They're just garbage. `walk_stats` is
the only warning you get:

| Key | Meaning |
|---|---|
| `start_nodes` | Nodes carrying the schema's first type |
| `attempted` | `start_nodes * walks_per_node` |
| `truncated` | Walks that hit a dead end before the full length |
| `dropped` | Walks shorter than two nodes, thrown away |
| `mean_length` | Mean length of the surviving walks |
| `walk_length` | Requested length after rounding up to the schema |

A high `truncated` or `dropped` share means the schema rarely fits the graph.
If every walk dead-ends on the first hop, there is nothing to train on and you
get a `Node2VecError` instead.

## Row order

The rows come back grouped by type, not in `node_ids` order. `emb.ids` and
`emb.types` name each row, so index through them:

```python
genes = emb.vectors[emb.types == "gene"]
```

Nodes no walk ever reaches, a type outside the schema for instance, keep their
random initialisation. Filter them out before you use the vectors.

Threads, reproducibility and the training defaults work exactly as in
[node2vec](node2vec.md#threads-and-reproducibility).
