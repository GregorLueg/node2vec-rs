# node2vec-rs

node2vec and metapath2vec graph embeddings. The
[Rust crate](https://github.com/GregorLueg/node2vec-rs) does the work: random
walks in parallel, then skip-gram with negative sampling trained by lock-free
Hogwild! SGD on the CPU. This package is a thin layer over it. Hand in an edge
list, get vectors back.

```bash
uv pip install node2vec-rs
```

```python
import node2vec_rs

src = ["a", "a", "b", "c", "c"]
dst = ["b", "c", "c", "d", "e"]

emb = node2vec_rs.node2vec(src, dst, dim=16, walk_length=20)
emb.vectors  # (5, 16) float32
emb.ids  # array(['a', 'b', 'c', 'd', 'e'])
```

Typed graph, knowledge graph style? Use metapath2vec:

```python
emb = node2vec_rs.metapath2vec(
    src=["g1", "g2", "g3"],
    dst=["p1", "p1", "p2"],
    node_ids=["g1", "g2", "g3", "p1", "p2"],
    node_types=["gene", "gene", "gene", "pathway", "pathway"],
    metapath="gene-pathway-gene",
)
emb.vectors, emb.ids, emb.types, emb.walk_stats
```

## Two functions, no classes

There is nothing to fit and keep: no `transform` on unseen nodes, no index to
query. So `node2vec()` and `metapath2vec()` are plain functions returning an
[`Embedding`](api.md#node2vec_rs.Embedding), a frozen dataclass of the vectors
and the ids naming each row.

Ids can be strings or integers, and they come back in the dtype they went in
as. Edges go in as two equal-length arrays plus an optional weight array, so a
polars frame, a networkx graph or a CSV are one line away:

```python
# polars
emb = node2vec_rs.node2vec(df["from"], df["to"], df["weight"])

# networkx
src, dst = zip(*G.edges())
emb = node2vec_rs.node2vec(src, dst)
```

## Where to go next

[node2vec](node2vec.md) covers `p`, `q`, memory and reproducibility.
[metapath2vec](metapath2vec.md) covers schemas, metapath2vec++ and the walk
statistics. The [API reference](api.md) has every argument.

This package versions separately from the Rust crate. Python 0.1.0 wraps
`node2vec-rs` 0.5.0.
