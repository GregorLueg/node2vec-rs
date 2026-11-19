[![PyPI](https://img.shields.io/pypi/v/node2vec-rs.svg)](https://pypi.org/project/node2vec-rs/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://gregorlueg.github.io/node2vec-rs/)
[![CI](https://github.com/GregorLueg/node2vec-rs/actions/workflows/python-test.yml/badge.svg)](https://github.com/GregorLueg/node2vec-rs/actions/workflows/python-test.yml)

# node2vec-rs

Python bindings for [`node2vec-rs`](https://github.com/GregorLueg/node2vec-rs):
node2vec and metapath2vec graph embeddings. The Rust crate does the work,
parallel random walks and skip-gram trained with lock-free Hogwild! SGD on the
CPU. This is a thin layer over it. Hand in an edge list, get vectors back.

**Documentation: <https://gregorlueg.github.io/node2vec-rs/>**

This package versions separately from the Rust crate.

## Install

```bash
uv pip install node2vec-rs
```

Wheels for Linux x86_64 and macOS (arm64 and x86_64), Python 3.10 and up.

## Use

```python
import node2vec_rs

src = ["a", "a", "b", "c", "c"]
dst = ["b", "c", "c", "d", "e"]

emb = node2vec_rs.node2vec(src, dst, dim=16, p=1.0, q=0.5)
emb.vectors  # (5, 16) float32
emb.ids  # node id per row
```

Typed graph, knowledge graph style? metapath2vec walks follow a schema over node
types:

```python
emb = node2vec_rs.metapath2vec(
    src=["g1", "g2", "g3"],
    dst=["p1", "p1", "p2"],
    node_ids=["g1", "g2", "g3", "p1", "p2"],
    node_types=["gene", "gene", "gene", "pathway", "pathway"],
    metapath="gene-pathway-gene",
    plus=True,  # metapath2vec++
)
emb.vectors, emb.ids, emb.types, emb.walk_stats
```

Ids can be strings or integers. Edges go in as arrays, lists, tuples or polars
Series. With more than one thread the Hogwild! updates race, so pass
`n_threads=1` if you need bit-identical vectors at a fixed seed.

## References

Grover and Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD
2016.

Dong, Chawla and Swami, *metapath2vec: Scalable Representation Learning for
Heterogeneous Networks*, KDD 2017.
