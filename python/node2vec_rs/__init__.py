"""node2vec and metapath2vec in Rust.

Thin bindings over the Hogwild! CPU trainer of `node2vec-rs`. Hand in an edge
list, get node vectors back.

    >>> import node2vec_rs
    >>> emb = node2vec_rs.node2vec(src, dst, dim=64, p=1.0, q=0.5)
    >>> emb.vectors, emb.ids

    >>> emb = node2vec_rs.metapath2vec(
    ...     src, dst, node_ids, node_types, "gene-pathway-gene", plus=True
    ... )
    >>> emb.vectors, emb.ids, emb.types, emb.walk_stats
"""

from ._api import Embedding, metapath2vec, node2vec
from ._node2vec_rs import Node2VecError, __version__

__all__ = ["Embedding", "Node2VecError", "__version__", "metapath2vec", "node2vec"]
