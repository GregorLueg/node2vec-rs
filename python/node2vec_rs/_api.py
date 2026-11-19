"""node2vec and metapath2vec entry points.

Both are one-shot: walk, train, hand back the vectors. There is no fitted state
worth keeping, so these are functions rather than an estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from beartype import beartype

from . import _node2vec_rs as _core

###########
# Globals #
###########

#: Node indices cross the FFI as `u32`.
_MAX_NODES: int = np.iinfo(np.uint32).max


@runtime_checkable
class _SupportsArray(Protocol):
    """Anything numpy can turn into an array, e.g. a polars Series."""

    def __array__(self) -> np.ndarray: ...


#: Node ids: strings or integers. Not `Ids`, which beartype rejects
#: for a plain tuple of strings, the very thing `zip(*G.edges())` hands back.
Ids: TypeAlias = np.ndarray | _SupportsArray | Sequence[str] | Sequence[int]

#: Edge weights.
Weights: TypeAlias = np.ndarray | _SupportsArray | Sequence[float] | Sequence[int]


@dataclass(frozen=True)
class Embedding:
    """Node vectors and the ids that name their rows.

    Attributes:
        vectors: `(n_nodes, dim)` float32 array.
        ids: Node id per row, in the dtype the ids came in as.
        types: Node type per row. metapath2vec only.
        walk_stats: What walk generation produced: start nodes, walks
            attempted, truncated and dropped, mean realised length and the
            walk length after rounding up to the schema. metapath2vec only. A
            high truncated or dropped count means the schema rarely fits the
            graph and the vectors are not worth much.
    """

    vectors: np.ndarray
    ids: np.ndarray
    types: np.ndarray | None = None
    walk_stats: dict[str, int | float] | None = None


#############
# Functions #
#############


@beartype
def node2vec(
    src: Ids,
    dst: Ids,
    weight: Weights | None = None,
    *,
    directed: bool = False,
    p: float = 1.0,
    q: float = 1.0,
    dim: int = 128,
    walks_per_node: int = 10,
    walk_length: int = 80,
    window: int = 10,
    epochs: int = 5,
    neg: int = 5,
    lr: float = 0.025,
    sample: float = 1e-3,
    normalise: bool = True,
    seed: int = 42,
    n_threads: int | None = None,
    verbose: bool = False,
) -> Embedding:
    """Embed the nodes of a homogeneous graph with node2vec.

    Second-order biased random walks feed a skip-gram model with negative
    sampling, trained with lock-free (Hogwild!) SGD.

    Memory is dominated by the transition table, keyed on `(previous,
    current)` node pairs, so it grows with the sum of squared degrees. Hub-heavy
    graphs get expensive fast; metapath2vec walks are first-order and do not
    have this problem.

    With more than one thread the threads race on the shared weights, so the
    vectors differ between runs at the same seed. Pass `n_threads=1` for
    bit-identical output. Walk generation is reproducible at any thread count.

    Args:
        src: Source node id per edge. Integers or strings.
        dst: Target node id per edge, same length as `src`.
        weight: Positive weight per edge. None weights every edge 1.0.
        directed: Keep edges one-way. Undirected graphs get both directions.
        p: Return parameter. Low values keep the walk local.
        q: In-out parameter. Low values push the walk outwards (DFS-like),
            high values keep it near the start (BFS-like).
        dim: Embedding dimension.
        walks_per_node: Walks started from every node.
        walk_length: Nodes per walk.
        window: Maximum skip-gram context window.
        epochs: Passes over the walks.
        neg: Negative samples per positive pair.
        lr: Initial learning rate, decayed linearly towards zero.
        sample: Subsampling threshold for frequent nodes. 0 disables it.
        normalise: L2-normalise every output vector.
        seed: Random seed.
        n_threads: Threads for this call. None uses every core.
        verbose: Print stage timings and one line per epoch. In a terminal
            the walks and training also get live progress bars.

    Returns:
        The embedding. Every node that appears in `src` or `dst` gets a row;
        rows are sorted by node id.

    Raises:
        ValueError: On mismatched edge arrays, a non-positive or non-finite
            weight, or an out-of-range parameter.
    """
    src_a, dst_a = _edge_arrays(src, dst)
    weight_a = _weights(weight, src_a.size)

    ids, inverse = np.unique(np.concatenate([src_a, dst_a]), return_inverse=True)
    if ids.size > _MAX_NODES:
        raise ValueError(f"at most {_MAX_NODES} nodes are supported, got {ids.size}")
    inverse = inverse.astype(np.uint32)

    _check_positive(p=p, q=q)
    params = _params(
        dim=dim,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        window=window,
        epochs=epochs,
        neg=neg,
        lr=lr,
        sample=sample,
        normalise=normalise,
        seed=seed,
        n_threads=n_threads,
        verbose=verbose,
    )

    vectors = _core.run_node2vec(
        inverse[: src_a.size],
        inverse[src_a.size :],
        weight_a,
        ids.size,
        directed=directed,
        p=p,
        q=q,
        params=params,
    )
    return Embedding(vectors=vectors, ids=ids)


@beartype
def metapath2vec(
    src: Ids,
    dst: Ids,
    node_ids: Ids,
    node_types: Ids,
    metapath: str,
    weight: Weights | None = None,
    *,
    plus: bool = False,
    directed: bool = False,
    dim: int = 128,
    walks_per_node: int = 10,
    walk_length: int = 80,
    window: int = 10,
    epochs: int = 5,
    neg: int = 5,
    lr: float = 0.025,
    sample: float = 1e-3,
    normalise: bool = True,
    seed: int = 42,
    n_threads: int | None = None,
    verbose: bool = False,
) -> Embedding:
    """Embed the nodes of a heterogeneous graph with metapath2vec.

    Walks follow a cyclic schema over node types, e.g. `"gene-pathway-gene"`,
    and start only on nodes of the schema's first type. The walk is
    first-order, so memory stays linear in the edge count.

    Same reproducibility caveat as `node2vec`: pass `n_threads=1` for
    bit-identical vectors.

    Args:
        src: Source node id per edge. Must appear in `node_ids`.
        dst: Target node id per edge, same length as `src`.
        node_ids: Every node id, unique. Integers or strings.
        node_types: Type name per entry of `node_ids`. At most 256 types.
        metapath: Hyphen-separated type names that close on the starting type,
            e.g. `"author-paper-venue-paper-author"`. Type names may not
            contain a hyphen.
        weight: Positive weight per edge. None makes the graph unweighted.
        plus: Draw negatives from the context node's own type, i.e.
            metapath2vec++.
        directed: Keep edges one-way. Undirected graphs get both directions.
        dim: Embedding dimension.
        walks_per_node: Walks started from every node of the starting type.
        walk_length: Nodes per walk, rounded up to close the schema.
        window: Maximum skip-gram context window.
        epochs: Passes over the walks.
        neg: Negative samples per positive pair.
        lr: Initial learning rate, decayed linearly towards zero.
        sample: Subsampling threshold for frequent nodes. 0 disables it.
        normalise: L2-normalise every output vector.
        seed: Random seed.
        n_threads: Threads for this call. None uses every core.
        verbose: Print stage timings and one line per epoch. In a terminal
            the walks and training also get live progress bars.

    Returns:
        The embedding with `types` and `walk_stats` set. Rows are grouped by
        type, not in `node_ids` order; `ids` and `types` name each row. Nodes
        no walk reaches keep their random initialisation.

    Raises:
        ValueError: On mismatched or duplicate ids, an edge naming an unknown
            node, or an out-of-range parameter.
        Node2VecError: If the core rejects the metapath or no node carries its
            starting type. Subclasses ValueError.
    """
    src_a, dst_a = _edge_arrays(src, dst)
    weight_a = _weights(weight, src_a.size)

    ids = np.asarray(node_ids)
    types = np.asarray(node_types)
    if ids.ndim != 1 or ids.shape != types.shape:
        raise ValueError(
            f"node_ids and node_types must be 1-D and the same length, got "
            f"shapes {ids.shape} and {types.shape}"
        )
    if ids.size > _MAX_NODES:
        raise ValueError(f"at most {_MAX_NODES} nodes are supported, got {ids.size}")
    if np.unique(ids).size != ids.size:
        raise ValueError("node_ids must be unique")

    params = _params(
        dim=dim,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        window=window,
        epochs=epochs,
        neg=neg,
        lr=lr,
        sample=sample,
        normalise=normalise,
        seed=seed,
        n_threads=n_threads,
        verbose=verbose,
    )

    ids_str = ids.astype(str)
    vectors, names, stats = _core.run_metapath2vec(
        _lookup(ids, src_a, "src"),
        _lookup(ids, dst_a, "dst"),
        weight_a,
        ids_str.tolist(),
        types.astype(str).tolist(),
        metapath,
        plus=plus,
        directed=directed,
        params=params,
    )

    rows = _lookup(ids_str, np.asarray(names), "core output")
    return Embedding(
        vectors=vectors, ids=ids[rows], types=types[rows], walk_stats=stats
    )


###########
# Helpers #
###########


@beartype
def _edge_arrays(src: Ids, dst: Ids) -> tuple[np.ndarray, np.ndarray]:
    """Coerce the edge endpoints to 1-D arrays of equal, non-zero length."""
    src_a = np.asarray(src)
    dst_a = np.asarray(dst)
    if src_a.ndim != 1 or src_a.shape != dst_a.shape:
        raise ValueError(
            f"src and dst must be 1-D and the same length, got shapes "
            f"{src_a.shape} and {dst_a.shape}"
        )
    if src_a.size == 0:
        raise ValueError("the graph has no edges")
    return src_a, dst_a


@beartype
def _weights(weight: Weights | None, n_edges: int) -> np.ndarray | None:
    """Coerce edge weights to contiguous float32 and check them."""
    if weight is None:
        return None
    w = np.ascontiguousarray(weight, dtype=np.float32)
    if w.shape != (n_edges,):
        raise ValueError(f"weight must have shape ({n_edges},), got {w.shape}")
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("every weight must be finite and positive")
    return w


@beartype
def _lookup(ids: np.ndarray, query: np.ndarray, what: str) -> np.ndarray:
    """Position of every `query` entry in `ids`.

    Sort-and-search rather than a dict, so it stays vectorised on millions of
    edges and works for integer and string ids alike.

    Returns:
        uint32 positions into `ids`.

    Raises:
        ValueError: If a query entry is not in `ids`.
    """
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    pos = np.searchsorted(sorted_ids, query).clip(max=ids.size - 1)
    missing = sorted_ids[pos] != query
    if missing.any():
        raise ValueError(
            f"{what} holds {int(missing.sum())} id(s) absent from node_ids, "
            f"e.g. {query[missing][0]!r}"
        )
    return order[pos].astype(np.uint32)


@beartype
def _check_positive(**values: float) -> None:
    """Raise unless every value is finite and strictly positive."""
    for name, v in values.items():
        if not (np.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be finite and positive, got {v}")


@beartype
def _params(
    *,
    dim: int,
    walks_per_node: int,
    walk_length: int,
    window: int,
    epochs: int,
    neg: int,
    lr: float,
    sample: float,
    normalise: bool,
    seed: int,
    n_threads: int | None,
    verbose: bool,
) -> dict[str, int | float | bool | None]:
    """Check the shared walk and training parameters and pack them for the core."""
    for name, v in {
        "dim": dim,
        "walks_per_node": walks_per_node,
        "window": window,
        "epochs": epochs,
        "neg": neg,
    }.items():
        if v < 1:
            raise ValueError(f"{name} must be at least 1, got {v}")
    if walk_length < 2:
        raise ValueError(f"walk_length must be at least 2, got {walk_length}")
    if n_threads is not None and n_threads < 1:
        raise ValueError(f"n_threads must be at least 1 or None, got {n_threads}")
    if not (np.isfinite(sample) and sample >= 0):
        raise ValueError(f"sample must be finite and non-negative, got {sample}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    _check_positive(lr=lr)

    return {
        "dim": dim,
        "walks_per_node": walks_per_node,
        "walk_length": walk_length,
        "window": window,
        "epochs": epochs,
        "neg": neg,
        "lr": lr,
        "sample": sample,
        "normalise": normalise,
        "seed": seed,
        "n_threads": n_threads,
        "verbose": verbose,
    }
