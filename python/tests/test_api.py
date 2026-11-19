"""End-to-end checks of the two entry points."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import node2vec_rs
from node2vec_rs import Node2VecError

###########
# Globals #
###########

#: Small and fast; the defaults are sized for real graphs.
FAST: dict[str, int] = {"dim": 16, "walks_per_node": 10, "walk_length": 20}


def two_cliques() -> tuple[np.ndarray, np.ndarray]:
    """Two 8-cliques, `a*` and `b*`, joined by one bridge edge."""
    edges = [
        (f"{side}{i}", f"{side}{j}")
        for side in "ab"
        for i, j in itertools.combinations(range(8), 2)
    ]
    edges.append(("a0", "b0"))
    src, dst = zip(*edges, strict=True)
    return np.array(src), np.array(dst)


def toy_het() -> dict[str, np.ndarray]:
    """Genes `g0..g5` in two groups, each sharing a pathway, plus a stray drug."""
    return {
        "node_ids": np.array(["g0", "g1", "g2", "g3", "g4", "g5", "pA", "pB", "d0"]),
        "node_types": np.array(["gene"] * 6 + ["pathway"] * 2 + ["drug"]),
        "src": np.array(["g0", "g1", "g2", "g3", "g4", "g5", "d0"]),
        "dst": np.array(["pA", "pA", "pA", "pB", "pB", "pB", "g0"]),
    }


############
# node2vec #
############


def test_node2vec_shape_and_ids() -> None:
    src, dst = two_cliques()
    emb = node2vec_rs.node2vec(src, dst, **FAST)
    assert emb.vectors.shape == (16, 16)
    assert emb.vectors.dtype == np.float32
    assert sorted(emb.ids.tolist()) == emb.ids.tolist()
    assert set(emb.ids.tolist()) == {f"{s}{i}" for s in "ab" for i in range(8)}
    assert emb.types is None
    assert emb.walk_stats is None


def test_node2vec_normalises_rows() -> None:
    src, dst = two_cliques()
    emb = node2vec_rs.node2vec(src, dst, **FAST)
    np.testing.assert_allclose(np.linalg.norm(emb.vectors, axis=1), 1.0, rtol=1e-5)


def test_node2vec_separates_cliques() -> None:
    src, dst = two_cliques()
    emb = node2vec_rs.node2vec(src, dst, n_threads=1, **FAST)
    side = np.array([i[0] for i in emb.ids])
    sim = emb.vectors @ emb.vectors.T
    same = sim[side[:, None] == side[None, :]].mean()
    other = sim[side[:, None] != side[None, :]].mean()
    assert same > other + 0.2


def test_node2vec_single_thread_is_reproducible() -> None:
    src, dst = two_cliques()
    a = node2vec_rs.node2vec(src, dst, n_threads=1, seed=7, **FAST)
    b = node2vec_rs.node2vec(src, dst, n_threads=1, seed=7, **FAST)
    c = node2vec_rs.node2vec(src, dst, n_threads=1, seed=8, **FAST)
    np.testing.assert_array_equal(a.vectors, b.vectors)
    assert not np.array_equal(a.vectors, c.vectors)


def test_node2vec_takes_tuples() -> None:
    src, dst = zip(*[("a", "b"), ("b", "c"), ("c", "a")], strict=True)
    emb = node2vec_rs.node2vec(src, dst, weight=(1, 2, 3), **FAST)
    assert emb.ids.tolist() == ["a", "b", "c"]


def test_node2vec_integer_ids_and_weights() -> None:
    src = np.array([10, 20, 30, 10])
    dst = np.array([20, 30, 10, 40])
    emb = node2vec_rs.node2vec(src, dst, weight=[1.0, 2.0, 0.5, 1.0], **FAST)
    assert emb.ids.tolist() == [10, 20, 30, 40]
    assert emb.vectors.shape == (4, 16)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weight": [1.0, -1.0, 1.0, 1.0]}, "weight"),
        ({"weight": [1.0, 1.0]}, "shape"),
        ({"p": 0.0}, "p must"),
        ({"dim": 0}, "dim"),
        ({"walk_length": 1}, "walk_length"),
        ({"n_threads": 0}, "n_threads"),
    ],
)
def test_node2vec_rejects_bad_input(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        node2vec_rs.node2vec([0, 1, 2, 0], [1, 2, 0, 3], **kwargs)


def test_node2vec_rejects_mismatched_edges() -> None:
    with pytest.raises(ValueError, match="same length"):
        node2vec_rs.node2vec([0, 1], [1])


################
# metapath2vec #
################


@pytest.mark.parametrize("plus", [False, True])
def test_metapath2vec_rows_line_up_with_ids_and_types(plus: bool) -> None:
    g = toy_het()
    emb = node2vec_rs.metapath2vec(
        g["src"],
        g["dst"],
        g["node_ids"],
        g["node_types"],
        "gene-pathway-gene",
        plus=plus,
        **FAST,
    )
    truth = dict(zip(g["node_ids"].tolist(), g["node_types"].tolist(), strict=True))
    assert emb.vectors.shape == (9, 16)
    assert sorted(emb.ids.tolist()) == sorted(truth)
    assert emb.types is not None
    assert all(truth[i] == t for i, t in zip(emb.ids, emb.types, strict=True))
    assert emb.walk_stats is not None
    assert emb.walk_stats["start_nodes"] == 6
    assert emb.walk_stats["dropped"] == 0


def test_metapath2vec_keeps_integer_id_dtype() -> None:
    emb = node2vec_rs.metapath2vec(
        [0, 1, 2],
        [3, 3, 3],
        [0, 1, 2, 3],
        ["a", "a", "a", "b"],
        "a-b-a",
        **FAST,
    )
    assert emb.ids.dtype.kind == "i"
    assert sorted(emb.ids.tolist()) == [0, 1, 2, 3]


def test_metapath2vec_unknown_type_raises_core_error() -> None:
    g = toy_het()
    with pytest.raises(Node2VecError, match="Unknown node type"):
        node2vec_rs.metapath2vec(
            g["src"], g["dst"], g["node_ids"], g["node_types"], "gene-protein-gene"
        )


def test_metapath2vec_unknown_edge_node() -> None:
    g = toy_het()
    with pytest.raises(ValueError, match="absent from node_ids"):
        node2vec_rs.metapath2vec(
            ["g0", "nope"],
            ["pA", "pA"],
            g["node_ids"],
            g["node_types"],
            "gene-pathway-gene",
        )


def test_metapath2vec_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        node2vec_rs.metapath2vec(
            ["x"], ["y"], ["x", "x", "y"], ["a", "a", "b"], "a-b-a"
        )


def test_metapath2vec_all_walks_dead_end() -> None:
    with pytest.raises(Node2VecError, match="dead-ended"):
        node2vec_rs.metapath2vec(
            ["a0"], ["c0"], ["a0", "b0", "c0"], ["a", "b", "c"], "a-b-a"
        )
