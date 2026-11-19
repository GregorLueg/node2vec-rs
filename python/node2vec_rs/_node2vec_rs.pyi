"""Type stub for the compiled extension.

`ty` reads this instead of importing the module, so linting needs no maturin
build. The typed user-facing surface is `_api.py`.
"""

import numpy as np

__version__: str

class Node2VecError(ValueError): ...

def run_node2vec(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None,
    n_nodes: int,
    /,
    *,
    directed: bool,
    p: float,
    q: float,
    params: dict[str, int | float | bool | None],
) -> np.ndarray: ...
def run_metapath2vec(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None,
    node_ids: list[str],
    node_types: list[str],
    metapath: str,
    /,
    *,
    plus: bool,
    directed: bool,
    params: dict[str, int | float | bool | None],
) -> tuple[np.ndarray, list[str], dict[str, int | float]]: ...
