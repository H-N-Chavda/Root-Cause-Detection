"""Fixtures for the phase 3 tests.

The synthetic system below is the backbone of the smoke test: if an algorithm
cannot recover a graph generated here, the wrapper is broken and no amount of
real-data output would reveal it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from causal_bench import paths
from causal_bench.graph.io import load_ground_truth_matrix

#: Contemporaneous X1->X2 and X3->X4; lagged X1(t-1)->X3 and X4(t-1)->X5.
SYNTHETIC_EDGES = [("X1", "X2"), ("X1", "X3"), ("X3", "X4"), ("X4", "X5")]
SYNTHETIC_NAMES = ["X1", "X2", "X3", "X4", "X5"]


@pytest.fixture(scope="session")
def synthetic_data() -> np.ndarray:
    """A linear system with Laplace innovations and a known graph.

    Laplace rather than Gaussian on purpose: VAR-LiNGAM identifies the
    contemporaneous order from non-Gaussian innovations, so a Gaussian-driven
    fixture would make its lag-0 result meaningless and the test would be
    checking nothing.
    """
    rng = np.random.default_rng(0)
    n, burn = 2000, 200
    total = n + burn
    e = rng.laplace(size=(total, 5))
    x = np.zeros((total, 5))
    for t in range(1, total):
        x[t, 0] = 0.3 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.9 * x[t, 0] + 0.5 * e[t, 1]
        x[t, 2] = 0.8 * x[t - 1, 0] + 0.2 * x[t - 1, 2] + 0.5 * e[t, 2]
        x[t, 3] = 0.9 * x[t, 2] + 0.5 * e[t, 3]
        x[t, 4] = 0.8 * x[t - 1, 3] + 0.5 * e[t, 4]
    return x[burn:]


@pytest.fixture(scope="session")
def synthetic_names() -> list[str]:
    return list(SYNTHETIC_NAMES)


@pytest.fixture(scope="session")
def synthetic_truth() -> np.ndarray:
    """The generating graph as an adjacency matrix, `matrix[i, j] == 1` for i->j."""
    index = {name: i for i, name in enumerate(SYNTHETIC_NAMES)}
    matrix = np.zeros((5, 5), dtype=int)
    for cause, effect in SYNTHETIC_EDGES:
        matrix[index[cause], index[effect]] = 1
    return matrix


@pytest.fixture(scope="session")
def synthetic_digraph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(SYNTHETIC_NAMES)
    graph.add_edges_from(SYNTHETIC_EDGES)
    return graph


@pytest.fixture(scope="session")
def te_ground_truth_matrix() -> np.ndarray:
    """The real 33x33 Tennessee Eastman ground truth."""
    return load_ground_truth_matrix(paths.ground_truth_path("TEGroundTruth.txt"))


@pytest.fixture(scope="session")
def te_full_names() -> list[str]:
    return [f"X{k + 1}" for k in range(33)]


@pytest.fixture(scope="session")
def te_observed_names() -> list[str]:
    """The 31 variables that have a column in the data. X27 and X31 do not."""
    return [f"X{k + 1}" for k in range(33) if k + 1 not in (27, 31)]
