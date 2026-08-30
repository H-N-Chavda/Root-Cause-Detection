"""Shared fixtures.

Everything here is deliberately small. `datasetTE.csv` is 1499 rows x 31
variables and `DatasetUF.csv` is 23132 rows; running PC over either takes long
enough that a test suite built on them stops being run. The slice fixtures cut
the real files down to a size the suite can afford, and the synthetic fixtures
avoid touching `data/` at all.

`data/` is read only. Every fixture that produces a file writes into pytest's
`tmp_path`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_bench import paths

# Tennessee variables chosen so that the ground-truth sub-graph over them is
# non-empty (9 of the 32 edges survive the restriction): X6->X9, X7->X22,
# X8->X9, X9->X7, X11->X13, X11->X14, X12->X11, X22->X11, X22->X33.
# load_ground_truth refuses an empty target, so the subset cannot be arbitrary.
TE_SLICE_COLUMNS = ["X6", "X7", "X8", "X9", "X11", "X12", "X13", "X14", "X22", "X33"]
TE_SLICE_ROWS = 300


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return paths.REPO_ROOT


@pytest.fixture(scope="session")
def te_dataset_path() -> Path:
    return paths.dataset_path("datasetTE.csv")


@pytest.fixture(scope="session")
def te_ground_truth_path() -> Path:
    return paths.ground_truth_path("TEGroundTruth.txt")


@pytest.fixture
def small_ground_truth() -> np.ndarray:
    """A 4x4 directed ground truth: X1 -> X2 -> X3, X4 isolated."""
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 1] = 1
    matrix[1, 2] = 1
    return matrix


@pytest.fixture
def small_ground_truth_file(tmp_path: Path, small_ground_truth: np.ndarray) -> Path:
    """`small_ground_truth` written out in the tab-separated matrix format the
    real ground-truth files use."""
    path = tmp_path / "small_gt.txt"
    path.write_text(
        "\n".join("\t".join(str(v) for v in row) for row in small_ground_truth) + "\n"
    )
    return path


@pytest.fixture
def small_dataset_csv(tmp_path: Path) -> Path:
    """A 4-variable, 200-row synthetic CSV matching `small_ground_truth`."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = x1 + rng.normal(scale=0.3, size=200)
    x3 = x2 + rng.normal(scale=0.3, size=200)
    x4 = rng.normal(size=200)
    path = tmp_path / "small.csv"
    pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4}).to_csv(path, index=False)
    return path


@pytest.fixture
def te_slice_csv(tmp_path: Path, te_dataset_path: Path) -> Path:
    """The first `TE_SLICE_ROWS` rows of the real Tennessee data, restricted to
    `TE_SLICE_COLUMNS`.

    Real values and real X<k> column names, so the by-name alignment against the
    full 33x33 ground truth is exercised for real, at a size the suite can run.
    """
    frame = pd.read_csv(te_dataset_path, usecols=TE_SLICE_COLUMNS, nrows=TE_SLICE_ROWS)
    path = tmp_path / "datasetTE_slice.csv"
    frame.to_csv(path, index=False)
    return path
