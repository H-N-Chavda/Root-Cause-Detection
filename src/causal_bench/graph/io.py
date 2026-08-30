"""Load and save graph objects as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .representation import CausalGraph


def _sanitise(value: Any) -> Any:
    """numpy scalars and non-finite floats made JSON safe.

    Same rule the EDA report uses: `json.dump` emits bare NaN and Infinity,
    which are not valid JSON, so they become null.
    """
    import math

    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return _sanitise(value.tolist())
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_graph(graph: CausalGraph, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitise(graph.to_dict())
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", "utf-8")
    return path


def load_graph(path: str | Path) -> CausalGraph:
    return CausalGraph.from_dict(json.loads(Path(path).read_text()))


def load_ground_truth_matrix(path: str | Path) -> np.ndarray:
    """A ground-truth adjacency matrix in its original, unaligned shape.

    Deliberately not `causal_bench.io.load_ground_truth`, which aligns the
    matrix onto the dataset's columns and drops the rest -- correct for scoring
    but it destroys the missing-node discrepancy this phase must reason about.
    """
    path = Path(path)
    for delimiter in (None, ",", "\t"):
        try:
            matrix = np.loadtxt(path, delimiter=delimiter)
        except Exception:
            continue
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            return (matrix != 0).astype(int)
    raise ValueError(f"could not read {path} as a square adjacency matrix")
