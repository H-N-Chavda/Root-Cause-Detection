"""Shape, dtypes, integrity, and the data/ground-truth node reconciliation.

Nothing here is specific to any dataset: every function takes a dataframe (and,
where relevant, a ground-truth matrix) plus the config, and returns a plain
JSON-serialisable mapping.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import StructureConfig

#: Columns named <prefix><integer>, e.g. "sensor_12" or "tag7". Used only to line
#: the data up with a ground-truth matrix that carries no names of its own.
_INDEXED_NAME = re.compile(r"^([^\d]*?)(\d+)$")


def describe_shape(frame: pd.DataFrame) -> dict[str, Any]:
    """Shape, dtypes, memory, and the index's type and monotonicity."""
    index = frame.index
    interval = _sampling_interval(index)
    return {
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "index": {
            "type": type(index).__name__,
            "dtype": str(index.dtype),
            "is_monotonic_increasing": bool(index.is_monotonic_increasing),
            "is_unique": bool(index.is_unique),
            "first": str(index[0]) if len(index) else None,
            "last": str(index[-1]) if len(index) else None,
        },
        "sampling_interval": interval,
    }


def _sampling_interval(index: pd.Index) -> dict[str, Any]:
    """Infers the spacing between consecutive rows.

    Returns the modal step and the fraction of steps equal to it, so an index
    with occasional gaps is distinguishable from a regular one. `regular` is
    None when the index carries no order information at all.
    """
    if len(index) < 3:
        return {"regular": None, "step": None, "unit": None, "fraction_at_step": None}

    if isinstance(index, pd.DatetimeIndex):
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return {"regular": None, "step": None, "unit": None, "fraction_at_step": None}
        modal = deltas.mode().iloc[0]
        return {
            "regular": bool((deltas == modal).mean() > 0.99),
            "step": float(modal.total_seconds()),
            "unit": "seconds",
            "fraction_at_step": float((deltas == modal).mean()),
        }

    if pd.api.types.is_numeric_dtype(index):
        deltas = pd.Series(np.diff(np.asarray(index, dtype=float)))
        modal = deltas.mode().iloc[0]
        return {
            "regular": bool(np.isclose(deltas, modal).mean() > 0.99),
            "step": float(modal),
            "unit": "index units",
            "fraction_at_step": float(np.isclose(deltas, modal).mean()),
        }

    return {"regular": None, "step": None, "unit": None, "fraction_at_step": None}


def check_integrity(frame: pd.DataFrame, cfg: StructureConfig) -> dict[str, Any]:
    """Missing values, constant and near-constant columns, duplicates, and
    columns whose support is discrete rather than continuous.

    A discrete or binary column matters because every normality test below
    assumes a continuous distribution; reporting the Shapiro-Wilk p-value of a
    two-valued column is meaningless, so those columns are flagged here and the
    distribution stage marks their verdicts as not applicable.
    """
    n_rows = len(frame)
    numeric = frame.select_dtypes(include=[np.number])

    missing = {str(c): int(frame[c].isna().sum()) for c in frame.columns}

    constant: list[str] = []
    near_constant: list[dict[str, Any]] = []
    discrete: list[dict[str, Any]] = []

    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        n_unique = int(series.nunique())
        std = float(series.std(ddof=0))
        mean = float(series.mean())
        # Coefficient of variation is undefined at a zero mean; fall back to the
        # raw standard deviation so a zero-centred constant is still caught.
        cv = abs(std / mean) if mean != 0.0 else std

        if n_unique <= 1 or std == 0.0:
            constant.append(str(column))
        elif cv < cfg.near_constant_cv or n_unique < cfg.min_unique:
            near_constant.append({"column": str(column), "cv": cv, "n_unique": n_unique})

        unique_ratio = n_unique / len(series)
        is_integer_valued = bool(np.allclose(series.to_numpy(), np.round(series)))
        if n_unique == 2 or unique_ratio < cfg.discrete_unique_ratio or is_integer_valued:
            discrete.append(
                {
                    "column": str(column),
                    "n_unique": n_unique,
                    "unique_ratio": unique_ratio,
                    "integer_valued": is_integer_valued,
                    "binary": n_unique == 2,
                }
            )

    duplicate_rows = int(frame.duplicated().sum())
    duplicate_columns = _duplicate_columns(numeric)

    return {
        "missing_per_column": missing,
        "missing_total": int(sum(missing.values())),
        "columns_with_missing": [c for c, n in missing.items() if n > 0],
        "constant_columns": constant,
        "near_constant_columns": near_constant,
        "discrete_columns": discrete,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_fraction": duplicate_rows / n_rows if n_rows else 0.0,
        "duplicate_column_groups": duplicate_columns,
        "non_numeric_columns": [
            str(c) for c in frame.columns if c not in set(numeric.columns)
        ],
    }


def _duplicate_columns(frame: pd.DataFrame) -> list[list[str]]:
    """Groups of columns that are exactly equal.

    pandas has no built-in for this (`duplicated` works on rows), so the columns
    are hashed on their byte content and only the colliding groups compared.
    """
    by_hash: dict[bytes, list[str]] = {}
    for column in frame.columns:
        key = pd.util.hash_pandas_object(frame[column], index=False).values.tobytes()
        by_hash.setdefault(key, []).append(str(column))
    return [group for group in by_hash.values() if len(group) > 1]


def reconcile_with_ground_truth(
    columns: Sequence[str], ground_truth: np.ndarray | None
) -> dict[str, Any]:
    """Lines the data's columns up against a ground-truth adjacency matrix.

    Ground-truth files are bare numeric matrices with no names, so the node set
    can only be reconstructed from the data's own naming scheme. When every
    column is <prefix><integer>, node k of the matrix is taken to be
    <prefix><k+1> -- the same convention the algorithms module uses -- and the
    two sets are differenced. When the names do not follow that scheme the check
    degrades to a dimension comparison and says so.

    Also reports the ground truth's own structural properties (edge count,
    self-loops, acyclicity), because an evaluation target with a cycle or a
    self-loop is not reachable by any DAG-based method.
    """
    columns = [str(c) for c in columns]
    result: dict[str, Any] = {
        "n_data_columns": len(columns),
        "ground_truth_available": ground_truth is not None,
    }
    if ground_truth is None:
        return result

    matrix = np.asarray(ground_truth)
    n_nodes = int(matrix.shape[0])
    binary = (matrix != 0).astype(int)
    off_diagonal = binary.copy()
    np.fill_diagonal(off_diagonal, 0)

    result.update(
        {
            "ground_truth_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "ground_truth_square": bool(matrix.shape[0] == matrix.shape[1]),
            "ground_truth_edges": int(off_diagonal.sum()),
            "ground_truth_self_loops": int(np.trace(binary)),
            "ground_truth_reciprocal_pairs": int(
                sum(
                    1
                    for i in range(n_nodes)
                    for j in range(i + 1, n_nodes)
                    if off_diagonal[i, j] and off_diagonal[j, i]
                )
            ),
            "ground_truth_acyclic": _is_acyclic(off_diagonal),
        }
    )

    parsed = [_INDEXED_NAME.match(c) for c in columns]
    prefixes = {m.group(1) for m in parsed if m}
    if not all(parsed) or len(prefixes) != 1:
        result.update(
            {
                "alignment": "unavailable",
                "reason": "column names are not a single <prefix><integer> family, "
                "so ground-truth rows cannot be named",
                "dimension_match": bool(n_nodes == len(columns)),
            }
        )
        return result

    prefix = prefixes.pop()
    data_nodes = {f"{prefix}{int(m.group(2))}" for m in parsed if m}
    gt_nodes = {f"{prefix}{k + 1}" for k in range(n_nodes)}

    def _order(names: set[str]) -> list[str]:
        return sorted(names, key=lambda s: int(s[len(prefix) :]))

    absent_from_data = _order(gt_nodes - data_nodes)
    absent_indices = {int(n[len(prefix) :]) - 1 for n in absent_from_data}
    lost_edges = [
        [f"{prefix}{i + 1}", f"{prefix}{j + 1}"]
        for i in range(n_nodes)
        for j in range(n_nodes)
        if off_diagonal[i, j] and (i in absent_indices or j in absent_indices)
    ]

    result.update(
        {
            "alignment": "by_name",
            "node_prefix": prefix,
            "n_ground_truth_nodes": n_nodes,
            "in_both": _order(data_nodes & gt_nodes),
            "absent_from_data": absent_from_data,
            "absent_from_ground_truth": _order(data_nodes - gt_nodes),
            "unrecoverable_edges": lost_edges,
            "n_unrecoverable_edges": len(lost_edges),
            "usable_edges": int(off_diagonal.sum()) - len(lost_edges),
        }
    )
    return result


def _is_acyclic(adjacency: np.ndarray) -> bool:
    """Cycle check by repeated removal of sink nodes (Kahn's algorithm).

    networkx is not a dependency of this project and adding one for a 15-line
    topological sort is not worth it.
    """
    remaining = set(range(adjacency.shape[0]))
    out_degree = {i: int(adjacency[i].sum()) for i in remaining}
    changed = True
    while changed and remaining:
        changed = False
        for node in list(remaining):
            if out_degree[node] == 0:
                remaining.discard(node)
                for parent in range(adjacency.shape[0]):
                    if parent in remaining and adjacency[parent, node]:
                        out_degree[parent] -= 1
                changed = True
    return not remaining


def analyse(
    frame: pd.DataFrame,
    cfg: StructureConfig,
    ground_truth: np.ndarray | None = None,
) -> dict[str, Any]:
    """Whole structure stage."""
    return {
        "shape": describe_shape(frame),
        "integrity": check_integrity(frame, cfg),
        "ground_truth_reconciliation": reconcile_with_ground_truth(
            list(frame.columns), ground_truth
        ),
    }
