from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class PCResult:
    adjacency: Dict[int, Set[int]]
    directed_edges: Set[Tuple[int, int]]
    sepset: Dict[Tuple[int, int], frozenset[int]]


def load_dataset(path: str | Path) -> Tuple[List[str], np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "DateTime" in df.columns:
        df = df.drop(columns=["DateTime"])

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1)
    if numeric_df.empty:
        raise ValueError("No numeric columns were found in the dataset")

    data = numeric_df.to_numpy(dtype=float)
    return numeric_df.columns.tolist(), data


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors must have the same length")

    if x.size < 2:
        return 0.0

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if np.isclose(denom, 0.0):
        return 0.0

    corr = float(np.dot(x_centered, y_centered) / denom)
    return float(np.clip(corr, -0.999999999, 0.999999999))


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray]) -> float:
    if len(z) == 0:
        return _correlation(x, y)

    remaining = list(z)
    base_xy = _partial_correlation(x, y, remaining[:-1])
    base_xz = _partial_correlation(x, remaining[-1], remaining[:-1])
    base_yz = _partial_correlation(y, remaining[-1], remaining[:-1])

    denom = math.sqrt(max(0.0, (1.0 - base_xz * base_xz) * (1.0 - base_yz * base_yz)))
    if np.isclose(denom, 0.0):
        return 0.0

    return (base_xy - base_xz * base_yz) / denom


def _p_value_for_partial_correlation(x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray], n_samples: int) -> float:
    if len(z) >= n_samples - 2:
        return 1.0

    corr = _partial_correlation(x, y, z)
    if abs(corr) >= 1.0:
        return 0.0

    fisher_z = 0.5 * math.sqrt(n_samples - len(z) - 3) * math.log((1.0 + corr) / (1.0 - corr))
    normal_tail = 1.0 - 0.5 * (1.0 + math.erf(abs(fisher_z) / math.sqrt(2.0)))
    return float(2.0 * normal_tail)


def _conditional_independence_test(x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray], n_samples: int) -> float:
    return _p_value_for_partial_correlation(x, y, z, n_samples)


def pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.05,
    max_cond_set_size: Optional[int] = None,
    variable_names: Optional[Sequence[str]] = None,
) -> PCResult:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")

    n_samples, n_vars = data.shape
    if n_vars < 2:
        raise ValueError("At least two variables are required")

    if max_cond_set_size is None:
        max_cond_set_size = max(0, n_vars - 2)

    adjacency: Dict[int, Set[int]] = {i: set(range(n_vars)) - {i} for i in range(n_vars)}
    sepset: Dict[Tuple[int, int], frozenset[int]] = {}

    for conditioning_size in range(max_cond_set_size + 1):
        changed = False
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if j not in adjacency[i]:
                    continue

                neighbors_i = adjacency[i] - {j}
                if len(neighbors_i) < conditioning_size:
                    continue

                for cond_set in combinations(sorted(neighbors_i), conditioning_size):
                    x = data[:, i]
                    y = data[:, j]
                    z = [data[:, idx] for idx in cond_set]
                    p_value = _conditional_independence_test(x, y, z, n_samples)
                    if p_value >= alpha:
                        adjacency[i].remove(j)
                        adjacency[j].remove(i)
                        sepset[(i, j)] = frozenset(cond_set)
                        sepset[(j, i)] = frozenset(cond_set)
                        changed = True
                        break

        if not changed:
            break

    directed_edges: Set[Tuple[int, int]] = set()
    for a in range(n_vars):
        for b in range(n_vars):
            if a == b or b not in adjacency[a]:
                continue
            for c in adjacency[b]:
                if a == c or c in adjacency[a]:
                    continue
                if (a, c) in sepset and b not in sepset[(a, c)]:
                    directed_edges.add((a, b))
                    directed_edges.add((c, b))

    return PCResult(adjacency=adjacency, directed_edges=directed_edges, sepset=sepset)


def summarize_graph(result: PCResult, variable_names: Optional[Sequence[str]] = None) -> str:
    if variable_names is None:
        variable_names = [str(i) for i in range(len(result.adjacency))]

    lines = [f"Variables: {', '.join(variable_names)}"]
    lines.append("Undirected skeleton:")
    for i, neighbors in sorted(result.adjacency.items()):
        names = [variable_names[n] for n in sorted(neighbors)]
        lines.append(f"  {variable_names[i]} -> {', '.join(names) if names else '(none)'}")

    lines.append("Directed edges:")
    if result.directed_edges:
        for parent, child in sorted(result.directed_edges):
            lines.append(f"  {variable_names[parent]} -> {variable_names[child]}")
    else:
        lines.append("  (none detected)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(
    path: str | Path,
    variable_names: Sequence[str],
) -> Tuple[np.ndarray, str]:
    """
    Loads a ground-truth causal graph as an n x n binary adjacency matrix,
    where entry [i, j] == 1 means a directed edge variable_i -> variable_j.

    Tries two formats, since CIPCaD-Bench ground-truth files are not
    guaranteed to be one or the other without inspecting them directly:

      1. Matrix format: a whitespace/comma-separated n x n block of 0/1
         values, one row per line.
      2. Edge-list format: one edge per line, either as
         "source,target" / "source target" names, or as
         "source_index,target_index" integers.

    Returns (matrix, format_used) so the caller can sanity-check which
    parser actually fired.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    n = len(variable_names)
    name_to_idx = {name: i for i, name in enumerate(variable_names)}

    raw_lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

    # --- Attempt 1: matrix format ---
    try:
        candidate = np.loadtxt(path, delimiter=None)
        if candidate.ndim == 2 and candidate.shape[0] == candidate.shape[1] == n:
            return (candidate != 0).astype(int), "matrix"
    except Exception:
        pass
    try:
        candidate = np.loadtxt(path, delimiter=",")
        if candidate.ndim == 2 and candidate.shape[0] == candidate.shape[1] == n:
            return (candidate != 0).astype(int), "matrix"
    except Exception:
        pass

    # --- Attempt 2: edge-list format ---
    matrix = np.zeros((n, n), dtype=int)
    parsed_any = False
    for line in raw_lines:
        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) < 2:
            continue
        src_raw, tgt_raw = parts[0], parts[1]

        src = name_to_idx.get(src_raw)
        tgt = name_to_idx.get(tgt_raw)
        if src is None or tgt is None:
            try:
                src = int(src_raw)
                tgt = int(tgt_raw)
            except ValueError:
                continue
        if src is None or tgt is None or not (0 <= src < n and 0 <= tgt < n):
            continue

        matrix[src, tgt] = 1
        parsed_any = True

    if parsed_any:
        return matrix, "edge_list"

    raise ValueError(
        f"Could not parse ground truth file {path} as either a matrix "
        f"({n}x{n} expected) or an edge list against variable_names."
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _result_to_matrix(result: PCResult, n_vars: int) -> np.ndarray:
    """Converts PCResult.directed_edges into an n x n binary matrix."""
    matrix = np.zeros((n_vars, n_vars), dtype=int)
    for parent, child in result.directed_edges:
        matrix[parent, child] = 1
    return matrix


def compute_metrics(
    result: PCResult,
    ground_truth: np.ndarray,
    n_vars: int,
) -> Dict[str, float]:
    predicted = _result_to_matrix(result, n_vars)

    tp = fp = fn = tn = 0
    shd = 0

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            pred = predicted[i, j]
            true = ground_truth[i, j]

            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 1:
                fn += 1
            else:
                tn += 1

            if pred != true:
                shd += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a TPR
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "Precision": round(precision, 4),
        "Recall_TPR": round(recall, 4),
        "F1": round(f1, 4),
        "FDR": round(fdr, 4),
        "FPR": round(fpr, 4),
        "SHD": shd,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    order = ["Precision", "Recall_TPR", "F1", "FDR", "FPR", "SHD", "TP", "FP", "FN", "TN"]
    lines = ["Evaluation against ground truth:"]
    for key in order:
        lines.append(f"  {key:<12}: {metrics[key]}")
    return "\n".join(lines)
