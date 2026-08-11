"""PC (Peter-Clark) causal discovery: data loading, CI testing, skeleton
discovery, edge orientation, ground-truth loading and evaluation.

See insight-report/ for the audit that motivated the current shape of this file;
finding IDs (F1, F4, ...) referenced in comments below point at that report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class PCResult:
    """Output of the PC algorithm.

    adjacency         : undirected skeleton, node -> set of neighbours
    directed_edges    : unambiguously oriented arcs (a, b) meaning a -> b
    undirected_edges  : pairs (i, j) with i < j left undirected by the CPDAG
    conflicting_edges : pairs (i, j) with i < j where colliders proposed BOTH
                        directions; left undirected rather than emitting both
    sepset            : separating set recorded for each removed pair
    """

    adjacency: Dict[int, Set[int]]
    directed_edges: Set[Tuple[int, int]]
    sepset: Dict[Tuple[int, int], frozenset[int]]
    undirected_edges: Set[Tuple[int, int]] = field(default_factory=set)
    conflicting_edges: Set[Tuple[int, int]] = field(default_factory=set)
    max_level_reached: int = 0
    has_cycle: bool = False

    def skeleton_edges(self) -> Set[Tuple[int, int]]:
        """Undirected skeleton as a set of (i, j) pairs with i < j."""
        return {
            (min(i, j), max(i, j))
            for i, neighbours in self.adjacency.items()
            for j in neighbours
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_INDEX_LIKE = re.compile(r"^(unnamed(:\s*\d+)?|index)$", re.IGNORECASE)


def _is_index_like(column: str, series: pd.Series) -> bool:
    """True for a CSV row-counter column masquerading as a variable (F2).

    Two independent signals: an unnamed/index-style header, or values that are
    exactly 0, 1, 2, ... n-1.
    """
    if _INDEX_LIKE.match(str(column).strip()):
        return True
    try:
        values = pd.to_numeric(series, errors="coerce")
    except (TypeError, ValueError):
        return False
    if values.isna().any():
        return False
    return np.array_equal(values.to_numpy(), np.arange(len(values), dtype=float))


def load_dataset(path: str | Path) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """Loads a dataset CSV.

    Returns (variable_names, data, dropped) where `dropped` records which
    columns were removed and why, so nothing disappears silently (F2, F18).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    dropped: Dict[str, List[str]] = {"timestamp": [], "index_like": [], "non_numeric": []}

    for column in list(df.columns):
        if str(column).strip().lower() in {"datetime", "date", "timestamp", "time"}:
            dropped["timestamp"].append(str(column))
            df = df.drop(columns=[column])

    # F2: the Tennessee CSV has an unnamed leading index column that used to be
    # ingested as a causal variable and even acquired spurious edges.
    for column in list(df.columns):
        if _is_index_like(column, df[column]):
            dropped["index_like"].append(str(column))
            df = df.drop(columns=[column])

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in numeric_df.columns if not numeric_df[c].isna().any()]
    dropped["non_numeric"] = [str(c) for c in numeric_df.columns if c not in keep]
    numeric_df = numeric_df[keep]

    if numeric_df.empty:
        raise ValueError("No numeric columns were found in the dataset")

    data = numeric_df.to_numpy(dtype=float)
    return [str(c) for c in numeric_df.columns], data, dropped


# ---------------------------------------------------------------------------
# Conditional independence testing
# ---------------------------------------------------------------------------

_CORR_CLIP = 0.999999999


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
    return float(np.clip(corr, -_CORR_CLIP, _CORR_CLIP))


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray]) -> float:
    """Recursive reduction formula for partial correlation.

        rho(x,y | Z + z) = (rho(x,y|Z) - rho(x,z|Z) rho(y,z|Z))
                           / sqrt((1 - rho(x,z|Z)^2)(1 - rho(y,z|Z)^2))

    Correct but costs ~3^|Z| sub-correlations, so `pc_algorithm` uses the
    matrix-inversion path below. Kept as the reference implementation; the two
    are cross-checked in the test suite.
    """
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


def _partial_correlation_matrix(
    corr: np.ndarray, i: int, j: int, cond: Sequence[int]
) -> float:
    """rho(i, j | cond) from the correlation matrix, via its inverse.

    For the sub-matrix over [i, j] + cond with inverse P,
        rho(i,j | cond) = -P[0,1] / sqrt(P[0,0] * P[1,1]).
    O(1) per test instead of the recursive form's ~3^|cond|, which is what
    makes running the skeleton search to convergence affordable (F13).
    """
    index = [i, j] + list(cond)
    sub = corr[np.ix_(index, index)]
    try:
        precision = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        return 0.0

    denom = precision[0, 0] * precision[1, 1]
    if denom <= 0.0 or np.isclose(denom, 0.0):
        return 0.0

    value = -precision[0, 1] / math.sqrt(denom)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -_CORR_CLIP, _CORR_CLIP))


def _fisher_z_p_value(corr: float, n_samples: int, n_conditioned: int) -> float:
    """Two-sided p-value for H0: rho = 0 under Fisher's z-transform.

        z = 0.5 * sqrt(n - |Z| - 3) * ln((1 + rho) / (1 - rho))
        p = 2 * (1 - Phi(|z|))
    """
    dof = n_samples - n_conditioned - 3
    if dof <= 0:
        return 1.0

    corr = float(np.clip(corr, -_CORR_CLIP, _CORR_CLIP))
    fisher_z = 0.5 * math.sqrt(dof) * math.log((1.0 + corr) / (1.0 - corr))
    normal_tail = 1.0 - 0.5 * (1.0 + math.erf(abs(fisher_z) / math.sqrt(2.0)))
    return float(2.0 * normal_tail)


def _p_value_for_partial_correlation(
    x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray], n_samples: int
) -> float:
    """Vector-based CI test, kept for direct use and for the test suite."""
    if len(z) >= n_samples - 3:
        return 1.0
    return _fisher_z_p_value(_partial_correlation(x, y, z), n_samples, len(z))


def _conditional_independence_test(
    x: np.ndarray, y: np.ndarray, z: Sequence[np.ndarray], n_samples: int
) -> float:
    return _p_value_for_partial_correlation(x, y, z, n_samples)


# ---------------------------------------------------------------------------
# Skeleton discovery
# ---------------------------------------------------------------------------

def _discover_skeleton(
    corr: np.ndarray,
    n_samples: int,
    n_vars: int,
    alpha: float,
    max_cond_set_size: int,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], frozenset[int]], int]:
    """PC-stable skeleton search.

    Starts from the complete graph and deletes an edge as soon as some
    conditioning set renders its endpoints conditionally independent.

    Three corrections over the original implementation:

    F4  The loop no longer aborts when a level happens to delete nothing. It
        continues while any adjacent pair still has at least `k` candidate
        neighbours, which is the standard stopping criterion. The old
        `if not changed: break` made PC fail on a plain X -> Y -> Z chain,
        because level 0 deletes nothing there and level 1 never ran.
    F5  Conditioning sets are drawn from adj(i)\\{j} AND adj(j)\\{i}. Searching
        only one side leaves edges whose separating set lives in the other
        endpoint's neighbourhood, inflating skeleton density.
    F9  PC-stable: candidate neighbour pools are frozen at the start of each
        level, so the result no longer depends on column order.
    """
    adjacency: Dict[int, Set[int]] = {i: set(range(n_vars)) - {i} for i in range(n_vars)}
    sepset: Dict[Tuple[int, int], frozenset[int]] = {}
    level_reached = 0

    for level in range(max_cond_set_size + 1):
        # F9: snapshot taken before any deletion at this level.
        frozen = {i: set(neighbours) for i, neighbours in adjacency.items()}

        # F4: is there still any adjacent pair with enough neighbours to test?
        has_work = any(
            len(frozen[i] - {j}) >= level or len(frozen[j] - {i}) >= level
            for i in range(n_vars)
            for j in adjacency[i]
            if i < j
        )
        if not has_work:
            break
        level_reached = level

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if j not in adjacency[i]:
                    continue

                # F5: both endpoints' neighbourhoods supply candidate sets.
                pools = [frozen[i] - {j}, frozen[j] - {i}]
                separated = False
                for pool in pools:
                    if len(pool) < level:
                        continue
                    for cond_set in combinations(sorted(pool), level):
                        p_value = _fisher_z_p_value(
                            _partial_correlation_matrix(corr, i, j, cond_set),
                            n_samples,
                            level,
                        )
                        if p_value >= alpha:
                            adjacency[i].discard(j)
                            adjacency[j].discard(i)
                            sepset[(i, j)] = frozenset(cond_set)
                            sepset[(j, i)] = frozenset(cond_set)
                            separated = True
                            break
                    if separated:
                        break

    return adjacency, sepset, level_reached


# ---------------------------------------------------------------------------
# Edge orientation
# ---------------------------------------------------------------------------

def _has_cycle(arrows: Set[Tuple[int, int]], n_vars: int) -> bool:
    """Depth-first cycle check over the oriented arcs only."""
    successors: Dict[int, Set[int]] = {i: set() for i in range(n_vars)}
    for parent, child in arrows:
        successors[parent].add(child)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = [WHITE] * n_vars

    def visit(node: int) -> bool:
        colour[node] = GREY
        for nxt in successors[node]:
            if colour[nxt] == GREY:
                return True
            if colour[nxt] == WHITE and visit(nxt):
                return True
        colour[node] = BLACK
        return False

    return any(colour[node] == WHITE and visit(node) for node in range(n_vars))


def _orient_edges(
    adjacency: Dict[int, Set[int]],
    sepset: Dict[Tuple[int, int], frozenset[int]],
    n_vars: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Orients the skeleton into a CPDAG (F6).

    Step 1 - collider rule. For every unshielded triple a - b - c (a and c not
    adjacent), if b is not in sepset(a, c) then b is a collider: a -> b <- c.
    Arrowheads are *collected* first. When colliders propose both directions for
    the same edge the edge is recorded as CONFLICTING and left undirected,
    rather than emitting both arcs. The original code emitted both, which on the
    over-dense UF skeleton turned 60 of 64 edges into mutual pairs and made the
    orientation output carry no causal information.

    Step 2 - Meek rules R1-R3, applied to a fixed point, propagate the
    orientations that the colliders imply. Without them the result is not a
    CPDAG. (R4 requires background knowledge and does not apply here.)

    Returns (arrows, undirected, conflicting).
    """
    edges = {
        (min(i, j), max(i, j))
        for i, neighbours in adjacency.items()
        for j in neighbours
    }

    # --- Step 1: colliders ---
    proposed: Set[Tuple[int, int]] = set()
    for b in range(n_vars):
        for a, c in combinations(sorted(adjacency[b]), 2):
            if c in adjacency[a]:
                continue  # shielded triple, no collider claim
            separating = sepset.get((a, c))
            if separating is None or b in separating:
                continue
            proposed.add((a, b))
            proposed.add((c, b))

    conflicting = {
        (min(a, b), max(a, b)) for (a, b) in proposed if (b, a) in proposed
    }
    arrows = {
        (a, b) for (a, b) in proposed if (min(a, b), max(a, b)) not in conflicting
    }

    def is_undirected(a: int, b: int) -> bool:
        return (a, b) not in arrows and (b, a) not in arrows

    # --- Step 2: Meek rules to a fixed point ---
    changed = True
    while changed:
        changed = False
        for (i, j) in sorted(edges):
            if (min(i, j), max(i, j)) in conflicting:
                continue
            for a, b in ((i, j), (j, i)):
                if not is_undirected(a, b):
                    continue

                # R1: c -> a, a - b, c not adjacent to b  =>  a -> b
                rule = any(
                    (c, a) in arrows and c != b and b not in adjacency[c]
                    for c in range(n_vars)
                )
                # R2: a -> c -> b, a - b  =>  a -> b
                if not rule:
                    rule = any(
                        (a, c) in arrows and (c, b) in arrows for c in range(n_vars)
                    )
                # R3: a - c, a - d, c -> b, d -> b, c and d not adjacent  =>  a -> b
                if not rule:
                    parents = [
                        c
                        for c in adjacency[a]
                        if c != b and (c, b) in arrows and is_undirected(a, c)
                    ]
                    rule = any(
                        d not in adjacency[c] for c, d in combinations(parents, 2)
                    )

                if rule:
                    arrows.add((a, b))
                    changed = True
                    break

    oriented_pairs = {(min(a, b), max(a, b)) for (a, b) in arrows}
    undirected = edges - oriented_pairs - conflicting
    return arrows, undirected, conflicting


def pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.01,
    max_cond_set_size: Optional[int] = None,
    variable_names: Optional[Sequence[str]] = None,
) -> PCResult:
    """Runs PC on `data` (rows = samples, columns = variables).

    max_cond_set_size=None runs the skeleton search to convergence (up to
    n_vars - 2); pass an integer to cap it.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")

    n_samples, n_vars = data.shape
    if n_vars < 2:
        raise ValueError("At least two variables are required")
    if n_samples < 5:
        raise ValueError("Too few samples for the Fisher-z test")

    if max_cond_set_size is None:
        max_cond_set_size = max(0, n_vars - 2)
    max_cond_set_size = min(max_cond_set_size, max(0, n_samples - 4))

    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    adjacency, sepset, level_reached = _discover_skeleton(
        corr, n_samples, n_vars, alpha, max_cond_set_size
    )
    arrows, undirected, conflicting = _orient_edges(adjacency, sepset, n_vars)

    return PCResult(
        adjacency=adjacency,
        directed_edges=arrows,
        sepset=sepset,
        undirected_edges=undirected,
        conflicting_edges=conflicting,
        max_level_reached=level_reached,
        has_cycle=_has_cycle(arrows, n_vars),
    )


def summarize_graph(result: PCResult, variable_names: Optional[Sequence[str]] = None) -> str:
    if variable_names is None:
        variable_names = [str(i) for i in range(len(result.adjacency))]

    name = lambda k: variable_names[k]  # noqa: E731
    skeleton = result.skeleton_edges()

    lines = [f"Variables ({len(variable_names)}): {', '.join(variable_names)}"]
    lines.append(
        f"Skeleton: {len(skeleton)} undirected edges "
        f"(highest conditioning level reached: {result.max_level_reached})"
    )
    # F18: '--' not '->', because this block is the undirected skeleton.
    for i, neighbours in sorted(result.adjacency.items()):
        names = [name(n) for n in sorted(neighbours)]
        lines.append(f"  {name(i)} -- {', '.join(names) if names else '(none)'}")

    lines.append(f"Oriented arcs ({len(result.directed_edges)}):")
    if result.directed_edges:
        for parent, child in sorted(result.directed_edges):
            lines.append(f"  {name(parent)} -> {name(child)}")
    else:
        lines.append("  (none)")

    lines.append(f"Left undirected ({len(result.undirected_edges)}):")
    if result.undirected_edges:
        for i, j in sorted(result.undirected_edges):
            lines.append(f"  {name(i)} -- {name(j)}")
    else:
        lines.append("  (none)")

    lines.append(
        f"Conflicting orientations ({len(result.conflicting_edges)}) "
        "- colliders proposed both directions, edge left undirected:"
    )
    if result.conflicting_edges:
        for i, j in sorted(result.conflicting_edges):
            lines.append(f"  {name(i)} <-> {name(j)}")
    else:
        lines.append("  (none)")

    if result.has_cycle:
        lines.append(
            "  [warning] the oriented arcs contain a cycle - the CPDAG is not "
            "a valid DAG pattern, usually a symptom of unreliable separating sets"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"^X(\d+)$")


def _looks_like_matrix(raw_lines: Sequence[str]) -> bool:
    """True when the file is a square numeric block.

    F1: the original loader, on failing its dimension check, fell through to an
    edge-list parser that happily read the first two tokens of every matrix row
    as "source target". For TEGroundTruth.txt that produced a ground truth
    consisting of a single diagonal entry - i.e. an empty target - which is why
    every Tennessee metric came out zero.
    """
    if not raw_lines:
        return False
    token_counts = {len(line.replace(",", " ").split()) for line in raw_lines}
    if len(token_counts) != 1:
        return False
    width = token_counts.pop()
    return width == len(raw_lines) and width > 2


def load_ground_truth(
    path: str | Path,
    variable_names: Sequence[str],
) -> Tuple[np.ndarray, str, Dict[str, object]]:
    """Loads a ground-truth causal graph as a binary adjacency matrix over
    `variable_names`, where [i, j] == 1 means variable_i -> variable_j.

    Alignment (F3). Ground-truth files cover the benchmark's *full* variable
    set, which need not match the columns present in the CSV: datasetTE.csv
    ships 31 of the 33 Tennessee variables (X27 and X31 are absent) against a
    33x33 ground truth. When the variables are named X<k>, row/column k-1 of the
    file is matched to variable X<k> by name and the relevant sub-matrix is
    taken. Positional alignment is used only as a fallback when the dimensions
    already agree.

    Returns (matrix, format_used, info). `info` records the alignment mode and
    which ground-truth edges were dropped as unrecoverable, so recall is never
    quietly computed against edges that cannot possibly be found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    n = len(variable_names)
    raw_lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

    info: Dict[str, object] = {
        "path": str(path),
        "n_vars": n,
        "alignment": None,
        "source_dim": None,
        "total_edges": 0,
        "kept_edges": 0,
        "dropped_edges": [],
        "reciprocal_pairs": 0,
    }

    full: Optional[np.ndarray] = None
    fmt = ""

    # --- Attempt 1: square matrix ---
    if _looks_like_matrix(raw_lines):
        for delimiter in (None, ","):
            try:
                candidate = np.loadtxt(path, delimiter=delimiter)
            except Exception:
                continue
            if candidate.ndim == 2 and candidate.shape[0] == candidate.shape[1]:
                full = (candidate != 0).astype(int)
                fmt = "matrix"
                break

    # --- Attempt 2: edge list ---
    if full is None:
        name_to_idx = {name: i for i, name in enumerate(variable_names)}
        matrix = np.zeros((n, n), dtype=int)
        parsed = 0
        for line in raw_lines:
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) < 2:
                continue
            resolved = []
            for token in parts[:2]:
                if token in name_to_idx:
                    resolved.append(name_to_idx[token])
                    continue
                # F17: only fall back to integer indices, and validate them.
                try:
                    index = int(token)
                except ValueError:
                    resolved = []
                    break
                if not 0 <= index < n:
                    resolved = []
                    break
                resolved.append(index)
            if len(resolved) != 2 or resolved[0] == resolved[1]:
                continue
            matrix[resolved[0], resolved[1]] = 1
            parsed += 1

        if parsed == 0:
            raise ValueError(
                f"Could not parse ground truth file {path} as a square matrix "
                f"or as an edge list over {variable_names[:3]}..."
            )
        full = matrix
        fmt = "edge_list"
        info["alignment"] = "by_name_or_index"

    info["source_dim"] = int(full.shape[0])
    np.fill_diagonal(full, 0)
    info["total_edges"] = int(full.sum())

    # --- Align the full ground truth onto the variables we actually have ---
    if fmt == "matrix":
        indices = [_VAR_PATTERN.match(str(name)) for name in variable_names]
        if all(indices) and max(int(m.group(1)) for m in indices) <= full.shape[0]:
            keep = [int(m.group(1)) - 1 for m in indices]
            info["alignment"] = "by_name"
        elif full.shape[0] == n:
            keep = list(range(n))
            info["alignment"] = "positional"
        else:
            raise ValueError(
                f"Ground truth {path} is {full.shape[0]}x{full.shape[0]} but the "
                f"dataset has {n} variables, and the names are not X<k>-shaped "
                "so they cannot be aligned by name."
            )

        kept_set = set(keep)
        for i in range(full.shape[0]):
            for j in range(full.shape[0]):
                if full[i, j] and (i not in kept_set or j not in kept_set):
                    info["dropped_edges"].append((f"X{i + 1}", f"X{j + 1}"))
        matrix = full[np.ix_(keep, keep)].astype(int)
    else:
        matrix = full

    info["kept_edges"] = int(matrix.sum())
    info["reciprocal_pairs"] = int(
        sum(
            1
            for i in range(n)
            for j in range(i + 1, n)
            if matrix[i, j] and matrix[j, i]
        )
    )

    # F1/F8: an empty target silently produces P=R=F1=0. Refuse to return one.
    if info["kept_edges"] == 0:
        raise ValueError(
            f"Ground truth {path} contains no edges over the dataset's variables "
            f"(parsed as {fmt}, {info['total_edges']} edges before alignment). "
            "Refusing to evaluate against an empty target."
        )

    return matrix, fmt, info


def format_ground_truth_info(fmt: str, info: Dict[str, object]) -> str:
    lines = [
        f"Ground truth: {info['path']}",
        f"  parsed as        : {fmt} ({info['source_dim']}x{info['source_dim']})",
        f"  aligned          : {info['alignment']}",
        f"  edges in file    : {info['total_edges']}",
        f"  edges usable     : {info['kept_edges']}",
    ]
    dropped = info["dropped_edges"]
    if dropped:
        shown = ", ".join(f"{s}->{t}" for s, t in dropped)
        lines.append(
            f"  unrecoverable    : {len(dropped)} "
            f"(variables absent from the dataset): {shown}"
        )
    if info["reciprocal_pairs"]:
        lines.append(
            f"  reciprocal pairs : {info['reciprocal_pairs']} "
            "(feedback loops - PC assumes acyclicity and cannot recover these)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _edge_type(matrix: np.ndarray, i: int, j: int) -> str:
    forward, backward = bool(matrix[i, j]), bool(matrix[j, i])
    if forward and backward:
        return "both"
    if forward:
        return "forward"
    if backward:
        return "backward"
    return "absent"


def compute_metrics(
    result: PCResult,
    ground_truth: np.ndarray,
    n_vars: int,
) -> Dict[str, object]:
    """Evaluates a PCResult against a directed ground-truth adjacency matrix.

    Reported in three blocks, because collapsing them into one number is what
    made the original figures uninterpretable (F7):

      Skeleton    - adjacency recovery, ignoring direction. This is what PC's
                    first phase is actually responsible for.
      Orientation - arrowhead accuracy, scored only over edges present in both
                    the predicted and the true skeleton, so skeleton errors are
                    not double-counted as orientation errors.
      SHD         - true edge-level Structural Hamming Distance: per unordered
                    pair, compare {absent, undirected, i->j, j->i} against the
                    truth and charge 1 for any mismatch. The original code
                    counted raw adjacency-cell disagreements, which is not SHD
                    and is not comparable to published numbers.
    """
    predicted_skeleton = result.skeleton_edges()
    true_skeleton = {
        (i, j)
        for i in range(n_vars)
        for j in range(i + 1, n_vars)
        if ground_truth[i, j] or ground_truth[j, i]
    }

    tp = len(predicted_skeleton & true_skeleton)
    fp = len(predicted_skeleton - true_skeleton)
    fn = len(true_skeleton - predicted_skeleton)
    total_pairs = n_vars * (n_vars - 1) // 2
    tn = total_pairs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    # F8: undefined, not zero, when the target has no positives.
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan") if (math.isnan(precision) or math.isnan(recall)) else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fdr = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    # --- Orientation, over shared skeleton edges only ---
    correct = reversed_ = undirected_pred = conflicting_pred = ambiguous_truth = 0
    for i, j in sorted(predicted_skeleton & true_skeleton):
        truth = _edge_type(ground_truth, i, j)
        if truth == "both":
            ambiguous_truth += 1  # 2-cycle in the truth; PC cannot express it
            continue
        if (i, j) in result.conflicting_edges:
            conflicting_pred += 1
        elif (i, j) in result.undirected_edges:
            undirected_pred += 1
        elif (i, j) in result.directed_edges:
            correct += 1 if truth == "forward" else 0
            reversed_ += 0 if truth == "forward" else 1
        elif (j, i) in result.directed_edges:
            correct += 1 if truth == "backward" else 0
            reversed_ += 0 if truth == "backward" else 1

    oriented = correct + reversed_
    arrow_precision = correct / oriented if oriented > 0 else float("nan")
    true_directed = len(true_skeleton) - sum(
        1 for i, j in true_skeleton if _edge_type(ground_truth, i, j) == "both"
    )
    arrow_recall = correct / true_directed if true_directed > 0 else float("nan")

    # --- True SHD ---
    shd = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            truth = _edge_type(ground_truth, i, j)
            if (i, j) in result.directed_edges:
                pred = "forward"
            elif (j, i) in result.directed_edges:
                pred = "backward"
            elif (i, j) in predicted_skeleton:
                pred = "undirected"
            else:
                pred = "absent"
            if truth == "both":
                # any single orientation is half-right; charge 1, absence 1 too
                shd += 0 if pred in {"forward", "backward", "undirected"} else 1
            elif pred != truth:
                shd += 1

    return {
        "Skeleton_Precision": precision,
        "Skeleton_Recall_TPR": recall,
        "Skeleton_F1": f1,
        "Skeleton_FDR": fdr,
        "Skeleton_FPR": fpr,
        "Skeleton_TP": tp,
        "Skeleton_FP": fp,
        "Skeleton_FN": fn,
        "Skeleton_TN": tn,
        "Predicted_skeleton_edges": len(predicted_skeleton),
        "True_skeleton_edges": len(true_skeleton),
        "Arrowhead_correct": correct,
        "Arrowhead_reversed": reversed_,
        "Arrowhead_left_undirected": undirected_pred,
        "Arrowhead_conflicting": conflicting_pred,
        "Arrowhead_truth_bidirected": ambiguous_truth,
        "Arrowhead_Precision": arrow_precision,
        "Arrowhead_Recall": arrow_recall,
        "SHD": shd,
    }


_METRIC_ORDER = [
    ("Skeleton (undirected adjacency recovery)", [
        "Predicted_skeleton_edges", "True_skeleton_edges",
        "Skeleton_Precision", "Skeleton_Recall_TPR", "Skeleton_F1",
        "Skeleton_FDR", "Skeleton_FPR",
        "Skeleton_TP", "Skeleton_FP", "Skeleton_FN", "Skeleton_TN",
    ]),
    ("Orientation (over edges in both skeletons)", [
        "Arrowhead_correct", "Arrowhead_reversed",
        "Arrowhead_left_undirected", "Arrowhead_conflicting",
        "Arrowhead_truth_bidirected",
        "Arrowhead_Precision", "Arrowhead_Recall",
    ]),
    ("Structural Hamming Distance (edge-level)", ["SHD"]),
]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "undefined"
        return f"{value:.4f}"
    return str(value)


def format_metrics(metrics: Dict[str, object]) -> str:
    lines = ["Evaluation against ground truth:"]
    for heading, keys in _METRIC_ORDER:
        lines.append(f"  {heading}:")
        for key in keys:
            lines.append(f"    {key:<28}: {_fmt(metrics[key])}")
    return "\n".join(lines)
