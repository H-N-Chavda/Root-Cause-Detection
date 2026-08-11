"""Driver: runs the PC algorithm on every benchmark case and writes Results.txt.

Previously main.py ran a single hard-coded dataset, so the three blocks in
Results.txt could only have been produced by editing this file between runs and
pasting the output together (F14). Everything is now driven from CASES below,
and the report records the parameters, data shape, ground-truth provenance and
the git commit it was generated from.

Usage:
    python main.py              # write Results.txt
    python main.py --no-write   # print to stdout only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import (
    compute_metrics,
    format_ground_truth_info,
    format_metrics,
    load_dataset,
    load_ground_truth,
    pc_algorithm,
    summarize_graph,
)

BASE_DIR = Path(__file__).resolve().parent

# Headline parameters. Deliberately kept at the values used for the original
# Results.txt so the before/after comparison isolates the effect of the fixes
# rather than mixing in a parameter change.
ALPHA = 0.01
MAX_COND_SET_SIZE = 2

CASES: List[Tuple[str, str, str]] = [
    ("Ultra Processed Food", "DatasetUF.csv", "UFGroundTruth.txt"),
    (
        "Ultra Processed Food (with Internal Machine Dependencies)",
        "DatasetUF.csv",
        "UFIMDGroundTruth.txt",
    ),
    ("Tennessee Eastman", "datasetTE.csv", "TEGroundTruth.txt"),
]

ALPHA_SWEEP = [0.05, 0.01, 1e-3, 1e-5, 1e-10]
COND_SWEEP = [0, 1, 2, 3, None]
THIN_SWEEP = [1, 5, 20, 50]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _describe_dropped(dropped: Dict[str, List[str]]) -> str:
    parts = []
    for reason, columns in dropped.items():
        if columns:
            parts.append(f"{reason}={columns}")
    return ", ".join(parts) if parts else "none"


def run_case(
    title: str,
    dataset_path: str,
    ground_truth_path: str,
    alpha: float = ALPHA,
    max_cond_set_size: Optional[int] = MAX_COND_SET_SIZE,
) -> str:
    names, data, dropped = load_dataset(BASE_DIR / dataset_path)
    result = pc_algorithm(
        data, alpha=alpha, max_cond_set_size=max_cond_set_size, variable_names=names
    )

    cap = "converged" if max_cond_set_size is None else str(max_cond_set_size)
    lines = [
        "=" * 78,
        f"{title}",
        "=" * 78,
        f"Dataset          : {dataset_path}  ({data.shape[0]} samples, {data.shape[1]} variables)",
        f"Dropped columns  : {_describe_dropped(dropped)}",
        f"Parameters       : alpha = {alpha}, max_cond_set_size = {cap}",
        "",
    ]

    ground_truth, fmt, info = load_ground_truth(BASE_DIR / ground_truth_path, names)
    lines.append(format_ground_truth_info(fmt, info))
    lines.append("")
    lines.append(summarize_graph(result, variable_names=names))
    lines.append("")
    lines.append(format_metrics(compute_metrics(result, ground_truth, len(names))))
    return "\n".join(lines)


def _sensitivity_row(
    dataset_path: str,
    ground_truth_path: str,
    alpha: float,
    max_cond: Optional[int],
    thin: int,
) -> str:
    names, data, _ = load_dataset(BASE_DIR / dataset_path)
    if thin > 1:
        data = data[::thin]
    result = pc_algorithm(data, alpha=alpha, max_cond_set_size=max_cond)
    ground_truth, _, _ = load_ground_truth(BASE_DIR / ground_truth_path, names)
    m = compute_metrics(result, ground_truth, len(names))
    cap = "conv" if max_cond is None else str(max_cond)
    return (
        f"    alpha={alpha:<9} k={cap:<5} thin={thin:<4} n={data.shape[0]:<6}"
        f" edges={m['Predicted_skeleton_edges']:<4}"
        f" P={m['Skeleton_Precision']:.3f} R={m['Skeleton_Recall_TPR']:.3f}"
        f" F1={m['Skeleton_F1']:.3f} SHD={m['SHD']}"
    )


def run_sensitivity() -> str:
    """Sweeps alpha, conditioning-set size and time thinning.

    The thinning sweep is the honest way to present F10: both datasets are
    autocorrelated time series (lag-1 up to 0.96), so the Fisher-z test's i.i.d.
    assumption is violated and nominal p-values are anti-conservative. Rather
    than silently changing the test, the sensitivity is measured and reported.
    """
    lines = ["=" * 78, "Sensitivity analysis", "=" * 78]
    for title, dataset_path, ground_truth_path in CASES:
        lines.append(f"\n  {title}")
        lines.append("  -- alpha sweep (k = 2, no thinning) --")
        for alpha in ALPHA_SWEEP:
            lines.append(_sensitivity_row(dataset_path, ground_truth_path, alpha, 2, 1))
        lines.append("  -- conditioning-set size sweep (alpha = 0.01) --")
        for max_cond in COND_SWEEP:
            lines.append(
                _sensitivity_row(dataset_path, ground_truth_path, ALPHA, max_cond, 1)
            )
        lines.append("  -- time-thinning sweep (alpha = 0.01, k = 2) --")
        for thin in THIN_SWEEP:
            lines.append(
                _sensitivity_row(dataset_path, ground_truth_path, ALPHA, 2, thin)
            )
    return "\n".join(lines)


def run_all() -> str:
    header = [
        "PC Algorithm - Results",
        f"Generated from commit {_git_commit()} by PC_Algorithm/main.py",
        f"Headline parameters: alpha = {ALPHA}, max_cond_set_size = {MAX_COND_SET_SIZE}",
        "",
        "Skeleton metrics score undirected adjacency recovery; orientation metrics",
        "score arrowheads only over edges present in both skeletons; SHD is the",
        "standard edge-level Structural Hamming Distance. See insight-report/ for",
        "why these are reported separately.",
        "",
    ]
    blocks = [run_case(*case) for case in CASES]
    return "\n".join(header) + "\n\n" + "\n\n\n".join(blocks) + "\n\n\n" + run_sensitivity() + "\n"


if __name__ == "__main__":
    report = run_all()
    print(report)
    if "--no-write" not in sys.argv:
        (BASE_DIR / "Results.txt").write_text(report)
        print(f"\n[written to {BASE_DIR / 'Results.txt'}]")
