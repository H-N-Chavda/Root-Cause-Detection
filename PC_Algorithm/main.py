from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


from utils import (
    load_dataset, pc_algorithm, summarize_graph, load_ground_truth, compute_metrics, format_metrics
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pc_algorithm(
    dataset_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
) -> str:

    if dataset_path is None:
        dataset_path = "DatasetUF.csv"

    if ground_truth_path is None:
        ground_truth_path = "UFGroundTruth.txt"

    variable_names, data = load_dataset(dataset_path)
    result = pc_algorithm(data, alpha=0.01, max_cond_set_size=2, variable_names=variable_names)

    output_lines = [summarize_graph(result, variable_names=variable_names)]

    try:
        ground_truth_matrix, fmt_used = load_ground_truth(ground_truth_path, variable_names)
        output_lines.append(f"\n(Ground truth parsed as: {fmt_used}, from {ground_truth_path})")
        metrics = compute_metrics(result, ground_truth_matrix, n_vars=len(variable_names))
        output_lines.append(format_metrics(metrics))
    except (FileNotFoundError, ValueError) as e:
        output_lines.append(f"\n[Ground-truth evaluation skipped: {e}]")

    return "\n".join(output_lines)


if __name__ == "__main__":
    print(run_pc_algorithm("DatasetUF.csv", "UFGroundTruth.txt"))
