from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils import load_dataset, pc_algorithm, summarize_graph


def run_pc_algorithm(dataset_path: Optional[str] = None) -> str:
    base_dir = Path(__file__).resolve().parent
    if dataset_path is None:
        dataset_path = base_dir.parent / "dataset" / "CPCaD-Bench" / "UltraProcessed_Food" / "DatasetUF.csv"

    variable_names, data = load_dataset(dataset_path)
    result = pc_algorithm(data, alpha=0.01, max_cond_set_size=2, variable_names=variable_names)
    return summarize_graph(result, variable_names=variable_names)


if __name__ == "__main__":
    print(run_pc_algorithm("DatasetUF.csv"))
