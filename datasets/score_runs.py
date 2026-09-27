"""Scores every graph in a run directory against that run's ground truth.

    python score_runs.py results/qtank/run-parcorr-tau20 [more runs ...]

Writes `scores.json` next to the graphs and prints a table. Scoring only -- it
never re-runs an algorithm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_bench.graph import load_graph
from causal_bench.scoring import score_edges

REPO = Path(__file__).resolve().parent.parent


def score_run(run_dir: Path) -> list[dict]:
    truth_path = run_dir / "ground_truth_edges.json"
    if not truth_path.exists():
        raise FileNotFoundError(
            f"{truth_path} not found; a run directory must carry the ground "
            "truth it was scored against, so scores cannot drift from the data."
        )
    truth = json.loads(truth_path.read_text())["edges"]

    rows = []
    for path in sorted((run_dir / "graphs").glob("*.json")):
        dataset, _, algorithm = path.stem.partition("__")
        score = score_edges(load_graph(path), truth)
        rows.append({"dataset": dataset, "algorithm": algorithm, **score.to_dict()})
    (run_dir / "scores.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    for run in args.runs:
        run = run if run.is_absolute() else REPO / run
        rows = score_run(run)
        print(f"\n=== {run.relative_to(REPO)} ===")
        head = (
            f"{'dataset':>8} {'algorithm':>11} | {'raw':>4} {'self':>4} {'uniq':>4} "
            f"| {'TP':>3} {'rev':>3} {'und':>3} {'FP':>3} {'FN':>3} "
            f"| {'prec':>5} {'rec':>5} {'F1':>5} {'SHD':>4} {'lag/e':>5}"
        )
        print(head)
        print("-" * len(head))
        for r in rows:
            print(
                f"{r['dataset']:>8} {r['algorithm']:>11} | "
                f"{r['n_raw_lagged_edges']:>4} {r['n_self_loops']:>4} "
                f"{r['n_predicted']:>4} | "
                f"{len(r['true_positives']):>3} {len(r['reversed_edges']):>3} "
                f"{len(r['undirected_hits']):>3} {len(r['false_positives']):>3} "
                f"{len(r['false_negatives']):>3} | "
                f"{r['precision']:5.2f} {r['recall']:5.2f} {r['f1']:5.2f} "
                f"{r['shd']:>4} {r['lag_multiplicity']:5.2f}"
            )
        print(f"wrote {(run / 'scores.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
