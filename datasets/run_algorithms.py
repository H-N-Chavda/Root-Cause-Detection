"""Runs the four causal-discovery algorithms on the generated quadruple-tank data.

Execution only -- no scoring, no interpretation. Graphs and metadata are written
to results/qtank/<timestamp>/ for later comparison against docs/QTP_EXPECTATIONS.md.

    python run_algorithms.py [--algorithms pc,pcmci_plus,...] [--tau-max 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from causal_bench.algorithms import build
from causal_bench.graph import save_graph

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "generated"

# tau_max >= 20: the dominant time constant is ~90 s and Ts = 5 s, so a cause
# needs ~18 samples to show its effect. A smaller window cannot see the process.
DEFAULT_TAU_MAX = 20

# LSTE is nonparametric and pays for every (variable, variable, lag) triple, so
# it gets a shorter window than the others; at tau_max=20 with 7 variables it is
# 980 CMIknn tests and does not finish in reasonable time.
CONFIGS = {
    "pc": lambda tau: {"pc_alpha": 0.01},
    "pcmci_plus": lambda tau: {"tau_max": tau, "pc_alpha": 0.01},
    "var_lingam": lambda tau: {
        "tau_max": tau,
        "run_bootstrap": True,
        "bootstrap_samples": 100,
        "bootstrap_threshold": 0.9,
    },
    "lste": lambda tau: {
        "tau_max": min(tau, 8),
        "alpha": 0.05,
        "sig_samples": 100,
        "two_stage": True,
        "fdr_method": "fdr_bh",
    },
}


def lste_config(args) -> dict:
    """LSTE overrides from the CLI.

    `two_stage` only takes effect when `sig_samples_final` is also set -- see
    lste.py, where it is gated as `two_stage and sig_samples_final`. Passing
    two_stage=True alone silently runs a single full-shuffle pass.
    """
    return {
        "tau_max": args.lste_tau,
        "alpha": 0.05,
        "sig_samples": args.lste_sig,
        "sig_samples_final": args.lste_sig_final,
        "two_stage": args.lste_sig_final is not None,
        "fdr_method": "fdr_bh",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algorithms", default="pc,pcmci_plus,var_lingam,lste")
    parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    parser.add_argument("--datasets", default="P_minus,P_plus")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every Nth row, raising the effective sampling interval. "
        "stride 3 turns Ts=5s into Ts=15s, so a given tau_max spans 3x more "
        "real time for ~1/3 the samples -- cheaper AND wider for LSTE.",
    )
    parser.add_argument("--lste-sig", type=int, default=100)
    parser.add_argument("--lste-sig-final", type=int, default=None)
    parser.add_argument("--lste-tau", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("qtank")

    run_dir = args.out or (
        REPO / "results" / "qtank" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    truth = json.loads((DATA / "ground_truth_edges.json").read_text())
    var_names = truth["columns"]

    summary = []
    for ds in args.datasets.split(","):
        raw = np.genfromtxt(DATA / f"qtank_{ds}.csv", delimiter=",", names=True)
        data = np.column_stack([raw[c] for c in var_names])[:: args.stride]
        log.info(
            "%s: %d rows x %d vars (stride %d, Ts=%gs)",
            ds, *data.shape, args.stride, 5.0 * args.stride,
        )

        for name in args.algorithms.split(","):
            params = (
                lste_config(args) if name == "lste" else CONFIGS[name](args.tau_max)
            )
            log.info("running %s on %s (%s)", name, ds, params)
            started = time.time()
            # standardize=True: levels span ~1.4-13 cm against ~3 V inputs, and
            # LSTE's kNN estimator measures distance in raw units.
            graph = build(name, seed=0, **params).run(
                data, var_names, standardize=True
            )
            elapsed = time.time() - started

            stem = f"{ds}__{name}"
            save_graph(graph, run_dir / "graphs" / f"{stem}.json")
            row = {
                "dataset": ds,
                "algorithm": name,
                "params": params,
                "completed": graph.meta.get("completed"),
                "error": graph.meta.get("error"),
                "n_edges": graph.n_edges(),
                "runtime_seconds": round(elapsed, 1),
            }
            summary.append(row)
            log.info(
                "%s: %s, %d edges, %.1fs",
                stem,
                "ok" if row["completed"] else f"FAILED ({row['error']})",
                row["n_edges"],
                elapsed,
            )

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "ground_truth_edges.json").write_text(json.dumps(truth, indent=2))
    print(f"\nwrote {run_dir}")
    for r in summary:
        state = "ok" if r["completed"] else "FAILED"
        print(f"  {r['dataset']:>8} {r['algorithm']:>11}  {state:>6}  "
              f"{r['n_edges']:>3} edges  {r['runtime_seconds']:>7.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
