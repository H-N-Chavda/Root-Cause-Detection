"""Command line entry point: runs the PC algorithm over the configured cases.

Replaces the former `PC_Algorithm/main.py`. Same report, same numbers; what
changed is that the parameters come from `configs/default.yaml` instead of
module-level constants, paths come from `causal_bench.paths`, progress goes
through `logging` instead of `print`, and the report is written into a fresh
timestamped directory under `results/` rather than overwriting a tracked file.

    causal-bench                              # all configured cases
    causal-bench --dataset datasetTE.csv \\
                 --ground-truth TEGroundTruth.txt
    causal-bench --config configs/default.yaml --output-dir /tmp/out
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .algorithms.pc import compute_metrics, format_metrics, pc_algorithm, summarize_graph
from .config import CaseConfig, Config, ConfigError
from .io import format_ground_truth_info, load_dataset, load_ground_truth
from .utils.logging import configure_logging, get_logger

log = get_logger(__name__)

REPORT_FILENAME = "results.txt"
LOG_FILENAME = "run.log"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _describe_dropped(dropped: dict[str, list[str]]) -> str:
    parts = [f"{reason}={columns}" for reason, columns in dropped.items() if columns]
    return ", ".join(parts) if parts else "none"


def run_case(case: CaseConfig, config: Config) -> str:
    """Runs one benchmark case and renders its report block."""
    alpha = config.algorithm.alpha
    max_cond_set_size = config.algorithm.max_cond_set_size

    log.info("case %s: loading %s", case.name, case.dataset)
    names, data, dropped = load_dataset(case.dataset_path)
    log.info(
        "case %s: %d samples x %d variables (dropped: %s)",
        case.name,
        data.shape[0],
        data.shape[1],
        _describe_dropped(dropped),
    )

    log.info("case %s: running PC (alpha=%s, k=%s)", case.name, alpha, max_cond_set_size)
    result = pc_algorithm(
        data, alpha=alpha, max_cond_set_size=max_cond_set_size, variable_names=names
    )
    log.info(
        "case %s: %d skeleton edges, %d oriented arcs",
        case.name,
        len(result.skeleton_edges()),
        len(result.directed_edges),
    )

    cap = "converged" if max_cond_set_size is None else str(max_cond_set_size)
    lines = [
        "=" * 78,
        f"{case.name}",
        "=" * 78,
        f"Dataset          : {case.dataset}  "
        f"({data.shape[0]} samples, {data.shape[1]} variables)",
        f"Dropped columns  : {_describe_dropped(dropped)}",
        f"Parameters       : alpha = {alpha}, max_cond_set_size = {cap}",
        "",
    ]

    ground_truth, fmt, info = load_ground_truth(case.ground_truth_path, names)
    lines.append(format_ground_truth_info(fmt, info))
    lines.append("")
    lines.append(summarize_graph(result, variable_names=names))
    lines.append("")
    lines.append(format_metrics(compute_metrics(result, ground_truth, len(names))))
    return "\n".join(lines)


def _sensitivity_row(
    case_data: tuple[list[str], Any, Any],
    alpha: float,
    max_cond: int | None,
    thin: int,
) -> str:
    """One sweep row. `case_data` is loaded once per case by `run_sensitivity`;
    thinning only drops rows, so the variable names and the aligned ground truth
    are identical across every row and re-reading the CSV per row is waste."""
    names, data, ground_truth = case_data
    if thin > 1:
        data = data[::thin]
    result = pc_algorithm(data, alpha=alpha, max_cond_set_size=max_cond)
    m = compute_metrics(result, ground_truth, len(names))
    cap = "conv" if max_cond is None else str(max_cond)
    return (
        f"    alpha={alpha:<9} k={cap:<5} thin={thin:<4} n={data.shape[0]:<6}"
        f" edges={m['Predicted_skeleton_edges']:<4}"
        f" P={m['Skeleton_Precision']:.3f} R={m['Skeleton_Recall_TPR']:.3f}"
        f" F1={m['Skeleton_F1']:.3f} SHD={m['SHD']}"
    )


def run_sensitivity(config: Config) -> str:
    """Sweeps alpha, conditioning-set size and time thinning.

    The thinning sweep is the honest way to present F10: both datasets are
    autocorrelated time series (lag-1 up to 0.96), so the Fisher-z test's i.i.d.
    assumption is violated and nominal p-values are anti-conservative. Rather
    than silently changing the test, the sensitivity is measured and reported.
    """
    sweep = config.sensitivity
    alpha = config.algorithm.alpha
    lines = ["=" * 78, "Sensitivity analysis", "=" * 78]

    for case in config.cases:
        log.info("sensitivity: %s", case.name)
        names, data, _ = load_dataset(case.dataset_path)
        ground_truth, _, _ = load_ground_truth(case.ground_truth_path, names)
        case_data = (names, data, ground_truth)

        lines.append(f"\n  {case.name}")
        lines.append("  -- alpha sweep (k = 2, no thinning) --")
        for value in sweep.alpha_sweep:
            lines.append(_sensitivity_row(case_data, value, 2, 1))
        lines.append(f"  -- conditioning-set size sweep (alpha = {alpha}) --")
        for max_cond in sweep.cond_sweep:
            lines.append(_sensitivity_row(case_data, alpha, max_cond, 1))
        lines.append(f"  -- time-thinning sweep (alpha = {alpha}, k = 2) --")
        for thin in sweep.thin_sweep:
            lines.append(_sensitivity_row(case_data, alpha, 2, thin))
    return "\n".join(lines)


def run_all(config: Config) -> str:
    """Builds the full report for every configured case."""
    cap = (
        "converged"
        if config.algorithm.max_cond_set_size is None
        else str(config.algorithm.max_cond_set_size)
    )
    header = [
        "PC Algorithm - Results",
        f"Generated from commit {_git_commit()} by causal_bench.cli",
        f"Config: {config.source or '<defaults>'}",
        f"Headline parameters: alpha = {config.algorithm.alpha}, max_cond_set_size = {cap}",
        "",
        "Skeleton metrics score undirected adjacency recovery; orientation metrics",
        "score arrowheads only over edges present in both skeletons; SHD is the",
        "standard edge-level Structural Hamming Distance. See docs/legacy/ for",
        "why these are reported separately.",
        "",
    ]
    blocks = [run_case(case, config) for case in config.cases]
    report = "\n".join(header) + "\n\n" + "\n\n\n".join(blocks)
    if config.sensitivity.enabled:
        report += "\n\n\n" + run_sensitivity(config)
    return report + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-bench",
        description="Run the PC causal discovery algorithm over the benchmark cases.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"YAML config file (default: {paths.DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="NAME",
        help="run a single ad-hoc case on this dataset instead of the configured "
        "cases; a bare name resolves under data/raw/. Requires --ground-truth.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        metavar="NAME",
        help="ground-truth file for --dataset; a bare name resolves under "
        "data/ground_truth/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"parent directory for the timestamped run folder "
        f"(default: {paths.RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="skip the sensitivity sweeps regardless of the config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the report to stdout without creating a run directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosity of progress logging (default: INFO)",
    )
    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Folds the command-line overrides into the loaded config."""
    from dataclasses import replace

    if args.dataset or args.ground_truth:
        if not (args.dataset and args.ground_truth):
            raise ConfigError("--dataset and --ground-truth must be given together")
        case = CaseConfig(
            name=f"{Path(args.dataset).stem} vs {Path(args.ground_truth).stem}",
            dataset=args.dataset,
            ground_truth=args.ground_truth,
        )
        config = replace(config, cases=(case,))

    if args.no_sensitivity:
        config = replace(config, sensitivity=replace(config.sensitivity, enabled=False))

    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    run_dir: Path | None = None
    if not args.dry_run:
        run_dir = paths.new_run_dir(args.output_dir)
    configure_logging(
        level=getattr(logging, args.log_level),
        log_file=run_dir / LOG_FILENAME if run_dir else None,
    )

    try:
        config = _apply_overrides(Config.load(args.config), args)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return 2

    # The PC implementation is deterministic, but seed numpy anyway so that any
    # future sampling in this pipeline reproduces from the config alone.
    np.random.seed(config.run.seed)

    log.info("starting run")
    log.info("paths:\n%s", paths.describe())
    log.info("config:\n%s", config.describe())

    try:
        report = run_all(config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("run failed: %s", exc)
        return 1

    if run_dir is None:
        print(report)
        log.info("dry run: nothing written")
        return 0

    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")
    print(report)

    summary = [
        "",
        "Run summary",
        "-----------",
        f"  cases       : {len(config.cases)}",
        f"  sensitivity : {'on' if config.sensitivity.enabled else 'off'}",
        f"  commit      : {_git_commit()}",
        f"  report      : {report_path}",
        f"  log         : {run_dir / LOG_FILENAME}",
    ]
    print("\n".join(summary))
    log.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
