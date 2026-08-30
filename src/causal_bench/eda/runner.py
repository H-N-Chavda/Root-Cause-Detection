"""Orchestrates the EDA stages and writes the two outputs.

Stage order matters in one place only: `suitability` reads the findings the
other five stages produced, so it runs last. Everything before it is
independent.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import paths
from ..config import EdaConfig
from ..io import load_dataset
from ..utils.logging import get_logger
from . import (
    conditioning,
    distribution,
    figures,
    linearity,
    report,
    structure,
    suitability,
    temporal,
)

log = get_logger(__name__)

JSON_FILENAME = "eda_report.json"
MARKDOWN_FILENAME = "EDA_REPORT.md"
FIGURES_DIRNAME = "figures"

#: Header names treated as a time index rather than as a variable. Same set the
#: dataset loader drops, so the EDA sees exactly the columns the algorithms do.
_TIME_COLUMNS = {"datetime", "date", "timestamp", "time"}


def load_frame(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Builds the analysis dataframe from a dataset file.

    The values come from `causal_bench.io.load_dataset`, so the EDA characterises
    exactly the columns the algorithms will be handed -- index counters, constant
    columns and non-numeric columns already removed, and the removals reported.
    The raw file is then re-read only to recover a timestamp column for use as
    the index, which the loader discards and which the sampling-interval and
    regime checks want.
    """
    names, values, dropped = load_dataset(dataset_path)
    frame = pd.DataFrame(values, columns=names)

    raw_header = pd.read_csv(dataset_path, nrows=0)
    time_columns = [
        c for c in raw_header.columns if str(c).strip().lower() in _TIME_COLUMNS
    ]
    index_source = None
    if time_columns:
        column = time_columns[0]
        stamps = pd.read_csv(dataset_path, usecols=[column]).iloc[:, 0]
        parsed = pd.to_datetime(stamps, errors="coerce")
        if parsed.notna().all() and len(parsed) == len(frame):
            frame.index = pd.DatetimeIndex(parsed)
            frame.index.name = str(column)
            index_source = str(column)

    return frame, {
        "path": str(dataset_path),
        "name": dataset_path.name,
        "dropped_by_loader": {k: v for k, v in dropped.items() if v},
        "index_source": index_source,
    }


def load_raw_ground_truth(path: Path) -> np.ndarray | None:
    """Reads a ground-truth adjacency matrix in its original, unaligned shape.

    Deliberately not `causal_bench.io.load_ground_truth`: that function aligns
    the matrix onto the dataset's columns and drops the rest, which is correct
    for scoring but destroys the very discrepancy this stage exists to report.
    """
    for delimiter in (None, ",", "\t"):
        try:
            matrix = np.loadtxt(path, delimiter=delimiter)
        except Exception:
            continue
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            return matrix
    log.warning("could not read %s as a square matrix; skipping reconciliation", path)
    return None


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


def run(
    dataset_path: Path,
    cfg: EdaConfig,
    out_dir: Path,
    ground_truth_path: Path | None = None,
    reference_path: Path | None = None,
    config_path: Path | None = None,
    command: str = "causal-bench eda",
) -> dict[str, Any]:
    """Runs every stage and writes `eda_report.json`, `EDA_REPORT.md`, `figures/`.

    Returns the findings mapping, so a caller (or a test) can assert on the
    numbers without re-reading the JSON.
    """
    # One seed for the whole run. numpy's global state covers anything that
    # reaches for it implicitly; the estimators that take an explicit
    # random_state are handed `cfg.seed` at their call sites.
    np.random.seed(cfg.seed)

    started = time.time()
    frame, dataset_meta = load_frame(dataset_path)
    log.info(
        "loaded %s: %d rows x %d columns%s",
        dataset_path.name,
        frame.shape[0],
        frame.shape[1],
        f" (dropped {dataset_meta['dropped_by_loader']})"
        if dataset_meta["dropped_by_loader"]
        else "",
    )

    ground_truth = load_raw_ground_truth(ground_truth_path) if ground_truth_path else None

    findings: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def stage(name: str, function: Any) -> None:
        log.info("stage: %s", name)
        mark = time.time()
        findings[name] = function()
        timings[name] = round(time.time() - mark, 3)

    stage(
        "structure",
        lambda: structure.analyse(frame, cfg.structure, ground_truth),
    )
    discrete = {
        entry["column"] for entry in findings["structure"]["integrity"]["discrete_columns"]
    }
    stage(
        "distribution",
        lambda: distribution.analyse(
            frame, cfg.distribution, cfg.nongaussianity, cfg.significance, discrete
        ),
    )
    stage("linearity", lambda: linearity.analyse(frame, cfg.linearity, cfg.seed))
    stage("temporal", lambda: temporal.analyse(frame, cfg.temporal, cfg.significance))
    stage(
        "conditioning",
        lambda: conditioning.analyse(frame, cfg.conditioning, cfg.temporal),
    )
    # Reads the five stages above; must run last.
    stage("suitability", lambda: suitability.analyse(findings, cfg))

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / FIGURES_DIRNAME
    log.info("stage: figures")
    produced = figures.build_all(frame, findings, figure_dir, cfg.figures)

    reference_result = None
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        reference_result = report.compare_to_reference(findings, reference)
        findings["verification"] = reference_result
        log.info(
            "reference check: %d/%d match",
            reference_result["n_match"],
            reference_result["n_checks"],
        )
        for check in reference_result["checks"]:
            if not check["match"]:
                log.warning(
                    "reference mismatch: %s expected %s, computed %s",
                    check["path"],
                    check["expected"],
                    check["actual"],
                )

    meta = {
        "dataset_name": dataset_meta["name"],
        "dataset_path": dataset_meta["path"],
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "index_source": dataset_meta["index_source"],
        "dropped_by_loader": dataset_meta["dropped_by_loader"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_path": str(config_path) if config_path else "<defaults>",
        "seed": cfg.seed,
        "command": command,
        "stage_seconds": timings,
        "total_seconds": round(time.time() - started, 3),
        "thresholds": cfg.thresholds(),
        "figures": produced,
    }
    findings["meta"] = meta

    report.write_json(findings, out_dir / JSON_FILENAME)
    markdown = report.build_markdown(findings, meta, produced, reference_result)
    report.write_markdown(markdown, out_dir / MARKDOWN_FILENAME)

    log.info(
        "wrote %s and %s (%d figures) in %.1fs",
        out_dir / JSON_FILENAME,
        out_dir / MARKDOWN_FILENAME,
        len(produced),
        meta["total_seconds"],
    )
    return findings
