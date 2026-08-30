"""Orchestrates the algorithm runs and the scoring.

Reads the Phase 2 EDA output for `tau_max` and the standardisation decision --
neither is chosen here -- runs each algorithm with and without prior knowledge,
collapses lagged graphs to summary graphs, and scores every result against the
targets from `targets.py`.

A failing algorithm is recorded and the run continues. A missing result is a
finding; a fabricated one is not.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..algorithms.registry import build, get_algorithm_class
from ..config import DiscoveryConfig
from ..graph.conversions import lagged_to_summary
from ..graph.io import load_ground_truth_matrix, save_graph
from ..graph.representation import CausalGraph
from ..io import load_dataset
from ..prior.knowledge import PriorKnowledge
from ..scoring.shd import align_to, structural_hamming_distance
from ..utils.logging import get_logger
from .targets import ScoringTargets, build_targets

log = get_logger(__name__)

GRAPHS_DIRNAME = "graphs"
FIGURES_DIRNAME = "figures"
SCORES_FILENAME = "scores.csv"
REPORT_FILENAME = "RUN_REPORT.md"
RESULTS_FILENAME = "run_results.json"


# ---------------------------------------------------------------------------
# Phase 2 handoff
# ---------------------------------------------------------------------------


def find_latest_eda_report(base: Path | None = None) -> Path | None:
    """The newest `eda_report.json` under `results/eda/`."""
    root = base or (paths.RESULTS_DIR / "eda")
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/eda_report.json"))
    return candidates[-1] if candidates else None


def load_eda_findings(path: Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    resolved = path or find_latest_eda_report()
    if resolved is None or not Path(resolved).exists():
        return None, None
    return json.loads(Path(resolved).read_text()), Path(resolved)


def tau_values_from_eda(
    findings: dict[str, Any] | None, cfg: DiscoveryConfig
) -> tuple[list[int], str]:
    """The lag orders to run, and where they came from.

    Never invents a lag order. An explicit `discovery.tau_max` in config wins;
    otherwise the Phase 2 VAR order selection decides, and because its criteria
    disagree the whole range is swept rather than one criterion being picked.
    """
    if cfg.tau_max is not None:
        return [cfg.tau_max], f"config discovery.tau_max = {cfg.tau_max}"
    if not findings:
        raise ValueError(
            "no tau_max in config and no EDA report to take one from. Run "
            "`causal-bench eda` first, or set discovery.tau_max explicitly; "
            "this phase will not choose a lag order on its own."
        )
    lag_order = findings["temporal"]["lag_order"]
    if "error" in lag_order:
        raise ValueError(f"EDA lag-order selection failed: {lag_order['error']}")
    selected = lag_order["selected"]
    low, high = int(lag_order["min_order"]), int(lag_order["max_order"])
    if cfg.tau_sweep and high > low:
        values = list(range(low, high + 1))
        reason = (
            f"EDA VAR order selection, swept over the full range {low}..{high} "
            f"because the criteria disagree ("
            + ", ".join(f"{k.upper()}={v}" for k, v in selected.items())
            + ")"
        )
    else:
        values = [high]
        reason = (
            f"EDA VAR order selection, max criterion = {high} ("
            + ", ".join(f"{k.upper()}={v}" for k, v in selected.items())
            + ")"
        )
    return values, reason


def standardize_from_eda(
    findings: dict[str, Any] | None, cfg: DiscoveryConfig
) -> tuple[bool, str]:
    if cfg.standardize is not None:
        return cfg.standardize, f"config discovery.standardize = {cfg.standardize}"
    if not findings:
        return False, "no EDA report; defaulted to False"
    advice = findings["suitability"]["preprocessing"]["standardization"]
    return bool(advice["required"]), f"EDA preprocessing advice: {advice['reason']}"


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


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _score_graph(
    graph: CausalGraph, targets: ScoringTargets, undirected_weight: float
) -> dict[str, Any]:
    """Collapses to a summary graph and scores it against every target."""
    summary = graph if graph.tau_max == 0 else lagged_to_summary(graph)
    aligned = align_to(summary, targets.var_names)
    return {
        name: structural_hamming_distance(
            aligned, target, undirected_weight=undirected_weight
        ).to_dict()
        for name, target in targets.by_name().items()
    }


def run_one(
    name: str,
    data: np.ndarray,
    var_names: list[str],
    cfg: DiscoveryConfig,
    targets: ScoringTargets,
    tau: int | None,
    prior: Any | None,
    prior_label: str,
    standardize: bool,
    eda_findings: dict[str, Any] | None,
    run_bootstrap: bool = True,
) -> tuple[dict[str, Any], CausalGraph]:
    """Builds, runs and scores one algorithm configuration."""
    params = cfg.params_for(name)
    cls = get_algorithm_class(name)
    if cls.handles_lags and tau is not None:
        params["tau_max"] = tau
    if not run_bootstrap:
        params["run_bootstrap"] = False
    algorithm = build(name, seed=cfg.seed, **params)

    graph = algorithm.run(
        data,
        var_names,
        prior_knowledge=prior,
        standardize=standardize,
        eda_findings=eda_findings,
    )

    record: dict[str, Any] = {
        "algorithm": name,
        "prior_knowledge": prior_label,
        "tau_max": tau if cls.handles_lags else 0,
        "handles_lags": cls.handles_lags,
        "n_edges": graph.n_edges(),
        "n_directed": len(graph.directed_edges()),
        "n_undirected": len(graph.undirected_edges()),
        "runtime_seconds": graph.meta.get("runtime_seconds"),
        "completed": graph.meta.get("completed"),
        "assumptions_met": graph.meta.get("assumptions_met"),
        "assumption_notes": graph.meta.get("assumption_notes", []),
        "error": graph.meta.get("error"),
        "meta": graph.meta,
    }
    if graph.meta.get("completed"):
        summary = graph if graph.tau_max == 0 else lagged_to_summary(graph)
        record["n_summary_edges"] = summary.n_edges()
        record["scores"] = _score_graph(graph, targets, cfg.undirected_weight)
    else:
        record["n_summary_edges"] = 0
        record["scores"] = None
    return record, graph


def run(
    dataset_path: Path,
    ground_truth_path: Path,
    cfg: DiscoveryConfig,
    out_dir: Path,
    algorithms: Sequence[str] | None = None,
    eda_report_path: Path | None = None,
    config_path: Path | None = None,
    command: str = "causal-bench run",
    make_figures: bool = True,
) -> dict[str, Any]:
    """Runs every requested algorithm and writes the outputs."""
    started = time.time()
    np.random.seed(cfg.seed)

    names, values, dropped = load_dataset(dataset_path)
    log.info(
        "loaded %s: %d rows x %d columns%s",
        dataset_path.name,
        values.shape[0],
        values.shape[1],
        f" (dropped {dict((k, v) for k, v in dropped.items() if v)})"
        if any(dropped.values())
        else "",
    )

    matrix = load_ground_truth_matrix(ground_truth_path)
    targets = build_targets(matrix, names)

    findings, eda_path = load_eda_findings(eda_report_path)
    if findings is None:
        log.warning("no EDA report found; assumption checks will be empty")
    taus, tau_reason = tau_values_from_eda(findings, cfg)
    standardize, standardize_reason = standardize_from_eda(findings, cfg)
    log.info("tau values %s (%s)", taus, tau_reason)
    log.info("standardize=%s (%s)", standardize, standardize_reason)

    prior_knowledge = None
    prior_validation = None
    if cfg.prior_knowledge and cfg.run_with_prior_knowledge:
        prior_path = Path(cfg.prior_knowledge)
        if not prior_path.is_absolute():
            prior_path = paths.REPO_ROOT / prior_path
        prior_knowledge = PriorKnowledge.from_yaml(prior_path)
        prior_validation = prior_knowledge.validate_against(names)
        log.info(
            "prior knowledge: %d/%d edges applicable",
            prior_validation["n_edges_applicable"],
            prior_validation["n_edges_declared"],
        )

    requested = list(algorithms or cfg.algorithms)
    graphs_dir = out_dir / GRAPHS_DIRNAME
    records: list[dict[str, Any]] = []
    saved_graphs: dict[str, CausalGraph] = {}

    for name in requested:
        cls = get_algorithm_class(name)
        if not cls.handles_lags:
            tau_list: list[int | None] = [None]
        elif cls.tau_sweep_is_informative:
            tau_list = list(taus)
        else:
            # The widest window only; see `tau_sweep_is_informative`.
            tau_list = [max(taus)]
        prior_variants: list[tuple[str, Any]] = [("none", None)]
        if prior_knowledge is not None and cls.supports_prior_knowledge:
            prior_variants.append(("controller_loops", prior_knowledge))

        for tau in tau_list:
            for prior_label, prior_object in prior_variants:
                converted = _convert_prior(prior_object, name, names, tau or 0)
                label = _run_label(name, tau, prior_label, cls.handles_lags)
                # The VAR-LiNGAM lag-0 bootstrap costs about as much as every
                # other run combined, and it answers one question -- is the
                # contemporaneous order identified -- that does not change with
                # tau or with prior knowledge. It therefore runs once, on the
                # baseline configuration: no prior knowledge, lowest tau, which
                # is also the VAR(1) the phase 2 residual analysis used.
                bootstrap_here = prior_label == "none" and tau == tau_list[0]
                log.info(
                    "running %s%s",
                    label,
                    " (with lag-0 bootstrap)"
                    if bootstrap_here and name == "var_lingam"
                    else "",
                )
                record, graph = run_one(
                    name=name,
                    data=values,
                    var_names=names,
                    cfg=cfg,
                    targets=targets,
                    tau=tau,
                    prior=converted,
                    prior_label=prior_label,
                    standardize=standardize,
                    eda_findings=findings,
                    run_bootstrap=bootstrap_here,
                )
                record["label"] = label
                records.append(record)
                save_graph(graph, graphs_dir / f"{label}.json")
                saved_graphs[label] = graph

    figures: dict[str, str] = {}
    if make_figures:
        from .figures import build_all

        figures = build_all(saved_graphs, targets, out_dir / FIGURES_DIRNAME)

    results: dict[str, Any] = {
        "meta": {
            "dataset": str(dataset_path),
            "ground_truth": str(ground_truth_path),
            "n_rows": int(values.shape[0]),
            "n_variables": int(values.shape[1]),
            "var_names": names,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "config_path": str(config_path) if config_path else "<defaults>",
            "eda_report": str(eda_path) if eda_path else None,
            "seed": cfg.seed,
            "command": command,
            "tau_values": taus,
            "tau_source": tau_reason,
            "standardize": standardize,
            "standardize_source": standardize_reason,
            "undirected_weight": cfg.undirected_weight,
            "algorithms_requested": requested,
            "total_seconds": round(time.time() - started, 2),
            "figures": figures,
            "library_versions": _library_versions(),
        },
        "targets": targets.provenance,
        "prior_knowledge": prior_validation,
        "runs": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    from .report import write_report, write_scores_csv

    write_scores_csv(results, out_dir / SCORES_FILENAME)
    write_report(results, out_dir / REPORT_FILENAME)
    (out_dir / RESULTS_FILENAME).write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False) + "\n", "utf-8"
    )
    log.info(
        "wrote %s, %s and %d graph(s) in %.1fs",
        out_dir / SCORES_FILENAME,
        out_dir / REPORT_FILENAME,
        len(records),
        results["meta"]["total_seconds"],
    )
    return results


def _run_label(name: str, tau: int | None, prior_label: str, lagged: bool) -> str:
    parts = [name]
    if lagged and tau is not None:
        parts.append(f"tau{tau}")
    if prior_label != "none":
        parts.append("pk")
    return "_".join(parts)


def _convert_prior(
    prior: PriorKnowledge | None, algorithm: str, var_names: list[str], tau_max: int
) -> Any | None:
    """One YAML source, converted into whichever encoding the library wants."""
    if prior is None:
        return None
    if algorithm in ("pc", "pcmci_plus", "lste"):
        # PC runs at tau_max = 0; the others need the full lag range.
        return prior.to_tigramite(var_names, tau_max=max(tau_max, 0))
    if algorithm == "var_lingam":
        return prior.to_lingam(var_names)
    raise ValueError(f"no prior-knowledge encoding for {algorithm!r}")


def _library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for package in ("tigramite", "lingam", "networkx", "numpy", "scipy", "statsmodels"):
        try:
            out[package] = md.version(package)
        except Exception:
            out[package] = "unknown"
    return out


def _json_safe(value: Any) -> Any:
    import math

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.str_, Path)):
        return str(value)
    return value
