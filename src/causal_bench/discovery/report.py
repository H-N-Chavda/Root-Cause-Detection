"""Renders the run results as `scores.csv` and `RUN_REPORT.md`.

No computation beyond formatting; every number is looked up from the results
mapping the runner produced, so the CSV, the markdown and the JSON cannot
disagree.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Column order in scores.csv. Long format: one row per (run, target).
CSV_COLUMNS = [
    "algorithm",
    "prior_knowledge",
    "tau_max",
    "target",
    "shd",
    "true_positives",
    "false_positives",
    "false_negatives",
    "reversed_edges",
    "unoriented_edges",
    "n_predicted_edges",
    "n_true_edges",
    "undirected_weight",
    "n_edges_lagged",
    "runtime_seconds",
    "completed",
    "assumptions_met",
]

PRIMARY_TARGET = "gt_projected"


def write_scores_csv(results: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in _score_rows(results):
            writer.writerow(row)
    return path


def _score_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in results["runs"]:
        base = {
            "algorithm": run["algorithm"],
            "prior_knowledge": run["prior_knowledge"],
            "tau_max": run["tau_max"],
            "n_edges_lagged": run["n_edges"],
            "runtime_seconds": run["runtime_seconds"],
            "completed": run["completed"],
            "assumptions_met": run["assumptions_met"],
        }
        if not run["scores"]:
            rows.append({**base, "target": "n/a", "shd": None})
            continue
        for target, score in run["scores"].items():
            rows.append(
                {
                    **base,
                    "target": target,
                    "shd": score["shd"],
                    "true_positives": score["true_positives"],
                    "false_positives": score["false_positives"],
                    "false_negatives": score["false_negatives"],
                    "reversed_edges": score["reversed_edges"],
                    "unoriented_edges": score["unoriented_edges"],
                    "n_predicted_edges": score["n_predicted_edges"],
                    "n_true_edges": score["n_true_edges"],
                    "undirected_weight": score["undirected_weight"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        text = f"{value:.{places}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _cell(value: Any) -> str:
    # Evidence strings contain pipes (|r| = 0.95); an unescaped one splits the row.
    return str(value).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def build_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    targets = results["targets"]
    runs = results["runs"]
    parts: list[str] = []
    add = parts.append

    add("# Causal discovery run report\n")
    add(
        f"Generated {meta['timestamp']} from commit `{meta['git_commit']}` by "
        f"`causal-bench run`. Seed {meta['seed']}. "
        f"Dataset `{Path(meta['dataset']).name}` "
        f"({meta['n_rows']} rows x {meta['n_variables']} variables).\n"
    )

    # -- headline ---------------------------------------------------------
    add("## Results\n")
    add(
        f"SHD against **{PRIMARY_TARGET}**, the primary target: the "
        f"{targets['gt_projected_edges']}-edge latent projection onto the "
        f"{targets['n_observed']} observed variables. Lower is better; an "
        f"empty graph would score {targets['gt_projected_edges']}.\n"
    )
    add(_headline_table(runs))

    add(
        f"\nAn undirected edge scored against a directed one costs "
        f"{meta['undirected_weight']} rather than 1, because a CPDAG's "
        "undirected edge is an abstention, not an error.\n"
    )

    # -- targets ----------------------------------------------------------
    add("\n## Scoring targets\n")
    add(
        _table(
            ["Target", "Nodes", "Edges", "What it is"],
            [
                [
                    "gt_full",
                    targets["n_ground_truth_nodes"],
                    targets["gt_full_edges"],
                    "raw ground truth; reference only, nothing is scored against it",
                ],
                [
                    "gt_induced",
                    targets["n_observed"],
                    targets["gt_induced_edges"],
                    "induced subgraph; every edge touching a latent node dropped",
                ],
                [
                    "gt_projected",
                    targets["n_observed"],
                    targets["gt_projected_edges"],
                    "**primary**; latent projection, the graph the data can support",
                ],
            ],
        )
    )
    latents = targets["latent_nodes"]
    if latents:
        add(
            f"\nLatent nodes, derived from the data-versus-ground-truth "
            f"reconciliation rather than configured: **{', '.join(latents)}**. "
            f"Induction drops "
            + ", ".join(f"`{a} -> {b}`" for a, b in targets["edges_dropped_by_induction"])
            + "; the projection recovers "
            + (
                ", ".join(f"`{a} -> {b}`" for a, b in targets["edges_added_by_projection"])
                if targets["edges_added_by_projection"]
                else "nothing"
            )
            + " because a latent mediator leaves its parent connected to its child.\n"
        )

    cpdag = targets["gt_full_cpdag"]
    add(
        f"\n**CPDAG floor.** The full ground truth has {cpdag['v_structures']} "
        f"v-structures, so its CPDAG has {cpdag['directed']} directed and "
        f"{cpdag['undirected']} undirected edges. Those "
        f"{cpdag['undirected']} are undirectable from observational data by any "
        f"method, which is why the `_cpdag` targets exist: scoring a "
        f"CPDAG-returning algorithm against a DAG charges it for a limit it "
        f"cannot beat. On the projected target the CPDAG is "
        f"{targets['gt_projected_cpdag']['directed']} directed and "
        f"{targets['gt_projected_cpdag']['undirected']} undirected.\n"
    )

    # -- full table -------------------------------------------------------
    add("\n## Full score table\n")
    add(_full_table(runs))

    # -- assumptions ------------------------------------------------------
    add("\n## Assumptions and failures\n")
    add(_assumptions_section(runs))

    # -- per algorithm ----------------------------------------------------
    add("\n## Per-algorithm notes\n")
    add(_per_algorithm_notes(runs))

    # -- prior knowledge --------------------------------------------------
    add("\n## Prior knowledge\n")
    add(_prior_section(results))

    # -- parameters -------------------------------------------------------
    add("\n## Parameters and provenance\n")
    add(
        _table(
            ["Setting", "Value", "Source"],
            [
                ["seed", meta["seed"], "config discovery.seed"],
                [
                    "tau values",
                    ", ".join(str(t) for t in meta["tau_values"]),
                    meta["tau_source"],
                ],
                ["standardize", _fmt(meta["standardize"]), meta["standardize_source"]],
                [
                    "undirected weight",
                    meta["undirected_weight"],
                    "config discovery.undirected_weight",
                ],
                ["EDA report", meta["eda_report"] or "none found", "phase 2 handoff"],
                ["config", meta["config_path"], "--config"],
            ],
        )
    )
    add("\n**Library versions**\n")
    add(
        _table(
            ["Package", "Version"],
            [[k, v] for k, v in meta["library_versions"].items()],
        )
    )

    if meta.get("figures"):
        add("\n## Figures\n")
        for key, filename in meta["figures"].items():
            add(f"### {key.replace('_', ' ')}\n")
            add(f"![{key}](figures/{filename})\n")

    add("\n## Reproducing this run\n")
    add("```bash")
    add(meta["command"])
    add("```")
    add(f"\nTotal runtime {meta['total_seconds']:.0f}s.\n")
    return "\n".join(parts) + "\n"


def _headline_table(runs: list[dict[str, Any]]) -> str:
    """One row per algorithm: its best configuration against the primary target."""
    rows = []
    for run in sorted(runs, key=lambda r: _primary_shd(r) or float("inf")):
        score = (run["scores"] or {}).get(PRIMARY_TARGET)
        rows.append(
            [
                run["label"],
                _fmt(score["shd"]) if score else "did not run",
                score["true_positives"] if score else "-",
                score["false_positives"] if score else "-",
                score["false_negatives"] if score else "-",
                score["reversed_edges"] if score else "-",
                score["unoriented_edges"] if score else "-",
                run["n_summary_edges"],
                _fmt(run["runtime_seconds"], 1),
                _fmt(run["assumptions_met"]),
            ]
        )
    return _table(
        [
            "Run",
            "SHD",
            "TP",
            "FP",
            "FN",
            "Rev",
            "Unor",
            "Edges",
            "Seconds",
            "Assumptions met",
        ],
        rows,
    )


def _primary_shd(run: dict[str, Any]) -> float | None:
    score = (run["scores"] or {}).get(PRIMARY_TARGET)
    return score["shd"] if score else None


def _full_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        if not run["scores"]:
            rows.append([run["label"], "n/a", "did not run", "-", "-", "-", "-", "-"])
            continue
        for target, score in run["scores"].items():
            rows.append(
                [
                    run["label"],
                    target,
                    _fmt(score["shd"]),
                    score["true_positives"],
                    score["false_positives"],
                    score["false_negatives"],
                    score["reversed_edges"],
                    score["unoriented_edges"],
                ]
            )
    return _table(
        ["Run", "Target", "SHD", "TP", "FP", "FN", "Reversed", "Unoriented"], rows
    )


def _assumptions_section(runs: list[dict[str, Any]]) -> str:
    failed = [r for r in runs if not r["completed"]]
    violated = [r for r in runs if r["completed"] and r["assumptions_met"] is False]
    clean = [r for r in runs if r["completed"] and r["assumptions_met"] is True]

    lines = []
    if failed:
        lines.append("**Did not complete**\n")
        lines.append(
            _table(
                ["Run", "Error"], [[r["label"], r["error"] or "unknown"] for r in failed]
            )
        )
    else:
        lines.append("Every requested run completed.\n")

    if violated:
        lines.append(
            "\n**Completed but mis-specified.** These ran to completion under "
            "assumptions the data violates. The graph exists; whether it means "
            "anything is what the note says.\n"
        )
        rows = []
        for run in violated:
            for note in run["assumption_notes"]:
                rows.append([run["label"], note])
        lines.append(_table(["Run", "Violated assumption and the number"], rows))
    if clean:
        lines.append(
            "\nRuns with no measured assumption violation: "
            + ", ".join(f"`{r['label']}`" for r in clean)
            + ".\n"
        )
    return "\n".join(lines)


def _per_algorithm_notes(runs: list[dict[str, Any]]) -> str:
    lines = []
    for run in runs:
        meta = run["meta"]
        notes = []
        if "collider_rule" in meta:
            notes.append(f"collider rule: `{meta['collider_rule']}`")
        if "skeleton" in meta:
            notes.append(f"skeleton: {meta['skeleton']}")
        if "independence_test" in meta:
            notes.append(f"independence test: `{meta['independence_test']}`")
        if "estimator" in meta:
            notes.append(f"estimator: {meta['estimator']}")
        if "n_contemporaneous_edges" in meta:
            notes.append(
                f"{meta['n_lagged_edges']} lagged edges and "
                f"{meta['n_contemporaneous_edges']} contemporaneous"
            )
        if "n_tests" in meta:
            notes.append(f"{meta['n_tests']} independence tests")
        if meta.get("fdr_method") and meta["fdr_method"] != "none":
            notes.append(
                f"{meta['n_significant_uncorrected']} significant uncorrected, "
                f"{meta['n_significant_corrected']} after {meta['fdr_method']}"
            )
        if not notes:
            continue
        lines.append(f"\n**`{run['label']}`** -- " + "; ".join(notes) + ".")

        bootstrap = meta.get("lag0_bootstrap")
        if bootstrap and bootstrap.get("available"):
            lines.append(_bootstrap_block(bootstrap))
        errors = meta.get("error_independence")
        if errors and errors.get("available"):
            lines.append(
                f"\n  Error independence: "
                f"{errors['n_rejected_at_0.05']}/{errors['n_pairs']} residual "
                f"pairs reject independence at 0.05 (min p "
                f"{_fmt(errors['min_p_value'], 4)}). {errors['note']}."
            )
    return "\n".join(lines)


def _bootstrap_block(bootstrap: dict[str, Any]) -> str:
    return (
        f"\n  Lag-0 bootstrap over {bootstrap['n_samples']} resamples: "
        f"{bootstrap['n_distinct_lag0_edges_seen']} distinct contemporaneous "
        f"edges appeared at least once, of which "
        f"{bootstrap['n_edges_selected_at_least_half_the_time']} were selected "
        f"in at least half the resamples. Highest selection frequency "
        f"{_fmt(bootstrap['max_selection_frequency'])}, median "
        f"{_fmt(bootstrap['median_selection_frequency'])}, mean direction-flip "
        f"rate {_fmt(bootstrap['mean_direction_flip_rate'])}. "
        f"{bootstrap['interpretation']}"
    )


def _prior_section(results: dict[str, Any]) -> str:
    validation = results.get("prior_knowledge")
    if not validation:
        return "No prior knowledge was supplied.\n"

    lines = [
        f"Source: `{validation['source']}`. "
        f"{validation['n_edges_applicable']} of "
        f"{validation['n_edges_declared']} declared edges apply to this dataset."
    ]
    if validation["variables_not_in_data"]:
        lines.append(
            f" {validation['n_edges_dropped']} were dropped because they name "
            f"variables with no column in the data "
            f"({', '.join(validation['variables_not_in_data'])})."
        )
    lines.append("\n")
    lines.append(
        _table(
            ["Cause", "Effect", "Lag", "Kind"],
            [
                [e["cause"], e["effect"], e["lag"], e["kind"]]
                for e in validation["applicable_edges"]
            ],
        )
    )
    lines.append(
        "\n**This prior is an oracle, not domain knowledge.** Which pairs are "
        "candidates was measured (the Phase 2 near-unity correlations), but the "
        "direction of every forced edge was taken from the ground truth file. "
        "A with-prior-knowledge score therefore shows that the machinery works "
        "and gives a best case; it is not independent validation of anything.\n"
    )
    lines.append(
        "\n**The ground truth cannot confirm a controller loop.** It is acyclic "
        "with no reciprocal pairs, so the feedback structure a controller loop "
        "would produce is absent from the target by construction. The "
        "machinery is built and tested; this dataset cannot exercise it.\n"
    )
    return "".join(lines)


def write_report(results: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(results), encoding="utf-8")
    return path
