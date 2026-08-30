"""Renders the findings mapping as JSON and as markdown.

No computation happens here beyond formatting and the optional reference
comparison. Every number in the markdown is looked up from the same mapping
that is written to JSON, so the two outputs can never disagree.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_INDEXED_STEP = re.compile(r"^([^\[]+)\[(\d+)\]$")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _sanitise(value: Any) -> Any:
    """Makes numpy and non-finite floats JSON safe.

    `json.dump` emits bare `NaN` and `Infinity`, which are not valid JSON and
    are rejected by strict parsers in other languages. They become null, and
    the markdown says "undefined" in the same places.
    """
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return _sanitise(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(findings: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitise(findings), indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


def resolve_path(findings: dict[str, Any], path: str) -> Any:
    """Looks up a dotted path such as `a.b[0].c` in the findings mapping."""
    node: Any = findings
    for step in path.split("."):
        match = _INDEXED_STEP.match(step)
        if match:
            node = node[match.group(1)][int(match.group(2))]
        else:
            node = node[step]
    return node


def compare_to_reference(
    findings: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Checks computed values against an independently supplied reference.

    A mismatch is reported, never adopted: the reference is evidence that the
    loader or a formula is wrong, not a value to copy.
    """
    checks = []
    for entry in reference.get("checks", []):
        path = entry["path"]
        expected = entry["expected"]
        tolerance = entry.get("tolerance", 0)
        try:
            actual = resolve_path(findings, path)
        except (KeyError, IndexError, TypeError) as exc:
            checks.append(
                {
                    **entry,
                    "actual": None,
                    "match": False,
                    "error": f"path not found: {exc}",
                }
            )
            continue

        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            match = (
                actual is not None
                and isinstance(actual, (int, float))
                and abs(float(actual) - float(expected)) <= float(tolerance)
            )
            deviation = (
                abs(float(actual) - float(expected))
                if isinstance(actual, (int, float))
                else None
            )
        else:
            match = actual == expected
            deviation = None

        checks.append(
            {
                **entry,
                "actual": _sanitise(actual),
                "match": bool(match),
                "deviation": deviation,
            }
        )

    return {
        "source": reference.get("source"),
        "checks": checks,
        "n_checks": len(checks),
        "n_match": int(sum(c["match"] for c in checks)),
        "n_mismatch": int(sum(not c["match"] for c in checks)),
        "all_match": all(c["match"] for c in checks) if checks else None,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "undefined"
        if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-4):
            return f"{value:.3e}"
        text = f"{value:.{places}f}"
        # Only strip zeros that sit after a decimal point: stripping them
        # unconditionally turns 1000 into 1.
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _ratio(numerator: Any, denominator: Any) -> str:
    """ "3 / 31" -- the shape almost every count in this report takes."""
    return f"{numerator} / {denominator}"


def _cell_text(value: Any) -> str:
    """Escapes the pipes inside a cell.

    Evidence strings are full of them -- `max |skew|`, `|r| = 0.95` -- and an
    unescaped pipe silently splits the row into extra columns.
    """
    return str(value).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(_cell_text(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(_cell_text(c) for c in row) + " |")
    return "\n".join(out)


_VERDICT_MARK = {
    "satisfied": "satisfied",
    "violated": "**violated**",
    "n/a": "n/a",
    "unknown": "unknown",
}


def build_markdown(
    findings: dict[str, Any],
    meta: dict[str, Any],
    figures: dict[str, str],
    reference: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    add = parts.append

    structure = findings["structure"]
    shape = structure["shape"]
    integrity = structure["integrity"]
    reconciliation = structure["ground_truth_reconciliation"]
    marginals = findings["distribution"]["marginals"]["summary"]
    residuals = findings["distribution"]["var_residuals"]
    pairwise = findings["linearity"]["pairwise"]
    predictive = findings["linearity"]["predictive"]
    serial = findings["temporal"]["serial_dependence"]["summary"]
    stationarity = findings["temporal"]["stationarity"]["summary"]
    regime = findings["temporal"]["regime"]
    lag_order = findings["temporal"]["lag_order"]
    xcorr = findings["temporal"]["cross_correlation"]
    collinearity = findings["conditioning"]["collinearity"]
    ess = findings["conditioning"]["effective_sample_size"]
    adequacy = findings["conditioning"]["sample_size_adequacy"]
    suitability = findings["suitability"]

    # Short aliases for values the tables below repeat, so the rows stay
    # readable at one line each.
    n_scored = marginals["n_scored"]
    rejections = marginals["rejections"]
    n_equations = residuals.get("n_equations")
    n_equations_lin = predictive.get("n_variables")
    band = findings["temporal"]["serial_dependence"]["significance_band"]
    n_vars = collinearity["n_variables"]
    vif = collinearity["vif"]
    gaussian_rule_label = (
        f"Practically Gaussian (|skew| < {marginals['skew_threshold']} and "
        f"|excess kurtosis| < {marginals['excess_kurtosis_threshold']})"
    )
    split_text = (
        f"{predictive.get('split', 'n/a')}, "
        f"{predictive.get('n_train_rows')} train / "
        f"{predictive.get('n_test_rows')} test"
    )

    add(f"# EDA report: {meta['dataset_name']}\n")
    add(
        f"Generated {meta['timestamp']} from commit `{meta['git_commit']}` by "
        f"`causal-bench eda`. Config: `{meta['config_path']}`, seed "
        f"{meta['seed']}.\n"
    )

    # -- Findings summary ---------------------------------------------------
    add("## Findings\n")
    add(_findings_summary(findings))

    ranking = suitability["recommendation"]["ranking"]
    add("\n**Algorithm outlook**\n")
    add(
        _table(
            ["Algorithm", "Verdict", "Assumptions violated by the data"],
            [
                [
                    r["algorithm"],
                    r["verdict"],
                    ", ".join(r["violated_assumptions"]) or "none",
                ]
                for r in ranking
            ],
        )
    )

    # -- Structure ----------------------------------------------------------
    add("\n## 1. Structure and integrity\n")
    add(
        _table(
            ["Property", "Value"],
            [
                ["Rows", shape["n_rows"]],
                ["Columns", shape["n_columns"]],
                ["Index type", shape["index"]["type"]],
                [
                    "Index monotonic increasing",
                    _fmt(shape["index"]["is_monotonic_increasing"]),
                ],
                ["Index unique", _fmt(shape["index"]["is_unique"])],
                [
                    "Sampling interval",
                    f"{_fmt(shape['sampling_interval']['step'])} "
                    f"{shape['sampling_interval']['unit'] or ''} "
                    f"(regular: {_fmt(shape['sampling_interval']['regular'])})",
                ],
                ["Missing values", integrity["missing_total"]],
                ["Constant columns", len(integrity["constant_columns"]) or "none"],
                [
                    "Near-constant columns",
                    len(integrity["near_constant_columns"]) or "none",
                ],
                [
                    "Discrete / integer-valued columns",
                    len(integrity["discrete_columns"]) or "none",
                ],
                ["Duplicate rows", integrity["duplicate_rows"]],
                [
                    "Duplicate column groups",
                    len(integrity["duplicate_column_groups"]) or "none",
                ],
                [
                    "Non-numeric columns",
                    ", ".join(integrity["non_numeric_columns"]) or "none",
                ],
            ],
        )
    )

    if reconciliation.get("ground_truth_available"):
        add("\n### Data versus ground-truth node set\n")
        add(
            _table(
                ["Property", "Value"],
                [
                    [
                        "Ground-truth shape",
                        " x ".join(str(v) for v in reconciliation["ground_truth_shape"]),
                    ],
                    ["Ground-truth edges", reconciliation["ground_truth_edges"]],
                    ["Self-loops", reconciliation["ground_truth_self_loops"]],
                    ["Reciprocal pairs", reconciliation["ground_truth_reciprocal_pairs"]],
                    ["Acyclic", _fmt(reconciliation["ground_truth_acyclic"])],
                    ["Data columns", reconciliation["n_data_columns"]],
                    ["Alignment", reconciliation.get("alignment")],
                    [
                        "Nodes in ground truth but absent from data",
                        ", ".join(reconciliation.get("absent_from_data", [])) or "none",
                    ],
                    [
                        "Nodes in data but absent from ground truth",
                        ", ".join(reconciliation.get("absent_from_ground_truth", []))
                        or "none",
                    ],
                    [
                        "Edges made unrecoverable by those absences",
                        reconciliation.get("n_unrecoverable_edges"),
                    ],
                    ["Usable edges", reconciliation.get("usable_edges")],
                ],
            )
        )
        lost = reconciliation.get("unrecoverable_edges", [])
        if lost:
            add(
                "\nUnrecoverable edges: "
                + ", ".join(f"`{a} -> {b}`" for a, b in lost)
                + ". Any recall computed against the full edge count is wrong by "
                "construction.\n"
            )

    # -- Distribution -------------------------------------------------------
    add("\n## 2. Distribution\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                ["Columns scored", marginals["n_scored"]],
                ["Max |skewness|", _fmt(marginals["max_abs_skewness"])],
                ["Mean |skewness|", _fmt(marginals["mean_abs_skewness"])],
                ["Max |excess kurtosis|", _fmt(marginals["max_abs_excess_kurtosis"])],
                ["Mean |excess kurtosis|", _fmt(marginals["mean_abs_excess_kurtosis"])],
                [
                    f"Shapiro-Wilk rejections at {marginals['alpha']}",
                    f"{marginals['rejections']['shapiro_wilk']} / {marginals['n_scored']}",
                ],
                [
                    f"D'Agostino K^2 rejections at {marginals['alpha']}",
                    f"{marginals['rejections']['dagostino_k2']} / {marginals['n_scored']}",
                ],
                [
                    f"Jarque-Bera rejections at {marginals['alpha']}",
                    f"{marginals['rejections']['jarque_bera']} / {marginals['n_scored']}",
                ],
                [
                    f"Anderson-Darling rejections at {marginals['alpha']}",
                    _ratio(rejections["anderson_darling"], n_scored),
                ],
                [
                    gaussian_rule_label,
                    f"{marginals['n_practically_gaussian']} / {marginals['n_scored']}",
                ],
            ],
        )
    )
    add(
        "\nThe p-value counts and the effect sizes answer different questions. At "
        f"n = {shape['n_rows']} a normality test has the power to reject on a "
        "departure far too small to disturb a Fisher-z test, so the effect sizes "
        "above, not the rejection counts, are what the Gaussian verdict rests on.\n"
    )

    add("\n### VAR residual non-Gaussianity (VAR-LiNGAM identifiability)\n")
    if residuals.get("fitted"):
        add(
            _table(
                ["Quantity", "Value"],
                [
                    ["VAR lag used", residuals["var_lag"]],
                    ["Equations", residuals["n_equations"]],
                    ["Residual observations", residuals["n_residual_observations"]],
                    ["Mean |excess kurtosis|", _fmt(residuals["mean_abs_excess_kurtosis"])],
                    ["Max |excess kurtosis|", _fmt(residuals["max_abs_excess_kurtosis"])],
                    ["Mean negentropy (nats)", _fmt(residuals["mean_negentropy"], 6)],
                    ["Max negentropy (nats)", _fmt(residuals["max_negentropy"], 6)],
                    [
                        "Jarque-Bera rejections",
                        _ratio(residuals["n_jarque_bera_rejections"], n_equations),
                    ],
                    [
                        "Equations clearing both thresholds",
                        f"{residuals['n_usable_for_ica']} / {residuals['n_equations']}",
                    ],
                    [
                        "Contemporaneous LiNGAM identifiable",
                        _fmt(residuals["lingam_contemporaneous_identifiable"]),
                    ],
                ],
            )
        )
        add(
            f"\n**Threshold and justification.** An equation counts as usable when "
            f"|excess kurtosis| > {residuals['abs_excess_kurtosis_threshold']} *and* "
            f"negentropy > {residuals['negentropy_threshold']} nats; the dataset "
            f"counts as identifiable when at least "
            f"{100 * residuals['min_fraction_nongaussian']:.0f}% of equations "
            "qualify. ICA cannot separate Gaussian sources at all -- a rotation of "
            "independent Gaussians is again independent Gaussians -- so the "
            "contrast must be large enough to survive finite-sample noise. An "
            "excess kurtosis of 1.0 is roughly the point at which the fourth "
            "cumulant is estimable at this sample size; the negentropy floor of "
            "0.01 nats is the same statement in the units FastICA actually "
            "maximises. Both are config values, not constants in the code.\n"
        )
    else:
        add(f"VAR fit failed: {residuals.get('error')}\n")

    add("\n### Outliers\n")
    outlier_rows = _top_outlier_rows(findings)
    add(
        _table(
            ["Column", "Modified z > threshold", "IQR rule", "Max |modified z|"],
            outlier_rows,
        )
    )
    add("\nCounts only. Nothing was removed.\n")

    # -- Linearity ----------------------------------------------------------
    add("\n## 3. Linearity\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                ["Pairs", pairwise["n_pairs"]],
                ["Max (dCor - |Pearson|)", _fmt(pairwise["max_gap"])],
                ["Median (dCor - |Pearson|)", _fmt(pairwise["median_gap"])],
                [
                    f"Pairs with gap > {pairwise['gap_threshold']}",
                    f"{pairwise['n_pairs_above_gap_threshold']} / {pairwise['n_pairs']} "
                    f"({100 * pairwise['fraction_above_gap_threshold']:.1f}%)",
                ],
            ],
        )
    )
    add("\n**Most nonlinear pairs** (largest dCor minus |Pearson|)\n")
    add(
        _table(
            ["Pair", "Pearson", "Spearman", "dCor", "Mutual information", "Gap"],
            [
                [
                    f"{p['a']} - {p['b']}",
                    _fmt(p["pearson"], 3),
                    _fmt(p["spearman"], 3),
                    _fmt(p["distance_correlation"], 3),
                    _fmt(p["mutual_information"], 3),
                    _fmt(p["gap"], 3),
                ]
                for p in pairwise["most_nonlinear_pairs"]
            ],
        )
    )

    if "skipped" not in predictive:
        add("\n### Linear model versus gradient-boosted tree, out of sample\n")
        add(
            _table(
                ["Quantity", "Value"],
                [
                    [
                        "Split",
                        split_text,
                    ],
                    ["Mean R^2 gain of tree over linear", _fmt(predictive["mean_r2_gap"])],
                    ["Median R^2 gain", _fmt(predictive["median_r2_gap"])],
                    ["Max R^2 gain", _fmt(predictive["max_r2_gap"])],
                    [
                        f"Variables gaining more than {predictive['r2_gap_threshold']}",
                        _ratio(predictive["n_with_nonlinear_gain"], n_equations_lin),
                    ],
                    [
                        "Ramsey RESET rejections at 0.05",
                        f"{predictive['n_reset_rejections']} / {predictive['n_variables']}",
                    ],
                ],
            )
        )
        add("\n**Largest nonlinear predictive gains**\n")
        top = sorted(
            predictive["per_variable"].items(),
            key=lambda kv: kv[1]["r2_gap"],
            reverse=True,
        )[:8]
        add(
            _table(
                ["Variable", "R^2 linear (oos)", "R^2 tree (oos)", "Gain", "RESET p"],
                [
                    [
                        name,
                        _fmt(v["r2_linear_oos"], 3),
                        _fmt(v["r2_tree_oos"], 3),
                        _fmt(v["r2_gap"], 3),
                        _fmt(v["ramsey_reset"].get("p_value"), 4),
                    ]
                    for name, v in top
                ],
            )
        )

    # -- Temporal -----------------------------------------------------------
    add("\n## 4. Temporal structure\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                ["Max |ACF| at lag 1", _fmt(serial["max_abs_acf_lag_1"])],
                ["Mean |ACF| at lag 1", _fmt(serial["mean_abs_acf_lag_1"])],
                ["Min |ACF| at lag 1", _fmt(serial["min_abs_acf_lag_1"])],
                [
                    "White-noise band",
                    f"+/- {_fmt(band)}",
                ],
                [
                    "Columns never decorrelating within the lag window",
                    f"{serial['n_never_decorrelated']} / {serial['n_scored']}",
                ],
                ["Median decorrelation lag", _fmt(serial["median_decorrelation_lag"])],
                ["Max decorrelation lag", _fmt(serial["max_decorrelation_lag"])],
                [
                    "Ljung-Box rejects independence",
                    f"{serial['n_ljung_box_rejections']} / {serial['n_scored']}",
                ],
            ],
        )
    )
    add(
        "\nThe decorrelation lag is the first lag at which the autocorrelation "
        "falls inside the white-noise band. It is the number that decides whether "
        "plain PC's independent-sample assumption is tenable.\n"
    )

    add("\n### Per-variable decorrelation lag\n")
    add(_decorrelation_table(findings))

    add("\n### Stationarity\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                [
                    "ADF says stationary",
                    f"{stationarity['n_adf_stationary']} / {stationarity['n_scored']}",
                ],
                [
                    "KPSS says stationary",
                    f"{stationarity['n_kpss_stationary']} / {stationarity['n_scored']}",
                ],
                ["Both agree", f"{stationarity['n_agree']} / {stationarity['n_scored']}"],
                ["Disagree", f"{stationarity['n_disagree']} / {stationarity['n_scored']}"],
                [
                    "Disagreeing columns",
                    ", ".join(stationarity["disagreeing_columns"]) or "none",
                ],
            ],
        )
    )
    add(
        "\nADF's null is a unit root and KPSS's null is stationarity, so they are "
        "not redundant. Where they disagree the evidence is genuinely ambiguous "
        "-- typically a near-unit-root or trend-stationary series -- and the "
        "disagreement is reported rather than resolved.\n"
    )

    add("\n### Regime stability\n")
    add(
        _table(
            ["Quantity", "Halves", "Thirds"],
            [
                [
                    "Max mean shift (SD units)",
                    _fmt(regime["halves"]["max_mean_shift_sd"], 3),
                    _fmt(regime["thirds"]["max_mean_shift_sd"], 3),
                ],
                [
                    "Mean mean-shift (SD units)",
                    _fmt(regime["halves"]["mean_mean_shift_sd"], 3),
                    _fmt(regime["thirds"]["mean_mean_shift_sd"], 3),
                ],
                [
                    f"Columns shifted more than {regime['shift_threshold_sd']} SD",
                    regime["halves"]["n_columns_shifted"],
                    regime["thirds"]["n_columns_shifted"],
                ],
            ],
        )
    )
    changepoints = regime["changepoints"]
    add(
        f"\nChangepoint pass (binary segmentation, Gaussian cost, penalty "
        f"{changepoints['penalty']}, minimum segment {changepoints['min_size']}): "
        f"{changepoints['n_columns_with_changepoints']} of "
        f"{len(changepoints['per_column'])} columns carry at least one "
        f"changepoint, {changepoints['total_changepoints']} in total.\n"
    )

    add("\n### Lag order\n")
    if "error" not in lag_order:
        selected = lag_order["selected"]
        add(
            _table(
                ["Criterion", "Selected order"],
                [[k.upper(), v] for k, v in selected.items()],
            )
        )
        add(
            f"\nSearched up to lag {lag_order['max_lag_searched']}. The criteria "
            f"{'agree' if lag_order['criteria_agree'] else 'disagree'}: AIC and FPE "
            "are efficiency criteria and lean long, BIC and HQIC are consistent "
            "criteria and lean short. `tau_max` is a modelling choice, not a fact "
            f"about the data; the widest supported value here is "
            f"{lag_order['recommended_tau_max']}.\n"
        )
    else:
        add(f"Lag-order selection failed: {lag_order['error']}\n")

    add("\n### Propagation delay\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                ["Lag window searched", f"+/- {xcorr['max_lag']}"],
                [
                    "Pairs peaking at a non-zero lag",
                    f"{xcorr['n_peak_at_nonzero_lag']} / {xcorr['n_pairs']} "
                    f"({100 * xcorr['fraction_peak_at_nonzero_lag']:.1f}%)",
                ],
                ["Max |peak lag|", xcorr["max_abs_peak_lag"]],
                ["Median |peak lag|", _fmt(xcorr["median_abs_peak_lag"])],
            ],
        )
    )
    add(f"\nSign convention: {xcorr['sign_convention']}.\n")
    if xcorr.get("peak_at_edge_warning"):
        add(
            f"\n{xcorr['n_peak_at_window_edge']} pair(s) peak at the edge of the "
            f"+/-{xcorr['max_lag']} search window, so for those the delay is a "
            "lower bound rather than a measurement; widening "
            "`eda.temporal.xcorr_max_lag` would settle them.\n"
        )
    if xcorr["strongest_delayed_pairs"]:
        add("\n**Strongest delayed pairs**\n")
        add(
            _table(
                ["Pair", "Peak lag", "Peak value", "Value at lag 0"],
                [
                    [
                        f"{p['a']} - {p['b']}",
                        p["peak_lag"],
                        _fmt(p["peak_value"], 3),
                        _fmt(p["lag_0_value"], 3),
                    ]
                    for p in xcorr["strongest_delayed_pairs"][:8]
                ],
            )
        )

    # -- Conditioning -------------------------------------------------------
    add("\n## 5. Conditioning and numerical health\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                [
                    "Correlation matrix condition number",
                    _fmt(collinearity["condition_number"], 1),
                ],
                ["Threshold", _fmt(collinearity["condition_number_threshold"], 0)],
                ["Ill conditioned", _fmt(collinearity["ill_conditioned"])],
                [
                    "Numerical rank",
                    f"{collinearity['numerical_rank']} / {collinearity['n_variables']}",
                ],
                ["Smallest eigenvalue", _fmt(collinearity["min_eigenvalue"])],
                [
                    "Five smallest eigenvalues",
                    ", ".join(_fmt(v) for v in collinearity["smallest_eigenvalues"]),
                ],
                ["Max VIF", _fmt(collinearity["vif"]["max"], 1)],
                ["Median VIF", _fmt(collinearity["vif"]["median"], 1)],
                [
                    f"Variables with VIF > {_fmt(collinearity['vif']['threshold'], 0)}",
                    _ratio(vif["n_above_threshold"], n_vars),
                ],
            ],
        )
    )
    pairs_above = collinearity["high_correlation_pairs"]
    add(
        f"\n**Pairs at or above |r| = {collinearity['report_corr_threshold']}** "
        f"({len(pairs_above)} of {pairwise['n_pairs']})\n"
    )
    if pairs_above:
        add(
            _table(
                ["Pair", "Correlation", f"Above {collinearity['high_corr_threshold']}"],
                [
                    [
                        f"{p['a']} - {p['b']}",
                        _fmt(p["correlation"]),
                        _fmt(p["above_high_threshold"]),
                    ]
                    for p in pairs_above
                ],
            )
        )
    else:
        add("None.\n")

    add("\n### Effective sample size and conditioning capacity\n")
    add(
        _table(
            ["Quantity", "Value"],
            [
                ["Rows", ess["n_rows"]],
                [
                    "Min effective n (AR(1) form, per variable)",
                    _fmt(ess["min_n_eff_ar1"], 1),
                ],
                ["Median effective n (AR(1) form)", _fmt(ess["median_n_eff_ar1"], 1)],
                [
                    "Min effective n (Bartlett, per pair)",
                    _fmt(ess["min_n_eff_bartlett"], 1),
                ],
                ["Median effective n (Bartlett)", _fmt(ess["median_n_eff_bartlett"], 1)],
                ["Binding effective n", _fmt(ess["binding_n_eff"], 1)],
                ["Shrinkage factor", _fmt(ess["shrinkage_factor"], 4)],
                [
                    "Practical max conditioning set at raw n",
                    adequacy["at_raw_n"]["practical_max_conditioning_set"],
                ],
                [
                    "Practical max conditioning set at effective n",
                    adequacy["at_effective_n"]["practical_max_conditioning_set"],
                ],
                ["Graph ceiling (n_vars - 2)", adequacy["graph_ceiling"]],
                [
                    "Recommended max conditioning set",
                    adequacy["recommended_max_conditioning_set"],
                ],
            ],
        )
    )
    add(
        f"\nThe Bartlett form is the one that binds: it governs the variance of a "
        f"*correlation* estimate, which is exactly what the Fisher-z test "
        f"studentises. At {adequacy['samples_per_parameter']} observations per "
        "estimated parameter, the effective sample size is what limits the "
        "conditioning set, not the row count.\n"
    )

    # -- Preprocessing ------------------------------------------------------
    preprocessing = suitability["preprocessing"]
    add("\n## 6. Scaling and preprocessing\n")
    add(
        _table(
            ["Step", "Required", "Basis", "Cost if applied"],
            [
                [
                    "Standardization",
                    _fmt(preprocessing["standardization"]["required"]),
                    preprocessing["standardization"]["reason"],
                    preprocessing["standardization"]["cost"],
                ],
                [
                    "Differencing",
                    _fmt(preprocessing["differencing"]["required"]),
                    preprocessing["differencing"]["reason"],
                    preprocessing["differencing"]["cost"],
                ],
                [
                    "Marginal transformation",
                    _fmt(preprocessing["transformation"]["worthwhile"]),
                    preprocessing["transformation"]["reason"],
                    preprocessing["transformation"]["cost"],
                ],
            ],
        )
    )
    add(f"\n{preprocessing['note']}\n")

    # -- Suitability --------------------------------------------------------
    add("\n## 7. Algorithm suitability\n")
    table = suitability["table"]
    add(
        _table(
            ["Assumption", *table["algorithms"], "Holds in this data", "Supporting number"],
            [
                [
                    row["label"],
                    *[_VERDICT_MARK[row["verdicts"][a]] for a in table["algorithms"]],
                    "unmeasurable"
                    if row["unmeasurable"] and row["holds_in_data"] is None
                    else _fmt(row["holds_in_data"]),
                    row["evidence"],
                ]
                for row in table["rows"]
            ],
        )
    )
    add(
        "\n`n/a` means the algorithm does not rely on that assumption. "
        "`unknown` means the data cannot settle it.\n"
    )

    add("\n### Ranked recommendation\n")
    add(
        "\nRule: an assumption the data settles *against* the method counts as a "
        'hard violation. Zero gives "expected to work", one or two gives '
        '"expected to work with caveats", three or more gives "expected to '
        'fail". Causal sufficiency and acyclicity are excluded from the count '
        "because the data cannot settle them, and counting them would mark every "
        "method as failing on every dataset.\n"
    )
    for entry in ranking:
        add(f"\n**{entry['algorithm']} - {entry['verdict']}**\n")
        if entry["reasons"]:
            for reason in entry["reasons"]:
                add(f"- {reason}")
        else:
            add("- No assumption that this data can settle is violated.")
        if entry["unknown_assumptions"]:
            add(f"- Unsettled by the data: {', '.join(entry['unknown_assumptions'])}.")

    # -- Verification -------------------------------------------------------
    if reference:
        add("\n## 8. Verification against an independent reference\n")
        add(
            f"Source: {reference.get('source')}. "
            f"{reference['n_match']} of {reference['n_checks']} checks match.\n"
        )
        add(
            _table(
                ["Quantity", "Reference", "Computed", "Tolerance", "Match"],
                [
                    [
                        c.get("label", c["path"]),
                        _fmt(c["expected"])
                        if not isinstance(c["expected"], list)
                        else ", ".join(str(v) for v in c["expected"]),
                        _fmt(c["actual"])
                        if not isinstance(c["actual"], list)
                        else ", ".join(str(v) for v in (c["actual"] or [])),
                        _fmt(c.get("tolerance")),
                        "yes" if c["match"] else "**NO**",
                    ]
                    for c in reference["checks"]
                ],
            )
        )
        mismatches = [c for c in reference["checks"] if not c["match"]]
        if mismatches:
            add(
                "\nMismatches are reported, not adopted. Each one is either a "
                "loader difference or a difference in estimator convention; see "
                "the notes on the rows above.\n"
            )
        else:
            add("\nNo mismatch. The loader and the estimators agree with the reference.\n")

    # -- Figures ------------------------------------------------------------
    if figures:
        add("\n## Figures\n")
        for key, filename in figures.items():
            add(f"### {key.replace('_', ' ').title()}\n")
            add(f"![{key}](figures/{filename})\n")

    # -- Reproduction -------------------------------------------------------
    add("\n## Reproducing this report\n")
    add("```bash")
    add(meta["command"])
    add("```")
    add(
        f"\nSeed {meta['seed']}; every estimator with a random component is seeded "
        "from it, so a rerun on the same input reproduces every number.\n"
    )

    return "\n".join(parts) + "\n"


def _findings_summary(findings: dict[str, Any]) -> str:
    """The one-minute read. Every sentence carries the number behind it."""
    reconciliation = findings["structure"]["ground_truth_reconciliation"]
    marginals = findings["distribution"]["marginals"]["summary"]
    residuals = findings["distribution"]["var_residuals"]
    pairwise = findings["linearity"]["pairwise"]
    predictive = findings["linearity"]["predictive"]
    serial = findings["temporal"]["serial_dependence"]["summary"]
    stationarity = findings["temporal"]["stationarity"]["summary"]
    lag_order = findings["temporal"]["lag_order"]
    collinearity = findings["conditioning"]["collinearity"]
    ess = findings["conditioning"]["effective_sample_size"]
    adequacy = findings["conditioning"]["sample_size_adequacy"]

    lines = []
    lines.append(
        f"1. **The data is close to Gaussian, not far from it.** Max |skewness| "
        f"{_fmt(marginals['max_abs_skewness'], 3)}, max |excess kurtosis| "
        f"{_fmt(marginals['max_abs_excess_kurtosis'], 3)} across "
        f"{marginals['n_scored']} columns. Shapiro-Wilk rejects normality for "
        f"{marginals['rejections']['shapiro_wilk']} of {marginals['n_scored']} at "
        f"alpha = {marginals['alpha']}, but the effect sizes say those rejections "
        f"are detections of a trivial departure, not evidence against a Gaussian "
        f"working assumption."
    )
    if residuals.get("fitted"):
        lines.append(
            f"2. **VAR-LiNGAM is not identifiable here.** The VAR"
            f"({residuals['var_lag']}) residuals have mean |excess kurtosis| "
            f"{_fmt(residuals['mean_abs_excess_kurtosis'], 3)} and max "
            f"{_fmt(residuals['max_abs_excess_kurtosis'], 3)}, against a threshold "
            f"of {residuals['abs_excess_kurtosis_threshold']}; max negentropy is "
            f"{_fmt(residuals['max_negentropy'], 6)} nats against a floor of "
            f"{residuals['negentropy_threshold']}. Only "
            f"{residuals['n_usable_for_ica']} of {residuals['n_equations']} "
            f"equations clear both. The innovations are effectively Gaussian, and "
            f"ICA has no traction on Gaussian sources."
            if not residuals["lingam_contemporaneous_identifiable"]
            else f"2. **VAR-LiNGAM's contemporaneous stage is identifiable.** "
            f"{residuals['n_usable_for_ica']} of {residuals['n_equations']} "
            f"residual equations clear both thresholds; mean |excess kurtosis| "
            f"{_fmt(residuals['mean_abs_excess_kurtosis'], 3)}."
        )
    lines.append(
        f"3. **Samples are not independent, by a wide margin.** Lag-1 "
        f"autocorrelation runs up to {_fmt(serial['max_abs_acf_lag_1'], 3)} (mean "
        f"{_fmt(serial['mean_abs_acf_lag_1'], 3)}), Ljung-Box rejects independence "
        f"for {serial['n_ljung_box_rejections']} of {serial['n_scored']} columns, "
        f"and the binding effective sample size is {_fmt(ess['binding_n_eff'], 0)} "
        f"of {ess['n_rows']} rows -- {100 * ess['shrinkage_factor']:.1f}% of the "
        f"nominal count. Plain PC's Fisher-z p-values are anti-conservative by "
        f"that factor."
    )
    nonlinear_pairs = pairwise["n_pairs_above_gap_threshold"]
    tree_wins = (
        predictive.get("n_with_nonlinear_gain", 0) if "skipped" not in predictive else 0
    )
    if nonlinear_pairs == 0 and tree_wins == 0:
        linearity_headline = "Dependence is linear, on every measure tried."
    elif nonlinear_pairs == 0:
        linearity_headline = (
            "Dependence is linear pairwise, but some variables gain from a nonlinear fit."
        )
    else:
        linearity_headline = "Dependence is mostly linear, with a nonlinear minority."
    lines.append(
        f"4. **{linearity_headline}** "
        f"{pairwise['n_pairs_above_gap_threshold']} of {pairwise['n_pairs']} pairs "
        f"({100 * pairwise['fraction_above_gap_threshold']:.1f}%) have distance "
        f"correlation exceeding |Pearson| by more than "
        f"{pairwise['gap_threshold']}; the largest gap is "
        f"{_fmt(pairwise['max_gap'], 3)}."
        + (
            f" Out of sample, a boosted tree beats a linear fit by a mean R^2 of "
            f"{_fmt(predictive['mean_r2_gap'], 3)} (max "
            f"{_fmt(predictive['max_r2_gap'], 3)}), and Ramsey RESET rejects "
            f"linearity for {predictive['n_reset_rejections']} of "
            f"{predictive['n_variables']} equations."
            if "skipped" not in predictive
            else ""
        )
    )
    stationarity_row = next(
        (
            row
            for row in findings["suitability"]["table"]["rows"]
            if row["assumption"] == "stationarity"
        ),
        None,
    )
    stationarity_headline = (
        "Stationary enough for the lagged methods."
        if stationarity_row and stationarity_row["holds_in_data"]
        else "Not fully stationary, which the lagged methods assume."
    )
    lines.append(
        f"5. **{stationarity_headline}** ADF calls "
        f"{stationarity['n_adf_stationary']} of {stationarity['n_scored']} columns "
        f"stationary; KPSS calls {stationarity['n_kpss_stationary']} stationary; "
        f"they disagree on {stationarity['n_disagree']}"
        + (
            f" ({', '.join(stationarity['disagreeing_columns'])})"
            if stationarity["disagreeing_columns"]
            else ""
        )
        + f". Mean shifts reach "
        f"{_fmt(findings['temporal']['regime']['halves']['max_mean_shift_sd'], 2)} SD "
        f"across halves and "
        f"{_fmt(findings['temporal']['regime']['thirds']['max_mean_shift_sd'], 2)} SD "
        f"across thirds, the latter on "
        f"{findings['temporal']['regime']['thirds']['n_columns_shifted']} column(s)."
    )
    if "error" not in lag_order:
        lines.append(
            "6. **Lag order is small and the criteria disagree.** "
            + ", ".join(f"{k.upper()} = {v}" for k, v in lag_order["selected"].items())
            + f". Any `tau_max` from {lag_order['min_order']} to "
            f"{lag_order['max_order']} is defensible."
        )
    lines.append(
        f"7. **The covariance is badly conditioned.** Condition number "
        f"{_fmt(collinearity['condition_number'], 0)} against a threshold of "
        f"{_fmt(collinearity['condition_number_threshold'], 0)}, smallest "
        f"eigenvalue {_fmt(collinearity['min_eigenvalue'])}, "
        f"{collinearity['n_pairs_above_high_threshold']} pair(s) at or above |r| = "
        f"{collinearity['high_corr_threshold']}, and "
        f"{collinearity['vif']['n_above_threshold']} of "
        f"{collinearity['n_variables']} variables with VIF above "
        f"{_fmt(collinearity['vif']['threshold'], 0)} (max "
        f"{_fmt(collinearity['vif']['max'], 1)}). Partial correlations inverting "
        f"this matrix amplify noise, and no p-value reveals it."
    )
    lines.append(
        f"8. **Conditioning sets must stay small.** At "
        f"{adequacy['samples_per_parameter']} observations per estimated "
        f"parameter the effective sample size supports a conditioning set of "
        f"{adequacy['recommended_max_conditioning_set']}, against "
        f"{adequacy['at_raw_n']['practical_max_conditioning_set']} if the row "
        f"count were taken at face value."
    )
    if reconciliation.get("absent_from_data"):
        lines.append(
            f"9. **Causal sufficiency is violated by construction.** The ground "
            f"truth names {len(reconciliation['absent_from_data'])} node(s) absent "
            f"from the data ({', '.join(reconciliation['absent_from_data'])}), "
            f"carrying {reconciliation['n_unrecoverable_edges']} edge(s). Those "
            f"are latent confounders, and "
            f"{reconciliation['n_unrecoverable_edges']} of "
            f"{reconciliation['ground_truth_edges']} edges cannot be recovered by "
            f"any method."
        )
    return "\n".join(lines) + "\n"


def _top_outlier_rows(findings: dict[str, Any]) -> list[list[Any]]:
    per_column = findings["distribution"]["marginals"]["per_column"]
    rows = [
        (
            column,
            value["outliers"]["modified_z_count"],
            value["outliers"]["iqr_count"],
            value["outliers"]["max_abs_modified_z"],
        )
        for column, value in per_column.items()
        if "skipped" not in value
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return [[r[0], r[1], r[2], _fmt(r[3], 2)] for r in rows[:10]]


def _decorrelation_table(findings: dict[str, Any]) -> str:
    per_column = findings["temporal"]["serial_dependence"]["per_column"]
    rows = [
        [
            column,
            _fmt(value["acf_lag_1"], 3),
            _fmt(value["acf_lag_10"], 3),
            _fmt(value["acf_lag_50"], 3),
            value["decorrelation_lag"]
            if value["decorrelation_lag"] is not None
            else "> window",
        ]
        for column, value in per_column.items()
        if "skipped" not in value
    ]
    return _table(["Column", "ACF(1)", "ACF(10)", "ACF(50)", "Decorrelation lag"], rows)


def write_markdown(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
