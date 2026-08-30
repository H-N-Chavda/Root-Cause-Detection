"""Maps measured findings onto the assumptions the four algorithms rely on.

Nothing is measured here. Every cell reads a number produced upstream and turns
it into a verdict against a threshold from the config, carrying the number with
it so the table is auditable rather than assertive.

Two of the nine assumptions -- causal sufficiency and acyclicity -- are not
identifiable from observational data. They are kept in the table with an
explicit `unmeasurable` status rather than dropped, because silently omitting
them would make the remaining seven look like a complete answer.
"""

from __future__ import annotations

from typing import Any

from ..config import EdaConfig

ALGORITHMS = ("PC", "PCMCI+", "VAR-LiNGAM", "LSTE")

#: What each algorithm needs from the data.
#:
#: "required"       the assumption must hold
#: "required_false" the assumption must NOT hold (LiNGAM needs non-Gaussianity)
#: "not_required"   the method does not rely on it
#:
#: PC and PCMCI+ are read with their standard partial-correlation (Fisher-z /
#: ParCorr) independence test, which is what this project uses; both admit
#: nonparametric tests that would relax the Gaussian and linear rows. LSTE is
#: read as a transfer-entropy-style estimator: nonparametric in functional form
#: and distribution, explicitly lagged, and therefore dependent on stationarity
#: and on sample size instead.
REQUIREMENTS: dict[str, dict[str, str]] = {
    "gaussian_data": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "not_required",
        "LSTE": "not_required",
    },
    "non_gaussian_residuals": {
        "PC": "not_required",
        "PCMCI+": "not_required",
        "VAR-LiNGAM": "required",
        "LSTE": "not_required",
    },
    "linear_relationships": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "not_required",
    },
    "independent_samples": {
        "PC": "required",
        "PCMCI+": "not_required",
        "VAR-LiNGAM": "not_required",
        "LSTE": "not_required",
    },
    "stationarity": {
        "PC": "not_required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "required",
    },
    "causal_sufficiency": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "required",
    },
    "acyclicity": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "not_required",
    },
    "adequate_sample_size": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "required",
    },
    "well_conditioned_covariance": {
        "PC": "required",
        "PCMCI+": "required",
        "VAR-LiNGAM": "required",
        "LSTE": "not_required",
    },
}

LABELS: dict[str, str] = {
    "gaussian_data": "Gaussian data (marginals)",
    "non_gaussian_residuals": "Non-Gaussian VAR residuals",
    "linear_relationships": "Linear relationships",
    "independent_samples": "Independent samples",
    "stationarity": "Stationarity",
    "causal_sufficiency": "Causal sufficiency",
    "acyclicity": "Acyclicity",
    "adequate_sample_size": "Adequate sample size",
    "well_conditioned_covariance": "Well-conditioned covariance",
}


def _holds_gaussian(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    summary = findings["distribution"]["marginals"]["summary"]
    max_skew = summary["max_abs_skewness"]
    max_kurt = summary["max_abs_excess_kurtosis"]
    n_non = summary["n_practically_non_gaussian"]
    holds = bool(
        max_skew is not None
        and max_kurt is not None
        and max_skew < cfg.distribution.skew_threshold
        and max_kurt < cfg.distribution.excess_kurtosis_threshold
    )
    return {
        "holds": holds,
        "evidence": (
            f"max |skew| = {max_skew:.3f} (< {cfg.distribution.skew_threshold}), "
            f"max |excess kurtosis| = {max_kurt:.3f} "
            f"(< {cfg.distribution.excess_kurtosis_threshold}); "
            f"{n_non}/{summary['n_scored']} columns exceed either threshold; "
            f"Shapiro-Wilk rejects {summary['rejections']['shapiro_wilk']}"
            f"/{summary['n_scored']} at alpha={summary['alpha']}"
        ),
    }


def _holds_non_gaussian_residuals(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    residuals = findings["distribution"]["var_residuals"]
    if not residuals.get("fitted"):
        return {"holds": None, "evidence": f"VAR fit failed: {residuals.get('error')}"}
    return {
        "holds": bool(residuals["lingam_contemporaneous_identifiable"]),
        "evidence": (
            f"VAR({residuals['var_lag']}) residuals: mean |excess kurtosis| = "
            f"{residuals['mean_abs_excess_kurtosis']:.3f}, max = "
            f"{residuals['max_abs_excess_kurtosis']:.3f} "
            f"(threshold {cfg.nongaussianity.abs_excess_kurtosis_threshold}); "
            f"max negentropy = {residuals['max_negentropy']:.5f} "
            f"(threshold {cfg.nongaussianity.negentropy_threshold}); "
            f"{residuals['n_usable_for_ica']}/{residuals['n_equations']} equations "
            f"clear both"
        ),
    }


def _holds_linear(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    pairwise = findings["linearity"]["pairwise"]
    predictive = findings["linearity"]["predictive"]
    if "skipped" in predictive:
        gap_text = "predictive comparison skipped"
        gap_ok = True
    else:
        gap_text = (
            f"mean out-of-sample R^2 gain of tree over linear = "
            f"{predictive['mean_r2_gap']:.3f}, max = {predictive['max_r2_gap']:.3f}; "
            f"{predictive['n_with_nonlinear_gain']}/{predictive['n_variables']} "
            f"variables gain more than {cfg.linearity.r2_gap_threshold}; "
            f"Ramsey RESET rejects {predictive['n_reset_rejections']}"
            f"/{predictive['n_variables']}"
        )
        gap_ok = predictive["mean_r2_gap"] <= cfg.linearity.r2_gap_threshold
    return {
        "holds": bool(gap_ok and pairwise["fraction_above_gap_threshold"] < 0.5),
        "evidence": (
            f"{pairwise['n_pairs_above_gap_threshold']}/{pairwise['n_pairs']} pairs "
            f"have distance correlation exceeding |Pearson| by more than "
            f"{cfg.linearity.dcor_pearson_gap} (max gap {pairwise['max_gap']:.3f}, "
            f"median {pairwise['median_gap']:.3f}); {gap_text}"
        ),
    }


def _holds_independent_samples(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    serial = findings["temporal"]["serial_dependence"]["summary"]
    ess = findings["conditioning"]["effective_sample_size"]
    return {
        "holds": bool(
            serial["n_ljung_box_rejections"] == 0 and ess["shrinkage_factor"] > 0.5
        ),
        "evidence": (
            f"Ljung-Box rejects independence for "
            f"{serial['n_ljung_box_rejections']}/{serial['n_scored']} columns; "
            f"max lag-1 ACF = {serial['max_abs_acf_lag_1']:.3f}, mean = "
            f"{serial['mean_abs_acf_lag_1']:.3f}; binding effective n = "
            f"{ess['binding_n_eff']:.0f} of {ess['n_rows']} rows "
            f"({100 * ess['shrinkage_factor']:.1f}%)"
        ),
    }


def _holds_stationarity(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    stat = findings["temporal"]["stationarity"]["summary"]
    regime = findings["temporal"]["regime"]
    halves, thirds = regime["halves"], regime["thirds"]
    changepoints = regime["changepoints"]
    all_stationary = stat["n_adf_stationary"] == stat["n_scored"]
    # Both splits count. A drift that reverses within the series cancels across
    # halves and only shows up against thirds, so testing halves alone would
    # miss exactly the regime shape these processes are most likely to have.
    shifted = max(halves["n_columns_shifted"], thirds["n_columns_shifted"])
    return {
        "holds": bool(all_stationary and shifted == 0),
        "evidence": (
            f"ADF says stationary for {stat['n_adf_stationary']}/{stat['n_scored']}, "
            f"KPSS for {stat['n_kpss_stationary']}/{stat['n_scored']}; the two "
            f"disagree on {stat['n_disagree']}; max mean shift = "
            f"{halves['max_mean_shift_sd']:.2f} SD across halves and "
            f"{thirds['max_mean_shift_sd']:.2f} SD across thirds "
            f"({halves['n_columns_shifted']} and {thirds['n_columns_shifted']} "
            f"columns respectively above {regime['shift_threshold_sd']} SD); "
            f"{changepoints['n_columns_with_changepoints']} columns carry a "
            f"detected changepoint"
        ),
    }


def _holds_causal_sufficiency(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    reconciliation = findings["structure"]["ground_truth_reconciliation"]
    absent = reconciliation.get("absent_from_data", [])
    lost = reconciliation.get("n_unrecoverable_edges")
    detail = ""
    if absent:
        detail = (
            f" The ground truth names {len(absent)} node(s) absent from the data "
            f"({', '.join(absent)}), carrying {lost} edge(s); those are confounders "
            "by construction, so sufficiency is known to be violated here."
        )
    return {
        "holds": False if absent else None,
        "unmeasurable": True,
        "evidence": (
            "Not identifiable from observational data: no test distinguishes a "
            "latent common cause from a direct edge without further assumptions." + detail
        ),
    }


def _holds_acyclicity(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    reconciliation = findings["structure"]["ground_truth_reconciliation"]
    if not reconciliation.get("ground_truth_available"):
        return {
            "holds": None,
            "unmeasurable": True,
            "evidence": "Not identifiable from observational data, and no ground "
            "truth supplied to check.",
        }
    acyclic = reconciliation.get("ground_truth_acyclic")
    reciprocal = reconciliation.get("ground_truth_reciprocal_pairs", 0)
    return {
        "holds": bool(acyclic),
        "unmeasurable": True,
        "evidence": (
            "Not identifiable from the data itself. The supplied ground truth is "
            f"{'acyclic' if acyclic else 'cyclic'} with {reciprocal} reciprocal "
            f"pair(s) and {reconciliation.get('ground_truth_self_loops')} self-loop(s), "
            "which is evidence about the target, not about the data."
        ),
    }


def _holds_sample_size(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    adequacy = findings["conditioning"]["sample_size_adequacy"]
    ess = findings["conditioning"]["effective_sample_size"]
    recommended = adequacy["recommended_max_conditioning_set"]
    return {
        "holds": bool(recommended >= 2),
        "evidence": (
            f"{ess['n_rows']} rows, binding effective n = {ess['binding_n_eff']:.0f} "
            f"after the autocorrelation discount; at "
            f"{adequacy['samples_per_parameter']} samples per parameter that "
            f"supports a conditioning set of {recommended} "
            f"(raw-n bound would be "
            f"{adequacy['at_raw_n']['practical_max_conditioning_set']})"
        ),
    }


def _holds_conditioning(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    coll = findings["conditioning"]["collinearity"]
    return {
        "holds": bool(not coll["ill_conditioned"] and coll["full_rank"]),
        "evidence": (
            f"correlation matrix condition number = {coll['condition_number']:.0f} "
            f"(threshold {coll['condition_number_threshold']:.0f}), rank "
            f"{coll['numerical_rank']}/{coll['n_variables']}, smallest eigenvalue "
            f"{coll['min_eigenvalue']:.2e}; "
            f"{coll['n_pairs_above_high_threshold']} pair(s) at or above "
            f"|r| = {coll['high_corr_threshold']}; "
            f"{coll['vif']['n_above_threshold']} variable(s) with VIF above "
            f"{coll['vif']['threshold']:.0f} (max {coll['vif']['max']:.1f})"
        ),
    }


_CHECKS = {
    "gaussian_data": _holds_gaussian,
    "non_gaussian_residuals": _holds_non_gaussian_residuals,
    "linear_relationships": _holds_linear,
    "independent_samples": _holds_independent_samples,
    "stationarity": _holds_stationarity,
    "causal_sufficiency": _holds_causal_sufficiency,
    "acyclicity": _holds_acyclicity,
    "adequate_sample_size": _holds_sample_size,
    "well_conditioned_covariance": _holds_conditioning,
}


def _cell(requirement: str, holds: bool | None) -> str:
    """One table cell: does this dataset meet what this algorithm needs?"""
    if requirement == "not_required":
        return "n/a"
    if holds is None:
        return "unknown"
    if requirement == "required":
        return "satisfied" if holds else "violated"
    return "satisfied" if not holds else "violated"  # required_false


def build_table(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    """The assumption-by-algorithm table, with the supporting number per row."""
    rows = []
    for key, check in _CHECKS.items():
        outcome = check(findings, cfg)
        holds = outcome["holds"]
        rows.append(
            {
                "assumption": key,
                "label": LABELS[key],
                "holds_in_data": holds,
                "unmeasurable": bool(outcome.get("unmeasurable", False)),
                "evidence": outcome["evidence"],
                "verdicts": {
                    algorithm: _cell(REQUIREMENTS[key][algorithm], holds)
                    for algorithm in ALGORITHMS
                },
                "requirements": REQUIREMENTS[key],
            }
        )
    return {"algorithms": list(ALGORITHMS), "rows": rows}


def _score(table: dict, algorithm: str) -> dict[str, Any]:
    violated = [r["label"] for r in table["rows"] if r["verdicts"][algorithm] == "violated"]
    unknown = [r["label"] for r in table["rows"] if r["verdicts"][algorithm] == "unknown"]
    # A violated assumption that the data cannot settle either way is a caveat,
    # not a failure; one the data does settle against the method is a failure.
    hard = [
        r["label"]
        for r in table["rows"]
        if r["verdicts"][algorithm] == "violated" and not r["unmeasurable"]
    ]
    return {"violated": violated, "hard_violations": hard, "unknown": unknown}


def recommend(table: dict, findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    """Ranked verdict per algorithm, each with the number behind it."""
    evidence_by_key = {r["assumption"]: r for r in table["rows"]}
    out = []
    for algorithm in ALGORITHMS:
        score = _score(table, algorithm)
        n_hard = len(score["hard_violations"])
        if n_hard == 0:
            verdict = "expected to work"
        elif n_hard <= 2:
            verdict = "expected to work with caveats"
        else:
            verdict = "expected to fail"
        reasons = [
            evidence_by_key[r["assumption"]]["evidence"]
            for r in table["rows"]
            if r["verdicts"][algorithm] == "violated" and not r["unmeasurable"]
        ]
        out.append(
            {
                "algorithm": algorithm,
                "verdict": verdict,
                "n_hard_violations": n_hard,
                "violated_assumptions": score["hard_violations"],
                "unknown_assumptions": score["unknown"],
                "reasons": reasons,
            }
        )
    order = {
        "expected to work": 0,
        "expected to work with caveats": 1,
        "expected to fail": 2,
    }
    out.sort(key=lambda r: (order[r["verdict"]], r["n_hard_violations"]))
    return {"ranking": out}


def preprocessing_advice(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    """What the data would need before modelling. Reports only; transforms nothing."""
    per_column = findings["distribution"]["marginals"]["per_column"]
    stds = [v["std"] for v in per_column.values() if "skipped" not in v and v["std"] > 0]
    ratio = (max(stds) / min(stds)) if stds else 1.0
    threshold = cfg.scaling.magnitude_ratio_threshold

    stationarity_summary = findings["temporal"]["stationarity"]
    non_stationary = [
        column
        for column, value in stationarity_summary["per_column"].items()
        if "skipped" not in value and not value["adf"]["stationary"]
    ]
    ambiguous = [
        column
        for column, value in stationarity_summary["per_column"].items()
        if "skipped" not in value and value["verdict"] == "ambiguous"
    ]

    marginal_summary = findings["distribution"]["marginals"]["summary"]
    skewed = [
        column
        for column, value in per_column.items()
        if "skipped" not in value
        and abs(value["skewness"]) >= cfg.distribution.skew_threshold
    ]

    return {
        "standardization": {
            "required": bool(ratio > threshold),
            "magnitude_ratio": float(ratio),
            "threshold": threshold,
            "reason": (
                f"largest/smallest column standard deviation = {ratio:.1f}"
                + (
                    f", above {threshold}: an unscaled distance or kernel "
                    "computation would be dominated by the widest variable"
                    if ratio > threshold
                    else f", below {threshold}"
                )
            ),
            "cost": "none for correlation-based methods, which are scale "
            "invariant; changes the units of any reported coefficient",
        },
        "differencing": {
            "required": bool(non_stationary),
            "columns": non_stationary,
            "ambiguous_columns": ambiguous,
            "reason": (
                f"{len(non_stationary)} column(s) fail the ADF stationarity test; "
                f"{len(ambiguous)} more have ADF and KPSS disagreeing"
            ),
            "cost": "differencing removes contemporaneous level information and "
            "changes the causal quantity being estimated from levels to changes; "
            "it also injects an MA(1) component that inflates the apparent lag order",
        },
        "transformation": {
            "worthwhile": bool(skewed),
            "skewed_columns": skewed,
            "reason": (
                f"{len(skewed)} column(s) have |skew| >= "
                f"{cfg.distribution.skew_threshold} (max observed "
                f"{marginal_summary['max_abs_skewness']:.3f})"
            ),
            "cost": "a monotone transform preserves Spearman and distance "
            "correlation but changes Pearson, so it alters exactly the test PC "
            "relies on; it also breaks the linear additive form VAR-LiNGAM assumes",
        },
        "applied": False,
        "note": "This stage reports. No transformation was applied to the data.",
    }


def analyse(findings: dict, cfg: EdaConfig) -> dict[str, Any]:
    """Whole suitability stage."""
    table = build_table(findings, cfg)
    return {
        "table": table,
        "recommendation": recommend(table, findings, cfg),
        "preprocessing": preprocessing_advice(findings, cfg),
    }
