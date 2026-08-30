"""The findings-to-assumptions mapping.

These tests drive the table with hand-built findings so the expected verdict is
unambiguous, rather than with a dataset whose properties are themselves under
investigation.
"""

from __future__ import annotations

import pytest

from causal_bench.config import EdaConfig
from causal_bench.eda import suitability


def _findings(**overrides):
    """A findings mapping in which every assumption holds, before overrides."""
    base = {
        "structure": {
            "ground_truth_reconciliation": {
                "ground_truth_available": True,
                "ground_truth_acyclic": True,
                "ground_truth_reciprocal_pairs": 0,
                "ground_truth_self_loops": 0,
                "ground_truth_edges": 10,
                "absent_from_data": [],
                "n_unrecoverable_edges": 0,
            }
        },
        "distribution": {
            "marginals": {
                "summary": {
                    "n_scored": 5,
                    "max_abs_skewness": 0.1,
                    "mean_abs_skewness": 0.05,
                    "max_abs_excess_kurtosis": 0.2,
                    "n_practically_non_gaussian": 0,
                    "rejections": {"shapiro_wilk": 0},
                    "alpha": 0.05,
                },
                "per_column": {
                    "X1": {"std": 1.0, "skewness": 0.1},
                    "X2": {"std": 1.2, "skewness": 0.1},
                },
            },
            "var_residuals": {
                "fitted": True,
                "var_lag": 1,
                "n_equations": 5,
                "mean_abs_excess_kurtosis": 3.0,
                "max_abs_excess_kurtosis": 3.5,
                "max_negentropy": 0.2,
                "n_usable_for_ica": 5,
                "lingam_contemporaneous_identifiable": True,
            },
        },
        "linearity": {
            "pairwise": {
                "n_pairs": 10,
                "n_pairs_above_gap_threshold": 0,
                "fraction_above_gap_threshold": 0.0,
                "max_gap": 0.02,
                "median_gap": 0.01,
            },
            "predictive": {
                "mean_r2_gap": 0.0,
                "max_r2_gap": 0.01,
                "n_with_nonlinear_gain": 0,
                "n_variables": 5,
                "n_reset_rejections": 0,
            },
        },
        "temporal": {
            "serial_dependence": {
                "summary": {
                    "n_scored": 5,
                    "n_ljung_box_rejections": 0,
                    "max_abs_acf_lag_1": 0.05,
                    "mean_abs_acf_lag_1": 0.03,
                }
            },
            "stationarity": {
                "summary": {
                    "n_scored": 5,
                    "n_adf_stationary": 5,
                    "n_kpss_stationary": 5,
                    "n_disagree": 0,
                    "disagreeing_columns": [],
                },
                "per_column": {
                    "X1": {"adf": {"stationary": True}, "verdict": "stationary"}
                },
            },
            "regime": {
                "shift_threshold_sd": 0.5,
                "halves": {"n_columns_shifted": 0, "max_mean_shift_sd": 0.1},
                "thirds": {"n_columns_shifted": 0, "max_mean_shift_sd": 0.1},
                "changepoints": {"n_columns_with_changepoints": 0},
            },
        },
        "conditioning": {
            "collinearity": {
                "ill_conditioned": False,
                "full_rank": True,
                "condition_number": 5.0,
                "condition_number_threshold": 1000.0,
                "numerical_rank": 5,
                "n_variables": 5,
                "min_eigenvalue": 0.4,
                "n_pairs_above_high_threshold": 0,
                "high_corr_threshold": 0.95,
                "vif": {"n_above_threshold": 0, "threshold": 10.0, "max": 1.2},
            },
            "effective_sample_size": {
                "n_rows": 1000,
                "binding_n_eff": 950.0,
                "shrinkage_factor": 0.95,
            },
            "sample_size_adequacy": {
                "samples_per_parameter": 10,
                "recommended_max_conditioning_set": 3,
                "at_raw_n": {"practical_max_conditioning_set": 98},
            },
        },
    }
    for path, value in overrides.items():
        node = base
        *parents, leaf = path.split(".")
        for step in parents:
            node = node[step]
        node[leaf] = value
    return base


@pytest.fixture
def cfg() -> EdaConfig:
    return EdaConfig()


def test_every_assumption_appears_for_every_algorithm(cfg):
    table = suitability.build_table(_findings(), cfg)
    assert set(table["algorithms"]) == {"PC", "PCMCI+", "VAR-LiNGAM", "LSTE"}
    assert len(table["rows"]) == 9
    for row in table["rows"]:
        assert set(row["verdicts"]) == set(table["algorithms"])
        assert row["evidence"], "every row must carry the number behind it"


def test_ideal_data_lets_everything_work(cfg):
    table = suitability.build_table(_findings(), cfg)
    ranking = suitability.recommend(table, _findings(), cfg)["ranking"]
    for entry in ranking:
        assert entry["verdict"] == "expected to work"
        assert entry["n_hard_violations"] == 0


def test_gaussian_residuals_break_only_var_lingam(cfg):
    findings = _findings(
        **{
            "distribution.var_residuals": {
                "fitted": True,
                "var_lag": 1,
                "n_equations": 5,
                "mean_abs_excess_kurtosis": 0.1,
                "max_abs_excess_kurtosis": 0.2,
                "max_negentropy": 0.0001,
                "n_usable_for_ica": 0,
                "lingam_contemporaneous_identifiable": False,
            }
        }
    )
    table = suitability.build_table(findings, cfg)
    row = next(r for r in table["rows"] if r["assumption"] == "non_gaussian_residuals")
    assert row["verdicts"]["VAR-LiNGAM"] == "violated"
    assert row["verdicts"]["PC"] == "n/a"
    assert row["verdicts"]["PCMCI+"] == "n/a"
    assert row["verdicts"]["LSTE"] == "n/a"


def test_autocorrelation_breaks_only_pc(cfg):
    """PC treats rows as independent draws; the lagged methods do not, so a
    dependent sample must penalise PC alone."""
    findings = _findings(
        **{
            "temporal.serial_dependence.summary": {
                "n_scored": 5,
                "n_ljung_box_rejections": 5,
                "max_abs_acf_lag_1": 0.98,
                "mean_abs_acf_lag_1": 0.9,
            },
            "conditioning.effective_sample_size": {
                "n_rows": 1000,
                "binding_n_eff": 40.0,
                "shrinkage_factor": 0.04,
            },
        }
    )
    table = suitability.build_table(findings, cfg)
    row = next(r for r in table["rows"] if r["assumption"] == "independent_samples")
    assert row["holds_in_data"] is False
    assert row["verdicts"]["PC"] == "violated"
    assert row["verdicts"]["PCMCI+"] == "n/a"
    assert row["verdicts"]["VAR-LiNGAM"] == "n/a"


def test_non_stationarity_spares_pc(cfg):
    """PC builds no time model, so a regime shift is not its problem."""
    findings = _findings(
        **{
            "temporal.regime.thirds": {
                "n_columns_shifted": 3,
                "max_mean_shift_sd": 2.0,
            }
        }
    )
    table = suitability.build_table(findings, cfg)
    row = next(r for r in table["rows"] if r["assumption"] == "stationarity")
    assert row["holds_in_data"] is False
    assert row["verdicts"]["PC"] == "n/a"
    for algorithm in ("PCMCI+", "VAR-LiNGAM", "LSTE"):
        assert row["verdicts"][algorithm] == "violated"


def test_a_thirds_only_shift_is_not_missed(cfg):
    """A drift that cancels across halves must still be caught by the thirds
    split; testing halves alone would report this data as stationary."""
    findings = _findings(
        **{
            "temporal.regime.halves": {
                "n_columns_shifted": 0,
                "max_mean_shift_sd": 0.05,
            },
            "temporal.regime.thirds": {
                "n_columns_shifted": 2,
                "max_mean_shift_sd": 1.4,
            },
        }
    )
    row = next(
        r
        for r in suitability.build_table(findings, cfg)["rows"]
        if r["assumption"] == "stationarity"
    )
    assert row["holds_in_data"] is False


def test_unmeasurable_assumptions_are_flagged_not_hidden(cfg):
    table = suitability.build_table(_findings(), cfg)
    unmeasurable = {r["assumption"] for r in table["rows"] if r["unmeasurable"]}
    assert unmeasurable == {"causal_sufficiency", "acyclicity"}
    for row in table["rows"]:
        if row["unmeasurable"]:
            assert "not identifiable" in row["evidence"].lower()


def test_a_latent_node_marks_causal_sufficiency_violated(cfg):
    findings = _findings(
        **{
            "structure.ground_truth_reconciliation": {
                "ground_truth_available": True,
                "ground_truth_acyclic": True,
                "ground_truth_reciprocal_pairs": 0,
                "ground_truth_self_loops": 0,
                "ground_truth_edges": 10,
                "absent_from_data": ["X9"],
                "n_unrecoverable_edges": 2,
            }
        }
    )
    row = next(
        r
        for r in suitability.build_table(findings, cfg)["rows"]
        if r["assumption"] == "causal_sufficiency"
    )
    assert row["holds_in_data"] is False
    assert "X9" in row["evidence"]
    # Still unmeasurable, so it must not be counted as a hard violation.
    assert row["unmeasurable"] is True


def test_unmeasurable_violations_do_not_force_a_failure_verdict(cfg):
    """Causal sufficiency is violated in every real dataset here. If it counted
    as a hard violation, every algorithm would read as failing regardless of the
    data, and the table would stop discriminating."""
    findings = _findings(
        **{
            "structure.ground_truth_reconciliation": {
                "ground_truth_available": True,
                "ground_truth_acyclic": True,
                "ground_truth_reciprocal_pairs": 0,
                "ground_truth_self_loops": 0,
                "ground_truth_edges": 10,
                "absent_from_data": ["X9"],
                "n_unrecoverable_edges": 2,
            }
        }
    )
    table = suitability.build_table(findings, cfg)
    ranking = suitability.recommend(table, findings, cfg)["ranking"]
    assert all(entry["verdict"] == "expected to work" for entry in ranking)


def test_preprocessing_reports_but_does_not_transform(cfg):
    advice = suitability.preprocessing_advice(_findings(), cfg)
    assert advice["applied"] is False
    assert advice["standardization"]["required"] is False
    assert advice["differencing"]["required"] is False
    for step in ("standardization", "differencing", "transformation"):
        assert advice[step]["cost"], "every recommendation must state its cost"


def test_preprocessing_flags_a_wide_magnitude_spread(cfg):
    findings = _findings(
        **{
            "distribution.marginals.per_column": {
                "X1": {"std": 0.001, "skewness": 0.1},
                "X2": {"std": 1000.0, "skewness": 0.1},
            }
        }
    )
    advice = suitability.preprocessing_advice(findings, cfg)
    assert advice["standardization"]["required"] is True
    assert advice["standardization"]["magnitude_ratio"] > 1e5
    assert advice["applied"] is False
