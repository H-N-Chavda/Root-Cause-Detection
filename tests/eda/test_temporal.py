"""Serial dependence, stationarity, regime shifts, lag order, delay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_bench.eda import temporal


def test_white_noise_decorrelates_immediately(gaussian_frame, eda_config):
    result = temporal.serial_dependence(
        gaussian_frame, eda_config.temporal, eda_config.significance
    )
    summary = result["summary"]
    assert summary["max_abs_acf_lag_1"] < 0.15
    assert summary["n_ljung_box_rejections"] == 0
    for column in gaussian_frame.columns:
        assert result["per_column"][column]["decorrelation_lag"] == 1


def test_autocorrelated_series_is_detected(ar_frame, eda_config):
    """An AR(2) with coefficient 0.6 at lag 1 must show a clear lag-1 ACF and
    have Ljung-Box reject independence on every column."""
    result = temporal.serial_dependence(
        ar_frame, eda_config.temporal, eda_config.significance
    )
    summary = result["summary"]
    assert summary["max_abs_acf_lag_1"] > 0.3
    assert summary["n_ljung_box_rejections"] == summary["n_scored"] == 3
    assert result["per_column"]["X1"]["decorrelation_lag"] > 1


def test_ar2_lag_order_is_recovered(ar_frame, eda_config):
    """The generating order is 2. AIC and FPE may overshoot, but the consistent
    criteria (BIC, HQIC) must land on it."""
    result = temporal.lag_order(ar_frame, eda_config.temporal)
    selected = result["selected"]
    assert selected["bic"] == 2
    assert selected["hqic"] == 2
    assert result["min_order"] >= 1


def test_stationary_series_passes_both_tests(ar_frame, eda_config):
    result = temporal.stationarity(ar_frame, eda_config.significance)
    summary = result["summary"]
    assert summary["n_adf_stationary"] == summary["n_scored"] == 3
    assert summary["n_kpss_stationary"] == 3
    assert summary["n_disagree"] == 0
    assert result["per_column"]["X1"]["verdict"] == "stationary"


def test_random_walk_fails_both_tests(random_walk_frame, eda_config):
    """A unit root: ADF must fail to reject its null, KPSS must reject its own.
    The two agreeing here is what makes the pair worth reporting."""
    result = temporal.stationarity(random_walk_frame, eda_config.significance)
    summary = result["summary"]
    assert summary["n_adf_stationary"] == 0
    assert summary["n_kpss_stationary"] == 0
    assert summary["n_disagree"] == 0
    assert result["per_column"]["X1"]["verdict"] == "non-stationary"
    assert result["per_column"]["X1"]["adf"]["null"] == "unit root"
    assert result["per_column"]["X1"]["kpss"]["null"] == "stationarity"


def test_regime_shift_is_detected_in_means_and_by_changepoint(
    regime_shift_frame, eda_config
):
    """A 4 SD level shift at the midpoint. The half-split must see it, and the
    changepoint pass must locate it near the midpoint."""
    result = temporal.regime_stability(regime_shift_frame, eda_config.temporal)

    halves = result["halves"]["per_column"]["X1"]
    assert halves["max_mean_shift_sd"] > 1.0
    assert result["halves"]["n_columns_shifted"] >= 1

    changepoints = result["changepoints"]["per_column"]["X1"]
    assert changepoints, "expected at least one changepoint"
    midpoint = len(regime_shift_frame) // 2
    assert min(abs(c - midpoint) for c in changepoints) < 30

    # The undisturbed column must stay clean.
    assert result["halves"]["per_column"]["X2"]["max_mean_shift_sd"] < 0.5


def test_stable_series_has_no_regime_shift(gaussian_frame, eda_config):
    result = temporal.regime_stability(gaussian_frame, eda_config.temporal)
    assert result["halves"]["n_columns_shifted"] == 0
    assert result["thirds"]["n_columns_shifted"] == 0
    assert result["changepoints"]["total_changepoints"] == 0


def test_known_delay_is_recovered(delayed_frame, eda_config):
    """X2 is X1 delayed by 7 samples, so the peak must sit at lag 7 with X1
    leading, matching the documented sign convention."""
    result = temporal.cross_correlation(delayed_frame, eda_config.temporal)
    pair = result["pairs"][0]
    assert {pair["a"], pair["b"]} == {"X1", "X2"}
    assert pair["a"] == "X1" and pair["b"] == "X2"
    assert pair["peak_lag"] == 7  # positive: `a` leads `b`, as documented
    assert abs(pair["peak_value"]) > 0.9
    assert abs(pair["lag_0_value"]) < 0.3


def test_simultaneous_pair_peaks_at_lag_zero(eda_config):
    generator = np.random.default_rng(0)
    x1 = generator.normal(size=600)
    frame = pd.DataFrame({"X1": x1, "X2": x1 + generator.normal(scale=0.1, size=600)})
    result = temporal.cross_correlation(frame, eda_config.temporal)
    assert result["pairs"][0]["peak_lag"] == 0
    assert result["n_peak_at_nonzero_lag"] == 0
