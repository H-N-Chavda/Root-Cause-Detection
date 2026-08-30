"""Normality and non-Gaussianity, on distributions with known moments."""

from __future__ import annotations

import numpy as np

from causal_bench.eda import distribution


def test_gaussian_data_passes_the_normality_check(gaussian_frame, eda_config):
    result = distribution.analyse_marginals(
        gaussian_frame, eda_config.distribution, eda_config.significance
    )
    summary = result["summary"]

    # Effect sizes, which is what the verdict rests on.
    assert summary["max_abs_skewness"] < eda_config.distribution.skew_threshold
    assert (
        summary["max_abs_excess_kurtosis"]
        < eda_config.distribution.excess_kurtosis_threshold
    )
    assert summary["n_practically_gaussian"] == summary["n_scored"] == 4
    # Genuinely Gaussian data should not be rejected wholesale either.
    assert summary["rejections"]["shapiro_wilk"] <= 1


def test_laplace_data_fails_the_normality_check(laplace_frame, eda_config):
    """Laplace has excess kurtosis 3, well above the threshold, and every test
    should reject at this sample size."""
    result = distribution.analyse_marginals(
        laplace_frame, eda_config.distribution, eda_config.significance
    )
    summary = result["summary"]

    assert (
        summary["max_abs_excess_kurtosis"]
        > eda_config.distribution.excess_kurtosis_threshold
    )
    assert summary["n_practically_gaussian"] == 0
    assert summary["rejections"]["shapiro_wilk"] == summary["n_scored"]
    assert summary["rejections"]["jarque_bera"] == summary["n_scored"]


def test_all_four_normality_tests_are_reported(gaussian_frame, eda_config):
    values = gaussian_frame["X1"].to_numpy()
    tests = distribution.normality_tests(values, eda_config.significance)
    for name in ("shapiro_wilk", "dagostino_k2", "jarque_bera", "anderson_darling"):
        assert name in tests
        assert "statistic" in tests[name]
        assert "reject" in tests[name]
    # Anderson-Darling has no p-value; it is judged against a critical value.
    assert tests["anderson_darling"]["p_value"] is None
    assert tests["anderson_darling"]["critical_value"] is not None


def test_negentropy_is_near_zero_for_a_gaussian():
    generator = np.random.default_rng(0)
    assert distribution.negentropy(generator.normal(size=5000)) < 1e-3


def test_negentropy_is_positive_for_a_laplace():
    generator = np.random.default_rng(0)
    gaussian = distribution.negentropy(generator.normal(size=5000))
    laplace = distribution.negentropy(generator.laplace(size=5000))
    assert laplace > 0.01
    assert laplace > 10 * gaussian


def test_negentropy_is_scale_and_location_invariant():
    """G is calibrated on a standard normal, so the estimate must not move when
    the sample is shifted or rescaled."""
    generator = np.random.default_rng(1)
    sample = generator.laplace(size=4000)
    base = distribution.negentropy(sample)
    rescaled = distribution.negentropy(3.0 * sample + 10.0)
    assert abs(rescaled - base) < 1e-9


def test_gaussian_var_residuals_are_not_ica_identifiable(eda_config):
    """A linear VAR driven by Gaussian noise leaves Gaussian residuals, so
    LiNGAM's contemporaneous stage must be reported as unidentifiable."""
    import pandas as pd

    generator = np.random.default_rng(0)
    n, burn = 900, 100
    series = np.zeros((n + burn, 3))
    noise = generator.normal(size=(n + burn, 3))
    for t in range(1, n + burn):
        series[t] = 0.5 * series[t - 1] + noise[t]
    frame = pd.DataFrame(series[burn:], columns=["X1", "X2", "X3"])

    result = distribution.var_residual_nongaussianity(
        frame, eda_config.nongaussianity, eda_config.significance
    )
    assert result["fitted"] is True
    assert result["lingam_contemporaneous_identifiable"] is False
    assert result["n_usable_for_ica"] == 0
    assert abs(result["mean_abs_excess_kurtosis"]) < 0.5


def test_laplace_var_residuals_are_ica_identifiable(eda_config):
    """The same VAR driven by Laplace innovations must flip the verdict. This is
    the pair that shows the residual test is measuring the innovations, not the
    marginals."""
    import pandas as pd

    generator = np.random.default_rng(0)
    n, burn = 900, 100
    series = np.zeros((n + burn, 3))
    noise = generator.laplace(size=(n + burn, 3))
    for t in range(1, n + burn):
        series[t] = 0.5 * series[t - 1] + noise[t]
    frame = pd.DataFrame(series[burn:], columns=["X1", "X2", "X3"])

    result = distribution.var_residual_nongaussianity(
        frame, eda_config.nongaussianity, eda_config.significance
    )
    assert result["fitted"] is True
    assert result["lingam_contemporaneous_identifiable"] is True
    assert result["n_usable_for_ica"] == result["n_equations"] == 3
    assert result["max_negentropy"] > eda_config.nongaussianity.negentropy_threshold


def test_outliers_are_counted_not_removed(eda_config):
    """A planted spike must be counted, and the row count must be untouched."""
    import pandas as pd

    generator = np.random.default_rng(0)
    values = generator.normal(size=500)
    values[10] = 50.0
    frame = pd.DataFrame({"X1": values})

    result = distribution.analyse_marginals(
        frame, eda_config.distribution, eda_config.significance
    )
    outliers = result["per_column"]["X1"]["outliers"]
    assert outliers["modified_z_count"] >= 1
    assert outliers["iqr_count"] >= 1
    assert outliers["max_abs_modified_z"] > eda_config.distribution.outlier_modified_z
    assert result["per_column"]["X1"]["n"] == 500


def test_discrete_columns_get_no_gaussian_verdict(eda_config):
    import pandas as pd

    generator = np.random.default_rng(0)
    frame = pd.DataFrame({"X1": generator.integers(0, 2, size=400).astype(float)})
    result = distribution.analyse_marginals(
        frame, eda_config.distribution, eda_config.significance, discrete_columns={"X1"}
    )
    assert result["per_column"]["X1"]["practically_gaussian"] is None
    assert result["per_column"]["X1"]["is_discrete"] is True
