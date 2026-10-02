"""Marginal distributions: normality tests and outlier counts.

Trimmed from the Phase-2 EDA module (commit e8f02e8) with two deliberate
changes: no `DistributionConfig` dependency (thresholds are plain keyword
arguments with the same defaults the old config shipped), and no
`var_residual_nongaussianity` (that needed statsmodels' VAR and a lag choice
that was Tennessee-Eastman-specific; not needed for the four-step pipeline
requested here).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Default MAD-based modified z-score threshold (Iglewicz & Hoaglin, 1993).
DEFAULT_MODIFIED_Z_THRESHOLD = 3.5
#: Default Tukey IQR fence multiplier.
DEFAULT_IQR_MULTIPLIER = 1.5


def _moments(values: np.ndarray) -> dict[str, Any]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "cv": abs(std / mean) if mean != 0.0 else float("nan"),
        "skewness": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "min": float(np.min(values)),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(np.max(values)),
        "iqr": q3 - q1,
    }


def normality_tests(sample: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Shapiro-Wilk, D'Agostino K^2, Jarque-Bera and Anderson-Darling.

    All four are reported because they weight departures differently:
    Shapiro-Wilk is an omnibus order-statistic test, D'Agostino and
    Jarque-Bera key on the third and fourth moments, Anderson-Darling
    weights the tails. Disagreement between them localises the departure.

    A binary (e.g. PRBS) input will fail every one of these emphatically and
    correctly -- that is the expected, physically meaningful result for such
    a column, not a bug in the test.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, p = stats.shapiro(x)
            out["shapiro_wilk"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as exc:  # pragma: no cover - degenerate input only
            out["shapiro_wilk"] = {"statistic": None, "p_value": None, "error": str(exc)}

        try:
            stat, p = stats.normaltest(x)
            out["dagostino_k2"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as exc:  # pragma: no cover
            out["dagostino_k2"] = {"statistic": None, "p_value": None, "error": str(exc)}

        stat, p = stats.jarque_bera(x)
        out["jarque_bera"] = {"statistic": float(stat), "p_value": float(p)}

        ad = stats.anderson(x, dist="norm")
        levels = np.asarray(ad.significance_level, dtype=float) / 100.0
        idx = int(np.argmin(np.abs(levels - alpha)))
        out["anderson_darling"] = {
            "statistic": float(ad.statistic),
            "critical_value": float(ad.critical_values[idx]),
            "significance_level_used": float(levels[idx]),
            "p_value": None,
            "reject": bool(ad.statistic > ad.critical_values[idx]),
        }

    for name in ("shapiro_wilk", "dagostino_k2", "jarque_bera"):
        p = out[name].get("p_value")
        out[name]["reject"] = bool(p is not None and p < alpha)

    out["n_reject"] = int(
        sum(
            bool(out[name].get("reject"))
            for name in ("shapiro_wilk", "dagostino_k2", "jarque_bera", "anderson_darling")
        )
    )
    out["unanimous_reject"] = out["n_reject"] == 4
    return out


def _single_column_outliers(
    values: np.ndarray,
    modified_z_threshold: float,
    iqr_multiplier: float,
) -> dict[str, Any]:
    """Counts only -- nothing is removed. Removing rows is a modelling
    decision, not an EDA one, and would break the time index."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        # 0.6745 is the 0.75 quantile of the standard normal: it rescales the
        # MAD to be a consistent estimator of sigma under normality.
        modified_z = 0.6745 * (values - median) / mad
        n_mz = int(np.sum(np.abs(modified_z) > modified_z_threshold))
        max_mz = float(np.max(np.abs(modified_z)))
        flagged = np.abs(modified_z) > modified_z_threshold
    else:
        n_mz, max_mz = 0, 0.0
        flagged = np.zeros(values.shape, dtype=bool)

    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    low = q1 - iqr_multiplier * iqr
    high = q3 + iqr_multiplier * iqr
    n_iqr = int(np.sum((values < low) | (values > high)))

    return {
        "modified_z_count": n_mz,
        "modified_z_fraction": n_mz / values.size,
        "max_abs_modified_z": max_mz,
        "iqr_count": n_iqr,
        "iqr_fraction": n_iqr / values.size,
        "iqr_low_fence": float(low),
        "iqr_high_fence": float(high),
        "mad": mad,
        "modified_z_flags": flagged,
    }


def outlier_summary(
    frame: pd.DataFrame,
    modified_z_threshold: float = DEFAULT_MODIFIED_Z_THRESHOLD,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> dict[str, Any]:
    """Per-column outlier counts (MAD modified z-score + IQR), plus moments.

    Returns ``{"per_column": {...}, "summary": {...}}``. Each per-column
    entry includes a boolean ``modified_z_flags`` array (same length as the
    column) so callers can overlay flagged points on a time-series figure
    without recomputing anything.
    """
    per_column: dict[str, Any] = {}
    for column in frame.columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size < 8 or np.std(values) == 0.0:
            per_column[str(column)] = {"skipped": "constant or too few observations"}
            continue
        per_column[str(column)] = {
            **_moments(values),
            **_single_column_outliers(values, modified_z_threshold, iqr_multiplier),
        }

    scored = {k: v for k, v in per_column.items() if "skipped" not in v}
    return {
        "per_column": per_column,
        "summary": {
            "n_scored": len(scored),
            "total_modified_z_outliers": int(
                sum(v["modified_z_count"] for v in scored.values())
            ),
            "total_iqr_outliers": int(sum(v["iqr_count"] for v in scored.values())),
            "modified_z_threshold": modified_z_threshold,
            "iqr_multiplier": iqr_multiplier,
        },
    }


#: Effect-size thresholds for the "practically Gaussian" verdict, below
#: which a formal-test rejection is a sample-size artifact rather than a
#: departure that matters. Matches the magnitude used in the Phase-2 module
#: (e8f02e8); not re-derived here, just not hidden behind a config object.
DEFAULT_SKEW_THRESHOLD = 0.5
DEFAULT_EXCESS_KURTOSIS_THRESHOLD = 1.0


def marginal_report(
    frame: pd.DataFrame,
    alpha: float = 0.05,
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
    excess_kurtosis_threshold: float = DEFAULT_EXCESS_KURTOSIS_THRESHOLD,
) -> dict[str, Any]:
    """Per-column normality verdict, for the before/after table in step 4.

    Reports both the four formal-test rejections AND an effect-size verdict
    (|skew| and |excess kurtosis| under threshold). The two can and do
    disagree at the sample sizes here: a Shapiro-Wilk p-value of 0.01 on a
    column with skew=0.02 and excess kurtosis=0.03 is not evidence the
    column is meaningfully non-Gaussian, it's evidence that n is in the
    thousands. Present both numbers rather than only the p-value count, or
    a working transform can look like a failed one on the slide.
    """
    per_column: dict[str, Any] = {}
    for column in frame.columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size < 8 or np.std(values) == 0.0:
            per_column[str(column)] = {"skipped": "constant or too few observations"}
            continue
        tests = normality_tests(values, alpha)
        skew = float(stats.skew(values, bias=False))
        excess_kurt = float(stats.kurtosis(values, fisher=True, bias=False))
        per_column[str(column)] = {
            **tests,
            "skewness": skew,
            "excess_kurtosis": excess_kurt,
            "practically_gaussian": bool(
                abs(skew) < skew_threshold and abs(excess_kurt) < excess_kurtosis_threshold
            ),
        }

    scored = {k: v for k, v in per_column.items() if "skipped" not in v}

    def _rejections(test: str) -> int:
        return int(sum(bool(v[test]["reject"]) for v in scored.values()))

    return {
        "per_column": per_column,
        "summary": {
            "n_scored": len(scored),
            "rejections": {
                test: _rejections(test)
                for test in ("shapiro_wilk", "dagostino_k2", "jarque_bera", "anderson_darling")
            },
            "n_unanimous_reject": int(
                sum(v["unanimous_reject"] for v in scored.values())
            ),
            "n_practically_gaussian": int(
                sum(v["practically_gaussian"] for v in scored.values())
            ),
            "alpha": alpha,
            "skew_threshold": skew_threshold,
            "excess_kurtosis_threshold": excess_kurtosis_threshold,
        },
    }
