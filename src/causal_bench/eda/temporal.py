"""Serial dependence, stationarity, regime stability, lag order, delays.

This stage decides more about algorithm choice than any other. Plain PC treats
rows as independent draws; if the autocorrelation is high, the effective sample
size is a fraction of the row count, the Fisher-z test's nominal p-values are
anti-conservative, and PC over-connects. PCMCI+ and VAR-LiNGAM model the lag
structure explicitly but assume stationarity, which the regime checks probe.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from ..config import TemporalConfig


def serial_dependence(
    frame: pd.DataFrame, cfg: TemporalConfig, alpha: float
) -> dict[str, Any]:
    """ACF and PACF to `max_lag`, the decorrelation lag, and Ljung-Box.

    `decorrelation_lag` is the first lag at which |ACF| drops inside the white
    noise band +-z/sqrt(n). It is the number that decides whether plain PC is
    applicable: it says roughly how far apart two rows must be before they can
    be treated as independent draws.
    """
    n_rows = len(frame)
    # Bartlett's band for a white-noise null. statsmodels can return confidence
    # intervals, but they widen with lag under its default (non-white-noise)
    # assumption; the flat band is the one that answers "is this lag distinct
    # from white noise", which is the question here.
    band = float(stats.norm.ppf(1 - alpha / 2) / np.sqrt(n_rows))
    max_lag = min(cfg.max_lag, n_rows // 2 - 1)

    per_column: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column].to_numpy(dtype=float)
        if np.std(series) == 0.0:
            per_column[str(column)] = {"skipped": "constant series"}
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(series, nlags=max_lag, fft=True)
            pacf_values = pacf(series, nlags=min(max_lag, n_rows // 2 - 2))
            lb = acorr_ljungbox(
                series, lags=[min(cfg.ljung_box_lags, max_lag)], return_df=True
            )

        inside = np.where(np.abs(acf_values[1:]) < band)[0]
        decorrelation_lag = int(inside[0] + 1) if inside.size else None

        per_column[str(column)] = {
            "acf": [float(v) for v in acf_values],
            "pacf": [float(v) for v in pacf_values],
            "acf_lag_1": float(acf_values[1]),
            "acf_lag_10": float(acf_values[10]) if len(acf_values) > 10 else None,
            "acf_lag_50": float(acf_values[50]) if len(acf_values) > 50 else None,
            # None means the ACF never entered the band within max_lag.
            "decorrelation_lag": decorrelation_lag,
            "significance_band": band,
            "ljung_box": {
                "lags": int(lb.index[-1]),
                "statistic": float(lb["lb_stat"].iloc[-1]),
                "p_value": float(lb["lb_pvalue"].iloc[-1]),
                "reject_independence": bool(lb["lb_pvalue"].iloc[-1] < alpha),
            },
        }

    scored = [v for v in per_column.values() if "skipped" not in v]
    lag1 = [abs(v["acf_lag_1"]) for v in scored]
    known = [v["decorrelation_lag"] for v in scored if v["decorrelation_lag"] is not None]

    return {
        "max_lag": max_lag,
        "significance_band": band,
        "per_column": per_column,
        "summary": {
            "n_scored": len(scored),
            "max_abs_acf_lag_1": max(lag1) if lag1 else None,
            "mean_abs_acf_lag_1": float(np.mean(lag1)) if lag1 else None,
            "min_abs_acf_lag_1": min(lag1) if lag1 else None,
            "n_never_decorrelated": len(scored) - len(known),
            "max_decorrelation_lag": max(known) if known else None,
            "median_decorrelation_lag": float(np.median(known)) if known else None,
            "n_ljung_box_rejections": int(
                sum(v["ljung_box"]["reject_independence"] for v in scored)
            ),
        },
    }


def stationarity(frame: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """ADF and KPSS on every column.

    The two carry opposite nulls -- ADF's null is a unit root, KPSS's null is
    stationarity -- so agreement is a strong signal and disagreement means the
    evidence is genuinely ambiguous (a near-unit-root or a trend-stationary
    series). Disagreements are flagged rather than resolved.
    """
    per_column: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column].to_numpy(dtype=float)
        if np.std(series) == 0.0:
            per_column[str(column)] = {"skipped": "constant series"}
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p = adfuller(series, autolag="AIC")[:2]
            kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]

        adf_stationary = bool(adf_p < alpha)
        kpss_stationary = bool(kpss_p >= alpha)
        per_column[str(column)] = {
            "adf": {
                "statistic": float(adf_stat),
                "p_value": float(adf_p),
                "stationary": adf_stationary,
                "null": "unit root",
            },
            "kpss": {
                "statistic": float(kpss_stat),
                # statsmodels clips the KPSS p-value to its lookup table's
                # range, so 0.01 and 0.1 mean "at most" and "at least".
                "p_value": float(kpss_p),
                "p_value_is_bound": bool(kpss_p in (0.01, 0.1)),
                "stationary": kpss_stationary,
                "null": "stationarity",
            },
            "agree": adf_stationary == kpss_stationary,
            "verdict": (
                "stationary"
                if adf_stationary and kpss_stationary
                else "non-stationary"
                if not adf_stationary and not kpss_stationary
                else "ambiguous"
            ),
        }

    scored = [v for v in per_column.values() if "skipped" not in v]
    return {
        "alpha": alpha,
        "per_column": per_column,
        "summary": {
            "n_scored": len(scored),
            "n_adf_stationary": int(sum(v["adf"]["stationary"] for v in scored)),
            "n_kpss_stationary": int(sum(v["kpss"]["stationary"] for v in scored)),
            "n_agree": int(sum(v["agree"] for v in scored)),
            "n_disagree": int(sum(not v["agree"] for v in scored)),
            "disagreeing_columns": [
                c for c, v in per_column.items() if "skipped" not in v and not v["agree"]
            ],
            "n_both_stationary": int(sum(v["verdict"] == "stationary" for v in scored)),
            "n_both_non_stationary": int(
                sum(v["verdict"] == "non-stationary" for v in scored)
            ),
        },
    }


def regime_stability(frame: pd.DataFrame, cfg: TemporalConfig) -> dict[str, Any]:
    """Split-sample mean and variance comparison, plus changepoint detection.

    Means are compared in units of the whole series' standard deviation, so the
    number is a scale-free effect size rather than a p-value that any split of
    1499 autocorrelated rows would reject.
    """
    values = frame.to_numpy(dtype=float)
    columns = [str(c) for c in frame.columns]
    n_rows = values.shape[0]

    def _segment_stats(splits: int) -> dict[str, Any]:
        bounds = [int(round(k * n_rows / splits)) for k in range(splits + 1)]
        per_column: dict[str, dict[str, Any]] = {}
        for position, column in enumerate(columns):
            series = values[:, position]
            overall_sd = float(np.std(series, ddof=1))
            if overall_sd == 0.0:
                continue
            means = [
                float(np.mean(series[a:b]))
                for a, b in zip(bounds, bounds[1:], strict=False)
            ]
            variances = [
                float(np.var(series[a:b], ddof=1))
                for a, b in zip(bounds, bounds[1:], strict=False)
            ]
            per_column[column] = {
                "segment_means": means,
                "segment_variances": variances,
                "max_mean_shift_sd": float((max(means) - min(means)) / overall_sd),
                # Ratio rather than difference: variance scale is arbitrary.
                "variance_ratio": float(max(variances) / min(variances))
                if min(variances) > 0
                else float("inf"),
            }
        shifts: list[float] = [float(v["max_mean_shift_sd"]) for v in per_column.values()]
        return {
            "n_segments": splits,
            "boundaries": bounds,
            "per_column": per_column,
            "max_mean_shift_sd": max(shifts) if shifts else None,
            "mean_mean_shift_sd": float(np.mean(shifts)) if shifts else None,
            "n_columns_shifted": int(
                sum(s > cfg.regime_shift_sd_threshold for s in shifts)
            ),
        }

    changepoints = {
        column: _binary_segmentation(values[:, position], cfg)
        for position, column in enumerate(columns)
    }
    counts = [len(v) for v in changepoints.values()]

    return {
        "shift_threshold_sd": cfg.regime_shift_sd_threshold,
        "halves": _segment_stats(2),
        "thirds": _segment_stats(3),
        "changepoints": {
            "penalty": cfg.changepoint_penalty,
            "min_size": cfg.changepoint_min_size,
            "max_breaks": cfg.changepoint_max_breaks,
            "per_column": changepoints,
            "n_columns_with_changepoints": int(sum(c > 0 for c in counts)),
            "total_changepoints": int(sum(counts)),
            "max_changepoints_in_a_column": max(counts) if counts else 0,
        },
    }


def _binary_segmentation(series: np.ndarray, cfg: TemporalConfig) -> list[int]:
    """Changepoint locations by binary segmentation on a Gaussian cost.

    Custom because no allowed dependency provides changepoint detection: scipy,
    statsmodels and scikit-learn all lack it, and pulling in `ruptures` for one
    routine is not worth a new dependency. The cost of a segment is its
    Gaussian negative log-likelihood at the maximum-likelihood variance,
    n/2 * log(sigma^2); a split is accepted when it reduces the total cost by
    more than `changepoint_penalty`. That is the standard binary-segmentation
    criterion with a fixed penalty.
    """
    series = np.asarray(series, dtype=float)

    def cost(a: int, b: int) -> float:
        segment = series[a:b]
        if segment.size < 2:
            return 0.0
        variance = float(np.var(segment))
        if variance <= 0:
            return 0.0
        return 0.5 * segment.size * np.log(variance)

    found: list[int] = []
    segments = [(0, len(series))]
    while len(found) < cfg.changepoint_max_breaks and segments:
        best = None
        for a, b in segments:
            if b - a < 2 * cfg.changepoint_min_size:
                continue
            base = cost(a, b)
            for split in range(a + cfg.changepoint_min_size, b - cfg.changepoint_min_size):
                gain = base - (cost(a, split) + cost(split, b))
                if gain > cfg.changepoint_penalty and (best is None or gain > best[0]):
                    best = (gain, split, a, b)
        if best is None:
            break
        _, split, a, b = best
        found.append(int(split))
        segments = [s for s in segments if s != (a, b)] + [(a, split), (split, b)]
    return sorted(found)


def lag_order(frame: pd.DataFrame, cfg: TemporalConfig) -> dict[str, Any]:
    """VAR order by AIC, BIC, HQIC and FPE.

    These routinely disagree: AIC and FPE are efficiency criteria and pick a
    larger order, BIC and HQIC are consistent criteria and pick a smaller one.
    All four are reported because the choice of `tau_max` for the lagged
    algorithms is a modelling decision, not a fact about the data.
    """
    numeric = frame.select_dtypes(include=[np.number]).dropna()
    max_lag = min(cfg.var_max_lag, max(1, len(numeric) // (5 * numeric.shape[1])))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(numeric.to_numpy(dtype=float))
            selection = model.select_order(maxlags=max_lag)
            chosen = {k: int(v) for k, v in selection.selected_orders.items()}
    except Exception as exc:
        return {"error": str(exc), "max_lag_searched": max_lag}

    return {
        "max_lag_searched": max_lag,
        "selected": chosen,
        "criteria_agree": len(set(chosen.values())) == 1,
        "min_order": min(chosen.values()),
        "max_order": max(chosen.values()),
        "recommended_tau_max": max(chosen.values()),
    }


def _lagged_correlation(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Pearson correlation between `x` and `y` shifted by `lag`.

    `lag > 0` compares x[t] with y[t + lag], so a large value means x leads y.
    Each lag is correlated on its own overlapping window rather than divided by
    a fixed length: the fixed-length form is biased toward zero at long lags and
    can exceed 1 at short ones, neither of which is wanted when the number is
    read as a correlation.
    """
    if lag > 0:
        u, v = x[: len(x) - lag], y[lag:]
    elif lag < 0:
        u, v = x[-lag:], y[: len(y) + lag]
    else:
        u, v = x, y
    if u.size < 3:
        return 0.0
    u = u - u.mean()
    v = v - v.mean()
    denominator = np.sqrt(np.dot(u, u) * np.dot(v, v))
    return float(np.dot(u, v) / denominator) if denominator > 0 else 0.0


def cross_correlation(frame: pd.DataFrame, cfg: TemporalConfig) -> dict[str, Any]:
    """Peak cross-correlation and its lag, for every unordered pair.

    The peak lag estimates the propagation delay between two process variables
    and is the input to a later ground-truth-free validation step: a claimed
    edge a -> b should not have its peak at a lag saying b leads a.

    Sign convention: `peak_lag > 0` means `a` leads `b` by that many samples.
    """
    values = frame.to_numpy(dtype=float)
    columns = [str(c) for c in frame.columns]
    n_rows = values.shape[0]
    max_lag = min(cfg.xcorr_max_lag, n_rows // 4)
    lags = list(range(-max_lag, max_lag + 1))

    pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(len(columns)), 2):
        x, y = values[:, i], values[:, j]
        ccf = np.array([_lagged_correlation(x, y, lag) for lag in lags])
        peak = int(np.argmax(np.abs(ccf)))
        pairs.append(
            {
                "a": columns[i],
                "b": columns[j],
                "peak_lag": int(lags[peak]),
                "peak_value": float(ccf[peak]),
                "lag_0_value": float(ccf[max_lag]),
            }
        )

    nonzero = [p for p in pairs if p["peak_lag"] != 0]
    # A peak sitting on the boundary means the true delay may be longer than the
    # window searched, so the estimate is a lower bound, not a measurement.
    at_edge = [p for p in pairs if abs(p["peak_lag"]) == max_lag]
    return {
        "max_lag": max_lag,
        "n_peak_at_window_edge": len(at_edge),
        "peak_at_edge_warning": bool(at_edge),
        "sign_convention": "peak_lag > 0 means `a` leads `b`",
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_peak_at_nonzero_lag": len(nonzero),
        "fraction_peak_at_nonzero_lag": len(nonzero) / len(pairs) if pairs else 0.0,
        "max_abs_peak_lag": max((abs(p["peak_lag"]) for p in pairs), default=0),
        "median_abs_peak_lag": float(np.median([abs(p["peak_lag"]) for p in pairs]))
        if pairs
        else None,
        "strongest_delayed_pairs": sorted(
            nonzero, key=lambda p: abs(p["peak_value"]), reverse=True
        )[:10],
    }


def analyse(frame: pd.DataFrame, cfg: TemporalConfig, alpha: float) -> dict[str, Any]:
    """Whole temporal stage."""
    return {
        "serial_dependence": serial_dependence(frame, cfg, alpha),
        "stationarity": stationarity(frame, alpha),
        "regime": regime_stability(frame, cfg),
        "lag_order": lag_order(frame, cfg),
        "cross_correlation": cross_correlation(frame, cfg),
    }
