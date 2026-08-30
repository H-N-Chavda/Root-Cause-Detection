"""Marginal distributions, normality, VAR-residual non-Gaussianity, outliers.

The distinction that matters for algorithm choice is between the *marginals*
and the *innovations*. PC's Fisher-z test cares about the marginals (really
about joint normality, of which marginal normality is a necessary condition).
VAR-LiNGAM cares about neither: it identifies the contemporaneous structure
from the non-Gaussianity of the VAR residuals. A variable can look
non-Gaussian and leave Gaussian residuals, and the reverse also happens, so
both are computed separately here.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, stats
from statsmodels.tsa.api import VAR

from ..config import DistributionConfig, NonGaussianityConfig

#: E{G(v)} for v ~ N(0,1) and G(u) = log cosh(u). Hyvarinen's negentropy
#: approximation needs this constant; it has no closed form, so it is
#: integrated numerically once at import. Deterministic, ~1e-11 accurate.
_E_G_GAUSSIAN: float = integrate.quad(
    lambda u: np.log(np.cosh(u)) * stats.norm.pdf(u), -12, 12
)[0]


#: Scaling constant for the log-cosh contrast, from Hyvarinen (1998), "New
#: approximations of differential entropy for independent component analysis
#: and projection pursuit". Without it the contrast is a bare squared
#: difference whose numeric scale means nothing; with it the estimate lands in
#: nats, on the same scale as the true negentropy (a unit-variance Laplace has
#: true negentropy 0.072 nats and is estimated at 0.106 here).
_NEGENTROPY_SCALE: float = 79.047


def negentropy(sample: np.ndarray) -> float:
    """Hyvarinen's robust negentropy approximation, in nats.

        J(y) ~= k [ E{G(y)} - E{G(v)} ]^2,   G(u) = log cosh(u),  v ~ N(0,1)

    This is (up to the choice of G) the contrast function FastICA maximises, so
    it is the quantity that decides whether an ICA-based method such as
    VAR-LiNGAM can separate the sources at all. It is exactly zero for a
    Gaussian and positive otherwise.

    It is an approximation, and a rough one away from the heavy-tailed case the
    log-cosh contrast is tuned for: on unit-variance references it returns
    0.106 against a true 0.072 for a Laplace, but 0.057 against a true 0.177
    for a uniform. It is used here to separate "effectively Gaussian" from "not"
    rather than to quantify an entropy, and that separation is wide -- a
    Gaussian returns ~1e-5, every non-Gaussian reference above returns >0.05.

    Written out rather than taken from a library because scikit-learn exposes
    this contrast only inside `FastICA`, with no public function returning the
    negentropy value itself.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    std = x.std(ddof=1)
    if std == 0.0:
        return 0.0
    # G is only calibrated against a standard normal, so y must be standardised.
    y = (x - x.mean()) / std
    contrast = np.mean(np.log(np.cosh(y))) - _E_G_GAUSSIAN
    return float(_NEGENTROPY_SCALE * contrast**2)


def _moments(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().to_numpy(dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        # Undefined at a zero mean; reported as NaN rather than a huge number.
        "cv": abs(std / mean) if mean != 0.0 else float("nan"),
        "skewness": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "min": float(np.min(values)),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(np.max(values)),
        "iqr": q3 - q1,
        "range": float(np.max(values) - np.min(values)),
    }


def normality_tests(sample: np.ndarray, alpha: float) -> dict[str, Any]:
    """Shapiro-Wilk, D'Agostino K^2, Jarque-Bera and Anderson-Darling.

    All four are reported because they weight departures differently:
    Shapiro-Wilk is an omnibus order-statistic test, D'Agostino and Jarque-Bera
    key on the third and fourth moments, Anderson-Darling weights the tails.
    Disagreement between them localises the departure, so it is informative and
    is deliberately not collapsed into one verdict.
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
        # anderson() returns critical values, not a p-value. Pick the tabulated
        # level closest to alpha and report which one was used.
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
    out["unanimous"] = out["n_reject"] in (0, 4)
    return out


def _outliers(values: np.ndarray, cfg: DistributionConfig) -> dict[str, Any]:
    """Counts only. Nothing is removed; that is a modelling decision, not an
    EDA one, and removing rows would break the time index every later stage
    depends on."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        # 0.6745 is the 0.75 quantile of the standard normal: it rescales the
        # MAD to be a consistent estimator of sigma under normality.
        modified_z = 0.6745 * (values - median) / mad
        n_mz = int(np.sum(np.abs(modified_z) > cfg.outlier_modified_z))
        max_mz = float(np.max(np.abs(modified_z)))
    else:
        n_mz, max_mz = 0, 0.0

    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    low = q1 - cfg.outlier_iqr_multiplier * iqr
    high = q3 + cfg.outlier_iqr_multiplier * iqr
    n_iqr = int(np.sum((values < low) | (values > high)))

    return {
        "modified_z_count": n_mz,
        "modified_z_fraction": n_mz / values.size,
        "max_abs_modified_z": max_mz,
        "iqr_count": n_iqr,
        "iqr_fraction": n_iqr / values.size,
        "mad": mad,
    }


def analyse_marginals(
    frame: pd.DataFrame,
    cfg: DistributionConfig,
    alpha: float,
    discrete_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Per-column moments, normality tests and outlier counts.

    The Gaussian verdict is driven by the *effect sizes* (|skew| and |excess
    kurtosis|) and only annotated with the p-values. At n in the thousands a
    normality test rejects on a departure far too small to affect a Fisher-z
    test, so a p-value alone cannot answer "is this Gaussian enough".
    """
    discrete_columns = discrete_columns or set()
    per_column: dict[str, Any] = {}

    for column in frame.columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size < 8 or np.std(values) == 0.0:
            per_column[str(column)] = {"skipped": "constant or too few observations"}
            continue

        moments = _moments(frame[column])
        tests = normality_tests(values, alpha)
        practically_gaussian = (
            abs(moments["skewness"]) < cfg.skew_threshold
            and abs(moments["excess_kurtosis"]) < cfg.excess_kurtosis_threshold
        )
        per_column[str(column)] = {
            **moments,
            "normality": tests,
            "outliers": _outliers(values, cfg),
            "is_discrete": str(column) in discrete_columns,
            # A discrete column has no meaningful continuous-density verdict.
            "practically_gaussian": (
                None if str(column) in discrete_columns else bool(practically_gaussian)
            ),
        }

    scored = [v for v in per_column.values() if "skipped" not in v]
    skews = [abs(v["skewness"]) for v in scored]
    kurts = [abs(v["excess_kurtosis"]) for v in scored]

    def _rejections(test: str) -> int:
        return int(sum(bool(v["normality"][test]["reject"]) for v in scored))

    return {
        "per_column": per_column,
        "summary": {
            "n_scored": len(scored),
            "max_abs_skewness": max(skews) if skews else None,
            "mean_abs_skewness": float(np.mean(skews)) if skews else None,
            "max_abs_excess_kurtosis": max(kurts) if kurts else None,
            "mean_abs_excess_kurtosis": float(np.mean(kurts)) if kurts else None,
            "rejections": {
                "shapiro_wilk": _rejections("shapiro_wilk"),
                "dagostino_k2": _rejections("dagostino_k2"),
                "jarque_bera": _rejections("jarque_bera"),
                "anderson_darling": _rejections("anderson_darling"),
            },
            "n_practically_gaussian": int(
                sum(v.get("practically_gaussian") is True for v in scored)
            ),
            "n_practically_non_gaussian": int(
                sum(v.get("practically_gaussian") is False for v in scored)
            ),
            "alpha": alpha,
            "skew_threshold": cfg.skew_threshold,
            "excess_kurtosis_threshold": cfg.excess_kurtosis_threshold,
        },
    }


def var_residual_nongaussianity(
    frame: pd.DataFrame, cfg: NonGaussianityConfig, alpha: float
) -> dict[str, Any]:
    """Fits a VAR and tests each equation's residuals for non-Gaussianity.

    This, not the marginal test above, is the quantity that decides whether
    VAR-LiNGAM's contemporaneous stage is identifiable. LiNGAM recovers the
    instantaneous mixing matrix by ICA on the innovations; ICA has no traction
    on Gaussian sources, because a rotation of independent Gaussians is again
    independent Gaussians. At most one Gaussian innovation is tolerable.

    Verdict rule: an equation counts as usable when |excess kurtosis| exceeds
    `abs_excess_kurtosis_threshold` AND its negentropy exceeds
    `negentropy_threshold`. The dataset counts as identifiable when at least
    `min_fraction_nongaussian` of the equations qualify. Both thresholds come
    from the config and are echoed in the output.
    """
    numeric = frame.select_dtypes(include=[np.number]).dropna()
    result: dict[str, Any] = {
        "var_lag": cfg.var_lag,
        "abs_excess_kurtosis_threshold": cfg.abs_excess_kurtosis_threshold,
        "negentropy_threshold": cfg.negentropy_threshold,
        "min_fraction_nongaussian": cfg.min_fraction_nongaussian,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VAR(numeric.to_numpy(dtype=float)).fit(cfg.var_lag)
            residuals = np.asarray(fitted.resid, dtype=float)
    except Exception as exc:
        result.update({"fitted": False, "error": str(exc)})
        return result

    per_equation: dict[str, Any] = {}
    for position, column in enumerate(numeric.columns):
        series = residuals[:, position]
        excess_kurtosis = float(stats.kurtosis(series, fisher=True, bias=False))
        skewness = float(stats.skew(series, bias=False))
        jb_stat, jb_p = stats.jarque_bera(series)
        neg = negentropy(series)
        per_equation[str(column)] = {
            "excess_kurtosis": excess_kurtosis,
            "skewness": skewness,
            "jarque_bera": {
                "statistic": float(jb_stat),
                "p_value": float(jb_p),
                "reject": bool(jb_p < alpha),
            },
            "negentropy": neg,
            "usable_for_ica": bool(
                abs(excess_kurtosis) > cfg.abs_excess_kurtosis_threshold
                and neg > cfg.negentropy_threshold
            ),
        }

    kurts = [abs(v["excess_kurtosis"]) for v in per_equation.values()]
    negs = [v["negentropy"] for v in per_equation.values()]
    n_usable = int(sum(v["usable_for_ica"] for v in per_equation.values()))
    n_eq = len(per_equation)
    fraction = n_usable / n_eq if n_eq else 0.0

    result.update(
        {
            "fitted": True,
            "n_equations": n_eq,
            "n_residual_observations": int(residuals.shape[0]),
            "per_equation": per_equation,
            "mean_abs_excess_kurtosis": float(np.mean(kurts)) if kurts else None,
            "max_abs_excess_kurtosis": float(np.max(kurts)) if kurts else None,
            "mean_negentropy": float(np.mean(negs)) if negs else None,
            "max_negentropy": float(np.max(negs)) if negs else None,
            "n_usable_for_ica": n_usable,
            "fraction_usable_for_ica": fraction,
            "n_jarque_bera_rejections": int(
                sum(v["jarque_bera"]["reject"] for v in per_equation.values())
            ),
            "lingam_contemporaneous_identifiable": bool(
                fraction >= cfg.min_fraction_nongaussian
            ),
        }
    )
    return result


def analyse(
    frame: pd.DataFrame,
    dist_cfg: DistributionConfig,
    ng_cfg: NonGaussianityConfig,
    alpha: float,
    discrete_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Whole distribution stage."""
    return {
        "marginals": analyse_marginals(frame, dist_cfg, alpha, discrete_columns),
        "var_residuals": var_residual_nongaussianity(frame, ng_cfg, alpha),
    }
