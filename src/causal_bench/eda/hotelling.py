"""Hotelling's T^2 statistic, for confirming multivariate Gaussianity after
the copula transform.

T^2_k = (x_k - xbar)' S^-1 (x_k - xbar)

computed per observation against the sample mean and sample covariance of
the (post-transform) data itself -- this is the standard Phase-I / individual-
observations control-chart setup (each point is checked against the overall
estimated mean and covariance, not against a separately-held reference
sample). If the copula transform worked, T^2 should behave like a scaled
F-distributed statistic and only ~alpha of points should exceed the UCL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def t2_series(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Hotelling's T^2 per row against the sample mean/covariance of `data`
    itself. Returns a 1-D array, one value per observation."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("t2_series expects a 2-D (n_obs, n_vars) array")
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    inv_cov = np.linalg.pinv(cov)  # pinv: robust to near-singular covariance
    centered = x - mean
    # (n, p) @ (p, p) @ (p, n) diagonal, done without forming the full matrix
    t2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    return t2


def ucl(n: int, p: int, alpha: float = 0.05) -> float:
    """Upper control limit for individual-observations Hotelling's T^2.

    UCL = [p (n-1)(n+1)] / [n (n-p)] * F_{alpha; p, n-p}

    Standard Phase-I individual-observations formula (e.g. Montgomery,
    *Introduction to Statistical Quality Control*, Hotelling's T^2 control
    chart, individual observations case). `n` is the number of observations
    used to estimate mean/covariance, `p` the number of variables.
    """
    if n <= p:
        raise ValueError(f"need n > p for the UCL formula (got n={n}, p={p})")
    f_crit = stats.f.ppf(1 - alpha, p, n - p)
    return float((p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit)


def exceedance_summary(
    t2: np.ndarray, ucl_value: float
) -> dict[str, float | int]:
    exceed = t2 > ucl_value
    return {
        "n": int(t2.size),
        "ucl": float(ucl_value),
        "n_exceeding": int(np.sum(exceed)),
        "fraction_exceeding": float(np.mean(exceed)),
        "max_t2": float(np.max(t2)),
    }
