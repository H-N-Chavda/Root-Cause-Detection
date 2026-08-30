"""Numerical health of the covariance structure, and sample-size adequacy.

Every constraint-based method here rests on partial correlations, and a partial
correlation is computed by inverting a sub-matrix of the correlation matrix. If
that matrix is ill conditioned the inverse amplifies estimation noise and the
conditional independence tests stop being reliable in a way no p-value reveals.
This stage measures that directly.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from ..config import ConditioningConfig, TemporalConfig


def collinearity(frame: pd.DataFrame, cfg: ConditioningConfig) -> dict[str, Any]:
    """Condition number, rank, eigenvalues, VIF, and the high-correlation pairs."""
    columns = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues = np.sort(np.linalg.eigvalsh(corr))
    smallest = eigenvalues[:5]
    largest = eigenvalues[-1]
    # Ratio of extreme eigenvalues: the standard 2-norm condition number of a
    # symmetric positive semi-definite matrix.
    condition_number = (
        float(largest / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")
    )
    rank = int(np.linalg.matrix_rank(corr))

    pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(len(columns)), 2):
        value = float(corr[i, j])
        if abs(value) >= cfg.report_corr_threshold:
            pairs.append(
                {
                    "a": columns[i],
                    "b": columns[j],
                    "correlation": value,
                    "above_high_threshold": bool(abs(value) >= cfg.high_corr_threshold),
                }
            )
    pairs.sort(key=lambda p: abs(float(p["correlation"])), reverse=True)

    vif = _vif(values, columns, cfg)

    return {
        "condition_number": condition_number,
        "condition_number_threshold": cfg.condition_number_threshold,
        "ill_conditioned": bool(condition_number > cfg.condition_number_threshold),
        "numerical_rank": rank,
        "full_rank": bool(rank == len(columns)),
        "n_variables": len(columns),
        "smallest_eigenvalues": [float(v) for v in smallest],
        "largest_eigenvalue": float(largest),
        "min_eigenvalue": float(eigenvalues[0]),
        "high_correlation_pairs": pairs,
        "report_corr_threshold": cfg.report_corr_threshold,
        "high_corr_threshold": cfg.high_corr_threshold,
        "n_pairs_above_report_threshold": len(pairs),
        "n_pairs_above_high_threshold": int(
            sum(bool(p["above_high_threshold"]) for p in pairs)
        ),
        "max_abs_off_diagonal_correlation": float(
            np.max(np.abs(corr - np.eye(len(columns))))
        ),
        "vif": vif,
    }


def _vif(values: np.ndarray, columns: list[str], cfg: ConditioningConfig) -> dict[str, Any]:
    """Variance inflation factor per variable.

    statsmodels wants the constant included, otherwise the VIFs are computed
    against a through-the-origin regression and come out inflated.
    """
    design = add_constant(values, has_constant="add")
    scores: dict[str, float] = {}
    for position, column in enumerate(columns):
        try:
            scores[column] = float(variance_inflation_factor(design, position + 1))
        except Exception:
            scores[column] = float("inf")
    finite = [v for v in scores.values() if np.isfinite(v)]
    above = [c for c, v in scores.items() if v > cfg.vif_threshold]
    return {
        "per_variable": scores,
        "threshold": cfg.vif_threshold,
        "n_above_threshold": len(above),
        "columns_above_threshold": above,
        "max": float(max(scores.values())) if scores else None,
        "median": float(np.median(finite)) if finite else None,
        "n_infinite": int(sum(not np.isfinite(v) for v in scores.values())),
    }


def effective_sample_size(
    frame: pd.DataFrame, temporal_cfg: TemporalConfig
) -> dict[str, Any]:
    """Sample size after discounting serial correlation.

    Two estimates, because they answer different questions:

    * Per variable, the AR(1) form  n_eff = n (1 - rho1) / (1 + rho1), the
      standard correction for the variance of a mean under serial correlation.
    * Per pair, Bartlett's formula  n_eff = n / (1 + 2 sum_k rho_x(k) rho_y(k)),
      which is the one that governs the variance of a *correlation* estimate --
      and a correlation is what the Fisher-z test actually studentises.

    Neither is in scipy or statsmodels as a public function, so both are
    written out here.
    """
    values = frame.to_numpy(dtype=float)
    columns = [str(c) for c in frame.columns]
    n_rows, n_vars = values.shape
    max_lag = min(temporal_cfg.max_lag, n_rows // 4)

    centred = values - values.mean(axis=0)
    scale = np.sum(centred**2, axis=0)
    autocorr = np.zeros((n_vars, max_lag + 1))
    autocorr[:, 0] = 1.0
    for lag in range(1, max_lag + 1):
        numerator = np.sum(centred[lag:] * centred[:-lag], axis=0)
        autocorr[:, lag] = np.divide(
            numerator, scale, out=np.zeros(n_vars), where=scale > 0
        )

    per_variable: dict[str, Any] = {}
    for position, column in enumerate(columns):
        rho1 = float(autocorr[position, 1])
        # rho1 -> 1 sends this to zero; clip so a near-unit-root series reports
        # "one effective observation", not a negative count.
        factor = (1.0 - rho1) / (1.0 + rho1) if rho1 > -1.0 else 1.0
        per_variable[column] = {
            "acf_lag_1": rho1,
            "n_eff_ar1": float(max(1.0, n_rows * factor)),
        }

    pairwise = []
    for i, j in combinations(range(n_vars), 2):
        overlap = float(np.sum(autocorr[i, 1 : max_lag + 1] * autocorr[j, 1 : max_lag + 1]))
        denominator = 1.0 + 2.0 * overlap
        pairwise.append(
            max(1.0, n_rows / denominator) if denominator > 0 else float(n_rows)
        )

    ar1 = [v["n_eff_ar1"] for v in per_variable.values()]
    return {
        "n_rows": n_rows,
        "max_lag_used": max_lag,
        "per_variable": per_variable,
        "min_n_eff_ar1": float(min(ar1)),
        "median_n_eff_ar1": float(np.median(ar1)),
        "max_n_eff_ar1": float(max(ar1)),
        "min_n_eff_bartlett": float(min(pairwise)),
        "median_n_eff_bartlett": float(np.median(pairwise)),
        "max_n_eff_bartlett": float(max(pairwise)),
        # The binding number for a conditional independence test is the worst
        # pair, not the average one.
        "binding_n_eff": float(min(pairwise)),
        "shrinkage_factor": float(min(pairwise) / n_rows),
    }


def sample_size_adequacy(
    n_vars: int,
    n_eff: float,
    n_rows: int,
    cfg: ConditioningConfig,
) -> dict[str, Any]:
    """Largest conditioning set the sample supports, at raw and effective n.

    Two bounds:

    * Hard: the Fisher-z degrees of freedom are n - |Z| - 3, so |Z| < n - 3 or
      the test is undefined. This is a floor, not a recommendation.
    * Practical: a partial correlation on |Z| conditioning variables fits
      |Z| + 2 parameters, and `samples_per_parameter` observations per
      parameter are required for the estimate to be stable. That gives
      |Z| <= n / samples_per_parameter - 2.

    Both are evaluated at the raw row count and at the effective sample size;
    the effective one is the honest answer on autocorrelated data.
    """

    def bounds(n: float) -> dict[str, Any]:
        hard = int(max(0, np.floor(n - 4)))
        practical = int(max(0, np.floor(n / cfg.samples_per_parameter - 2)))
        return {
            "n": float(n),
            "hard_max_conditioning_set": hard,
            "practical_max_conditioning_set": practical,
        }

    at_raw = bounds(n_rows)
    at_eff = bounds(n_eff)
    # PC's skeleton search can condition on up to n_vars - 2 variables; there is
    # no point reporting a bound above that.
    ceiling = max(0, n_vars - 2)

    return {
        "samples_per_parameter": cfg.samples_per_parameter,
        "graph_ceiling": ceiling,
        "at_raw_n": at_raw,
        "at_effective_n": at_eff,
        "recommended_max_conditioning_set": int(
            min(ceiling, at_eff["practical_max_conditioning_set"])
        ),
        "note": "recommended = min(n_vars - 2, effective-n practical bound)",
    }


def analyse(
    frame: pd.DataFrame,
    cfg: ConditioningConfig,
    temporal_cfg: TemporalConfig,
) -> dict[str, Any]:
    """Whole conditioning stage."""
    coll = collinearity(frame, cfg)
    ess = effective_sample_size(frame, temporal_cfg)
    adequacy = sample_size_adequacy(
        n_vars=frame.shape[1],
        n_eff=ess["binding_n_eff"],
        n_rows=ess["n_rows"],
        cfg=cfg,
    )
    return {
        "collinearity": coll,
        "effective_sample_size": ess,
        "sample_size_adequacy": adequacy,
    }
