"""Exploratory data analysis for causal-bench datasets.

Public surface
--------------
distribution.normality_tests   -- Shapiro-Wilk, D'Agostino K², Jarque-Bera,
                                   Anderson-Darling per variable
distribution.outlier_summary   -- MAD modified-z + IQR counts per variable
distribution.marginal_report   -- normality_tests run over every column,
                                   with rejection-count summary (used for the
                                   before/after table in step 4)
copula.gaussian_copula_fit     -- fit QuantileTransformer per column
copula.gaussian_copula_transform  -- apply fitted transformer
copula.gaussian_copula_inverse -- invert the transform
hotelling.t2_series            -- Hotelling T² statistic per observation
hotelling.ucl                  -- F-distribution upper control limit
hotelling.exceedance_summary   -- # / fraction of points exceeding the UCL
figures.*                      -- figure-generating helpers (return Path)

Entry point
-----------
Run ``python scripts/eda_normality.py --dataset P_plus`` (or ``P_minus``)
for the full four-step analysis on the four-tank dataset.
"""

from __future__ import annotations

from . import figures
from .copula import gaussian_copula_fit, gaussian_copula_inverse, gaussian_copula_transform
from .distribution import marginal_report, normality_tests, outlier_summary
from .hotelling import exceedance_summary, t2_series, ucl

__all__ = [
    "normality_tests",
    "outlier_summary",
    "marginal_report",
    "gaussian_copula_fit",
    "gaussian_copula_transform",
    "gaussian_copula_inverse",
    "t2_series",
    "ucl",
    "exceedance_summary",
    "figures",
]
