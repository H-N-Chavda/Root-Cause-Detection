"""Gaussian-copula-style marginal transform.

Each column is rank-transformed to its empirical CDF, then pushed through the
inverse standard normal CDF -- the standard "Gaussianization" / NORTA
approach, which is exactly what `sklearn.preprocessing.QuantileTransformer
(output_distribution="normal")` implements. One transformer is fit per
column (not one multivariate transformer) because the point of this step is
to Gaussianize each *marginal* independently; the dependence structure
between columns is deliberately left alone for the copula transform to
preserve, and is exactly what Hotelling's T^2 in the next step examines.

Caveat recorded here rather than hidden: a two-valued (PRBS-type) column
rank-transforms to a two-point mixture of normal quantiles, not to a
continuous Gaussian. It is a valid, invertible transform and T^2 remains
well-defined on the result, but normality tests on such a column will still
reject post-transform -- there is no marginal transform that makes a
two-valued variable look continuous. This is noted in the before/after
report, not silently smoothed over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

#: Transformer per column, keyed by column name.
CopulaTransformers = dict[str, QuantileTransformer]


def gaussian_copula_fit(
    frame: pd.DataFrame, random_state: int = 0
) -> CopulaTransformers:
    """Fits one QuantileTransformer(output_distribution='normal') per column.

    `n_quantiles` is set to the sample size (not sklearn's default cap of
    1000), so the empirical CDF has one quantile per observation rather than
    collapsing nearby points into shared bins. At the dataset sizes here
    (thousands of rows) this is cheap and matters: a tighter cap introduces
    tied values in the output, and Shapiro-Wilk in particular rejects on
    ties even when the pre-transform column was already continuous and
    well-behaved -- that would show up as a *worse* "after" verdict than
    "before" for reasons that have nothing to do with Gaussianity. For a
    two-valued column, n_quantiles this high still collapses to effectively
    2 output levels, which is correct for that column.
    """
    transformers: CopulaTransformers = {}
    for column in frame.columns:
        values = frame[[column]].to_numpy(dtype=float)
        n_quantiles = values.shape[0]
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=10**9,  # don't subsample; dataset sizes here are small
            random_state=random_state,
        )
        qt.fit(values)
        transformers[str(column)] = qt
    return transformers


def gaussian_copula_transform(
    frame: pd.DataFrame, transformers: CopulaTransformers
) -> pd.DataFrame:
    """Applies already-fitted per-column transformers. Column order and
    index are preserved; raises if a column has no fitted transformer."""
    missing = [c for c in frame.columns if str(c) not in transformers]
    if missing:
        raise KeyError(f"no fitted transformer for columns: {missing}")
    out = {}
    for column in frame.columns:
        qt = transformers[str(column)]
        values = frame[[column]].to_numpy(dtype=float)
        out[column] = qt.transform(values).ravel()
    return pd.DataFrame(out, index=frame.index)


def gaussian_copula_inverse(
    frame: pd.DataFrame, transformers: CopulaTransformers
) -> pd.DataFrame:
    """Inverts the transform column-by-column, back to original units."""
    missing = [c for c in frame.columns if str(c) not in transformers]
    if missing:
        raise KeyError(f"no fitted transformer for columns: {missing}")
    out = {}
    for column in frame.columns:
        qt = transformers[str(column)]
        values = frame[[column]].to_numpy(dtype=float)
        out[column] = qt.inverse_transform(values).ravel()
    return pd.DataFrame(out, index=frame.index)


def binary_columns(frame: pd.DataFrame, max_unique: int = 2) -> list[str]:
    """Columns with at most `max_unique` distinct values -- used by the
    report/figures to flag PRBS-type inputs that will fail normality tests
    both before and (less dramatically) after the copula transform."""
    return [
        str(c) for c in frame.columns if frame[c].nunique(dropna=True) <= max_unique
    ]
