"""Linear versus general dependence.

Two complementary views:

* Pairwise -- Pearson sees only linear structure, Spearman sees monotone
  structure, distance correlation and mutual information see any dependence.
  A pair with low |Pearson| and high distance correlation carries structure
  that a correlation-based conditional independence test cannot see, and PC's
  Fisher-z test is exactly such a test.
* Per variable -- a linear model against all others, versus a gradient-boosted
  tree, scored out of sample. A large R^2 gap means the linear methods are
  leaving real signal on the table.

Every out-of-sample split here is time ordered: the last `test_fraction` of
rows. A random split on an autocorrelated series leaks the future into the
training set through neighbouring rows and inflates both scores.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import dcor
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import linear_reset

from ..config import LinearityConfig


def pairwise_dependence(
    frame: pd.DataFrame, cfg: LinearityConfig, seed: int
) -> dict[str, Any]:
    """Four dependence measures for every unordered pair of columns."""
    columns = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=float)
    n_vars = len(columns)

    mi = _mutual_information_matrix(values, cfg, seed)

    pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(n_vars), 2):
        x, y = values[:, i], values[:, j]
        pearson, pearson_p = stats.pearsonr(x, y)
        spearman, spearman_p = stats.spearmanr(x, y)
        # avl is the O(n log n) algorithm for univariate inputs; the naive
        # O(n^2) default would make 465 pairs an order of magnitude slower for
        # an identical answer.
        distance = float(dcor.distance_correlation(x, y, method="avl"))
        pairs.append(
            {
                "a": columns[i],
                "b": columns[j],
                "pearson": float(pearson),
                "pearson_p": float(pearson_p),
                "spearman": float(spearman),
                "spearman_p": float(spearman_p),
                "distance_correlation": distance,
                "mutual_information": float(mi[i, j]),
                # Distance correlation is on the same [0, 1] scale as |Pearson|
                # and equals zero only under independence, so their difference
                # is a direct read-out of dependence Pearson cannot see.
                "gap": distance - abs(float(pearson)),
                # Monotone-but-nonlinear structure shows up here instead.
                "spearman_pearson_gap": abs(float(spearman)) - abs(float(pearson)),
            }
        )

    ranked = sorted(pairs, key=lambda p: p["gap"], reverse=True)
    gaps = np.array([p["gap"] for p in pairs])
    return {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "most_nonlinear_pairs": ranked[: cfg.top_pairs],
        "gap_threshold": cfg.dcor_pearson_gap,
        "n_pairs_above_gap_threshold": int(np.sum(gaps > cfg.dcor_pearson_gap)),
        "fraction_above_gap_threshold": float(np.mean(gaps > cfg.dcor_pearson_gap)),
        "max_gap": float(gaps.max()) if gaps.size else None,
        "mean_gap": float(gaps.mean()) if gaps.size else None,
        "median_gap": float(np.median(gaps)) if gaps.size else None,
    }


def _mutual_information_matrix(
    values: np.ndarray, cfg: LinearityConfig, seed: int
) -> np.ndarray:
    """Symmetric kNN mutual-information matrix (nats).

    `mutual_info_regression` scores many features against one target in a
    single call, so this costs one call per variable rather than one per pair.
    The estimator is not exactly symmetric in its two arguments, so MI(i,j) and
    MI(j,i) are averaged.
    """
    n_vars = values.shape[1]
    matrix = np.zeros((n_vars, n_vars), dtype=float)
    for target in range(n_vars):
        others = [k for k in range(n_vars) if k != target]
        scores = mutual_info_regression(
            values[:, others],
            values[:, target],
            n_neighbors=cfg.mi_n_neighbors,
            random_state=seed,
        )
        for position, source in enumerate(others):
            matrix[source, target] = scores[position]
    return (matrix + matrix.T) / 2.0


def linear_versus_nonparametric(
    frame: pd.DataFrame, cfg: LinearityConfig, seed: int
) -> dict[str, Any]:
    """Per variable: OLS against all others versus a boosted tree, out of sample.

    The RESET test is run on a full-sample OLS fit (it is a specification test
    for the model as a whole), while the R^2 comparison uses the time-ordered
    holdout. Reporting both keeps an in-sample specification failure separate
    from an out-of-sample predictive gain.
    """
    columns = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=float)
    n_rows, n_vars = values.shape
    split = int(round(n_rows * (1.0 - cfg.test_fraction)))
    if split < 20 or n_rows - split < 10:
        return {"skipped": f"too few rows ({n_rows}) for a {cfg.test_fraction} holdout"}

    per_variable: dict[str, Any] = {}
    for target in range(n_vars):
        others = [k for k in range(n_vars) if k != target]
        x, y = values[:, others], values[:, target]
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        linear = LinearRegression().fit(x_train, y_train)
        r2_linear = float(r2_score(y_test, linear.predict(x_test)))

        # HistGradientBoosting rather than GradientBoosting: same model class,
        # binned and an order of magnitude faster at this width, and fully
        # deterministic once seeded.
        tree = HistGradientBoostingRegressor(
            max_iter=cfg.gbt_n_estimators,
            max_depth=cfg.gbt_max_depth,
            learning_rate=cfg.gbt_learning_rate,
            random_state=seed,
            early_stopping=False,
        ).fit(x_train, y_train)
        r2_tree = float(r2_score(y_test, tree.predict(x_test)))

        reset = _reset_test(x, y)

        per_variable[columns[target]] = {
            "r2_linear_oos": r2_linear,
            "r2_tree_oos": r2_tree,
            "r2_gap": r2_tree - r2_linear,
            "nonlinear_gain": bool(r2_tree - r2_linear > cfg.r2_gap_threshold),
            "ramsey_reset": reset,
        }

    gaps = [v["r2_gap"] for v in per_variable.values()]
    n_reset_reject = int(
        sum(bool(v["ramsey_reset"].get("reject")) for v in per_variable.values())
    )
    return {
        "n_train_rows": split,
        "n_test_rows": n_rows - split,
        "split": "time-ordered (last fraction held out)",
        "test_fraction": cfg.test_fraction,
        "r2_gap_threshold": cfg.r2_gap_threshold,
        "per_variable": per_variable,
        "mean_r2_gap": float(np.mean(gaps)),
        "max_r2_gap": float(np.max(gaps)),
        "median_r2_gap": float(np.median(gaps)),
        "n_with_nonlinear_gain": int(
            sum(v["nonlinear_gain"] for v in per_variable.values())
        ),
        "n_variables": n_vars,
        "n_reset_rejections": n_reset_reject,
    }


def _reset_test(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ramsey RESET on the full-sample OLS fit.

    Rejection means powers of the fitted values add explanatory power, i.e. the
    linear specification is misspecified. It is a specification test, not an
    effect size, so the R^2 gap above is what quantifies how much it costs.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(y, sm.add_constant(x)).fit()
            result = linear_reset(fit, power=2, test_type="fitted", use_f=True)
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "reject": bool(result.pvalue < 0.05),
            "power": 2,
        }
    except Exception as exc:
        return {"statistic": None, "p_value": None, "reject": None, "error": str(exc)}


def analyse(frame: pd.DataFrame, cfg: LinearityConfig, seed: int) -> dict[str, Any]:
    """Whole linearity stage."""
    return {
        "pairwise": pairwise_dependence(frame, cfg, seed),
        "predictive": linear_versus_nonparametric(frame, cfg, seed),
    }
