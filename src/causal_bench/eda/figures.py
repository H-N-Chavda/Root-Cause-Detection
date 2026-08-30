"""Figure generation.

Split out of `report` because plotting is the one part of the pipeline that
touches a global (matplotlib's backend and rcParams) and the one part that can
be switched off entirely via `eda.figures.enabled` without changing a number.

Every figure is derived from the findings mapping that has already been
computed, never from a fresh calculation, so a figure cannot disagree with the
report beside it. Grids are capped at `max_grid_vars` panels so a wide dataset
still produces something readable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend: this runs headless in CI and must never try to open
# a window. Selected before pyplot is imported, which is the only order that
# works.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ..config import FiguresConfig  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def _grid(n_panels: int) -> tuple[int, int]:
    columns = min(4, n_panels)
    rows = math.ceil(n_panels / columns)
    return rows, columns


def _save(fig: plt.Figure, path: Path, dpi: int) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path.name


def histograms_with_normal(frame: pd.DataFrame, out_dir: Path, cfg: FiguresConfig) -> str:
    """Histogram per column with the maximum-likelihood normal density on top.

    The overlay is the visual counterpart of the skew/kurtosis numbers: a
    departure too small to see here is too small to break a Fisher-z test.
    """
    columns = list(frame.columns)[: cfg.max_grid_vars]
    rows, cols = _grid(len(columns))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    # strict=False on purpose: the axes grid is padded to a full rectangle,
    # so there are usually more axes than columns and the tail is blanked below.
    for ax, column in zip(np.ravel(np.atleast_1d(axes)), columns, strict=False):
        values = frame[column].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=40, density=True, alpha=0.65, edgecolor="none")
        grid = np.linspace(values.min(), values.max(), 200)
        ax.plot(grid, stats.norm.pdf(grid, values.mean(), values.std(ddof=1)), lw=1.2)
        ax.set_title(str(column), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(np.atleast_1d(axes))[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Marginal histograms with fitted normal density", fontsize=11)
    return _save(fig, out_dir / "histograms_with_normal.png", cfg.dpi)


def qq_grid(
    frame: pd.DataFrame, out_dir: Path, cfg: FiguresConfig, filename: str, title: str
) -> str:
    """Normal QQ plot per column."""
    columns = list(frame.columns)[: cfg.max_grid_vars]
    rows, cols = _grid(len(columns))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows))
    # strict=False on purpose: the axes grid is padded to a full rectangle,
    # so there are usually more axes than columns and the tail is blanked below.
    for ax, column in zip(np.ravel(np.atleast_1d(axes)), columns, strict=False):
        values = frame[column].dropna().to_numpy(dtype=float)
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(str(column), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        ax.get_lines()[0].set_markersize(1.5)
    for ax in np.ravel(np.atleast_1d(axes))[len(columns) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    return _save(fig, out_dir / filename, cfg.dpi)


def dependence_gap_heatmap(
    findings: dict[str, Any], columns: list[str], out_dir: Path, cfg: FiguresConfig
) -> str:
    """Distance correlation minus |Pearson|, as a matrix.

    A bright cell is a pair whose dependence a correlation-based independence
    test cannot see.
    """
    index = {c: i for i, c in enumerate(columns)}
    matrix = np.zeros((len(columns), len(columns)))
    for pair in findings["linearity"]["pairwise"]["pairs"]:
        i, j = index[pair["a"]], index[pair["b"]]
        matrix[i, j] = matrix[j, i] = pair["gap"]

    fig, ax = plt.subplots(figsize=(0.32 * len(columns) + 3, 0.32 * len(columns) + 2))
    image = ax.imshow(matrix, cmap="magma", vmin=0)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=6)
    ax.set_yticklabels(columns, fontsize=6)
    fig.colorbar(image, ax=ax, shrink=0.8, label="dCor - |Pearson|")
    ax.set_title("Nonlinear dependence not visible to Pearson", fontsize=11)
    return _save(fig, out_dir / "dependence_gap_heatmap.png", cfg.dpi)


def nonlinear_pair_scatter(
    frame: pd.DataFrame, findings: dict[str, Any], out_dir: Path, cfg: FiguresConfig
) -> str:
    """Scatter plots for the pairs with the largest dCor-minus-Pearson gap."""
    pairs = findings["linearity"]["pairwise"]["most_nonlinear_pairs"]
    if not pairs:
        return ""
    rows, cols = _grid(len(pairs))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    for ax, pair in zip(np.ravel(np.atleast_1d(axes)), pairs, strict=False):
        ax.scatter(frame[pair["a"]], frame[pair["b"]], s=2, alpha=0.35)
        ax.set_xlabel(pair["a"], fontsize=8)
        ax.set_ylabel(pair["b"], fontsize=8)
        ax.set_title(
            f"r={pair['pearson']:.2f}  dCor={pair['distance_correlation']:.2f}"
            f"  gap={pair['gap']:.2f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    for ax in np.ravel(np.atleast_1d(axes))[len(pairs) :]:
        ax.axis("off")
    fig.suptitle("Most nonlinear pairs", fontsize=11)
    return _save(fig, out_dir / "nonlinear_pairs_scatter.png", cfg.dpi)


def acf_pacf_grid(findings: dict[str, Any], out_dir: Path, cfg: FiguresConfig) -> str:
    """ACF and PACF stems per column, with the white-noise band."""
    per_column = findings["temporal"]["serial_dependence"]["per_column"]
    columns = [c for c, v in per_column.items() if "skipped" not in v][: cfg.max_grid_vars]
    band = findings["temporal"]["serial_dependence"]["significance_band"]
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(9, 1.5 * len(columns)), squeeze=False
    )
    for row, column in enumerate(columns):
        for col, key in enumerate(("acf", "pacf")):
            values = per_column[column][key]
            ax = axes[row][col]
            ax.bar(range(len(values)), values, width=0.7)
            ax.axhline(band, ls="--", lw=0.6, color="crimson")
            ax.axhline(-band, ls="--", lw=0.6, color="crimson")
            ax.set_ylim(-1.05, 1.05)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(column, fontsize=7)
            if row == 0:
                ax.set_title(key.upper(), fontsize=9)
    fig.suptitle("Autocorrelation and partial autocorrelation", fontsize=11)
    return _save(fig, out_dir / "acf_pacf_grid.png", cfg.dpi)


def rolling_moments(
    frame: pd.DataFrame, out_dir: Path, cfg: FiguresConfig, window: int = 100
) -> str:
    """Rolling mean and variance, standardised so all columns share an axis.

    This is the picture behind the regime-stability numbers: a drifting band
    here is what the half/third split reports as a mean shift in SD units.
    """
    columns = list(frame.columns)[: cfg.max_grid_vars]
    standardised = (frame[columns] - frame[columns].mean()) / frame[columns].std(ddof=0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(standardised.rolling(window).mean().to_numpy(), lw=0.7)
    axes[0].set_ylabel(f"rolling mean\n(window {window}, SD units)", fontsize=8)
    axes[1].plot(standardised.rolling(window).var().to_numpy(), lw=0.7)
    axes[1].set_ylabel(f"rolling variance\n(window {window})", fontsize=8)
    axes[1].set_xlabel("row", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("Rolling first and second moments", fontsize=11)
    return _save(fig, out_dir / "rolling_moments.png", cfg.dpi)


def cross_correlation_heatmaps(
    findings: dict[str, Any], columns: list[str], out_dir: Path, cfg: FiguresConfig
) -> str:
    """Peak cross-correlation value and the lag at which it occurs."""
    index = {c: i for i, c in enumerate(columns)}
    n = len(columns)
    peak = np.zeros((n, n))
    lag = np.zeros((n, n))
    for pair in findings["temporal"]["cross_correlation"]["pairs"]:
        i, j = index[pair["a"]], index[pair["b"]]
        peak[i, j] = peak[j, i] = abs(pair["peak_value"])
        lag[i, j] = pair["peak_lag"]
        lag[j, i] = -pair["peak_lag"]
    np.fill_diagonal(peak, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(0.5 * n + 6, 0.3 * n + 3))
    for ax, matrix, title, cmap, kwargs in (
        (axes[0], peak, "|peak cross-correlation|", "viridis", {"vmin": 0, "vmax": 1}),
        (axes[1], lag, "lag of peak (rows lead columns)", "coolwarm", {}),
    ):
        image = ax.imshow(matrix, cmap=cmap, **kwargs)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(columns, rotation=90, fontsize=5)
        ax.set_yticklabels(columns, fontsize=5)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.7)
    return _save(fig, out_dir / "cross_correlation_heatmaps.png", cfg.dpi)


def build_all(
    frame: pd.DataFrame,
    findings: dict[str, Any],
    out_dir: Path,
    cfg: FiguresConfig,
) -> dict[str, str]:
    """Every figure. Returns {key: filename}; a failing figure is logged and
    skipped rather than aborting a run that has already produced its numbers."""
    if not cfg.enabled:
        log.info("figures disabled by config")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [str(c) for c in frame.columns]
    residuals = _residual_frame(frame, findings)

    jobs: list[tuple[str, Any]] = [
        ("histograms", lambda: histograms_with_normal(frame, out_dir, cfg)),
        (
            "qq_marginals",
            lambda: qq_grid(
                frame, out_dir, cfg, "qq_marginals.png", "Normal QQ plots (marginals)"
            ),
        ),
        (
            "dependence_gap_heatmap",
            lambda: dependence_gap_heatmap(findings, columns, out_dir, cfg),
        ),
        ("nonlinear_pairs", lambda: nonlinear_pair_scatter(frame, findings, out_dir, cfg)),
        ("acf_pacf", lambda: acf_pacf_grid(findings, out_dir, cfg)),
        ("rolling_moments", lambda: rolling_moments(frame, out_dir, cfg)),
        (
            "cross_correlation",
            lambda: cross_correlation_heatmaps(findings, columns, out_dir, cfg),
        ),
    ]
    if residuals is not None:
        jobs.insert(
            2,
            (
                "qq_residuals",
                lambda: qq_grid(
                    residuals,
                    out_dir,
                    cfg,
                    "qq_var_residuals.png",
                    "Normal QQ plots (VAR residuals)",
                ),
            ),
        )

    produced: dict[str, str] = {}
    for name, job in jobs:
        try:
            filename = job()
            if filename:
                produced[name] = filename
                log.info("figure: %s", filename)
        except Exception as exc:
            log.warning("figure %s failed: %s", name, exc)
    return produced


def _residual_frame(frame: pd.DataFrame, findings: dict[str, Any]) -> pd.DataFrame | None:
    """Re-fits the VAR to recover the residual series for plotting.

    The findings mapping stores residual *statistics*, not the series -- putting
    1498 x 31 floats in the JSON report would dominate the file for no
    analytical gain -- so the fit is repeated here. It is deterministic, so the
    plotted residuals are exactly the ones the statistics describe.
    """
    residual_findings = findings["distribution"]["var_residuals"]
    if not residual_findings.get("fitted"):
        return None
    try:
        import warnings

        from statsmodels.tsa.api import VAR

        numeric = frame.select_dtypes(include=[np.number]).dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VAR(numeric.to_numpy(dtype=float)).fit(residual_findings["var_lag"])
        return pd.DataFrame(np.asarray(fitted.resid), columns=list(numeric.columns))
    except Exception as exc:  # pragma: no cover
        log.warning("could not rebuild VAR residuals for plotting: %s", exc)
        return None
