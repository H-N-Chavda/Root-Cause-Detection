"""Figure-generating helpers. Every function saves a PNG and returns its Path.

Adapted from the Phase-2 module's figures.py (commit e8f02e8): same
histogram/QQ logic, but DPI is a plain int argument instead of a
FiguresConfig object, so this file has no config-system dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: safe to import in scripts with no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DEFAULT_DPI = 160


def _grid(n_panels: int) -> tuple[int, int]:
    ncols = min(4, n_panels) or 1
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def histograms_with_normal(
    frame: pd.DataFrame,
    out_dir: Path,
    filename: str = "histograms_with_normal.png",
    title: str = "Marginal histograms with fitted normal density",
    dpi: int = DEFAULT_DPI,
) -> Path:
    columns = list(frame.columns)
    nrows, ncols = _grid(len(columns))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns, strict=False):
        values = frame[column].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=40, density=True, alpha=0.65, color="#4C72B0", edgecolor="white")
        if values.std() > 0:
            xs = np.linspace(values.min(), values.max(), 200)
            ax.plot(
                xs,
                stats.norm.pdf(xs, values.mean(), values.std(ddof=1)),
                color="#C44E52",
                linewidth=2,
                label="fitted normal",
            )
        ax.set_title(str(column), fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    return _save(fig, out_dir / filename, dpi)


def qq_grid(
    frame: pd.DataFrame,
    out_dir: Path,
    filename: str = "qq_marginals.png",
    title: str = "Normal QQ plots (marginals)",
    dpi: int = DEFAULT_DPI,
) -> Path:
    columns = list(frame.columns)
    nrows, ncols = _grid(len(columns))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns, strict=False):
        values = frame[column].dropna().to_numpy(dtype=float)
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(str(column), fontsize=10)
        ax.get_lines()[0].set_markerfacecolor("#4C72B0")
        ax.get_lines()[0].set_markeredgecolor("none")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color("#C44E52")

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    return _save(fig, out_dir / filename, dpi)


def outlier_boxplot_grid(
    frame: pd.DataFrame,
    out_dir: Path,
    filename: str = "outlier_boxplots.png",
    title: str = "Per-variable distribution and IQR outlier fences",
    dpi: int = DEFAULT_DPI,
) -> Path:
    columns = list(frame.columns)
    fig, ax = plt.subplots(figsize=(1.4 * len(columns) + 2, 4.5))
    # Standardise for display only, so all 7 variables share one axis
    # without the binary {0,1}-scaled inputs flattening the tank-level boxes.
    standardized = (frame - frame.mean()) / frame.std(ddof=1).replace(0, 1)
    ax.boxplot(
        [standardized[c].dropna().to_numpy() for c in columns],
        tick_labels=columns,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#C44E52", "markeredgecolor": "none"},
    )
    ax.set_ylabel("standardized value (z-score)")
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    return _save(fig, out_dir / filename, dpi)


def t2_control_chart(
    t2: np.ndarray,
    ucl_value: float,
    out_dir: Path,
    filename: str = "t2_control_chart.png",
    title: str = "Hotelling's T\u00b2 control chart (post-copula-transform)",
    dpi: int = DEFAULT_DPI,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(t2.size)
    exceed = t2 > ucl_value
    ax.plot(idx, t2, color="#4C72B0", linewidth=0.8, label="T\u00b2")
    ax.scatter(idx[exceed], t2[exceed], color="#C44E52", s=10, zorder=3, label="exceeds UCL")
    ax.axhline(ucl_value, color="#C44E52", linestyle="--", linewidth=1.2, label=f"UCL = {ucl_value:.2f}")
    ax.set_xlabel("observation index")
    ax.set_ylabel("T\u00b2")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    return _save(fig, out_dir / filename, dpi)


def normality_before_after_chart(
    before_rejections: dict[str, int],
    after_rejections: dict[str, int],
    n_scored: int,
    out_dir: Path,
    filename: str = "normality_before_after.png",
    title: str = "Normality-test rejections, before vs after copula transform",
    dpi: int = DEFAULT_DPI,
) -> Path:
    tests = list(before_rejections.keys())
    x = np.arange(len(tests))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, [before_rejections[t] for t in tests], width, label="before", color="#C44E52")
    ax.bar(x + width / 2, [after_rejections[t] for t in tests], width, label="after", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in tests], rotation=20, ha="right")
    ax.set_ylabel(f"# variables rejecting normality (of {n_scored})")
    ax.set_title(title, fontsize=12)
    ax.legend()
    return _save(fig, out_dir / filename, dpi)
