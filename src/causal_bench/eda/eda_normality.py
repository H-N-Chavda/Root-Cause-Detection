#!/usr/bin/env python
"""Four-step EDA pipeline for the four-tank dataset.

    1. Gaussianity check  -- QQ plots, histograms, normality tests
    2. Outlier check      -- MAD modified-z + IQR, boxplot grid
    3. Copula transform   -- per-column QuantileTransformer -> Gaussian
    4. T^2 confirmation   -- Hotelling's T^2 control chart + before/after
                             normality-rejection table

Usage:
    python scripts/eda_normality.py --dataset P_plus
    python scripts/eda_normality.py --dataset P_minus --alpha 0.01

Output goes under a fresh results/eda-<timestamp>/ directory (via
causal_bench.paths.new_run_dir), never into the repo tree itself.

Note on dataset location: datasets/generated/qtank_*.csv live outside
causal_bench.paths' RAW_DATA_DIR convention (data/raw/), so this script
resolves the path directly from REPO_ROOT rather than through
paths.dataset_path() -- that helper's relative-path branch resolves against
the current working directory, which is fragile if the script isn't run
from the repo root. Resolving via REPO_ROOT avoids that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_bench import paths
from causal_bench.eda import (
    figures,
    gaussian_copula_fit,
    gaussian_copula_transform,
    marginal_report,
    outlier_summary,
    t2_series,
    ucl,
)
from causal_bench.eda.copula import binary_columns
from causal_bench.io.datasets import load_dataset


def _dataset_path(name: str) -> Path:
    path = paths.REPO_ROOT / "datasets" / "generated" / f"qtank_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"expected {path} (run datasets/generate_qtank_data.py first "
            "if it doesn't exist yet)"
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["P_plus", "P_minus"], default="P_plus",
        help="four-tank operating point to analyse",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument("--dpi", type=int, default=figures.DEFAULT_DPI)
    args = parser.parse_args()

    csv_path = _dataset_path(args.dataset)
    var_names, data, dropped = load_dataset(csv_path)
    print(f"Loaded {csv_path.name}: {data.shape[0]} rows x {len(var_names)} vars")
    if any(dropped.values()):
        print(f"  dropped columns: {dropped}")

    import pandas as pd

    frame = pd.DataFrame(data, columns=var_names)
    binary_cols = binary_columns(frame)
    print(f"  binary (PRBS-type) columns, expected to fail normality: {binary_cols}")

    run_dir = paths.new_run_dir(prefix="eda")
    fig_dir = run_dir / "figures"
    print(f"Writing output to {run_dir}")

    report: dict = {"dataset": args.dataset, "n_rows": int(frame.shape[0]), "columns": var_names,
                     "binary_columns": binary_cols, "alpha": args.alpha}

    # --- Step 1: Gaussianity check -----------------------------------
    print("\n[1/4] Gaussianity check")
    before_normality = marginal_report(frame, alpha=args.alpha)
    figures.histograms_with_normal(frame, fig_dir, filename="step1_histograms.png", dpi=args.dpi)
    figures.qq_grid(frame, fig_dir, filename="step1_qq.png", dpi=args.dpi)
    print(f"  rejections (of {before_normality['summary']['n_scored']} vars): "
          f"{before_normality['summary']['rejections']}")
    report["before_normality"] = before_normality["summary"]

    # --- Step 2: Outlier check -----------------------------------------
    print("\n[2/4] Outlier check")
    outliers = outlier_summary(frame)
    figures.outlier_boxplot_grid(frame, fig_dir, filename="step2_outlier_boxplots.png", dpi=args.dpi)
    print(f"  total modified-z outliers: {outliers['summary']['total_modified_z_outliers']}, "
          f"total IQR outliers: {outliers['summary']['total_iqr_outliers']}")
    report["outliers"] = outliers["summary"]

    # --- Step 3: Copula transform ---------------------------------------
    print("\n[3/4] Copula transform")
    transformers = gaussian_copula_fit(frame)
    transformed = gaussian_copula_transform(frame, transformers)
    figures.histograms_with_normal(transformed, fig_dir, filename="step3_histograms_transformed.png",
                                    title="Marginal histograms after copula transform", dpi=args.dpi)
    figures.qq_grid(transformed, fig_dir, filename="step3_qq_transformed.png",
                     title="Normal QQ plots after copula transform", dpi=args.dpi)

    # --- Step 4: T^2 confirmation ---------------------------------------
    print("\n[4/4] T^2 confirmation")
    after_normality = marginal_report(transformed, alpha=args.alpha)
    print(f"  rejections after transform: {after_normality['summary']['rejections']}")
    figures.normality_before_after_chart(
        before_normality["summary"]["rejections"],
        after_normality["summary"]["rejections"],
        before_normality["summary"]["n_scored"],
        fig_dir, filename="step4_normality_before_after.png", dpi=args.dpi,
    )

    t2 = t2_series(transformed)
    ucl_value = ucl(n=t2.size, p=transformed.shape[1], alpha=args.alpha)
    exceed_frac = float((t2 > ucl_value).mean())
    figures.t2_control_chart(t2, ucl_value, fig_dir, filename="step4_t2_control_chart.png", dpi=args.dpi)
    print(f"  UCL={ucl_value:.2f}, fraction exceeding={exceed_frac:.3f} "
          f"(expect ~{args.alpha} if copula-transformed data is well-behaved)")

    report["after_normality"] = after_normality["summary"]
    report["t2"] = {"ucl": ucl_value, "fraction_exceeding": exceed_frac, "n": int(t2.size)}

    (run_dir / "eda_summary.json").write_text(json.dumps(report, indent=2))
    _write_markdown_summary(report, run_dir / "EDA_SUMMARY.md")
    print(f"\nDone. Figures + summary in {run_dir}")


def _write_markdown_summary(report: dict, out_path: Path) -> None:
    b, a = report["before_normality"], report["after_normality"]
    t2 = report["t2"]
    lines = [
        f"# EDA summary -- {report['dataset']} ({report['n_rows']} rows, "
        f"{len(report['columns'])} vars, alpha={report['alpha']})",
        "",
        f"Binary (PRBS-type) columns, expected to fail normality regardless "
        f"of transform: {', '.join(report['binary_columns']) or 'none'}",
        "",
        "## Normality rejections (of {} variables)".format(b["n_scored"]),
        "",
        "| Test | Before | After |",
        "|---|---|---|",
    ]
    for test in b["rejections"]:
        lines.append(f"| {test.replace('_', ' ')} | {b['rejections'][test]} | {a['rejections'][test]} |")
    lines += [
        f"| **practically Gaussian** (\\|skew\\|<{b['skew_threshold']}, "
        f"\\|excess kurtosis\\|<{b['excess_kurtosis_threshold']}) "
        f"| {b['n_practically_gaussian']} | {a['n_practically_gaussian']} |",
        "",
        "At n in the thousands, formal normality tests reject on departures "
        "too small to matter -- the effect-size row above is the one that "
        "actually answers whether the copula transform worked.",
        "",
        "## Outliers",
        "",
        f"- Modified-z (MAD) outliers: {report['outliers']['total_modified_z_outliers']}",
        f"- IQR outliers: {report['outliers']['total_iqr_outliers']}",
        "",
        "## Hotelling's T\u00b2 (post-copula-transform)",
        "",
        f"- UCL (alpha={report['alpha']}): {t2['ucl']:.2f}",
        f"- Fraction of {t2['n']} observations exceeding UCL: {t2['fraction_exceeding']:.3f}",
        f"  (expected ~{report['alpha']} under the null that the transform succeeded)",
    ]
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
