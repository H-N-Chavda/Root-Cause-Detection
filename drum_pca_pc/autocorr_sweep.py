"""Row-interval sweep: how autocorrelation in the PC input affects the PC algorithm.

The notebook feeds every PC_ROW_STEP-th normal-operation row to the PC algorithm. Consecutive rows are strongly
autocorrelated, which makes Fisher-z p-values too small and the graph too dense. This script tries several row
intervals and, for each one, measures:

  1. null false-positive rate: % of node pairs significant at depth 0 after one series of each pair is circularly
     shifted by >= MIN_LAG_MIN minutes (real cross-dependence removed, autocorrelation kept). Should be ~alpha.
  2. lag-1 autocorrelation of the PC nodes (median, 25-75 % band).
  3. rows used n and median effective sample size n_eff = n (1 - rho) / (1 + rho).
  4. skeleton stability: mean pairwise Jaccard overlap of the depth-PC_MAX_K skeletons from several start offsets.
  5. number of skeleton edges (and % of all node pairs).

Data loading, cleaning, flare masking and PCA are taken from drum_pca_pc.ipynb itself (its code cells are executed
here, so the inputs are identical to the notebook's; the PCA checkpoint is reused if present). The PC node set is
the notebook's (constant / collinear / near-dependent PCs dropped as in Step 4a) and is the same for every interval.

Usage (from anywhere in the repository):
    python drum_pca_pc/autocorr_sweep.py
    python drum_pca_pc/autocorr_sweep.py --steps 10,20,30,60,120,240,480 --offsets 5 --shifts 5

Outputs: drum_pca_pc/outputs/autocorr_sweep/  (metrics.csv, null_fpr_draws.csv, one PNG per plot, all_plots.png)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pc_fast  # noqa: E402

# notebook code cells executed, in this order (each marker must appear in exactly one code cell)
CELL_MARKERS = ["# ---- paths ----", "def load_parquet_folder", "def screen_matrix", "def label_flare_events",
                "def pca_block"]
QUIET = lambda *a, **k: None  # noqa: E731


def load_notebook_state() -> dict:
    """Run the notebook's config, load/clean, flare and PCA cells; return their variables."""
    nb = json.loads((HERE / "drum_pca_pc.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "drum_pca_pc_notebook"}
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        for marker in CELL_MARKERS:
            hits = [s for s in code if marker in s]
            assert len(hits) == 1, f"expected exactly one notebook code cell containing {marker!r}, found {len(hits)}"
            print(f"\n--- notebook cell: {marker}")
            exec(compile(hits[0], f"<notebook cell: {marker}>", "exec"), ns)
            ns["display"] = QUIET  # tables printed by the notebook are not needed here
        hits = [s for s in code if "def drop_dependent_nodes" in s]
        assert len(hits) == 1, "notebook cell with drop_dependent_nodes not found"
        exec(compile(hits[0].split("# ---- 4a")[0], "<notebook cell: Step 4 helpers>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def lag1_autocorr(D: np.ndarray) -> np.ndarray:
    a = D[:-1] - D[:-1].mean(0)
    b = D[1:] - D[1:].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def null_fpr(D: np.ndarray, alpha: float, min_lag_rows: int, n_shifts: int, rng) -> list[dict]:
    """Circular-shift null: every column rolled by its own random lag; a pair counts only if the two lags differ
    by >= min_lag_rows (circularly). Returns one dict per draw."""
    n, p = D.shape
    iu = np.triu_indices(p, 1)
    out = []
    for draw in range(n_shifts):
        lags = rng.integers(0, n, p)
        S = np.empty_like(D)
        for j in range(p):
            S[:, j] = np.roll(D[:, j], lags[j])
        d = np.abs(lags[:, None] - lags[None, :])
        d = np.minimum(d, n - d)[iu]
        ok = d >= min_lag_rows
        if not ok.any():
            out.append({"draw": draw, "pairs": 0, "fpr": np.nan})
            continue
        r = np.corrcoef(S.T)[iu][ok]
        out.append({"draw": draw, "pairs": int(ok.sum()), "fpr": float((pc_fast._fisher_p(r, n, 0) <= alpha).mean())})
    return out


def edge_set(A: np.ndarray) -> set:
    i, j = np.nonzero(np.triu(A, 1))
    return set(zip(i.tolist(), j.tolist()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="10,20,30,60,120,240,480",
                    help="row intervals to test (rows of the 1-min grid = minutes), comma separated")
    ap.add_argument("--offsets", type=int, default=5, help="start offsets per interval for the stability metric")
    ap.add_argument("--shifts", type=int, default=5, help="random circular-shift draws for the null test")
    ap.add_argument("--min-lag-min", type=float, default=1440, help="smallest circular shift in the null test, minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (default: <notebook OUT_DIR>/autocorr_sweep)")
    args = ap.parse_args()
    steps = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    assert steps and min(steps) >= 1, "steps must be positive integers"

    t_all = time.time()
    ns = load_notebook_state()
    alpha, max_k, base_step = ns["PC_ALPHA"], ns["PC_MAX_K"], ns["PC_ROW_STEP"]
    MODELS, normal = ns["MODELS"], ns["normal"]
    out_dir = Path(args.out) if args.out else Path(ns["OUT_DIR"]) / "autocorr_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # PC scores of every normal-operation row, all drums side by side (as notebook Step 4a, without thinning)
    blocks, names = [], []
    for m in MODELS.values():
        blocks.append(((normal[m["cols"]].to_numpy(float) - m["mu"]) / m["sd"]) @ m["P"])
        names += m["nodes"]
    Z = np.hstack(blocks)
    del blocks
    # node set exactly as the notebook chooses it (on its every-PC_ROW_STEP-th rows), then fixed for all intervals
    keep, _, _, drops = ns["drop_dependent_nodes"](Z[::base_step], names)
    Z = np.ascontiguousarray(Z[:, keep])
    ts = normal.index
    p = Z.shape[1]
    n_pairs = p * (p - 1) // 2
    print(f"\nPC scores: {Z.shape[0]:,} normal rows x {p} PC nodes ({len(drops)} left out as in the notebook); "
          f"alpha {alpha}, depth <= {max_k}; intervals {steps}")

    rng = np.random.default_rng(args.seed)
    rows, draws = [], []
    for s in steps:
        t0 = time.time()
        offsets = sorted({int(round(k * s / args.offsets)) for k in range(args.offsets)})
        D0 = Z[offsets[0]::s]
        n = len(D0)
        rho = lag1_autocorr(D0)
        rho_c = np.clip(rho, -0.999, 0.999)
        n_eff = n * (1 - rho_c) / (1 + rho_c)
        gaps = np.diff(ts[offsets[0]::s].asi8) / 6e10  # minutes between consecutive used rows
        min_lag_rows = int(np.ceil(args.min_lag_min / s))
        nd = null_fpr(D0, alpha, min_lag_rows, args.shifts, rng)
        for d in nd:
            draws.append({"step_min": s, **d})
        fprs = np.array([d["fpr"] for d in nd], float)

        skels = []
        for o in offsets:
            A, _, _, _ = pc_fast.skeleton(Z[o::s], alpha, max_k, log=QUIET)
            skels.append(edge_set(A))
        jac = [jaccard(a, b) for a, b in combinations(skels, 2)]
        n_edges = [len(e) for e in skels]

        row = {"step_min": s, "n_rows": n, "median_gap_min": float(np.median(gaps)), "max_gap_min": float(gaps.max()),
               "rho_median": float(np.nanmedian(rho)), "rho_q25": float(np.nanquantile(rho, 0.25)),
               "rho_q75": float(np.nanquantile(rho, 0.75)), "n_eff_median": float(np.nanmedian(n_eff)),
               "null_fpr_mean": float(np.nanmean(fprs)) if np.isfinite(fprs).any() else np.nan,
               "null_fpr_min": float(np.nanmin(fprs)) if np.isfinite(fprs).any() else np.nan,
               "null_fpr_max": float(np.nanmax(fprs)) if np.isfinite(fprs).any() else np.nan,
               "null_min_lag_rows": min_lag_rows, "n_offsets": len(offsets),
               "jaccard_mean": float(np.mean(jac)) if jac else np.nan,
               "jaccard_min": float(np.min(jac)) if jac else np.nan,
               "edges_offset0": n_edges[0], "edges_mean": float(np.mean(n_edges)),
               "edge_pct_offset0": 100 * n_edges[0] / n_pairs, "seconds": time.time() - t0}
        rows.append(row)
        print(f"step {s:>4} min: n {n:>7,} | rho {row['rho_median']:.3f} | n_eff {row['n_eff_median']:>9,.0f} | "
              f"null FPR {100 * row['null_fpr_mean']:5.1f} % | Jaccard {row['jaccard_mean']:.3f} | "
              f"edges {n_edges[0]:,} ({row['edge_pct_offset0']:.1f} %) | {row['seconds']:.1f} s")

    M = pd.DataFrame(rows)
    M.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(draws).to_csv(out_dir / "null_fpr_draws.csv", index=False)
    make_plots(M, alpha, n_pairs, out_dir)
    print(f"\nwritten to {out_dir} | total {time.time() - t_all:.0f} s")


def _xaxis(ax, M):
    ax.set_xscale("log")
    ax.set_xticks(M.step_min)
    ax.set_xticklabels([f"{s}\nn={n:,}" for s, n in zip(M.step_min, M.n_rows)], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("row interval (min); n = rows used")
    ax.grid(alpha=0.3)


def make_plots(M: pd.DataFrame, alpha: float, n_pairs: int, out_dir: Path):
    x = M.step_min

    def p1(ax):
        ax.fill_between(x, 100 * M.null_fpr_min, 100 * M.null_fpr_max, alpha=0.25, color="#1f4e79", lw=0,
                        label="range over shift draws")
        ax.plot(x, 100 * M.null_fpr_mean, "o-", color="#1f4e79", label="mean")
        ax.axhline(100 * alpha, ls="--", color="red", label=f"expected {100 * alpha:g} %")
        ax.set_ylabel("% null pairs significant (depth 0)")
        ax.set_title("1. False-positive rate on shifted (null) pairs")
        ax.set_ylim(0, None)
        ax.legend(fontsize=7)

    def p2(ax):
        ax.fill_between(x, M.rho_q25, M.rho_q75, alpha=0.25, color="#1f4e79", lw=0, label="25-75 %")
        ax.plot(x, M.rho_median, "o-", color="#1f4e79", label="median")
        ax.axhline(0, ls="--", color="gray")
        ax.set_ylabel("lag-1 autocorrelation")
        ax.set_title("2. Remaining autocorrelation of PC nodes")
        ax.legend(fontsize=7)

    def p3(ax):
        ax.plot(x, M.n_rows, "o-", color="#1f4e79", label="n (rows used)")
        ax.plot(x, M.n_eff_median, "s-", color="#c55a11", label="median n_eff")
        ax.set_yscale("log")
        ax.set_ylabel("sample size")
        ax.set_title("3. Rows used vs effective sample size")
        ax.legend(fontsize=7)

    def p4(ax):
        ax.plot(x, M.jaccard_mean, "o-", color="#1f4e79", label="mean")
        ax.plot(x, M.jaccard_min, "v--", color="#1f4e79", alpha=0.6, label="min")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Jaccard overlap of edge sets")
        ax.set_title("4. Skeleton stability across start offsets")
        ax.legend(fontsize=7)

    def p5(ax):
        ax.plot(x, M.edges_offset0, "o-", color="#1f4e79")
        ax.set_ylabel("skeleton edges")
        ax.set_title("5. Graph size (context only)")
        sec = ax.secondary_yaxis("right", functions=(lambda e: 100 * e / n_pairs, lambda q: q * n_pairs / 100))
        sec.set_ylabel("% of node pairs")

    panels = [("1_null_false_positive_rate", p1), ("2_autocorrelation", p2), ("3_sample_size", p3),
              ("4_skeleton_stability", p4), ("5_edges", p5)]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fn(ax)
        _xaxis(ax, M)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for (_, fn), ax in zip(panels, axes.flat):
        fn(ax)
        _xaxis(ax, M)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "all_plots.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
