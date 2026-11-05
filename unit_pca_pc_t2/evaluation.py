"""Moving-window T² (TA's Main.py), block metrics (TP / FN per flare event, FP per non-flare block, F1), root causes.

Reads only saved outputs (PC scores, final DAG, row labels), so it never re-runs the PC algorithm.

Definitions (agreed with the user, 2026-11-05; no alarm on/off delays, time is not counted):
- A minute is "above" when overall T² > its limit.
- Flare block = [onset - FLARE_PRE_WINDOW_MIN, end] of one flare event (>= FLARE_MIN_DURATION_MIN); non-flare block =
  the time between flare blocks.
- TP: a flare event with T² above at least once in [onset - FLARE_PRE_WINDOW_MIN, onset). One TP per event.
- FN: a flare event with T² never above in that window.
- FP: a non-flare block with T² above at least once, however long or often. One FP per block.
- TN: not counted. F1 = 2TP / (2TP + FP + FN).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as C
import helpers as H


# ---- MOVING-WINDOW T² ----
def causal_regressions(S: np.ndarray, par_idx: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Path coefficients and conditional variances from a correlation matrix (same maths as fit_causal_regressions).

    Args:
        S: Correlation matrix of the standardized PCs.
        par_idx: For each PC, the column positions of its parents.

    Returns:
        A tuple (B, sigma2): B[k, j] = coefficient of parent k for PC j; sigma2[j] = conditional variance (floored).
    """
    p = len(par_idx)
    B, sigma2 = np.zeros((p, p)), np.ones(p)
    for j, P in enumerate(par_idx):
        if P:
            beta = np.linalg.lstsq(S[np.ix_(P, P)], S[P, j], rcond=None)[0]
            B[P, j] = beta
            sigma2[j] = 1.0 - beta @ S[P, j]
    return B, np.maximum(sigma2, C.SIGMA2_MIN)


def moving_window_t2(X: np.ndarray, normal: np.ndarray, par_idx: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Overall T² and causal terms with a baseline re-estimated from the trailing window (TA's Main.py, l. 280-340).

    For each block of MW_STEP_MIN rows starting at s: the baseline is the flare-free rows among [s - WL, s),
    WL = MW_ROWS_PER_VAR x (PCs + 1). Their mean, std and correlation R standardize the block; T² = zᵀ inv(R) z and
    T_j = (z_j - β_jᵀ z_PA(j)) / σ_j. Differences from the TA: the TA moves one row at a time and his window includes
    the current row; here the step is MW_STEP_MIN and the window ends just before the block (no self-inclusion).

    Args:
        X: PC scores (rows = minutes, columns = PCs in DAG order).
        normal: Flare-free rows (not in a flare event or short flare).
        par_idx: Parent positions per PC, from the final DAG.

    Returns:
        A tuple (t2, T): T² per row and causal terms per row and PC; NaN where the window had fewer than
        MW_MIN_NORMAL flare-free rows (not scored).
    """
    n, p = X.shape
    WL = C.MW_ROWS_PER_VAR * (p + 1)
    t2, T = np.full(n, np.nan), np.full((n, p), np.nan)
    skipped, floored = 0, 0
    sd_floor = C.MW_SD_FLOOR_FRAC * X[normal].std(axis=0, ddof=1)
    for s in range(WL, n, C.MW_STEP_MIN):
        Xw = X[s - WL:s][normal[s - WL:s]]
        if len(Xw) < C.MW_MIN_NORMAL:
            skipped += 1
            continue
        m, sd = Xw.mean(axis=0), Xw.std(axis=0, ddof=1)
        floored += int((sd < sd_floor).any())
        sd = np.maximum(sd, sd_floor)
        Zw = (Xw - m) / sd
        S = Zw.T @ Zw / (len(Zw) - 1)
        d = np.sqrt(np.maximum(np.diag(S), 1e-12))
        S = S / np.outer(d, d) * np.outer(d > 0.5, d > 0.5)
        S = (S + S.T) / 2
        np.fill_diagonal(S, 1.0)
        B, sigma2 = causal_regressions(S, par_idx)
        Zb = (X[s:s + C.MW_STEP_MIN] - m) / sd
        t2[s:s + C.MW_STEP_MIN] = ((Zb @ np.linalg.inv(S)) * Zb).sum(axis=1)
        T[s:s + C.MW_STEP_MIN] = (Zb - Zb @ B) / np.sqrt(sigma2)
    H.log(f"moving-window T²: WL = {WL} rows, step {C.MW_STEP_MIN}; {skipped} steps not scored (< {C.MW_MIN_NORMAL} "
          f"flare-free rows); {floored} steps with a PC std floored; scored rows {int(np.isfinite(t2).sum())} of {n}")
    return t2, T

# ---- BLOCK METRICS ----
def top_pcs(ratio_row: np.ndarray, nodes: list[str], loadings: pd.DataFrame) -> dict[str, Any]:
    """Top TOP_N_ROOT PCs of one row by T²_j / own limit.

    Args:
        ratio_row: T²_j / limit_j per PC for one minute.
        nodes: PC names.
        loadings: PCA loadings (for the top tags of each PC).

    Returns:
        Dict pc1..pcN, unit1..N, ratio1..N, tags1..N and n_pcs_above (PCs above their own limit).
    """
    out: dict[str, Any] = {"n_pcs_above": int(np.nansum(ratio_row > 1))}
    order = np.argsort(-np.nan_to_num(ratio_row, nan=-1))[:C.TOP_N_ROOT]
    for k, j in enumerate(order, start=1):
        out.update({f"pc{k}": nodes[j], f"unit{k}": H.node_unit(nodes[j]), f"ratio{k}": float(ratio_row[j]),
                    f"tags{k}": H.top_tags_text(loadings, nodes[j])})
    return out


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True in a boolean array.

    Args:
        mask: Boolean array.

    Returns:
        (first row, last row) of each run, in time order.
    """
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(int(a), int(b) - 1) for a, b in zip(e[::2], e[1::2])]


def evaluate(t2: pd.DataFrame, t2_lim: float, ratio: np.ndarray, nodes: list[str], events: pd.DataFrame,
             loadings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per-event outcomes (TP / FN), non-flare blocks (FP) and F1. No alarm delays: a minute counts if T² > limit.

    The time line is split into flare blocks [onset - FLARE_PRE_WINDOW_MIN, end] (one per flare event) and non-flare
    blocks (the gaps between them). Each flare event is one TP if overall T² is above its limit at least once in
    [onset - FLARE_PRE_WINDOW_MIN, onset) (rows inside another flare event left out), else one FN. Each non-flare
    block is one FP if T² is above its limit at least once in it, however long or often. TN is not counted.

    Args:
        t2: All-row table with T2, above_limit, scored, flare_event (index = minutes).
        t2_lim: Overall T² limit.
        ratio: T²_j / own limit per row and PC.
        nodes: PC names in ratio's column order.
        events: Flare events (event_id, start, end, n_flagged, peak_flare_net, duration_min).
        loadings: PCA loadings.

    Returns:
        A tuple (outcomes, blocks, metrics): outcomes = one row per flare event; blocks = one row per non-flare block.
    """
    idx = t2.index
    n = len(idx)
    step = idx[1] - idx[0]
    w = int(round(pd.Timedelta(minutes=C.FLARE_PRE_WINDOW_MIN) / step))
    above, scored = t2["above_limit"].to_numpy(), t2["scored"].to_numpy()
    T2v, flare = t2["T2"].to_numpy(), t2["flare_event"].to_numpy()
    onset, end = idx.get_indexer(events["start"]), idx.get_indexer(events["end"])
    H.gate(bool((onset >= 0).all() and (end >= 0).all()), "every flare event's start and end are on the T² grid")

    out, in_flare_block = [], np.zeros(n, dtype=bool)
    for ev, o, e in zip(events.itertuples(index=False), onset, end):
        w0 = max(o - w, 0)
        in_flare_block[w0:e + 1] = True
        win = np.arange(w0, o)
        win = win[~flare[win]]
        sc = win[scored[win]]
        hit = sc[above[sc]]
        r = {"event_id": ev.event_id, "onset": ev.start, "end": ev.end, "duration_min": ev.duration_min,
             "n_flagged": ev.n_flagged, "peak_flare_net": ev.peak_flare_net, "window_start": idx[w0],
             "window_rows": len(win), "window_rows_scored": len(sc), "window_minutes_above": len(hit),
             "max_T2_window": float(np.max(T2v[sc])) if len(sc) else np.nan, "T2_limit": t2_lim,
             "above_at_window_start": bool(above[sc[0]]) if len(sc) else np.nan, "first_above": pd.NaT,
             "lead_min": np.nan, "analysed_minute": pd.NaT}
        if not len(sc):
            r["outcome"] = "not scored"
        elif len(hit):
            r.update({"outcome": "TP", "first_above": idx[hit[0]], "lead_min": (o - hit[0]) * step / pd.Timedelta("1min")})
            r.update(top_pcs(ratio[hit[0]], nodes, loadings))
        else:
            j = sc[int(np.argmax(T2v[sc]))]
            r.update({"outcome": "FN", "analysed_minute": idx[j]})
            r.update(top_pcs(ratio[j], nodes, loadings))
        r["figure"] = f"figures/events/E{ev.event_id:03d}_{r['outcome'].split()[0]}.png"
        out.append(r)
    outcomes = pd.DataFrame(out)
    outcomes["max_T2_window_over_limit"] = outcomes["max_T2_window"] / t2_lim

    rows = []
    for k, (a, b) in enumerate(runs_of(~in_flare_block), start=1):
        seg = np.arange(a, b + 1)
        sc = seg[scored[seg]]
        hit = sc[above[sc]]
        r = {"block_id": k, "start": idx[a], "end": idx[b], "rows": len(seg), "rows_scored": len(sc),
             "minutes_above": len(hit), "max_T2": float(np.max(T2v[sc])) if len(sc) else np.nan,
             "outcome": "not scored" if not len(sc) else ("FP" if len(hit) else "no detection"), "first_above": pd.NaT}
        if len(hit):
            r["first_above"] = idx[hit[0]]
            r.update(top_pcs(ratio[hit[0]], nodes, loadings))
        rows.append(r)
    blocks = pd.DataFrame(rows)

    oc = outcomes["outcome"].value_counts()
    tp, fn = int(oc.get("TP", 0)), int(oc.get("FN", 0))
    fp = int((blocks["outcome"] == "FP").sum()) if len(blocks) else 0
    metrics = {
        "flare events (>= 30 min)": len(events), "flare events not scored (no T² in the window)": int(oc.get("not scored", 0)),
        "flare events scored": tp + fn, "TP (flare events with T² above the limit in the window)": tp,
        "FN (flare events without)": fn, "non-flare blocks": len(blocks),
        "non-flare blocks not scored": int((blocks["outcome"] == "not scored").sum()) if len(blocks) else 0,
        "FP (non-flare blocks with T² above the limit)": fp,
        "F1 = 2TP / (2TP + FP + FN)": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
        "TP: T² already above the limit when the window opened": int(outcomes.loc[outcomes["outcome"] == "TP",
                                                                                    "above_at_window_start"].sum()),
        "median minutes from first T² above the limit to onset (TP)":
            float(outcomes.loc[outcomes["outcome"] == "TP", "lead_min"].median()) if tp else np.nan,
    }
    for k, v in metrics.items():
        H.log(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return outcomes, blocks, metrics


def rescore(out: Path) -> None:
    """Re-score the saved outputs with the moving-window baseline and the scaled limits; write all t2/ and table files.

    Keeps the fixed-baseline T² as column T2_fixed in t2/t2_all_rows.parquet and relabels preflare_event_id with
    FLARE_PRE_WINDOW_MIN. Overwrites t2_all_rows.parquet, terms_all_rows.parquet, term_limits.csv, limits.csv,
    preflare_first_ooc.csv; writes tables/event_outcomes.csv, tables/nonflare_blocks.csv, tables/metrics.csv.

    Args:
        out: Output folder holding a finished main.py run.
    """
    t2 = pd.read_parquet(out / "t2" / "t2_all_rows.parquet")
    scores = pd.read_parquet(out / "graph" / "pc_scores.parquet")
    dag = pd.read_csv(out / "graph" / "dag_edges.csv")
    events = pd.read_csv(out / "tables" / "flare_events.csv", parse_dates=["start", "end"])
    loadings = pd.read_csv(out / "tables" / "pca_loadings.csv", index_col=0)
    old_lim = pd.read_csv(out / "t2" / "limits.csv").iloc[0].to_dict()
    nodes = [c for c in scores.columns]
    H.gate(scores.index.equals(t2.index), "PC scores and T² rows share one time index")
    pos = {v: i for i, v in enumerate(nodes)}
    dag = dag[dag["target"] != C.FLARE_NET]
    par_idx = [[pos[s] for s in dag.loc[dag["target"] == v, "source"]] for v in nodes]
    normal = ~(t2["flare_event"] | t2["short_flare"]).to_numpy()
    if "T2_fixed" not in t2.columns:
        t2["T2_fixed"] = t2["T2"]
    t2["preflare_event_id"] = H.extract_preflare_window(t2.index, events, t2.index[1] - t2.index[0],
                                                        t2["flare_event"].to_numpy())
    t2v, T = moving_window_t2(scores.to_numpy(dtype=np.float64), normal, par_idx)
    p, WL = len(nodes), C.MW_ROWS_PER_VAR * (len(nodes) + 1)
    lim = C.T2_LIMIT_SCALE * H.t2_threshold(WL, p, 0, C.ALPHA0)
    lim_fixed = C.T2_LIMIT_SCALE * H.t2_threshold(int(old_lim["n_noc_rows"]), p, 0, C.ALPHA0)
    n_par = np.array([len(P) for P in par_idx])
    term_lim = np.array([C.T2_LIMIT_SCALE * H.t2_threshold(WL, 1, k, C.ALPHA_TERM) for k in n_par])
    t2["T2"], t2["scored"] = t2v, np.isfinite(t2v)
    t2["above_limit"] = np.nan_to_num(t2v, nan=0.0) > lim
    t2["above_limit_fixed"] = t2["T2_fixed"] > lim_fixed
    t2.to_parquet(out / "t2" / "t2_all_rows.parquet")
    pd.DataFrame(T.astype(np.float32), index=t2.index, columns=nodes).to_parquet(out / "t2" / "terms_all_rows.parquet")
    pd.DataFrame({"node": nodes, "n_parents": n_par, "t2_limit": term_lim}).to_csv(out / "t2" / "term_limits.csv",
                                                                                  index=False)
    H.log(f"limits x{C.T2_LIMIT_SCALE:g}: overall {lim:.2f} (m = WL = {WL}, p = {p}); per-PC "
          f"{term_lim.min():.3f} to {term_lim.max():.3f}; fixed-baseline overall limit x{C.T2_LIMIT_SCALE:g} = "
          f"{lim_fixed:.2f}")
    ratio = T ** 2 / term_lim
    outcomes, blocks, metrics = evaluate(t2, lim, ratio, nodes, events, loadings)
    outcomes.to_csv(out / "tables" / "event_outcomes.csv", index=False)
    blocks.to_csv(out / "tables" / "nonflare_blocks.csv", index=False)
    pd.Series(metrics, name="value").rename_axis("metric").to_csv(out / "tables" / "metrics.csv")
    noc, sc = t2["noc"].to_numpy(), t2["scored"].to_numpy()
    new_lim = dict(old_lim, t2_limit=lim, t2_limit_fixed_baseline=lim_fixed, limit_scale=C.T2_LIMIT_SCALE,
                   baseline="moving window", window_rows=WL, step_rows=C.MW_STEP_MIN,
                   pre_window_min=C.FLARE_PRE_WINDOW_MIN, n_events_found=metrics[
                       "TP (flare events with T² above the limit in the window)"],
                   pct_noc_above_t2_limit=100 * float(t2["above_limit"][noc & sc].mean()),
                   pct_noc_above_term_limit_mean=100 * float(np.nanmean(ratio[noc & sc] > 1)),
                   pct_preflare_above_t2_limit=100 * float(t2["above_limit"][(t2["preflare_event_id"] > 0) & sc].mean()))
    pd.DataFrame([new_lim]).to_csv(out / "t2" / "limits.csv", index=False)
    first = outcomes.rename(columns={"first_above": "first_ooc"}).assign(
        detected=outcomes["outcome"].eq("TP"), minutes_before_onset=outcomes["lead_min"])
    first.to_csv(out / "t2" / "preflare_first_ooc.csv", index=False)


# ---- EXCEL REPORT ----
def pc_frequency(outcomes: pd.DataFrame, blocks: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """How often each PC is a root cause: top-1 and top-3 for TP events, top-1 for FP blocks.

    Args:
        outcomes: Event outcome table from evaluate().
        blocks: Non-flare block table from evaluate().
        nodes: All PC names.

    Returns:
        Table indexed by PC with unit and the counts, sorted by top-1 count for TP events.
    """
    tp = outcomes[outcomes["outcome"] == "TP"]
    fp = blocks[blocks["outcome"] == "FP"] if "pc1" in blocks.columns else blocks.iloc[:0].assign(pc1=[])
    top3 = pd.concat([tp[f"pc{k}"] for k in range(1, C.TOP_N_ROOT + 1)]) if len(tp) else pd.Series(dtype=str)
    f = pd.DataFrame({"unit": [H.node_unit(v) for v in nodes]}, index=pd.Index(nodes, name="pc"))
    f["top1_TP_events"] = tp["pc1"].value_counts().reindex(nodes, fill_value=0) if len(tp) else 0
    f[f"top{C.TOP_N_ROOT}_TP_events"] = top3.value_counts().reindex(nodes, fill_value=0)
    f["top1_FP_blocks"] = fp["pc1"].value_counts().reindex(nodes, fill_value=0)
    return f.sort_values(["top1_TP_events", f"top{C.TOP_N_ROOT}_TP_events"], ascending=False)


def write_excel(out: Path) -> Path:
    """Write outputs/RCA_report.xlsx: metrics, events (one row per flare event), nonflare_blocks, pc_frequency,
    unit_frequency, definitions.

    Args:
        out: Output folder after rescore().

    Returns:
        Path of the workbook.
    """
    outcomes = pd.read_csv(out / "tables" / "event_outcomes.csv")
    blocks = pd.read_csv(out / "tables" / "nonflare_blocks.csv")
    metrics = pd.read_csv(out / "tables" / "metrics.csv")
    nodes = list(pd.read_csv(out / "t2" / "term_limits.csv")["node"])
    freq = pc_frequency(outcomes, blocks, nodes)
    unit = freq.groupby("unit").sum(numeric_only=True).sort_values("top1_TP_events", ascending=False)
    W = C.FLARE_PRE_WINDOW_MIN
    defs = pd.DataFrame({"term": ["above limit", "flare block", "non-flare block", "TP", "FN", "FP", "TN", "F1",
                                  "lead_min", "root cause", "ratio", "baseline", "limits"],
                         "definition": [
                             "overall T² > its limit at that minute (no on/off delay)",
                             f"[onset - {W} min, flare end] of one flare event (>= {C.FLARE_MIN_DURATION_MIN} min)",
                             "time between flare blocks",
                             f"flare event with T² above the limit at least once in [onset - {W} min, onset) "
                             "(minutes inside another flare event left out)",
                             f"flare event with T² never above the limit in [onset - {W} min, onset)",
                             "non-flare block with T² above the limit at least once (one FP per block)",
                             "not counted",
                             "2 TP / (2 TP + FP + FN)",
                             "minutes from the first minute above the limit in the window to flare onset",
                             f"top {C.TOP_N_ROOT} PCs by T²_j / own limit at the first minute above the limit "
                             "(FN events: at the window's highest T²)",
                             "T²_j / own limit; above 1 = PC out of control",
                             f"moving window: {C.MW_ROWS_PER_VAR} x (PCs + 1) flare-free trailing rows, "
                             f"re-estimated every {C.MW_STEP_MIN} min (TA's Main.py)",
                             f"Eq. 13 with m = window length, x{C.T2_LIMIT_SCALE:g} (overall and per-PC)"]})
    path = out / "RCA_report.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        sheets = (("metrics", metrics), ("events", outcomes), ("nonflare_blocks", blocks), ("definitions", defs))
        for name, df in sheets[:3]:
            df.to_excel(xw, sheet_name=name, index=False)
        freq.to_excel(xw, sheet_name="pc_frequency")
        unit.to_excel(xw, sheet_name="unit_frequency")
        defs.to_excel(xw, sheet_name="definitions", index=False)
        for name, df in sheets:
            for j, c in enumerate(df.columns):
                xw.sheets[name].set_column(j, j, min(60, max(10, len(str(c)) + 2)))
    H.log(f"saved {path.name}")
    return path


# ---- FIGURES ----
def plot_event(D: dict[str, Any], e: Any, r: pd.Series, path: Path, plt: Any, mdates: Any) -> None:
    """One flare event: flare_net, overall T² with its limit, and the top-3 PCs' T²_j with their own limits.

    Args:
        D: Plot inputs (flare, t2, terms, term_lim, t2_lim, thr, interval).
        e: Event row (event_id, start, end, duration_min, peak_flare_net).
        r: The event's row in event_outcomes.csv.
        path: PNG to write.
        plt: matplotlib.pyplot.
        mdates: matplotlib.dates.
    """
    col = C.PLOT_COLORS
    W = C.FLARE_PRE_WINDOW_MIN
    a = e.start - pd.Timedelta(minutes=C.PLOT_ZOOM_BEFORE_MIN)
    b = min(e.end, e.start + pd.Timedelta(minutes=C.PLOT_ZOOM_AFTER_MIN * 4)) + D["interval"]
    pcs = [r.get(f"pc{k}") for k in range(1, C.TOP_N_ROOT + 1) if isinstance(r.get(f"pc{k}"), str)]
    fig, axs = plt.subplots(2 + len(pcs), 1, figsize=(12, 2.1 * (2 + len(pcs))), sharex=True)
    f = D["flare"][C.FLARE_NET].loc[a:b]
    axs[0].plot(f.index, f.to_numpy(), color=col["flare"], lw=1)
    axs[0].axhline(D["thr"], color=col["limit"], ls="--", lw=0.8, label=f"threshold {D['thr']:g} t/h")
    axs[0].set_ylabel("flare_net\n(t/h)")
    t = D["t2"]["T2"].loc[a:b]
    axs[1].plot(t.index, t.clip(lower=1e-2).to_numpy(), color=col["t2"], lw=1)
    axs[1].axhline(D["t2_lim"], color=col["limit"], ls="--", lw=0.8, label=f"limit {D['t2_lim']:.1f}")
    mark = r.get("first_above") if r.get("outcome") == "TP" else r.get("analysed_minute")
    if isinstance(mark, (str, pd.Timestamp)) and str(mark) != "nan":
        cx = pd.Timestamp(mark)
        for ax in axs:
            ax.axvline(cx, color="#8e44ad", lw=1.2, ls=":")
        axs[1].plot([cx], [max(D["t2"]["T2"].get(cx, np.nan), 1e-2)], "o", color="#8e44ad", label="analysed minute")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("overall T²")
    for k, pc in enumerate(pcs):
        ax = axs[2 + k]
        y = (D["terms"][pc].loc[a:b].astype(np.float64) ** 2).clip(lower=1e-3)
        ax.plot(y.index, y.to_numpy(), color=plt.cm.tab10(k), lw=1)
        ax.axhline(D["term_lim"][pc], color=col["limit"], ls="--", lw=0.8, label=f"limit {D['term_lim'][pc]:.2f}")
        ax.set_yscale("log")
        ax.set_ylabel(f"T²_j {pc}\n#{k + 1}", fontsize=8)
        ax.text(0.01, 0.92, f"{H.node_unit(pc)} | {r.get(f'tags{k + 1}', '')} | ratio at analysed minute "
                f"{r.get(f'ratio{k + 1}', np.nan):.1f}", transform=ax.transAxes, fontsize=8, va="top")
    for ax in axs:
        ax.axvspan(e.start - pd.Timedelta(minutes=W), e.start, color=col["preflare"], alpha=0.08, lw=0)
        ax.axvspan(e.start, e.end + D["interval"], color=col["event"], alpha=0.18, lw=0)
        ax.grid(alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        if h:
            u = dict(zip(l, h))
            ax.legend(u.values(), u.keys(), loc="upper right", fontsize=7)
    H.format_time_axis(axs[-1], mdates, a, b)
    lead = f", T² first above the limit {r['lead_min']:.0f} min before onset" if r.get("outcome") == "TP" else ""
    fig.suptitle(f"E{e.event_id}: onset {e.start:%d %b %Y %H:%M}, {e.duration_min:.0f} min, peak {e.peak_flare_net:.2f} "
                 f"t/h. Outcome: {r.get('outcome')}{lead}. Orange = flare, pink = {W}-min window",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_all_events(D: dict[str, Any], out: Path) -> None:
    """One PNG per flare event in figures/events/ (file name = the 'figure' column of event_outcomes.csv).

    Args:
        D: Plot inputs from helpers.load_plot_inputs() after rescore().
        out: Output folder.
    """
    plt, mdates = H.plot_modules()
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv",
                     parse_dates=["onset", "first_above", "analysed_minute"]).set_index("event_id")
    folder = out / "figures" / "events"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("E*.png"):
        old.unlink()
    for e in D["events"].itertuples(index=False):
        r = oc.loc[e.event_id]
        plot_event(D, e, r, out / r["figure"], plt, mdates)
    H.log(f"saved {len(D['events'])} event figures in figures/events/")


def plot_root_cause_frequency(out: Path, path: Path) -> None:
    """Figure 6: root-cause frequency per PC (top-1 and top-3, TP events), per PC for FP blocks, and per unit.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, _ = H.plot_modules()
    blocks = pd.read_csv(out / "tables" / "nonflare_blocks.csv")
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv")
    nodes = list(pd.read_csv(out / "t2" / "term_limits.csv")["node"])
    f = pc_frequency(oc, blocks, nodes)
    units = sorted(f["unit"].unique())
    ucol = {u: plt.cm.tab20(i % 20) for i, u in enumerate(units)}
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    n_tp, n_fp = int((oc["outcome"] == "TP").sum()), int((blocks["outcome"] == "FP").sum())
    for ax, c, title in ((axs[0, 0], "top1_TP_events", f"Top-1 root cause, {n_tp} TP events"),
                         (axs[0, 1], f"top{C.TOP_N_ROOT}_TP_events", f"In the top {C.TOP_N_ROOT}, {n_tp} TP events"),
                         (axs[1, 0], "top1_FP_blocks", f"Top-1 PC of FP blocks ({n_fp} FP)")):
        s = f[c].sort_values(ascending=False)
        ax.bar(range(len(s)), s.to_numpy(), color=[ucol[f.at[v, 'unit']] for v in s.index])
        ax.set_xticks(range(len(s)), s.index, rotation=70, ha="right", fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)
    u = f.groupby("unit")[["top1_TP_events", "top1_FP_blocks"]].sum().sort_values("top1_TP_events")
    y = np.arange(len(u))
    axs[1, 1].barh(y - 0.2, u["top1_TP_events"], 0.4, label="TP events (top-1)", color="#1f5fa8")
    axs[1, 1].barh(y + 0.2, u["top1_FP_blocks"], 0.4, label="FP blocks (top-1)", color="#c0392b")
    axs[1, 1].set_yticks(y, u.index)
    axs[1, 1].legend()
    axs[1, 1].set_title("By unit: TP events vs FP blocks", loc="left")
    axs[1, 1].grid(axis="x", alpha=0.3)
    H.save_figure(fig, path, plt)


def plot_metrics(out: Path, path: Path) -> None:
    """Figure 7: TP / FN / FP counts with F1, lead time of TP events, events by month, FP blocks by month.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, _ = H.plot_modules()
    m = pd.read_csv(out / "tables" / "metrics.csv").set_index("metric")["value"].astype(float)
    bl = pd.read_csv(out / "tables" / "nonflare_blocks.csv", parse_dates=["start"])
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv", parse_dates=["onset"])
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    tp, fn = int((oc["outcome"] == "TP").sum()), int((oc["outcome"] == "FN").sum())
    fp = int((bl["outcome"] == "FP").sum())
    axs[0, 0].bar(["TP", "FN", "FP"], [tp, fn, fp], color=["#2ca02c", "#c0392b", "#e67e22"])
    for x, v in enumerate([tp, fn, fp]):
        axs[0, 0].text(x, v, str(v), ha="center", va="bottom", fontsize=13)
    axs[0, 0].set_title(f"F1 = 2TP / (2TP + FP + FN) = {m['F1 = 2TP / (2TP + FP + FN)']:.3f}", loc="left")
    axs[0, 0].set_ylabel("count (events / blocks)")
    lead = oc.loc[oc["outcome"] == "TP", "lead_min"]
    axs[0, 1].hist(lead, bins=range(0, C.FLARE_PRE_WINDOW_MIN + 2), color="#2ca02c")
    axs[0, 1].set_title(f"First minute above the limit -> onset, TP events (n = {len(lead)})", loc="left")
    axs[0, 1].set_xlabel("minutes before onset")
    tab = oc.groupby([oc["onset"].dt.strftime("%Y-%m"), "outcome"]).size().unstack(fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=axs[1, 0], color=[{"TP": "#2ca02c", "FN": "#c0392b",
             "not scored": "#9e9e9e"}.get(c, "#555") for c in tab.columns])
    axs[1, 0].set_title("Flare events by month and outcome", loc="left")
    tabb = bl.groupby([bl["start"].dt.strftime("%Y-%m"), "outcome"]).size().unstack(fill_value=0)
    tabb.plot(kind="bar", stacked=True, ax=axs[1, 1], color=[{"FP": "#e67e22", "no detection": "#1f5fa8",
              "not scored": "#9e9e9e"}.get(c, "#555") for c in tabb.columns])
    axs[1, 1].set_title("Non-flare blocks by month (start) and outcome", loc="left")
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.tick_params(axis="x", rotation=0)
    H.save_figure(fig, path, plt)


def plot_baseline_drift(out: Path, path: Path) -> None:
    """Figure 8: fixed (NOC) vs moving-window baseline: daily median T² on flare-free minutes and share in alarm.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, mdates = H.plot_modules()
    t = pd.read_parquet(out / "t2" / "t2_all_rows.parquet")
    lim = pd.read_csv(out / "t2" / "limits.csv").iloc[0]
    q = t[~t["flare_event"] & ~t["short_flare"]]
    d = q.resample("1D").agg({"T2_fixed": "median", "T2": "median", "above_limit_fixed": "mean", "above_limit": "mean"})
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axs[0].plot(d.index, d["T2_fixed"], color="#7f7f7f", label="fixed baseline (Jul-Sep NOC)")
    axs[0].plot(d.index, d["T2"], color="#1f5fa8", label="moving-window baseline")
    axs[0].axhline(lim["t2_limit_fixed_baseline"], color="#7f7f7f", ls="--", lw=0.8, label="fixed limit")
    axs[0].axhline(lim["t2_limit"], color="#1f5fa8", ls="--", lw=0.8, label="moving-window limit")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("daily median T²\n(flare-free minutes)")
    axs[1].plot(d.index, 100 * d["above_limit_fixed"], color="#7f7f7f", label="fixed baseline")
    axs[1].plot(d.index, 100 * d["above_limit"], color="#1f5fa8", label="moving-window baseline")
    axs[1].set_ylabel("% flare-free minutes\nabove limit")
    for ax in axs:
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    H.format_time_axis(axs[1], mdates, d.index.min(), d.index.max() + pd.Timedelta(days=1))
    axs[0].set_title("Why the baseline moves: with a fixed Jul-Sep baseline T² drifts above the limit and stays there",
                     loc="left")
    H.save_figure(fig, path, plt)


def plot_root_cause_by_week(out: Path, path: Path) -> None:
    """Figure 9: heatmap of top-1 root-cause PC of detected events per week.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, _ = H.plot_modules()
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv", parse_dates=["onset"])
    tp = oc[oc["outcome"] == "TP"]
    if tp.empty:
        H.log("no detected events: Figure 9 skipped")
        return
    tab = tp.groupby([tp["pc1"], tp["onset"].dt.to_period("W").dt.start_time]).size().unstack(fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(16, 0.4 * len(tab) + 2))
    im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(tab)), tab.index)
    ax.set_xticks(range(len(tab.columns)), [c.strftime("%d %b") for c in tab.columns], rotation=70)
    fig.colorbar(im, ax=ax, label="detected events with this top-1 PC")
    ax.set_title("Top-1 root-cause PC of detected events, per week", loc="left")
    H.save_figure(fig, path, plt)
