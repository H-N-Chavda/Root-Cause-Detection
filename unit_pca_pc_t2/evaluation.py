"""Moving-window T² (TA's Main.py), ISA-18.2 style alarms, TP/FP/TN/FN metrics, root cause per alarm and event.

Reads only saved outputs (PC scores, final DAG, row labels), so it never re-runs the PC algorithm.

Definitions (agreed with the user, 2026-11-05):
- Alarm: switches on when overall T² stays above its limit ALARM_ON_DELAY_MIN minutes; switches off when T² stays
  below it ALARM_OFF_DELAY_MIN minutes. One activation = one alarm, however long it stays on.
- TP: an alarm switches on outside a flare and a flare event starts within PRED_HORIZON_MIN minutes.
- FP: an alarm switches on outside a flare and no flare event starts within PRED_HORIZON_MIN minutes.
- FN: a flare event with no alarm switching on in its PRED_HORIZON_MIN-minute window ("already in alarm" listed apart).
- TN: a 30-minute flare-free block outside every pre-flare window with no alarm on at any time.
- Alarms that switch on during a flare (event or short) are not predictions.
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
    WL = MW_ROWS_PER_VAR x (PCs + 1). Their mean, std and correlation R standardize the block; T² = zᵀ pinv(R) z and
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
        t2[s:s + C.MW_STEP_MIN] = ((Zb @ np.linalg.pinv(S)) * Zb).sum(axis=1)
        T[s:s + C.MW_STEP_MIN] = (Zb - Zb @ B) / np.sqrt(sigma2)
    H.log(f"moving-window T²: WL = {WL} rows, step {C.MW_STEP_MIN}; {skipped} steps not scored (< {C.MW_MIN_NORMAL} "
          f"flare-free rows); {floored} steps with a PC std floored; scored rows {int(np.isfinite(t2).sum())} of {n}")
    return t2, T


# ---- ALARMS ----
def alarm_states(above: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Alarm on/off with on-delay and off-delay (ISA-18.2 style).

    Args:
        above: T² above its limit per row (False where not scored).

    Returns:
        A tuple (on, episodes): on = alarm state per row; episodes = (crossing row, activation row, last row on).
    """
    on = np.zeros(len(above), dtype=bool)
    eps, state, run_up, run_dn, cross = [], False, 0, 0, -1
    for i, a in enumerate(above):
        if not state:
            run_up = run_up + 1 if a else 0
            if run_up == 1:
                cross = i
            if run_up >= C.ALARM_ON_DELAY_MIN:
                state, run_dn = True, 0
                eps.append([cross, i, -1])
                on[cross:i + 1] = True
        else:
            on[i] = True
            run_dn = 0 if a else run_dn + 1
            if run_dn >= C.ALARM_OFF_DELAY_MIN:
                state, run_up = False, 0
                eps[-1][2] = i - C.ALARM_OFF_DELAY_MIN
                on[i - C.ALARM_OFF_DELAY_MIN + 1:i + 1] = False
    if eps and eps[-1][2] < 0:
        eps[-1][2] = len(above) - 1
    return on, [tuple(e) for e in eps]


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


def evaluate(t2: pd.DataFrame, ratio: np.ndarray, nodes: list[str], events: pd.DataFrame,
             loadings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Alarm table, per-event outcome table and the TP / FP / TN / FN counts.

    Args:
        t2: All-row table with T2, above_limit, scored, flare_event, short_flare (index = minutes).
        ratio: T²_j / own limit per row and PC.
        nodes: PC names in ratio's column order.
        events: Flare events (event_id, start, end, duration_min, peak_flare_net).
        loadings: PCA loadings.

    Returns:
        A tuple (alarms, outcomes, metrics).
    """
    idx = t2.index
    n, H_ = len(idx), C.PRED_HORIZON_MIN
    flare = (t2["flare_event"] | t2["short_flare"]).to_numpy()
    scored = t2["scored"].to_numpy()
    on, eps = alarm_states(t2["above_limit"].to_numpy())
    onset = idx.get_indexer(events["start"])
    eid = events["event_id"].to_numpy()

    rows = []
    for c, a, e in eps:
        r = {"crossing": idx[c], "activation": idx[a], "end": idx[e], "duration_min": e - c + 1,
             "T2_at_crossing": float(t2["T2"].iat[c]), "class": "FP", "event_id": np.nan, "lead_min": np.nan}
        if flare[a]:
            r["class"] = "during flare"
        else:
            k = np.searchsorted(onset, a, side="right")
            if k < len(onset) and onset[k] >= 0 and onset[k] - a <= H_:
                r.update({"class": "TP", "event_id": int(eid[k]), "lead_min": int(onset[k] - a)})
        r.update(top_pcs(ratio[c], nodes, loadings))
        rows.append(r)
    alarms = pd.DataFrame(rows)

    out = []
    in_window = np.zeros(n, dtype=bool)
    for ev, o in zip(events.itertuples(index=False), onset):
        w0 = max(o - H_, 0)
        in_window[w0:o] = True
        r = {"event_id": ev.event_id, "onset": ev.start, "duration_min": ev.duration_min,
             "peak_flare_net": ev.peak_flare_net, "window_rows": o - w0}
        tp = alarms[(alarms["class"] == "TP") & (alarms["event_id"] == ev.event_id)] if len(alarms) else alarms
        if o - w0 == 0 or not scored[w0:o].any():
            r["outcome"] = "not scored"
        elif len(tp):
            f = tp.iloc[0]
            r.update({"outcome": "TP", "crossing": f["crossing"], "activation": f["activation"],
                      "lead_min": f["lead_min"]})
            r.update({k: f[k] for k in f.index if k[:-1] in ("pc", "unit", "ratio", "tags") or k == "n_pcs_above"})
        else:
            r["outcome"] = "FN (already in alarm)" if on[w0] else "FN (no alarm)"
            seg = t2["T2"].to_numpy()[w0:o]
            if np.isfinite(seg).any():
                j = w0 + int(np.nanargmax(seg))
                r.update({"crossing": idx[j]})
                r.update(top_pcs(ratio[j], nodes, loadings))
        out.append(r)
    outcomes = pd.DataFrame(out)

    elig = ~flare & ~in_window & scored
    act = np.zeros(n, dtype=bool)
    for _, a, _ in eps:
        act[a] = True
    blocks = {"TN": 0, "FP block": 0, "standing alarm": 0}
    edges = np.flatnonzero(np.diff(np.r_[0, elig.astype(int), 0]))
    for b, e in zip(edges[::2], edges[1::2]):
        for s in range(b, e - H_ + 1, H_):
            sl = slice(s, s + H_)
            blocks["TN" if not on[sl].any() else ("FP block" if act[sl].any() else "standing alarm")] += 1

    cls = alarms["class"].value_counts() if len(alarms) else pd.Series(dtype=int)
    oc = outcomes["outcome"].value_counts()
    tp_ev, tp_al, fp = int(oc.get("TP", 0)), int(cls.get("TP", 0)), int(cls.get("FP", 0))
    fn = int(oc.get("FN (no alarm)", 0) + oc.get("FN (already in alarm)", 0))
    tn = blocks["TN"]
    tot = tp_ev + fn + fp + tn
    metrics = {
        "actual flare events (scored)": tp_ev + fn, "flare events not scored": int(oc.get("not scored", 0)),
        "alarms (activations)": len(alarms), "alarms during a flare (not predictions)": int(cls.get("during flare", 0)),
        "predicted flare events (alarms outside flares)": tp_al + fp, "correct predictions (TP alarms)": tp_al,
        "incorrect predictions (FP alarms)": fp, "TP (events detected)": tp_ev, "FN (events missed)": fn,
        "FN: no alarm": int(oc.get("FN (no alarm)", 0)), "FN: already in alarm": int(oc.get("FN (already in alarm)", 0)),
        "TN (30-min flare-free blocks, no alarm)": tn, "30-min blocks with an FP activation": blocks["FP block"],
        "30-min blocks inside a standing alarm (not counted)": blocks["standing alarm"],
        "FP alarm minutes": int(alarms.loc[alarms["class"] == "FP", "duration_min"].sum()) if len(alarms) else 0,
        "base accuracy (TP+TN)/(TP+FN+FP+TN)": (tp_ev + tn) / tot if tot else np.nan,
        "precision TP alarms/(TP+FP alarms)": tp_al / (tp_al + fp) if tp_al + fp else np.nan,
        "recall TP events/(TP+FN events)": tp_ev / (tp_ev + fn) if tp_ev + fn else np.nan,
        "median lead time TP (min)": float(alarms.loc[alarms["class"] == "TP", "lead_min"].median()) if tp_al else np.nan,
    }
    for k, v in metrics.items():
        H.log(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return alarms, outcomes, metrics


def rescore(out: Path) -> None:
    """Re-score the saved outputs with the moving-window baseline and the scaled limits; write all t2/ and table files.

    Keeps the fixed-baseline T² as column T2_fixed in t2/t2_all_rows.parquet. Overwrites t2_all_rows.parquet,
    terms_all_rows.parquet, term_limits.csv, limits.csv, preflare_first_ooc.csv; writes tables/alarms.csv,
    tables/event_outcomes.csv, tables/metrics.csv.

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
    alarms, outcomes, metrics = evaluate(t2, ratio, nodes, events, loadings)
    alarms.to_csv(out / "tables" / "alarms.csv", index=False)
    outcomes.to_csv(out / "tables" / "event_outcomes.csv", index=False)
    pd.Series(metrics, name="value").rename_axis("metric").to_csv(out / "tables" / "metrics.csv")
    noc, sc = t2["noc"].to_numpy(), t2["scored"].to_numpy()
    new_lim = dict(old_lim, t2_limit=lim, t2_limit_fixed_baseline=lim_fixed, limit_scale=C.T2_LIMIT_SCALE,
                   baseline="moving window", window_rows=WL, step_rows=C.MW_STEP_MIN,
                   n_events_found=metrics["TP (events detected)"],
                   pct_noc_above_t2_limit=100 * float(t2["above_limit"][noc & sc].mean()),
                   pct_noc_above_term_limit_mean=100 * float(np.nanmean(ratio[noc & sc] > 1)),
                   pct_preflare_above_t2_limit=100 * float(t2["above_limit"][(t2["preflare_event_id"] > 0) & sc].mean()))
    pd.DataFrame([new_lim]).to_csv(out / "t2" / "limits.csv", index=False)
    first = outcomes.rename(columns={"crossing": "first_ooc"}).assign(
        detected=outcomes["outcome"].eq("TP"),
        minutes_before_onset=lambda d: (d["onset"] - pd.to_datetime(d["first_ooc"])) / pd.Timedelta(minutes=1))
    first.to_csv(out / "t2" / "preflare_first_ooc.csv", index=False)


# ---- EXCEL REPORT ----
def pc_frequency(alarms: pd.DataFrame, outcomes: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """How often each PC is a root cause: top-1 and top-3 for detected events, top-1 for FP alarms.

    Args:
        alarms: Alarm table from evaluate().
        outcomes: Event outcome table from evaluate().
        nodes: All PC names.

    Returns:
        Table indexed by PC with unit and the counts, sorted by top-1 count for detected events.
    """
    tp = outcomes[outcomes["outcome"] == "TP"]
    fp = alarms[alarms["class"] == "FP"]
    top3 = pd.concat([tp[f"pc{k}"] for k in range(1, C.TOP_N_ROOT + 1)])
    f = pd.DataFrame({"unit": [H.node_unit(v) for v in nodes]}, index=pd.Index(nodes, name="pc"))
    f["top1_detected_events"] = tp["pc1"].value_counts().reindex(nodes, fill_value=0)
    f[f"top{C.TOP_N_ROOT}_detected_events"] = top3.value_counts().reindex(nodes, fill_value=0)
    f["top1_FP_alarms"] = fp["pc1"].value_counts().reindex(nodes, fill_value=0)
    return f.sort_values(["top1_detected_events", f"top{C.TOP_N_ROOT}_detected_events"], ascending=False)


def write_excel(out: Path) -> Path:
    """Write outputs/RCA_report.xlsx: metrics, events, alarms, pc_frequency, unit_frequency, definitions.

    Args:
        out: Output folder after rescore().

    Returns:
        Path of the workbook.
    """
    alarms = pd.read_csv(out / "tables" / "alarms.csv")
    outcomes = pd.read_csv(out / "tables" / "event_outcomes.csv")
    metrics = pd.read_csv(out / "tables" / "metrics.csv")
    nodes = list(pd.read_csv(out / "t2" / "term_limits.csv")["node"])
    freq = pc_frequency(alarms, outcomes, nodes)
    unit = freq.groupby("unit").sum(numeric_only=True).sort_values("top1_detected_events", ascending=False)
    defs = pd.DataFrame({"term": ["alarm", "TP", "FP", "FN", "TN", "root cause", "ratio", "baseline", "limits"],
                         "definition": [
                             f"overall T² above its limit {C.ALARM_ON_DELAY_MIN} min in a row switches the alarm on; "
                             f"{C.ALARM_OFF_DELAY_MIN} min below switches it off; one activation = one alarm",
                             f"alarm switches on outside a flare and a flare event starts within {C.PRED_HORIZON_MIN} min",
                             f"alarm switches on outside a flare and no flare event starts within {C.PRED_HORIZON_MIN} min",
                             f"flare event with no alarm switching on in its {C.PRED_HORIZON_MIN} min window",
                             "30-min flare-free block outside all pre-flare windows with no alarm on",
                             f"top {C.TOP_N_ROOT} PCs by T²_j / own limit at the alarm's first minute above the limit "
                             "(FN events: at the window's highest T²)",
                             "T²_j / own limit; above 1 = PC out of control",
                             f"moving window: {C.MW_ROWS_PER_VAR} x (PCs + 1) flare-free trailing rows, "
                             f"re-estimated every {C.MW_STEP_MIN} min (TA's Main.py)",
                             f"Eq. 13 with m = window length, x{C.T2_LIMIT_SCALE:g} (overall and per-PC)"]})
    path = out / "RCA_report.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        metrics.to_excel(xw, sheet_name="metrics", index=False)
        outcomes.to_excel(xw, sheet_name="events", index=False)
        alarms.to_excel(xw, sheet_name="alarms", index=False)
        freq.to_excel(xw, sheet_name="pc_frequency")
        unit.to_excel(xw, sheet_name="unit_frequency")
        defs.to_excel(xw, sheet_name="definitions", index=False)
        for name, df in (("metrics", metrics), ("events", outcomes), ("alarms", alarms), ("definitions", defs)):
            for j, c in enumerate(df.columns):
                xw.sheets[name].set_column(j, j, min(60, max(10, len(str(c)) + 2)))
    H.log(f"saved {path.name}")
    return path


# ---- FIGURES ----
def plot_event(D: dict[str, Any], e: Any, r: pd.Series, alarm_spans: list[tuple], path: Path, plt: Any,
               mdates: Any) -> None:
    """One flare event: flare_net, overall T² with its limit and alarm, and the top-3 PCs as lines with own limits.

    Args:
        D: Plot inputs (flare, t2, terms, term_lim, t2_lim, thr, interval).
        e: Event row (event_id, start, end, duration_min, peak_flare_net).
        r: The event's row in event_outcomes.csv.
        alarm_spans: (start, end) of every alarm.
        path: PNG to write.
        plt: matplotlib.pyplot.
        mdates: matplotlib.dates.
    """
    col = C.PLOT_COLORS
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
    for k, (s, en) in enumerate(alarm_spans):
        if en >= a and s <= b:
            axs[1].axvspan(max(s, a), min(en, b), color="#f1c40f", alpha=0.25, lw=0, label="alarm on" if k >= 0 else None)
    if isinstance(r.get("crossing"), str) or isinstance(r.get("crossing"), pd.Timestamp):
        cx = pd.Timestamp(r["crossing"])
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
        ax.axvspan(e.start - pd.Timedelta(minutes=C.PRED_HORIZON_MIN), e.start, color=col["preflare"], alpha=0.08, lw=0)
        ax.axvspan(e.start, e.end + D["interval"], color=col["event"], alpha=0.18, lw=0)
        ax.grid(alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        if h:
            u = dict(zip(l, h))
            ax.legend(u.values(), u.keys(), loc="upper right", fontsize=7)
    H.format_time_axis(axs[-1], mdates, a, b)
    lead = f", alarm {r['lead_min']:.0f} min before onset" if r.get("outcome") == "TP" else ""
    fig.suptitle(f"E{e.event_id}: onset {e.start:%d %b %Y %H:%M}, {e.duration_min:.0f} min, peak {e.peak_flare_net:.2f} "
                 f"t/h. Outcome: {r.get('outcome')}{lead}. Orange = flare, pink = 30-min window, yellow = alarm on",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_all_events(D: dict[str, Any], out: Path) -> None:
    """One PNG per flare event in figures/events/ (replaces the stacked-share Figure 3).

    Args:
        D: Plot inputs from helpers.load_plot_inputs() after rescore().
        out: Output folder.
    """
    plt, mdates = H.plot_modules()
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv", parse_dates=["onset"]).set_index("event_id")
    al = pd.read_csv(out / "tables" / "alarms.csv", parse_dates=["crossing", "end"])
    spans = list(zip(al["crossing"], al["end"] + D["interval"]))
    folder = out / "figures" / "events"
    folder.mkdir(parents=True, exist_ok=True)
    for e in D["events"].itertuples(index=False):
        r = oc.loc[e.event_id]
        plot_event(D, e, r, spans, folder / f"E{e.event_id:03d}_{str(r['outcome']).split()[0]}.png", plt, mdates)
    H.log(f"saved {len(D['events'])} event figures in figures/events/")


def plot_root_cause_frequency(out: Path, path: Path) -> None:
    """Figure 6: root-cause frequency per PC (top-1 and top-3, detected events), per PC for FP alarms, and per unit.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, _ = H.plot_modules()
    alarms = pd.read_csv(out / "tables" / "alarms.csv")
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv")
    nodes = list(pd.read_csv(out / "t2" / "term_limits.csv")["node"])
    f = pc_frequency(alarms, oc, nodes)
    units = sorted(f["unit"].unique())
    ucol = {u: plt.cm.tab20(i % 20) for i, u in enumerate(units)}
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    n_tp = int((oc["outcome"] == "TP").sum())
    for ax, c, title in ((axs[0, 0], "top1_detected_events", f"Top-1 root cause, {n_tp} detected events"),
                         (axs[0, 1], f"top{C.TOP_N_ROOT}_detected_events",
                          f"In the top {C.TOP_N_ROOT}, {n_tp} detected events"),
                         (axs[1, 0], "top1_FP_alarms",
                          f"Top-1 PC of false alarms ({int((alarms['class'] == 'FP').sum())} FP)")):
        s = f[c].sort_values(ascending=False)
        ax.bar(range(len(s)), s.to_numpy(), color=[ucol[f.at[v, 'unit']] for v in s.index])
        ax.set_xticks(range(len(s)), s.index, rotation=70, ha="right", fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)
    u = f.groupby("unit")[["top1_detected_events", "top1_FP_alarms"]].sum().sort_values("top1_detected_events")
    y = np.arange(len(u))
    axs[1, 1].barh(y - 0.2, u["top1_detected_events"], 0.4, label="detected events (top-1)", color="#1f5fa8")
    axs[1, 1].barh(y + 0.2, u["top1_FP_alarms"], 0.4, label="false alarms (top-1)", color="#c0392b")
    axs[1, 1].set_yticks(y, u.index)
    axs[1, 1].legend()
    axs[1, 1].set_title("By unit: true vs false alarms", loc="left")
    axs[1, 1].grid(axis="x", alpha=0.3)
    H.save_figure(fig, path, plt)


def plot_metrics(out: Path, path: Path) -> None:
    """Figure 7: confusion counts, lead time of TP alarms, duration of FP alarms, outcome by month.

    Args:
        out: Output folder.
        path: PNG to write.
    """
    plt, _ = H.plot_modules()
    m = pd.read_csv(out / "tables" / "metrics.csv").set_index("metric")["value"].astype(float)
    al = pd.read_csv(out / "tables" / "alarms.csv")
    oc = pd.read_csv(out / "tables" / "event_outcomes.csv", parse_dates=["onset"])
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    ax = axs[0, 0]
    cm = np.array([[m["TP (events detected)"], m["FN (events missed)"]],
                   [m["incorrect predictions (FP alarms)"], m["TN (30-min flare-free blocks, no alarm)"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{['TP', 'FN', 'FP', 'TN'][2 * i + j]}\n{int(v)}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["alarm", "no alarm"])
    ax.set_yticks([0, 1], ["flare followed", "no flare"])
    ax.set_title(f"Accuracy {m['base accuracy (TP+TN)/(TP+FN+FP+TN)']:.2f}, precision "
                 f"{m['precision TP alarms/(TP+FP alarms)']:.2f}, recall {m['recall TP events/(TP+FN events)']:.2f}",
                 loc="left")
    tp, fp = al[al["class"] == "TP"], al[al["class"] == "FP"]
    axs[0, 1].hist(tp["lead_min"], bins=range(0, C.PRED_HORIZON_MIN + 2), color="#2ca02c")
    axs[0, 1].set_title(f"Lead time of TP alarms (alarm on -> flare onset), n = {len(tp)}", loc="left")
    axs[0, 1].set_xlabel("minutes")
    axs[1, 0].hist(fp["duration_min"].clip(upper=600), bins=40, color="#c0392b")
    axs[1, 0].set_title(f"Duration of FP alarms (capped at 600 min), n = {len(fp)}, total "
                        f"{int(fp['duration_min'].sum())} min", loc="left")
    axs[1, 0].set_xlabel("minutes")
    tab = oc.groupby([oc["onset"].dt.strftime("%Y-%m"), "outcome"]).size().unstack(fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=axs[1, 1], color=[{"TP": "#2ca02c", "FN (no alarm)": "#c0392b",
             "FN (already in alarm)": "#e67e22", "not scored": "#9e9e9e"}.get(c, "#555") for c in tab.columns])
    axs[1, 1].set_title("Flare events by month and outcome", loc="left")
    axs[1, 1].tick_params(axis="x", rotation=0)
    for ax in axs.ravel()[1:]:
        ax.grid(alpha=0.3)
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
