"""Stage 4 - Hotelling T² and Q (SPE) with the 26-PC model: limits, two-level decomposition,
alarm events, detection metrics against the flare labels, and links to the Stage 3 graph.

Writes outputs/data/{t2q, unit_contrib_signed, unit_contrib_clipped}.parquet, s4_summary.json and t2_* tables.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from common import (DATA, EDGE_DIRECTED, EDGE_UNDIRECTED, EVENT_GAP, K, MATCH_WINDOW, PCS, TAB,
                    flags_to_events, gate, load_json, log, present_units, read_flare_events,
                    read_tag_map, save_json, stage)

CHUNK = 50_000          # rows per block, keeps the n x tags contribution matrix out of memory

stage("Stage 4 - T² / Q analysis")
z = np.load(DATA / "pca.npz")
P, lam, mean, eig = z["P"], z["lam"], z["mean"], z["eig_full"]
cols = [str(c) for c in z["columns"]]
X_train = pd.read_parquet(DATA / "X_train.parquet")
X = pd.concat([X_train, pd.read_parquet(DATA / "X_test.parquet")]).sort_index()
gate(list(X.columns) == cols, "data columns match the PCA model")
labels = pd.read_parquet(DATA / "labels.parquet").reindex(X.index)
is_train = (labels["set"] == "TRAIN").to_numpy()
flare = labels["flare"].to_numpy() == 1
interval = pd.Timedelta(seconds=load_json("audit.json")["interval_s"])
has_labels = load_json("s1_summary.json")["has_labels"]
tm = read_tag_map()
unit_of = dict(zip(tm["column"], tm["unit"]))
units = present_units(cols, unit_of)
M = np.zeros((len(cols), len(units)))                 # tag -> unit indicator
M[np.arange(len(cols)), [units.index(unit_of[c]) for c in cols]] = 1.0


def var_contrib(xc):
    """Signed per-tag T² contributions, Kourti-MacGregor form summed over the 26 PCs:
    c_j = x_j * sum_a p_ja t_a / lambda_a. They can be negative and sum over tags exactly to T²."""
    t = xc @ P
    return xc * ((t / lam) @ P.T)


# ---- statistics per sample
Xv = X.to_numpy()
n = len(Xv)
T2, Q = np.empty(n), np.empty(n)
C = np.empty((n, K))
U_signed, U_clipped = np.empty((n, len(units))), np.empty((n, len(units)))
max_var_err = 0.0
for a in range(0, n, CHUNK):
    b = min(a + CHUNK, n)
    xc = Xv[a:b] - mean
    t = xc @ P
    C[a:b] = t ** 2 / lam                                  # PC-wise contributions
    T2[a:b] = np.einsum("ij,ij->i", t / lam, t)            # t' Lambda^-1 t
    Q[a:b] = ((xc - t @ P.T) ** 2).sum(axis=1)             # residual outside the 26 PCs
    vc = xc * ((t / lam) @ P.T)
    max_var_err = max(max_var_err, float(np.abs(vc.sum(axis=1) - T2[a:b]).max()))
    U_signed[a:b] = vc @ M
    U_clipped[a:b] = np.clip(vc, 0, None) @ M

gate(np.allclose(C.sum(axis=1), T2), "sum of PC contributions == T² (np.allclose)")
gate(max_var_err <= 1e-6 * max(1.0, T2.max()), f"signed tag contributions sum to T² (max abs error {max_var_err:.2e})")

# ---- control limits from TRAIN
n_tr = int(is_train.sum())


def t2_ucl(alpha):
    return K * (n_tr ** 2 - 1) / (n_tr * (n_tr - K)) * stats.f.ppf(1 - alpha, K, n_tr - K)


theta = [float(np.sum(eig[K:] ** i)) for i in (1, 2, 3)]


def q_ucl(alpha):
    """Jackson-Mudholkar limit from the residual eigenvalues lambda_27..lambda_p."""
    th1, th2, th3 = theta
    if th1 <= 0 or th2 <= 0:
        return np.nan
    h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
    ca = stats.norm.ppf(1 - alpha)
    return th1 * (ca * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)


lim = {"T2_95": t2_ucl(0.05), "T2_99": t2_ucl(0.01), "Q_95": q_ucl(0.05), "Q_99": q_ucl(0.01),
       "T2_99_empirical_train": float(np.percentile(T2[is_train], 99)),
       "Q_99_empirical_train": float(np.percentile(Q[is_train], 99))}
q_method = "Jackson-Mudholkar"
if not np.isfinite(lim["Q_99"]):
    lim["Q_95"], lim["Q_99"] = float(np.percentile(Q[is_train], 95)), lim["Q_99_empirical_train"]
    q_method = "empirical TRAIN percentiles (no residual eigenvalues for Jackson-Mudholkar)"
pd.DataFrame({"limit": list(lim), "value": list(lim.values())}).to_csv(TAB / "control_limits.csv", index=False)
log(f"limits: {', '.join(f'{k}={v:.3g}' for k, v in lim.items())}; Q limit used: {q_method}")

alarm_T2 = T2 > lim["T2_99"]
alarm_Q = Q > lim["Q_99"]

# ---- alarm events (T² > 99 % UCL; alarms <= EVENT_GAP samples apart merged)
idx = X.index
fl = read_flare_events()
ev = flags_to_events(idx, alarm_T2, EVENT_GAP * interval)
log(f"events: {len(ev)} (T² > 99% UCL, gap tolerance {EVENT_GAP} samples = {EVENT_GAP * interval})")
ev_rows, peak_rows, ev_pcs = [], [], {}
for k, e in enumerate(ev.itertuples(), start=1):
    i0, i1 = idx.searchsorted(e.start), idx.searchsorted(e.end, side="right")
    sl = slice(i0, i1)
    pk = i0 + int(np.argmax(T2[sl]))
    top_pc = np.argsort(C[sl].mean(axis=0))[::-1][:3]
    top_u = np.argsort(U_signed[sl].mean(axis=0))[::-1][:3]
    top_t = np.argsort(var_contrib(Xv[sl] - mean).mean(axis=0))[::-1][:10]
    vc_pk = var_contrib(Xv[pk:pk + 1] - mean)[0]
    for j in np.argsort(vc_pk)[::-1][:30]:
        peak_rows.append({"event_id": k, "column": cols[j], "unit": unit_of[cols[j]],
                          "contrib_signed": vc_pk[j], "contrib_clipped": max(vc_pk[j], 0.0)})
    match = fl[(fl["start"] - MATCH_WINDOW <= e.start) & (e.start <= fl["end"])]
    if len(match):
        m = match.loc[(match["start"] - e.start).abs().idxmin()]
        flare_id, onset, lead = int(m["event_id"]), m["start"], (m["start"] - e.start).total_seconds() / 60
    else:
        flare_id, onset, lead = np.nan, pd.NaT, np.nan
    ev_pcs[k] = [PCS[i] for i in top_pc]
    ev_rows.append({"event_id": k, "start": e.start, "end": e.end,
                    "duration_min": (e.end - e.start + interval).total_seconds() / 60,
                    "n_alarm_samples": int(e.n_flagged), "set": "TRAIN" if is_train[sl].mean() > 0.5 else "TEST",
                    "peak_time": idx[pk], "peak_T2": T2[pk],
                    "area_above_ucl99": float(np.clip(T2[sl] - lim["T2_99"], 0, None).sum()),
                    "top3_pcs": "; ".join(ev_pcs[k]), "top3_units": "; ".join(units[i] for i in top_u),
                    "top10_tags": "; ".join(cols[i] for i in top_t),
                    "flare_event_id": flare_id, "flare_onset": onset, "lead_to_flare_min": lead})
events = pd.DataFrame(ev_rows, columns=["event_id", "start", "end", "duration_min", "n_alarm_samples", "set",
                                        "peak_time", "peak_T2", "area_above_ucl99", "top3_pcs", "top3_units",
                                        "top10_tags", "flare_event_id", "flare_onset", "lead_to_flare_min"])
events.to_csv(TAB / "t2_events.csv", index=False)
pd.DataFrame(peak_rows, columns=["event_id", "column", "unit", "contrib_signed", "contrib_clipped"]).to_csv(
    TAB / "event_peak_tag_contrib.csv", index=False)

# ---- detection metrics (labels never invented: everything below needs flare_data events)
metrics = {"far_train_T2_95": float((T2[is_train] > lim["T2_95"]).mean()),
           "far_train_T2_99": float(alarm_T2[is_train].mean()), "far_train_Q_99": float(alarm_Q[is_train].mean()),
           "far_train_T2_or_Q_99": float((alarm_T2 | alarm_Q)[is_train].mean()),
           "test_alarm_rate_T2_99": float(alarm_T2[~is_train].mean()), "test_alarm_rate_Q_99": float(alarm_Q[~is_train].mean())}
det = pd.DataFrame(columns=["flare_event_id", "onset", "end", "observable", "detected", "first_alarm", "delay_min", "lead_min"])
if has_labels:
    det_rows = []
    for f in fl.itertuples():
        i0, i1 = idx.searchsorted(f.start - MATCH_WINDOW), idx.searchsorted(f.end, side="right")
        hits = np.flatnonzero(alarm_T2[i0:i1])
        first = idx[i0 + hits[0]] if len(hits) else pd.NaT
        delay = (first - f.start).total_seconds() / 60 if len(hits) else np.nan
        det_rows.append({"flare_event_id": f.event_id, "onset": f.start, "end": f.end, "observable": i1 > i0,
                         "detected": len(hits) > 0, "first_alarm": first, "delay_min": delay, "lead_min": -delay})
    det = pd.DataFrame(det_rows)
    obs = det[det["observable"]]
    test_flare, test_norm = ~is_train & flare, ~is_train & ~flare
    metrics.update({"n_flares": len(det), "n_flares_observable": len(obs), "n_flares_detected": int(obs["detected"].sum()),
                    "event_detection_rate": float(obs["detected"].mean()) if len(obs) else np.nan,
                    "median_detection_delay_min": float(obs["delay_min"].median()) if obs["detected"].any() else np.nan,
                    "sample_detection_rate_T2_99": float(alarm_T2[test_flare].mean()) if test_flare.any() else np.nan,
                    "sample_false_alarm_test_normal_T2_99": float(alarm_T2[test_norm].mean()) if test_norm.any() else np.nan})
    y = flare[~is_train]
    if 0 < y.sum() < len(y):
        for name, score in (("T2", T2), ("Q", Q)):
            fpr, tpr, thr = roc_curve(y, score[~is_train])
            metrics[f"auc_{name}"] = float(roc_auc_score(y, score[~is_train]))
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(TAB / f"roc_{name}.csv", index=False)
det.to_csv(TAB / "flare_detection.csv", index=False)
pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())}).to_csv(TAB / "detection_metrics.csv", index=False)
log("metrics: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

# ---- link to Stage 3: are the dominant PCs ancestors of the PC most elevated at flare onset / peak?
stable = pd.read_csv(TAB / "pc_edges_stable.csv")
graph_used = "stable graph (bootstrap freq >= 0.7)" if len(stable) else "full final CPDAG (no stable edges)"
E = stable if len(stable) else pd.read_csv(TAB / "pc_edges_final.csv")
D, Dp = nx.DiGraph(), nx.DiGraph()       # definite (directed edges) and possible (directed + undirected both ways)
D.add_nodes_from(PCS)
Dp.add_nodes_from(PCS)
for s, t, ty in E[["source", "target", "type"]].itertuples(index=False):
    if ty == EDGE_DIRECTED:
        D.add_edge(s, t)
        Dp.add_edge(s, t)
    elif ty == EDGE_UNDIRECTED:
        Dp.add_edge(s, t)
        Dp.add_edge(t, s)
dom = dict(pd.read_csv(TAB / "pc_node_labels.csv")[["PC", "dominant_unit"]].itertuples(index=False))
test_ev = events[events["set"] == "TEST"]
chosen = pd.concat([test_ev.nlargest(10, "area_above_ucl99"), test_ev[test_ev["flare_event_id"].notna()]]).drop_duplicates("event_id")
hyp = []
for e in chosen.itertuples():
    use_onset = pd.notna(e.flare_onset)
    ri = idx.get_indexer([e.flare_onset if use_onset else e.peak_time], method="nearest")[0]
    elev = PCS[int(np.argmax(C[ri]))]
    tops = ev_pcs[e.event_id]
    definite = [p for p in tops if p != elev and p in nx.ancestors(D, elev)]
    possible = [p for p in tops if p != elev and p not in definite and p in nx.ancestors(Dp, elev)]
    hyp.append({"event_id": e.event_id, "reference": "flare onset" if use_onset else "event peak",
                "reference_time": idx[ri], "elevated_pc": elev, "elevated_pc_unit": dom[elev],
                "top3_pcs": "; ".join(tops), "definite_ancestors": "; ".join(definite),
                "possible_ancestors": "; ".join(possible),
                "candidate_root_units": "; ".join(sorted({dom[p] for p in definite + possible})),
                "status": "HYPOTHESIS - not a conclusion"})
pd.DataFrame(hyp, columns=["event_id", "reference", "reference_time", "elevated_pc", "elevated_pc_unit", "top3_pcs",
                           "definite_ancestors", "possible_ancestors", "candidate_root_units", "status"]).to_csv(
    TAB / "root_cause_hypotheses.csv", index=False)

# ---- save per-sample results
out = pd.DataFrame({"set": labels["set"].to_numpy(), "flare": flare.astype(int), "T2": T2, "Q": Q,
                    "alarm_T2": alarm_T2.astype(int), "alarm_Q": alarm_Q.astype(int)}, index=idx)
out = pd.concat([out, pd.DataFrame(C, index=idx, columns=PCS)], axis=1)
out.to_parquet(DATA / "t2q.parquet")
pd.DataFrame(U_signed, index=idx, columns=units).to_parquet(DATA / "unit_contrib_signed.parquet")
pd.DataFrame(U_clipped, index=idx, columns=units).to_parquet(DATA / "unit_contrib_clipped.parquet")
save_json({"limits": lim, "q_limit_method": q_method, "n_train": n_tr, "n_events": len(events),
           "n_events_test": int((events["set"] == "TEST").sum()), "event_gap_samples": EVENT_GAP,
           "event_rule": "T² > 99% UCL", "match_window": str(MATCH_WINDOW), "metrics": metrics,
           "graph_used_for_links": graph_used, "n_hypotheses": len(hyp)}, "s4_summary.json")
gate(True, "T² / Q statistics, limits, events, metrics and hypotheses saved")
