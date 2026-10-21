"""Stage 3 - §3.3 pre-flight and the PC algorithm on the 26 PC scores.

CPDAG (edges kept as found), block-bootstrap edge stability, and a unit / tag reading of every edge.
Writes outputs/data/{pc_graphs.npz, s3_summary.json} and the pc_* tables.
"""
import time

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from common import (ALPHAS, BOOT_B, DATA, EDGE_BIDIRECTED, EDGE_DIRECTED, EDGE_UNDIRECTED, K, KCI_N,
                    LAG_TAU, MAX_STEP_SECONDS, NEAR_EMPTY_EDGES, PCS, SEED, STABLE_FREQ, TAB, gate,
                    load_json, loading_mass, log, present_units, read_tag_map, save_json, stage)


def run_pc(data, alpha, test="fisherz"):
    """causal-learn PC, stable variant. Returns (graph matrix, number of distinct CI tests, seconds)."""
    t0 = time.time()
    cg = pc(np.asarray(data, dtype=float), alpha=alpha, indep_test=test, stable=True, show_progress=False)
    cache = getattr(cg.test, "pvalue_cache", None)
    return cg.G.graph.copy(), (len(cache) if cache is not None else -1), time.time() - t0


def edge_list(G, names):
    """CPDAG edges in causal-learn's encoding: G[i,j] = -1 and G[j,i] = 1 means i -> j;
    both -1 = undirected; both 1 = bidirected."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                rows.append((names[i], names[j], EDGE_DIRECTED))
            elif a == 1 and b == -1:
                rows.append((names[j], names[i], EDGE_DIRECTED))
            elif a == -1 and b == -1:
                rows.append((names[i], names[j], EDGE_UNDIRECTED))
            elif a == 1 and b == 1:
                rows.append((names[i], names[j], EDGE_BIDIRECTED))
            else:
                rows.append((names[i], names[j], f"other({a},{b})"))
    return pd.DataFrame(rows, columns=["source", "target", "type"])


def skeleton(edges):
    return {frozenset(p) for p in zip(edges["source"], edges["target"])}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def acf_time(x, max_lag):
    """First lag at which the autocorrelation drops below 1/e (FFT autocorrelation)."""
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    ac = ac / ac[0]
    below = np.flatnonzero(ac < np.exp(-1))
    return int(below[0]) if len(below) else max_lag


def lagged_scores(T, tau, interval):
    """[PC(t), PC(t-1), ..., PC(t-tau)] on rows where every lag exists (respects time gaps)."""
    parts = [T.rename(columns=lambda c: f"{c}(t)")]
    for k in range(1, tau + 1):
        parts.append(T.shift(freq=k * interval).rename(columns=lambda c, k=k: f"{c}(t-{k})"))
    return pd.concat(parts, axis=1, join="inner")


stage("Stage 3 - PC algorithm")
T_train = pd.read_parquet(DATA / "T_train.parquet")
T_test = pd.read_parquet(DATA / "T_test.parquet")
labels = pd.read_parquet(DATA / "labels.parquet")
interval = pd.Timedelta(seconds=load_json("audit.json")["interval_s"])

# ---- §3.3 step 1: correlation of the scores
off = ~np.eye(K, dtype=bool)
c_tr, c_te = T_train.corr(), T_test.corr()
max_tr, max_te = float(np.abs(c_tr.to_numpy()[off]).max()), float(np.abs(c_te.to_numpy()[off]).max())
c_tr.to_csv(TAB / "pc_score_corr_train.csv")
c_te.to_csv(TAB / "pc_score_corr_test.csv")
log(f"§3.3 step 1: max |off-diagonal corr| T_train = {max_tr:.2e}, T_test = {max_te:.3f}")

variants, edges_by, graphs = [], {}, {}


def add_variant(name, data, alpha, test="fisherz", note=""):
    G, n_tests, secs = run_pc(data.to_numpy(), alpha, test)
    e = edge_list(G, list(data.columns))
    e.to_csv(TAB / f"pc_edges_{name}_alpha{str(alpha).replace('.', 'p')}.csv", index=False)
    edges_by[(name, alpha)], graphs[(name, alpha)] = e, G
    variants.append({"variant": name, "test": test, "alpha": alpha, "rows": len(data), "nodes": data.shape[1],
                     "edges": len(e), "directed": int((e["type"] == EDGE_DIRECTED).sum()),
                     "undirected": int((e["type"] == EDGE_UNDIRECTED).sum()),
                     "bidirected": int((e["type"] == EDGE_BIDIRECTED).sum()),
                     "ci_tests": n_tests, "seconds": round(secs, 1), "note": note})
    log(f"  PC {name} ({test}, alpha={alpha}, n={len(data)}): {len(e)} edges, {n_tests} CI tests, {secs:.1f}s")
    return e


# ---- §3.3 step 2: the planned PC on T_train
for a in ALPHAS:
    add_variant("T_train_fisherz", T_train, a, note="planned method")
n_base = len(edges_by[("T_train_fisherz", 0.05)])
near_empty = n_base < NEAR_EMPTY_EDGES
log(f"§3.3 step 2: planned PC on T_train -> {n_base} edges ({'NEAR-EMPTY' if near_empty else 'not near-empty'}, "
    f"threshold {NEAR_EMPTY_EDGES})")

# ---- §3.3 step 3: variants, only when the planned graph is near-empty
skipped_variants = []
if near_empty:
    log("REPORT: planned graph is near-empty, as expected for uncorrelated PCA scores. Running variants (a)-(c); "
        "the planned method is NOT replaced silently.")
    for a in ALPHAS:
        add_variant("T_test_fisherz", T_test, a, note="(a) TEST scores, TRAIN-fit PCA")
    T_flare = T_test[labels.loc[T_test.index, "flare"].to_numpy() == 1]
    if len(T_flare) >= 10 * K:
        for a in ALPHAS:
            add_variant("flare_only_fisherz", T_flare, a, note="(a) flare-window rows only")
    else:
        skipped_variants.append(f"flare-window-only: {len(T_flare)} flare rows (< {10 * K}) or no labels")
    sub = T_train.iloc[np.linspace(0, len(T_train) - 1, min(KCI_N, len(T_train))).astype(int)]
    add_variant("T_train_kci", sub, 0.05, test="kci",
                note=f"(b) KCI on an evenly spaced subsample of n={len(sub)} TRAIN rows; alpha=0.01 not run (runtime)")
    lag = lagged_scores(T_train, LAG_TAU, interval)
    for a in ALPHAS:
        add_variant("T_train_lagged_fisherz", lag, a, note=f"(c) DIAGNOSTIC ONLY: lags 0..{LAG_TAU}")
for s in skipped_variants:
    log(f"  skipped variant - {s}")

# ---- §3.3 step 4: the final variant (rule agreed with the project owner before the run)
if not near_empty:
    final_name, final_data = "T_train_fisherz", T_train
    reason = f"planned Fisher-Z PC on T_train found {n_base} edges (>= {NEAR_EMPTY_EDGES}); plan kept"
else:
    final_name, final_data = "T_test_fisherz", T_test
    reason = (f"planned graph near-empty ({n_base} < {NEAR_EMPTY_EDGES} edges); pre-agreed rule: final graph = "
              "Fisher-Z PC on T_test scores (variant a). KCI and lagged runs are comparisons only.")
log(f"final variant: {final_name} - {reason}")
final_edges = edges_by[(final_name, 0.05)].copy()
G_final = graphs[(final_name, 0.05)]

sens = []
for name in dict.fromkeys(v["variant"] for v in variants):
    a, b = edges_by.get((name, 0.05)), edges_by.get((name, 0.01))
    same_nodes = "lagged" not in name
    sens.append({"variant": name,
                 "jaccard_alpha0p05_vs_0p01": jaccard(skeleton(a), skeleton(b)) if b is not None else np.nan,
                 "jaccard_vs_final_alpha0p05": jaccard(skeleton(a), skeleton(final_edges)) if same_nodes else np.nan})
pd.DataFrame(variants).to_csv(TAB / "pc_variants_summary.csv", index=False)
pd.DataFrame(sens).to_csv(TAB / "pc_sensitivity.csv", index=False)

# ---- block bootstrap of the final variant
arr = final_data.to_numpy()
n = len(arr)
max_lag = max(1, min(n // 10, 5000))
L = max(1, int(np.median([acf_time(arr[:, i], max_lag) for i in range(K)])))
log(f"bootstrap: moving-block, block length L = {L} rows (median 1/e autocorrelation time of the 26 scores)")
rng = np.random.default_rng(SEED)
adj, dir_, und, bi = (np.zeros((K, K)) for _ in range(4))
B, b, reduction = BOOT_B, 0, ""
while b < B:
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    idx = (starts[:, None] + np.arange(L)).ravel()[:n]
    G, _, secs = run_pc(arr[idx], 0.05)
    adj += (G != 0) | (G.T != 0)
    dir_ += (G == -1) & (G.T == 1)          # dir_[i, j]: i -> j
    und += (G == -1) & (G.T == -1)
    bi += (G == 1) & (G.T == 1)
    b += 1
    if b == 1 and secs * BOOT_B > MAX_STEP_SECONDS:
        B = max(10, int(MAX_STEP_SECONDS // max(secs, 1e-9)))
        reduction = f"B reduced from {BOOT_B} to {B}: one PC run took {secs:.0f}s (B={BOOT_B} would exceed 10 min)"
        log(reduction)
    if b % 10 == 0:
        log(f"  bootstrap {b}/{B}")

pos = {p: i for i, p in enumerate(PCS)}
final_edges["boot_freq"] = [adj[pos[s], pos[t]] / B for s, t in zip(final_edges["source"], final_edges["target"])]
final_edges["stable"] = final_edges["boot_freq"] >= STABLE_FREQ
final_edges.to_csv(TAB / "pc_edges_final.csv", index=False)
stable_edges = final_edges[final_edges["stable"]]
stable_edges.to_csv(TAB / "pc_edges_stable.csv", index=False)
in_final = {frozenset((s, t)): f"{s} {ty} {t}" for s, t, ty in final_edges[["source", "target", "type"]].itertuples(index=False)}
boot_rows = []
for i in range(K):
    for j in range(i + 1, K):
        key = frozenset((PCS[i], PCS[j]))
        if adj[i, j] == 0 and key not in in_final:
            continue
        boot_rows.append({"pc_a": PCS[i], "pc_b": PCS[j], "freq": adj[i, j] / B, "freq_a_to_b": dir_[i, j] / B,
                          "freq_b_to_a": dir_[j, i] / B, "freq_undirected": und[i, j] / B,
                          "freq_bidirected": bi[i, j] / B, "final_edge": in_final.get(key, "")})
pd.DataFrame(boot_rows).to_csv(TAB / "pc_bootstrap_freq.csv", index=False)
log(f"final graph {len(final_edges)} edges; stable (freq >= {STABLE_FREQ}): {len(stable_edges)}; B = {B}")

# ---- map PCs and edges back to units and tags
z = np.load(DATA / "pca.npz")
P, cols = z["P"], [str(c) for c in z["columns"]]
tm = read_tag_map()
unit_of = dict(zip(tm["column"], tm["unit"]))
desc_of = dict(zip(tm["column"], tm["description"]))
lm = loading_mass(P, cols, unit_of, present_units(cols, unit_of))
evr = z["lam"] / z["eig_full"].sum()


def top_tags(pc_name, n_top=5):
    s = pd.Series(P[:, pos[pc_name]], index=cols)
    top = s.abs().sort_values(ascending=False).index[:n_top]
    return "; ".join(f"{t} ({unit_of[t]}, {s[t]:+.2f}{', ' + desc_of[t] if desc_of[t] else ''})" for t in top)


def dom_units(pc_name, n_top=2):
    r = lm.loc[pc_name].sort_values(ascending=False)[:n_top]
    return "; ".join(f"{u} ({v:.0%})" for u, v in r.items())


nodes = pd.DataFrame({"PC": PCS, "explained_var_ratio": evr,
                      "dominant_unit": [lm.loc[p].idxmax() for p in PCS],
                      "dominant_share": [lm.loc[p].max() for p in PCS],
                      "dominant_units": [dom_units(p) for p in PCS], "top5_tags": [top_tags(p) for p in PCS]})
nodes.to_csv(TAB / "pc_node_labels.csv", index=False)
interp = final_edges.copy()
for side in ("source", "target"):
    interp[f"{side}_units"] = interp[side].map(dom_units)
    interp[f"{side}_top5_tags"] = interp[side].map(top_tags)
interp.to_csv(TAB / "pc_edge_interpretation.csv", index=False)
gate(nodes["dominant_unit"].notna().all(), "every PC node has a unit-level label")

np.savez(DATA / "pc_graphs.npz", G_final=G_final, adj_freq=adj / B, dir_freq=dir_ / B,
         und_freq=und / B, bi_freq=bi / B)
save_json({"max_offdiag_corr_train": max_tr, "max_offdiag_corr_test": max_te, "n_edges_planned": n_base,
           "near_empty": near_empty, "near_empty_threshold": NEAR_EMPTY_EDGES, "final_variant": final_name,
           "final_reason": reason, "n_edges_final": len(final_edges), "n_edges_stable": len(stable_edges),
           "bootstrap_B": B, "bootstrap_reduction": reduction, "block_length": L, "kci_n": KCI_N,
           "skipped_variants": skipped_variants, "variants": variants}, "s3_summary.json")
gate(True, "graph, edge list, CI-test counts and bootstrap frequencies saved")
