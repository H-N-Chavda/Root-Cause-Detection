"""Stage 5 - figures 1-23 (300 dpi PNG + vector PDF + CSV of the plotted numbers) and RESULTS.md.

Reads only the artifacts saved by stages 0-4, so it can be rerun on its own.
"""
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from common import (ALPHAS, DATA, EDGE_BIDIRECTED, EDGE_DIRECTED, EDGE_UNDIRECTED, EVENT_GAP,
                    EXPECTED_N_PARAMS, FFILL_MAX_GAP, FIG_INDEX, FLARE_COLOR, K, KCI_N, MATCH_WINDOW,
                    PCS, PROJECT, SEED, STABLE_FREQ, TAB, TEST_COLOR, TRAIN_COLOR, UNIT_COLORS, UNITS,
                    bin_for_plot, load_json, loading_mass, log, md_table, present_units,
                    read_flare_events, read_tag_map, save_fig, shade_flares, stage, unit_handles)

pd.plotting.register_matplotlib_converters()
stage("Stage 5 - figures and RESULTS.md")
audit, s1, s2, s3, s4 = (load_json(f) for f in ("audit.json", "s1_summary.json", "s2_summary.json",
                                                 "s3_summary.json", "s4_summary.json"))
tm = read_tag_map()
unit_of = dict(zip(tm["column"], tm["unit"]))
X_train = pd.read_parquet(DATA / "X_train.parquet")
cols = list(X_train.columns)
units = present_units(cols, unit_of)
labels = pd.read_parquet(DATA / "labels.parquet")
fl = read_flare_events()
T = pd.concat([pd.read_parquet(DATA / "T_train.parquet"), pd.read_parquet(DATA / "T_test.parquet")]).sort_index()
t2q = pd.read_parquet(DATA / "t2q.parquet")
P = np.load(DATA / "pca.npz")["P"]
lm = loading_mass(P, cols, unit_of, units)
lim, metrics = s4["limits"], s4["metrics"]
cut = pd.Timestamp(s1["cut"])
interval = pd.Timedelta(seconds=audit["interval_s"])
events = pd.read_csv(TAB / "t2_events.csv", parse_dates=["start", "end", "peak_time", "flare_onset"])
test_events = events[events["set"] == "TEST"]
top5 = test_events.nlargest(5, "area_above_ucl99")
nodes = pd.read_csv(TAB / "pc_node_labels.csv")
fe = pd.read_csv(TAB / "pc_edges_final.csv")
se = fe[fe["stable"].astype(bool)]
skipped = []            # (figure, reason) for RESULTS.md


def ts_label(t):
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def mark_cut(ax):
    ax.axvline(cut, color="k", ls="--", lw=1, label="TRAIN/TEST cut (60 % of time)")


# ============================================================================ A. data and preprocessing
mu = pd.read_csv(TAB / "missing_by_unit.csv")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x, w = np.arange(len(mu)), 0.4
for ax, (before, after, ylab, title) in zip(axes, [
        ("n_params_raw", "n_params_clean", "Number of tags", "Parameters per unit"),
        ("missing_before_pct", "missing_after_pct", "Missing data (%)", "Missing data per unit")]):
    ax.bar(x - w / 2, mu[before], w, color="#BBBBBB", edgecolor="k", label="before cleaning")
    ax.bar(x + w / 2, mu[after], w, color=[UNIT_COLORS[u] for u in mu["unit"]], edgecolor="k",
           label="after cleaning (unit colour)")
    ax.set_xticks(x, mu["unit"], rotation=45, ha="right")
    ax.set_xlabel("Unit")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
save_fig(fig, "fig01_parameters_missing_by_unit", "Parameters and missing data per unit: before vs after cleaning",
         f"{int(mu['n_params_raw'].sum())} tags in {len(mu)} units -> {int(mu['n_params_clean'].sum())} kept; "
         f"mean missing {mu['missing_before_pct'].mean():.1f}% -> {mu['missing_after_pct'].mean():.1f}% "
         f"(after = after the <= {FFILL_MAX_GAP}-sample fill, kept tags)", mu)

cov = pd.read_csv(TAB / "coverage_daily.csv", index_col=0, parse_dates=True)
cu = [u for u in UNITS if u in cov.columns]
fig, ax = plt.subplots(figsize=(15, 0.5 * len(cu) + 2.5))
for k, u in enumerate(cu):
    days = cov.index[cov[u] > 0.5]
    ax.broken_barh([(mdates.date2num(d), 1.0) for d in days], (k - 0.4, 0.8), facecolors=UNIT_COLORS[u])
shade_flares(ax, fl)
mark_cut(ax)
ax.set_yticks(range(len(cu)), cu)
ax.invert_yaxis()
ax.xaxis_date()
ax.set_xlabel("Date")
ax.set_ylabel("Unit")
ax.legend(loc="upper right")
save_fig(fig, "fig02_time_coverage", "Time coverage per unit with flare events",
         f"Data from {cov.index.min():%Y-%m-%d} to {cov.index.max():%Y-%m-%d}; a bar = day with > 50% of the "
         f"unit's tags present; {len(fl)} flare events shaded", cov, index=True)

corr = X_train.corr()
cu_arr = np.array([unit_of[c] for c in cols])
same, off = cu_arr[:, None] == cu_arr[None, :], ~np.eye(len(cols), dtype=bool)
A = np.abs(corr.to_numpy())
within, between = np.nanmean(A[same & off]), np.nanmean(A[~same]) if (~same).any() else np.nan
fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
bounds = np.r_[0, np.cumsum([int((cu_arr == u).sum()) for u in units])]
for b in bounds[1:-1]:
    ax.axhline(b - 0.5, color="k", lw=0.8)
    ax.axvline(b - 0.5, color="k", lw=0.8)
centres = (bounds[:-1] + bounds[1:]) / 2 - 0.5
ax.set_xticks(centres, units, rotation=90)
ax.set_yticks(centres, units)
ax.set_xlabel("Tags, grouped by unit")
ax.set_ylabel("Tags, grouped by unit")
fig.colorbar(im, ax=ax, label="Pearson r (TRAIN, z-scored)", shrink=0.8)
save_fig(fig, "fig03_tag_correlation", f"Correlation of the {len(cols)} tags, ordered by unit",
         f"Mean |r| within units {within:.2f} vs between units {between:.2f} - block structure motivates PCA",
         corr, index=True)

# ============================================================================ B. PCA
var = pd.read_csv(TAB / "pca_variance.csv")
v = var.iloc[:min(60, len(var))]
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(v["PC"], v["eigenvalue"], "o-", color="#0072B2", ms=4, label="eigenvalue")
ax.axvline(K, color=FLARE_COLOR, ls="--", label=f"PC {K} (k used)")
ax.plot(K, var["eigenvalue"].iloc[K - 1], "*", color=FLARE_COLOR, ms=16)
ax.set_xlabel("Principal component index")
ax.set_ylabel("Eigenvalue λ (variance, z-score units²)")
ax.legend()
save_fig(fig, "fig04_scree", "Scree plot (TRAIN)",
         f"λ1 = {var['eigenvalue'].iloc[0]:.1f}, λ{K} = {var['eigenvalue'].iloc[K - 1]:.2f}; "
         f"first {K} PCs carry {s2['cum_var_k']:.1%} of TRAIN variance", v)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(var["PC"], 100 * var["cumulative"], color="#0072B2", lw=2)
for (thr, npc), c in zip(s2["n_pcs_for"].items(), ["#009E73", "#E69F00", "#CC79A7", "#56B4E9"]):
    ax.axhline(float(thr.rstrip("%")), color=c, ls=":", label=f"{thr}: {npc} PCs")
ax.axvline(K, color=FLARE_COLOR, ls="--", label=f"PC {K}: {100 * s2['cum_var_k']:.1f}%")
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance (%)")
ax.legend(loc="lower right")
save_fig(fig, "fig05_cumulative_variance", "Cumulative explained variance (TRAIN)",
         f"{K} PCs explain {s2['cum_var_k']:.1%}; 80/90/95/99% need "
         + "/".join(str(n) for n in s2["n_pcs_for"].values()) + " PCs", var[["PC", "cumulative"]])

fig, ax = plt.subplots(figsize=(15, 0.4 * len(units) + 3))
im = ax.imshow(lm.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(K), PCS, rotation=90)
ax.set_yticks(range(len(units)), units)
ax.set_xlabel("Principal component")
ax.set_ylabel("Unit")
fig.colorbar(im, ax=ax, label="Share of the PC's squared loadings")
save_fig(fig, "fig06_unit_loading_mass", "Which unit lives in which PC (loading mass)",
         "; ".join(f"{p} <- {lm.loc[p].idxmax()} ({lm.loc[p].max():.0%})" for p in PCS[:3]),
         lm.rename_axis("PC"), index=True)

load = pd.DataFrame(P, index=cols, columns=PCS)
npc = min(6, K)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
rows = []
for i, ax in enumerate(axes.ravel()[:npc]):
    s = load[PCS[i]]
    top = s.reindex(s.abs().sort_values(ascending=False).index[:10])[::-1]
    ax.barh(range(len(top)), top.to_numpy(), color=[UNIT_COLORS.get(unit_of[t], "#BBBBBB") for t in top.index], edgecolor="k")
    ax.set_yticks(range(len(top)), top.index)
    ax.set_xlabel("Loading")
    ax.set_title(f"{PCS[i]} (dominant: {lm.loc[PCS[i]].idxmax()})")
    rows += [{"PC": PCS[i], "column": t, "unit": unit_of[t], "loading": top[t]} for t in top.index[::-1]]
fig.legend(handles=unit_handles(units), loc="lower center", ncol=min(7, len(units)), title="unit")
first = rows[0]
save_fig(fig, "fig07_top_tag_loadings", f"Top-10 tag loadings of PC1-PC{npc}",
         f"PC1 led by {first['column']} ({first['unit']}); dominant units PC1-PC{npc}: "
         + ", ".join(lm.loc[p].idxmax() for p in PCS[:npc]), pd.DataFrame(rows), bottom=0.12)

lab = labels.reindex(T.index)
grp = np.where(lab["flare"].to_numpy() == 1, "flare window", np.where(lab["set"] == "TRAIN", "TRAIN", "TEST (non-flare)"))
rng = np.random.default_rng(SEED)
pts = []
for g in ("TRAIN", "TEST (non-flare)", "flare window"):
    i = np.flatnonzero(grp == g)
    if len(i) > 15000:
        i = np.sort(rng.choice(i, 15000, replace=False))
    pts.append(T.iloc[i][["PC1", "PC2", "PC3"]].assign(group=g))
pts = pd.concat(pts)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, yc in zip(axes, ("PC2", "PC3")):
    for g, c, sz in (("TRAIN", TRAIN_COLOR, 4), ("TEST (non-flare)", TEST_COLOR, 4), ("flare window", "#000000", 8)):
        d = pts[pts["group"] == g]
        ax.scatter(d["PC1"], d[yc], s=sz, c=c, alpha=0.4, label=f"{g} (n shown {len(d)})", rasterized=True)
    ax.set_xlabel("PC1 score (z-score units)")
    ax.set_ylabel(f"{yc} score (z-score units)")
    ax.legend(markerscale=3)
sd = T["PC1"].groupby(grp).std()
save_fig(fig, "fig08_score_plots", "PCA score plots (TRAIN vs TEST, flare windows highlighted)",
         "PC1 std: " + ", ".join(f"{g} {sd[g]:.2f}" for g in sd.index) + " (up to 15 000 random points per group shown)",
         pts.reset_index())

b9, how9 = bin_for_plot(T[PCS[:npc]], "mean")
fig, axes = plt.subplots(npc, 1, figsize=(15, 2.0 * npc + 1.5), sharex=True)
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.plot(b9.index, b9[PCS[i]], lw=0.6, color="#0072B2")
    shade_flares(ax, fl, label=(i == 0))
    mark_cut(ax)
    ax.set_ylabel(f"{PCS[i]} score")
np.atleast_1d(axes)[0].legend(loc="upper right")
np.atleast_1d(axes)[-1].set_xlabel(f"Time ({how9})")
absT = T[PCS[:npc]].abs()
pc_m = absT.max().idxmax()
t_m = absT[pc_m].idxmax()
save_fig(fig, "fig09_pc_timeseries", f"Scores of PC1-PC{npc} over time",
         f"Largest excursion: {pc_m} = {T.loc[t_m, pc_m]:.1f} at {ts_label(t_m)}", b9, index=True)

c_tr = pd.read_csv(TAB / "pc_score_corr_train.csv", index_col=0)
c_te = pd.read_csv(TAB / "pc_score_corr_test.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, c, name in zip(axes, (c_tr, c_te), ("TRAIN", "TEST")):
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K), PCS, rotation=90)
    ax.set_yticks(range(K), PCS)
    ax.set_title(f"{name} scores")
fig.colorbar(im, ax=axes, label="Pearson r", shrink=0.8)
save_fig(fig, "fig10_pc_score_correlation", "Correlation of the 26 PC scores: TRAIN vs TEST",
         f"max |off-diagonal r|: TRAIN {s3['max_offdiag_corr_train']:.1e} (zero by construction), "
         f"TEST {s3['max_offdiag_corr_test']:.2f} - why PC on TRAIN scores is expected to be near-empty",
         pd.concat({"TRAIN": c_tr, "TEST": c_te}), index=True)

# ============================================================================ C. PC algorithm
evr_of = dict(zip(nodes["PC"], nodes["explained_var_ratio"]))
dom_of = dict(zip(nodes["PC"], nodes["dominant_unit"]))
node_units = [u for u in UNITS if u in set(nodes["dominant_unit"])]


def pc_layout(edges):
    G = nx.Graph()
    G.add_nodes_from(PCS)
    G.add_edges_from(zip(edges["source"], edges["target"]))
    return nx.spring_layout(G, seed=SEED, k=1.8 / np.sqrt(K), iterations=200)


def draw_pc_graph(ax, edges, pos, highlight=(), marked=None):
    """Node colour = dominant unit, size = explained variance, edge width = bootstrap frequency."""
    G = nx.DiGraph()
    G.add_nodes_from(PCS)
    G.add_edges_from(zip(edges["source"], edges["target"]))
    sizes = [300 + 8000 * evr_of[p] for p in PCS]
    nx.draw_networkx_nodes(G, pos, nodelist=PCS, node_size=sizes, ax=ax,
                           node_color=[UNIT_COLORS.get(dom_of[p], "#BBBBBB") for p in PCS],
                           edgecolors=[FLARE_COLOR if p == marked else "k" for p in PCS],
                           linewidths=[4.0 if p == marked else 3.0 if p in highlight else 0.8 for p in PCS])
    styles = {EDGE_DIRECTED: ("-|>", "solid"), EDGE_UNDIRECTED: ("-", "solid"), EDGE_BIDIRECTED: ("<|-|>", "solid")}
    for kind in edges["type"].unique():
        sub = edges[edges["type"] == kind]
        arrow, line = styles.get(kind, ("-", "dashed"))
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(sub["source"], sub["target"])), ax=ax,
                               width=list(0.5 + 4 * sub["boot_freq"]), arrows=True, arrowstyle=arrow,
                               arrowsize=16, style=line, node_size=sizes, nodelist=PCS, edge_color="#444444")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11,
                            bbox=dict(facecolor="white", alpha=0.75, lw=0, boxstyle="round,pad=0.15"))
    ax.set_axis_off()


pos = pc_layout(fe)
counts = fe["type"].value_counts()
fig, ax = plt.subplots(figsize=(13, 11))
draw_pc_graph(ax, fe, pos)
ax.legend(handles=unit_handles(node_units), title="dominant unit", loc="upper left", bbox_to_anchor=(1.0, 1.0))
save_fig(fig, "fig11_pc_causal_graph", f"CPDAG among the {K} PCs ({s3['final_variant']}, α = 0.05)",
         f"{len(fe)} edges: {counts.get(EDGE_DIRECTED, 0)} directed, {counts.get(EDGE_UNDIRECTED, 0)} undirected, "
         f"{counts.get(EDGE_BIDIRECTED, 0)} bidirected; node size = explained variance, width = bootstrap frequency", fe)

bf = pd.read_csv(TAB / "pc_bootstrap_freq.csv")
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1, 1.4]})
axes[0].hist(bf["freq"] if len(bf) else [], bins=np.linspace(0, 1, 21), color="#0072B2", edgecolor="k")
axes[0].axvline(STABLE_FREQ, color=FLARE_COLOR, ls="--", label=f"stable threshold {STABLE_FREQ}")
axes[0].set_xlabel("Bootstrap selection frequency")
axes[0].set_ylabel("Number of PC pairs")
axes[0].legend()
draw_pc_graph(axes[1], se, pos)
axes[1].set_title(f"Stable edges only (frequency >= {STABLE_FREQ})")
save_fig(fig, "fig12_edge_stability", "Edge stability under block bootstrap",
         f"{len(se)} of {len(fe)} edges reach >= {STABLE_FREQ:.0%} (B = {s3['bootstrap_B']}, "
         f"block length {s3['block_length']} rows)", bf)

W = pd.DataFrame(0.0, index=pd.Index(units, name="from_unit"), columns=pd.Index(units, name="to_unit"))
for e in fe.itertuples():
    a, b = lm.loc[e.source].to_numpy(), lm.loc[e.target].to_numpy()
    if e.type == EDGE_DIRECTED:
        W += e.boot_freq * np.outer(a, b)
    else:                                   # undirected / bidirected: half the weight each way
        W += 0.5 * e.boot_freq * (np.outer(a, b) + np.outer(b, a))
wl = W.stack().rename("weight").reset_index()
wl_off = wl[(wl["from_unit"] != wl["to_unit"]) & (wl["weight"] > 0)].nlargest(20, "weight")
if wl_off.empty:
    skipped.append(("fig13_unit_level_graph", "the final PC graph has no edges, so there is nothing to aggregate"))
else:
    Gu = nx.DiGraph()
    Gu.add_nodes_from(units)
    Gu.add_weighted_edges_from(wl_off.itertuples(index=False))
    posu = nx.circular_layout(units)
    fig, ax = plt.subplots(figsize=(12, 11))
    nx.draw_networkx_nodes(Gu, posu, node_color=[UNIT_COLORS[u] for u in units], node_size=1800, edgecolors="k", ax=ax)
    nx.draw_networkx_edges(Gu, posu, edgelist=list(zip(wl_off["from_unit"], wl_off["to_unit"])), ax=ax,
                           width=list(1 + 7 * wl_off["weight"] / wl_off["weight"].max()), arrowstyle="-|>",
                           arrowsize=20, node_size=1800, connectionstyle="arc3,rad=0.12", edge_color="#444444")
    nx.draw_networkx_labels(Gu, {u: p * 1.2 for u, p in posu.items()}, font_size=11, ax=ax)
    ax.margins(0.25)
    ax.set_axis_off()
    t0 = wl_off.iloc[0]
    save_fig(fig, "fig13_unit_level_graph", "Unit-level graph: PC edges aggregated by loading mass",
             f"Strongest link {t0['from_unit']} -> {t0['to_unit']} (weight {t0['weight']:.2f}); top {len(wl_off)} "
             "links drawn; undirected PC edges count half each way", wl)

vs = pd.read_csv(TAB / "pc_variants_summary.csv")
sens = pd.read_csv(TAB / "pc_sensitivity.csv")
names = list(dict.fromkeys(vs["variant"]))
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
x = np.arange(len(names))
for j, a in enumerate(ALPHAS):
    vals = [vs.loc[(vs["variant"] == nm) & (vs["alpha"] == a), "edges"].sum()
            if ((vs["variant"] == nm) & (vs["alpha"] == a)).any() else np.nan for nm in names]
    axes[0].bar(x + (j - 0.5) * 0.38, vals, 0.38, label=f"α = {a}", color=[TRAIN_COLOR, TEST_COLOR][j], edgecolor="k")
s_ = sens.set_index("variant").reindex(names)
axes[1].bar(x - 0.19, s_["jaccard_alpha0p05_vs_0p01"], 0.38, label="α 0.05 vs 0.01", color=TRAIN_COLOR, edgecolor="k")
axes[1].bar(x + 0.19, s_["jaccard_vs_final_alpha0p05"], 0.38, label="vs final graph (α 0.05)", color=TEST_COLOR, edgecolor="k")
for ax, ylab in zip(axes, ("Number of edges", "Jaccard similarity of skeletons")):
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_xlabel("PC-algorithm variant")
    ax.set_ylabel(ylab)
    ax.legend()
j_final = s_.loc[s3["final_variant"], "jaccard_alpha0p05_vs_0p01"]
save_fig(fig, "fig14_pc_sensitivity", "Sensitivity of the PC graph to α and to the §3.3 variant",
         f"Planned T_train graph: {s3['n_edges_planned']} edges; final = {s3['final_variant']}, "
         f"α 0.05 vs 0.01 Jaccard {j_final:.2f}; lagged variant is a diagnostic only",
         vs.merge(sens, on="variant", how="left"))

# ============================================================================ D. T² / Q
def chart(ax, col, ucls):
    frames = []
    for g, c in (("TRAIN", TRAIN_COLOR), ("TEST", TEST_COLOR)):
        b, how = bin_for_plot(t2q.loc[t2q["set"] == g, [col]], "max")
        ax.plot(b.index, b[col], lw=0.6, color=c, label=f"{g} ({how})")
        frames.append(b.assign(set=g))
    for (name, val), ls in zip(ucls, ("--", "-")):
        ax.axhline(val, color="k", ls=ls, lw=1, label=f"{name} = {val:.3g}")
    shade_flares(ax, fl)
    mark_cut(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left", fontsize=11)
    return pd.concat(frames)


fig, ax = plt.subplots(figsize=(15, 5.5))
tab15 = chart(ax, "T2", [("95% UCL", lim["T2_95"]), ("99% UCL", lim["T2_99"])])
ax.set_ylabel("Hotelling T² (dimensionless)")
save_fig(fig, "fig15_t2_chart", "Hotelling T² monitoring chart (26 PCs)",
         f"{100 * metrics['test_alarm_rate_T2_99']:.1f}% of TEST vs {100 * metrics['far_train_T2_99']:.1f}% of TRAIN "
         f"samples above the 99% UCL ({lim['T2_99']:.1f})", tab15, index=True)

fig, axes = plt.subplots(2, 1, figsize=(15, 11))
tab16 = chart(axes[0], "Q", [("95% UCL", lim["Q_95"]), ("99% UCL", lim["Q_99"])])
axes[0].set_ylabel("Q / SPE (z-score units²)")
sc = []
for g, c in (("TRAIN", TRAIN_COLOR), ("TEST", TEST_COLOR)):
    d = t2q.loc[t2q["set"] == g, ["T2", "Q", "flare"]]
    if len(d) > 20000:
        d = d.iloc[np.sort(rng.choice(len(d), 20000, replace=False))]
    axes[1].scatter(d["T2"], d["Q"], s=4, c=c, alpha=0.4, label=g, rasterized=True)
    sc.append(d.assign(set=g))
axes[1].axvline(lim["T2_99"], color="k", ls="-", lw=1, label="T² 99% UCL")
axes[1].axhline(lim["Q_99"], color="k", ls="--", lw=1, label="Q 99% UCL")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Hotelling T²")
axes[1].set_ylabel("Q / SPE")
axes[1].legend(markerscale=3)
pd.concat(sc).to_csv(TAB / "fig16_q_chart_scatter_points.csv")
save_fig(fig, "fig16_q_chart", f"Q (SPE) chart and T²-Q quadrant plot (Q limit: {s4['q_limit_method']})",
         f"Q above its 99% UCL on {100 * metrics['test_alarm_rate_Q_99']:.1f}% of TEST vs "
         f"{100 * metrics['far_train_Q_99']:.1f}% of TRAIN samples (scatter: up to 20 000 random points per set)",
         tab16, index=True)

qs = np.linspace(0, 1, 1001)
q_tr = np.quantile(t2q.loc[t2q["set"] == "TRAIN", "T2"], qs)
q_te = np.quantile(t2q.loc[t2q["set"] == "TEST", "T2"], qs)
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(q_tr, qs, color=TRAIN_COLOR, lw=2, label="TRAIN")
ax.plot(q_te, qs, color=TEST_COLOR, lw=2, label="TEST")
ax.axvline(lim["T2_95"], color="k", ls="--", lw=1, label="95% UCL")
ax.axvline(lim["T2_99"], color="k", ls="-", lw=1, label="99% UCL")
ax.set_xscale("log")
ax.set_xlabel("Hotelling T²")
ax.set_ylabel("Empirical CDF")
ax.legend()
save_fig(fig, "fig17_t2_distribution", "Distribution of T²: TRAIN vs TEST (ECDF)",
         f"Median T²: TRAIN {np.median(q_tr):.1f}, TEST {np.median(q_te):.1f}; 99% UCL = {lim['T2_99']:.1f}",
         pd.DataFrame({"quantile": qs, "T2_train": q_tr, "T2_test": q_te}))

if top5.empty:
    for f_ in ("fig18_pc_contribution_heatmaps", "fig19_event_top_tags"):
        skipped.append((f_, "no T² alarm event in TEST"))
else:
    fig, axes = plt.subplots(len(top5), 1, figsize=(15, 3.2 * len(top5) + 1.5))
    long = []
    for ax, e in zip(np.atleast_1d(axes), top5.itertuples()):
        pad = max(e.end - e.start, 12 * interval)
        wdw, how = bin_for_plot(t2q.loc[e.start - pad: e.end + pad, PCS], "mean")
        vals = np.clip(wdw.to_numpy().T, 1e-3, None)
        im = ax.imshow(vals, aspect="auto", cmap="viridis", interpolation="nearest",
                       norm=LogNorm(vmin=1e-3, vmax=max(float(np.nanmax(vals)), 1e-2)),
                       extent=[mdates.date2num(wdw.index[0]), mdates.date2num(wdw.index[-1]), K - 0.5, -0.5])
        ax.axvline(e.start, color="white", lw=1)
        ax.axvline(e.end, color="white", lw=1)
        ax.xaxis_date()
        ax.set_yticks(range(0, K, 5), PCS[::5])
        ax.set_ylabel("PC")
        ax.set_title(f"Event {e.event_id}: {ts_label(e.start)}, peak T² {e.peak_T2:.0f}, top PCs {e.top3_pcs} ({how})")
        fig.colorbar(im, ax=ax, label="t_i²/λ_i")
        long.append(wdw.stack().rename("contribution").rename_axis(["timestamp", "PC"]).reset_index().assign(event_id=e.event_id))
    np.atleast_1d(axes)[-1].set_xlabel("Time (white lines = event start / end)")
    save_fig(fig, "fig18_pc_contribution_heatmaps", "PC contributions to T² around the top-5 TEST events",
             "Driving PC per event: " + ", ".join(f"#{e.event_id} {e.top3_pcs.split('; ')[0]}" for e in top5.itertuples())
             + " (events ranked by T² area above the 99% UCL)", pd.concat(long))

    pk = pd.read_csv(TAB / "event_peak_tag_contrib.csv", dtype={"column": str})
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    used = []
    for ax, e in zip(axes, top5.itertuples()):
        d = pk[pk["event_id"] == e.event_id].nlargest(15, "contrib_signed")[::-1]
        used.append(d)
        ax.barh(range(len(d)), d["contrib_signed"], color=[UNIT_COLORS.get(u, "#BBBBBB") for u in d["unit"]], edgecolor="k")
        ax.set_yticks(range(len(d)), d["column"])
        ax.set_xlabel("Signed contribution to T² at the peak")
        ax.set_title(f"Event {e.event_id}, peak {ts_label(e.peak_time)}")
    for ax in axes[len(top5):]:
        ax.set_axis_off()
    axes[-1].legend(handles=unit_handles(units), loc="center", title="unit", ncol=2)
    best = pd.concat(used).nlargest(1, "contrib_signed").iloc[0]
    save_fig(fig, "fig19_event_top_tags", "Top-15 tag contributions at the T² peak of the top-5 TEST events",
             f"Largest single-tag contribution: {best['column']} ({best['unit']}) = {best['contrib_signed']:.1f} in event "
             f"{best['event_id']}; clipped values in outputs/tables/event_peak_tag_contrib.csv", pd.concat(used))

uc = pd.read_parquet(DATA / "unit_contrib_clipped.parquet")
b20, how20 = bin_for_plot(uc, "mean")
vals = np.clip(b20.to_numpy().T, 1e-3, None)
fig, ax = plt.subplots(figsize=(15, 0.45 * len(units) + 3))
im = ax.imshow(vals, aspect="auto", cmap="viridis", interpolation="nearest",
               norm=LogNorm(vmin=1e-3, vmax=max(float(np.nanmax(vals)), 1e-2)),
               extent=[mdates.date2num(b20.index[0]), mdates.date2num(b20.index[-1]), len(units) - 0.5, -0.5])
for k, s in enumerate(fl["start"]):
    ax.axvline(s, color="white", lw=0.8, ls="--", label="flare onset" if k == 0 else None)
ax.axvline(cut, color=FLARE_COLOR, lw=1.5, label="TRAIN/TEST cut")
ax.xaxis_date()
ax.set_yticks(range(len(units)), units)
ax.set_xlabel(f"Time ({how20})")
ax.set_ylabel("Unit")
ax.legend(loc="upper right")
fig.colorbar(im, ax=ax, label="Clipped T² contribution (sum over the unit's tags)")
test_mean = uc[t2q["set"].to_numpy() == "TEST"].mean()
take20 = f"Highest mean clipped contribution in TEST: {test_mean.idxmax()}"
if t2q["flare"].sum():
    take20 += f"; during flare windows: {uc[t2q['flare'].to_numpy() == 1].mean().idxmax()}"
save_fig(fig, "fig20_unit_contribution_heatmap", "Unit-level T² contributions over time", take20, b20, index=True)

if test_events.empty:
    skipped.append(("fig21_event_table", "no T² alarm event in TEST"))
else:
    tt = test_events.nlargest(15, "area_above_ucl99")
    show = pd.DataFrame({"Event": tt["event_id"], "Start": tt["start"].dt.strftime("%Y-%m-%d %H:%M"),
                         "Duration (min)": tt["duration_min"].round(0).astype(int),
                         "Peak T²": tt["peak_T2"].round(1), "Top PCs": tt["top3_pcs"], "Top units": tt["top3_units"],
                         "Lead to flare (min)": tt["lead_to_flare_min"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")})
    fig, ax = plt.subplots(figsize=(20, 0.5 * len(show) + 2.5))
    ax.set_axis_off()
    table = ax.table(cellText=show.astype(str).to_numpy(), colLabels=list(show.columns), loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    table.auto_set_column_width(list(range(show.shape[1])))
    save_fig(fig, "fig21_event_table", "T² alarm events in TEST (top 15 by T² area above the 99% UCL)",
             f"{len(test_events)} TEST events; {int(test_events['flare_event_id'].notna().sum())} start within "
             f"{MATCH_WINDOW} before a flare onset (or during the flare)", show)

roc = {n_: pd.read_csv(TAB / f"roc_{n_}.csv") for n_ in ("T2", "Q") if (TAB / f"roc_{n_}.csv").exists()}
det = pd.read_csv(TAB / "flare_detection.csv")
leads = det.loc[det["detected"].astype(bool), "lead_min"] / 60 if len(det) else pd.Series(dtype=float)
if not s1["has_labels"]:
    skipped.append(("fig22_detection_performance", "no flare labels could be derived from flare_data"))
elif not roc and leads.empty:
    skipped.append(("fig22_detection_performance", "TEST lacks flare or normal samples and no flare was detected"))
else:
    panels = (["roc"] if roc else []) + (["lead"] if len(leads) else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.5), squeeze=False)
    parts = []
    for ax, kind in zip(axes[0], panels):
        if kind == "roc":
            for n_, c in (("T2", TRAIN_COLOR), ("Q", TEST_COLOR)):
                if n_ in roc:
                    ax.plot(roc[n_]["fpr"], roc[n_]["tpr"], color=c, lw=2, label=f"{n_} (AUC {metrics[f'auc_{n_}']:.2f})")
                    parts.append(roc[n_].assign(kind=f"roc_{n_}"))
            ax.plot([0, 1], [0, 1], color="#888888", ls=":")
            ax.set_xlabel("False-alarm rate (TEST normal samples)")
            ax.set_ylabel("Detection rate (TEST flare samples)")
            ax.set_title("ROC, sample level")
            ax.legend()
        else:
            ax.hist(leads, bins=20, color=TRAIN_COLOR, edgecolor="k")
            ax.set_xlabel("Lead time (h); positive = first alarm before flare onset")
            ax.set_ylabel("Number of flares")
            ax.set_title(f"Alarm-to-flare lead time (window {MATCH_WINDOW} before onset)")
            parts.append(pd.DataFrame({"lead_h": leads.to_numpy(), "kind": "lead"}))
    rate = metrics.get("event_detection_rate", np.nan)
    save_fig(fig, "fig22_detection_performance", "Detection performance of T² against the flare labels",
             f"{metrics.get('n_flares_detected', 0)}/{metrics.get('n_flares_observable', 0)} observable flares detected "
             f"({100 * rate:.0f}%); " + ", ".join(f"AUC {n_} {metrics[f'auc_{n_}']:.2f}" for n_ in roc)
             + (f"; median lead {leads.median():.1f} h" if len(leads) else ""), pd.concat(parts))

hyp = pd.read_csv(TAB / "root_cause_hypotheses.csv")
cand = test_events[test_events["event_id"].isin(hyp["event_id"])]
matched = cand[cand["flare_event_id"].notna()]
pick = (matched if len(matched) else cand).nlargest(1, "area_above_ucl99")
if pick.empty:
    skipped.append(("fig23_root_cause_storyline", "no TEST alarm event to illustrate"))
else:
    e = pick.iloc[0]
    h = hyp[hyp["event_id"] == e["event_id"]].iloc[0]
    tops = str(h["top3_pcs"]).split("; ")
    anc = [p for p in str(h["definite_ancestors"]).split("; ") + str(h["possible_ancestors"]).split("; ") if p in PCS]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    (a1, a2), (a3, a4) = axes
    pad = max(e["end"] - e["start"], 24 * interval)
    wdw = t2q.loc[e["start"] - pad: e["end"] + pad]
    a1.plot(wdw.index, wdw["T2"], color=TEST_COLOR, lw=1, label="T²")
    a1.axhline(lim["T2_99"], color="k", lw=1, label="99% UCL")
    a1.axvspan(e["start"], e["end"], color="#888888", alpha=0.25, label="alarm event")
    shade_flares(a1, fl[(fl["end"] >= wdw.index.min()) & (fl["start"] <= wdw.index.max())])
    a1.set_yscale("log")
    a1.set_xlabel("Time")
    a1.set_ylabel("Hotelling T²")
    a1.set_title(f"(i) T² around event {e['event_id']}")
    a1.legend(loc="upper left")
    mc = t2q.loc[e["start"]: e["end"], PCS].mean().sort_values(ascending=False)[:8]
    a2.bar(mc.index, mc.to_numpy(), edgecolor="k",
           color=[FLARE_COLOR if p == h["elevated_pc"] else TRAIN_COLOR if p in tops else "#BBBBBB" for p in mc.index])
    a2.set_xlabel("PC")
    a2.set_ylabel("Mean t_i²/λ_i over the event")
    a2.set_title(f"(ii) Top contributing PCs (red = most elevated at {h['reference']}: {h['elevated_pc']})")
    use_stable = str(s4["graph_used_for_links"]).startswith("stable")
    draw_pc_graph(a3, se if use_stable else fe, pos, highlight=tops, marked=h["elevated_pc"])
    a3.set_title(f"(iii) Position in the {'stable' if use_stable else 'full'} graph (thick = top PCs, red ring = elevated)")
    src = anc if anc else tops
    mass = lm.loc[src].sum().sort_values(ascending=False)
    mass = mass[mass > 0]
    a4.bar(mass.index, mass.to_numpy(), color=[UNIT_COLORS[u] for u in mass.index], edgecolor="k")
    a4.set_xticks(range(len(mass)), mass.index, rotation=45, ha="right")
    a4.set_xlabel("Unit")
    a4.set_ylabel("Summed loading mass")
    a4.set_title("(iv) Implied upstream units (loading mass of ancestor top PCs)" if anc else
                 "(iv) No top PC is an ancestor in the graph: unit mass of the top PCs shown")
    cru = h["candidate_root_units"] if isinstance(h["candidate_root_units"], str) else "none"
    story = pd.concat([mc.rename("value").rename_axis("key").reset_index().assign(panel="ii_pc_contribution"),
                       mass.rename("value").rename_axis("key").reset_index().assign(panel="iv_unit_mass")])
    save_fig(fig, "fig23_root_cause_storyline", "Root-cause hypothesis for one event (storyline)",
             f"Event {e['event_id']} ({ts_label(e['start'])}): elevated {h['elevated_pc']} ({h['elevated_pc_unit']}); "
             f"ancestor top PCs: {', '.join(anc) or 'none'}; candidate upstream units: {cru} - HYPOTHESIS", story)

# ============================================================================ RESULTS.md
def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}%"


recon = "match" if audit["n_params"] == EXPECTED_N_PARAMS else f"DISCREPANCY {audit['n_params'] - EXPECTED_N_PARAMS:+d}"
vs_md = vs[["variant", "test", "alpha", "rows", "edges", "directed", "undirected", "ci_tests", "seconds", "note"]]
top_ev = top5[["event_id", "start", "duration_min", "peak_T2", "top3_pcs", "top3_units", "lead_to_flare_min"]]
hyp_md = hyp[["event_id", "reference", "elevated_pc", "elevated_pc_unit", "top3_pcs", "definite_ancestors",
              "possible_ancestors", "candidate_root_units"]].head(10)
unit_links = wl_off.head(5)
L = ["# RESULTS - PCA -> PC algorithm -> T² decomposition", "",
     f"Generated by `src/s5_figures.py` on {date.today().isoformat()} from `outputs/`; no number is typed by hand.", "",
     "## 1. Data summary",
     f"- Time window: {audit.get('window', 'all data (no window)')}.",
     f"- Parameters audited: {audit['n_params']} vs {EXPECTED_N_PARAMS} ({recon}); units {len(audit['units'])}/13; "
     f"control data {'included' if audit['include_control'] else 'excluded'}.",
     f"- Final matrix: {s1['shape_final'][0]} rows x {s1['shape_final'][1]} tags (joined before cleaning: "
     f"{s1['shape_joined'][0]} x {s1['shape_joined'][1]}); interval {s1['interval']}; {s1['time_start']} to {s1['time_end']}.",
     f"- Split: {s1['split_rule']}. TRAIN {s1['n_train']} rows, TEST {s1['n_test']} rows, flare rows in TEST {s1['flare_rows_test']}.",
     f"- Dropped tags: {s1['n_dropped_missing']} (> 30 % missing), {s1['n_dropped_constant']} constant, "
     f"{s1['n_dropped_constant_train']} constant in TRAIN (`outputs/tables/dropped_tags.csv`); "
     f"incomplete rows dropped {s1['rows_dropped_incomplete']}.",
     f"- Short-gap fill N = {FFILL_MAX_GAP} samples: covers {s1['ffill_pct_runs']:.1f}% of NaN runs "
     f"({s1['ffill_pct_values']:.1f}% of missing values; `gap_lengths.csv`). Sentinels replaced: {s1['n_sentinels_replaced']}.",
     f"- Flare label: {audit['label_definition']}; {audit['n_flare_events']} events.", "",
     "## 2. PCA",
     f"- k = {K}: cumulative variance {s2['cum_var_k']:.1%}; PCs for 80/90/95/99 %: "
     + "/".join(str(v_) for v_ in s2["n_pcs_for"].values()) + f"; reconstruction R² (TRAIN) {s2['reconstruction_r2']:.4f}.",
     "- Dominant units: " + "; ".join(f"{d['PC']} {d['dominant_unit']} ({d['share']:.0%})" for d in s2["dominant_units"][:6]) + ".", "",
     "## 3. PC algorithm",
     f"- Planned Fisher-Z PC on T_train: {s3['n_edges_planned']} edges; max |off-diagonal r| TRAIN "
     f"{s3['max_offdiag_corr_train']:.1e}, TEST {s3['max_offdiag_corr_test']:.2f}.",
     f"- Final variant **{s3['final_variant']}**: {s3['final_reason']}",
     f"- Final graph {s3['n_edges_final']} edges; stable (>= {STABLE_FREQ} of B = {s3['bootstrap_B']}, block length "
     f"{s3['block_length']}): {s3['n_edges_stable']}. {s3['bootstrap_reduction']}",
     md_table(vs_md), "",
     "Key unit-level links (fig13 weights):" if len(unit_links) else "Key unit-level links: none (empty graph).",
     md_table(unit_links) if len(unit_links) else "", "",
     "## 4. T² / Q",
     f"- Limits (TRAIN, n = {s4['n_train']}): T² 95/99 % = {lim['T2_95']:.2f}/{lim['T2_99']:.2f}; Q 95/99 % = "
     f"{lim['Q_95']:.3g}/{lim['Q_99']:.3g} ({s4['q_limit_method']}); empirical TRAIN 99th pct T² {lim['T2_99_empirical_train']:.2f}, "
     f"Q {lim['Q_99_empirical_train']:.3g}.",
     f"- Events: {s4['n_events']} ({s4['n_events_test']} in TEST); rule {s4['event_rule']}, alarms <= {EVENT_GAP} samples apart merged.",
     f"- False-alarm rate on TRAIN: T² 99 % {pct(metrics['far_train_T2_99'])}, Q 99 % {pct(metrics['far_train_Q_99'])}, "
     f"T² or Q {pct(metrics['far_train_T2_or_Q_99'])}.",
     (f"- Flares: {metrics.get('n_flares_detected')} of {metrics.get('n_flares_observable')} observable detected "
      f"({pct(metrics.get('event_detection_rate'))}); median delay {metrics.get('median_detection_delay_min')} min "
      f"(negative = before onset; window {s4['match_window']}); sample detection {pct(metrics.get('sample_detection_rate_T2_99'))}; "
      + ", ".join(f"AUC {n_} {metrics[f'auc_{n_}']:.2f}" for n_ in ('T2', 'Q') if f'auc_{n_}' in metrics))
     if s1["has_labels"] else "- No flare labels: detection metrics not computed.",
     "Top-5 TEST events:" if len(top_ev) else "No TEST events.", md_table(top_ev) if len(top_ev) else "", "",
     "## 5. Linked root-cause hypotheses (HYPOTHESES, not conclusions)",
     f"- Graph used: {s4['graph_used_for_links']}. Ancestor = directed path to the PC most elevated at flare onset "
     "(or event peak); possible ancestor also uses undirected edges.",
     md_table(hyp_md) if len(hyp_md) else "- No events to link.", "",
     "## 6. Limitations",
     "- PCA mixes variables: edges are between components, not tags; unit readings rely on loading mass.",
     "- PC algorithm assumes i.i.d. samples; plant data is autocorrelated (block bootstrap only partly addresses this).",
     f"- Linear CI test (Fisher-Z) for the final graph; KCI ran only as a check on n = {KCI_N} rows.",
     "- No PFD / process ground truth; flare labels are derived from flare_data as stated in section 1.",
     "- T² limits assume multivariate normal, independent TRAIN samples.", "",
     "## 7. Figure index",
     *[f"- `{n_}`: {t_}" for n_, t_ in FIG_INDEX],
     *[f"- `{n_}`: SKIPPED - {r_}" for n_, r_ in skipped], "",
     "## 8. Suggested next step",
     "Re-test the stable unit-level links with a lag-aware method (e.g. PCMCI+) on the original tags of the candidate "
     "root-cause units, using the TEST flare windows, and check the resulting directions against the plant's process "
     "flow diagram with the operations team. Not started."]
(PROJECT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
log(f"figures saved: {len(FIG_INDEX)}; skipped: {len(skipped)}; RESULTS.md written")
for n_, r_ in skipped:
    log(f"  skipped {n_}: {r_}")
