"""Stage 6 - figures and tables for the later slides (the PPT itself is not built here).

Style, palette, binning and the T² / heatmap / per-event figures are copied from causal_t2/src/s5_figures.py and
edited for nodes (26 PCs + flare tags). New: PCA figures (scree, unit composition, loadings), the DAG node-link
diagram with PCs and flare tags in two colours, a degree summary and a contact sheet of the per-node charts.
One style for every figure: 14-16 pt text, labelled axes, saved with a tight bounding box (no clipped text).

Writes outputs/figures/** (PNG), outputs/tables/degree_summary.csv and outputs/figures_index.csv.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import (DATA, FIG, GRAPH, N_PCS, OUT, PLOT_MAX_POINTS, T2DIR, TAB, clear_notes, done,  # noqa: E402
                    flags_to_events, gate, load_json, log, read_flare_events, read_nodes, safe_name, stage)

S = "s6_figures"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                    "#184f95", "#0d366b"])
FLARE_SHADE, REF_SHADE = SERIES[1], SERIES[2]
PC_COLOR, FLARE_COLOR = SERIES[0], SERIES[1]          # the two node kinds
HATCHES = ["", "//", "\\\\", ".."]
TJ = r"$T_{j\cdot PA(j)}$"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True})
INDEX = []


def save(fig, rel, what, key):
    fig.savefig(FIG / rel, dpi=150, bbox_inches="tight")
    plt.close(fig)
    INDEX.append({"filename": f"figures/{rel}", "what_it_shows": what, "key_numbers": key})


def spans(flag_binned, rule):
    """Consecutive flagged bins -> (start, end) spans for shading."""
    ev = flags_to_events(flag_binned.index, flag_binned.to_numpy(dtype=bool), rule)
    return list(zip(ev["start"], ev["end"] + rule))


def shade(ax, flare_spans, ref_spans):
    for k, (s, e) in enumerate(ref_spans):
        ax.axvspan(s, e, color=REF_SHADE, alpha=0.12, lw=0, label="reference period" if k == 0 else None)
    for k, (s, e) in enumerate(flare_spans):
        ax.axvspan(s, e, color=FLARE_SHADE, alpha=0.15, lw=0, label="flare event" if k == 0 else None)


def time_axis(ax):
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_xlabel("Time")


t0 = time.time()
stage("Stage 6 - figures and tables")
clear_notes(S)
nodes = read_nodes()
unit_of = dict(zip(nodes["node"], nodes["unit"]))
flare_of = dict(zip(nodes["node"], nodes["is_flare"]))
s2, s3, s5 = load_json("s2_summary.json"), load_json("s3_summary.json"), load_json("s5_summary.json")
z_lim, t2_lim = s5["z_lim"], s5["t2_lim"]
T = pd.read_parquet(T2DIR / "T_terms.parquet")
t2 = pd.read_parquet(T2DIR / "t2.parquet")
meta = pd.read_parquet(DATA / "meta.parquet")
events = read_flare_events()
rc = pd.read_csv(TAB / "root_cause_nodes.csv", dtype={"node": str})
ev_tab = pd.read_csv(TAB / "root_cause_by_event.csv", dtype={"node": str})
parents = load_json("parents.json")
var = pd.read_csv(DATA / "pca_variance.csv")
top_tags = pd.read_csv(TAB / "pca_top_tags.csv", dtype={"tag": str})
share = pd.read_csv(TAB / "pca_unit_share.csv", index_col=0)
nodes_t2 = list(T.columns)

# ---- unit styles for tag-level figures: fixed hue order, then a hatch (never a generated colour)
units = list(share.columns)
STYLE = {u: (SERIES[i % 8], HATCHES[(i // 8) % 4]) for i, u in enumerate(units)}


def unit_patches(us):
    return [Patch(facecolor=STYLE[u][0], hatch=STYLE[u][1], edgecolor=SURFACE, label=u) for u in us]


def node_label(v):
    return v if flare_of[v] else f"{v} ({unit_of[v][4:-1]})"      # 'PC3 (LOUP)'


# ============================================================ 1) PCA
pcs = list(var["pc"])
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(1, len(pcs) + 1)
ax.bar(x, 100 * var["explained_variance_ratio"], color=PC_COLOR, label="explained variance per PC")
ax2 = ax.twinx()
ax2.plot(x, 100 * var["cumulative"], color=INK, marker="o", lw=2, label="cumulative")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Cumulative explained variance (%)")
ax2.grid(False)
ax.set_xticks(x)
ax.set_xticklabels(pcs, rotation=90, fontsize=11)
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax.set_title(f"PCA on {s2['n_process_tags']} process tags: {N_PCS} PCs keep {100 * s2['cum_variance']:.1f}% of the variance",
             loc="left")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=12)
save(fig, "pca_scree.png", "Scree plot: explained variance per PC (bars) and cumulative (line)",
     f"cum_26={100 * s2['cum_variance']:.2f}%; PC1={100 * var['explained_variance_ratio'].iloc[0]:.2f}%")

fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(share))
for u in units:
    v = 100 * share[u].to_numpy()
    ax.bar(np.arange(len(share)), v, bottom=bottom, color=STYLE[u][0], hatch=STYLE[u][1], edgecolor=SURFACE, label=u)
    bottom += v
ax.set_xticks(np.arange(len(share)))
ax.set_xticklabels(share.index, rotation=90, fontsize=11)
ax.set_xlabel("Principal component")
ax.set_ylabel("Share of squared loadings (%)")
ax.set_ylim(0, 100)
ax.set_title("Unit composition of each PC (share of squared loadings)", loc="left")
ax.legend(handles=unit_patches(units), loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, title="unit")
dom = share.idxmax(axis=1)
save(fig, "pca_unit_composition.png", "Stacked bars: share of each PC's squared loadings per unit",
     "dominant: " + "; ".join(f"{p}={dom[p]}" for p in share.index[:5]))

for pc in pcs[:4]:
    d = top_tags[top_tags["pc"] == pc].sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(d) + 2))
    for y, (u, v) in enumerate(d[["unit", "loading"]].itertuples(index=False)):
        ax.barh(y, v, color=STYLE[u][0], hatch=STYLE[u][1], edgecolor=SURFACE, height=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["tag"], fontsize=12)
    ax.axvline(0, color=INK2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Loading")
    k = pcs.index(pc)
    ax.set_title(f"{pc}: top {len(d)} tags by |loading| ({100 * var['explained_variance_ratio'].iloc[k]:.1f}% of variance)",
                 loc="left")
    ax.legend(handles=unit_patches(list(dict.fromkeys(d["unit"][::-1]))), loc="lower right", fontsize=11)
    save(fig, f"pca_loadings_{pc}.png", f"Top tags by |loading| for {pc}, coloured by unit",
         "top3=" + ", ".join(d.sort_values("rank")["tag"].head(3)))

# ============================================================ 2) DAG and degree summary
dag_edges = pd.read_csv(GRAPH / "dag_edges.csv", dtype={"source": str, "target": str})
dag_nodes = [str(t) for t in np.load(DATA / "pc_graph.npz", allow_pickle=True)["names"]]
dag = nx.DiGraph()
dag.add_nodes_from(dag_nodes)
dag.add_edges_from(zip(dag_edges["source"], dag_edges["target"]))
pc_n = [v for v in dag_nodes if not flare_of[v]]
fl_n = [v for v in dag_nodes if flare_of[v]]
pos = nx.shell_layout(dag, [fl_n, pc_n]) if fl_n and pc_n else nx.circular_layout(dag)
fig, ax = plt.subplots(figsize=(13, 12))
nx.draw_networkx_edges(dag, pos, ax=ax, edge_color=INK2, width=1.2, arrows=True, arrowsize=14, node_size=1100,
                       connectionstyle="arc3,rad=0.06")
nx.draw_networkx_nodes(dag, pos, nodelist=pc_n, ax=ax, node_color=PC_COLOR, node_size=1100, edgecolors=SURFACE)
nx.draw_networkx_nodes(dag, pos, nodelist=fl_n, ax=ax, node_color=FLARE_COLOR, node_shape="s", node_size=1100,
                       edgecolors=SURFACE)
nx.draw_networkx_labels(dag, pos, labels={v: v for v in pc_n}, ax=ax, font_size=10, font_color=SURFACE)
nx.draw_networkx_labels(dag, {v: (x_, y_ - 0.09) for v, (x_, y_) in pos.items() if v in fl_n},
                        labels={v: v[:40] for v in fl_n}, ax=ax, font_size=10, font_color=INK)
ax.axis("off")
ax.margins(0.12)
kc = s3["edges_by_kind"]
ax.set_title(f"DAG on {len(pc_n)} PCs + {len(fl_n)} flare tags: {len(dag_edges)} edges "
             f"(PC-PC {kc['pc_pc']}, PC -> flare {kc['pc_flare']}); alpha = {s3['alpha']}", loc="left")
ax.legend(handles=[Line2D([], [], ls="", marker="o", color=PC_COLOR, markersize=12, label="principal component"),
                   Line2D([], [], ls="", marker="s", color=FLARE_COLOR, markersize=12, label="flare tag")],
          loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=12)
save(fig, "dag_graph.png", "Final DAG: PCs (circles) and flare tags (squares), edges directed",
     f"nodes={len(dag_nodes)}; edges={len(dag_edges)}; pc_pc={kc['pc_pc']}; pc_flare={kc['pc_flare']}")

deg = pd.DataFrame({"node": dag_nodes, "kind": ["flare" if flare_of[v] else "PC" for v in dag_nodes],
                    "in_degree": [dag.in_degree(v) for v in dag_nodes], "out_degree": [dag.out_degree(v) for v in dag_nodes]})
deg.to_csv(TAB / "degree_summary.csv", index=False)
INDEX.append({"filename": "tables/degree_summary.csv", "what_it_shows": "In / out degree of every DAG node (table)",
              "key_numbers": f"nodes={len(deg)}; mean_in={deg['in_degree'].mean():.2f}"})
fig, ax = plt.subplots(figsize=(14, 6))
xx = np.arange(len(deg))
ax.bar(xx - 0.2, deg["in_degree"], 0.4, color=PC_COLOR, label="in-degree (parents)")
ax.bar(xx + 0.2, deg["out_degree"], 0.4, color=SERIES[2], label="out-degree (children)")
ax.set_xticks(xx)
ax.set_xticklabels([v[:18] for v in deg["node"]], rotation=90, fontsize=10)
ax.set_ylabel("number of edges")
ax.set_title(f"DAG degree summary: mean parents {deg['in_degree'].mean():.2f}, max {deg['in_degree'].max()}", loc="left")
ax.legend(fontsize=12)
save(fig, "dag_degree_summary.png", "In- and out-degree of every node in the DAG",
     f"mean_in={deg['in_degree'].mean():.2f}; max_in={deg['in_degree'].max()}; max_out={deg['out_degree'].max()}")

# ============================================================ 3) one T_j·PA(j) chart per node + contact sheet
span = T.index.max() - T.index.min()
rule = max(pd.Timedelta(minutes=1), (span / PLOT_MAX_POINTS).ceil("min"))
flare_sp = spans(meta["flare"].astype(float).resample(rule).max().fillna(0), rule)
ref_sp = spans(meta["reference"].astype(float).resample(rule).max().fillna(0), rule)
log(f"time series binned to {rule} ({len(flare_sp)} flare spans, {len(ref_sp)} reference spans shaded)")
Tmin, Tmax = T.resample(rule).min(), T.resample(rule).max()
for old in (FIG / "per_node").glob("*.png"):     # generated files from an earlier run
    old.unlink()
rcx = rc.set_index("node")
per_node = []
for i, t in enumerate(nodes_t2):
    lo, hi = Tmin[t], Tmax[t]
    big = float(np.nanmax(np.abs(np.r_[lo.to_numpy(), hi.to_numpy()])))
    ylim = max(1.5 * z_lim, float(np.nanpercentile(np.abs(np.r_[lo.dropna(), hi.dropna()]), 99.9)))
    fig, ax = plt.subplots(figsize=(12, 5))
    shade(ax, flare_sp, ref_sp)
    ax.fill_between(lo.index, lo.to_numpy(), hi.to_numpy(), color=FLARE_COLOR if flare_of[t] else PC_COLOR, lw=0,
                    step="post", label=f"{TJ} (min-max per {int(rule.total_seconds() // 60)} min)")
    ax.axhline(z_lim, color=INK2, ls="--", lw=1.5, label=f"±z(α'/2) = ±{z_lim:.2f}")
    ax.axhline(-z_lim, color=INK2, ls="--", lw=1.5)
    ax.set_ylim(-ylim, ylim)
    pct = rcx.loc[t, "pct_significant"]
    clip = f"; y clipped at ±{ylim:.1f}, max |T| = {big:.1f}" if big > ylim else ""
    ax.set_title(f"{node_label(t)}\n{len(parents[t])} parents, significant in {pct:.1f}% of samples{clip}",
                 loc="left", fontsize=14)
    ax.set_ylabel(TJ)
    time_axis(ax)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=11)
    rel = f"per_node/{i + 1:03d}_{safe_name(t)}.png"
    per_node.append(rel)
    save(fig, rel, f"{TJ} over time for {node_label(t)} with ±z limits, flare events and reference period shaded",
         f"parents={len(parents[t])}; significant={pct:.2f}%; max|T|={big:.1f}")
log(f"per-node charts: {len(nodes_t2)}")
del Tmin, Tmax

ncol = 5
nrow = int(np.ceil(len(per_node) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 2.2 * nrow), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, rel in zip(axes.ravel(), per_node):
    ax.imshow(plt.imread(FIG / rel))
fig.subplots_adjust(wspace=0.02, hspace=0.02)
save(fig, "per_node_contact_sheet.png", f"Grid of all {len(per_node)} per-node {TJ} charts", f"charts={len(per_node)}")

# ============================================================ 4) overall T², discrepancy, heatmap
t2max, t2med = t2["T2"].resample(rule).max(), t2["T2"].resample(rule).median()
fig, ax = plt.subplots(figsize=(13, 6))
shade(ax, flare_sp, ref_sp)
ax.plot(t2max.index, t2max.to_numpy(), color=SERIES[0], lw=1.2, label="T² (max per bin)")
ax.plot(t2med.index, t2med.to_numpy(), color="#0d366b", lw=1.5, label="T² (median per bin)")
ax.axhline(t2_lim, color=INK2, ls="--", lw=1.8, label=f"χ² limit = {t2_lim:.1f} (df = {s5['df']}, α = {s5['alpha0']})")
ax.set_yscale("log")
ax.set_ylabel(r"$T^2 = Z^{T}\Sigma^{-1}Z$ (log scale)")
ax.set_title(f"Overall T² on {s5['p']} nodes: above the limit in {s5['pct_ooc_all']:.1f}% of samples "
             f"({s5['pct_ooc_reference']:.1f}% in the reference period)", loc="left")
time_axis(ax)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
save(fig, "t2_overall.png", "Overall Hotelling T² (Eq. 7) with the χ² limit, flare events and reference period shaded",
     f"limit={t2_lim:.1f}; ooc_all={s5['pct_ooc_all']:.2f}%; ooc_reference={s5['pct_ooc_reference']:.2f}%")

dmed = t2["rel_discrepancy"].resample(rule).median()
d90 = t2["rel_discrepancy"].resample(rule).quantile(0.9)
fig, ax = plt.subplots(figsize=(13, 5.5))
shade(ax, flare_sp, ref_sp)
ax.fill_between(d90.index, 0, d90.to_numpy(), color="#9ec5f4", lw=0, step="post", label="90th percentile per bin")
ax.plot(dmed.index, dmed.to_numpy(), color=SERIES[0], lw=1.5, label="median per bin")
dis = s5["discrepancy"]
ax.set_ylabel(r"$|T^2 - \sum_j T^2_{j\cdot PA(j)}|\ /\ T^2$")
ax.set_title(f"Graph-quality diagnostic (Proposition 1): median relative discrepancy {dis['all']['median']:.2f} "
             f"(reference {dis['reference']['median']:.2f})", loc="left")
time_axis(ax)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
save(fig, "t2_discrepancy.png", "Relative gap between T² and the sum of the causal terms (0 for a correct graph)",
     f"median_all={dis['all']['median']:.3f}; p90_all={dis['all']['p90']:.3f}; "
     f"median_reference={dis['reference']['median']:.3f}")

rule_h = max(pd.Timedelta(hours=1), (span / 400).ceil("h"))
H = T.abs().resample(rule_h).mean()
Hm = H.to_numpy().T
vmax = max(z_lim, float(np.nanpercentile(Hm, 99)))
fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * len(nodes_t2) + 2)))
x0, x1 = mdates.date2num(H.index[0]), mdates.date2num(H.index[-1] + rule_h)
im = ax.imshow(Hm, aspect="auto", cmap=BLUES, vmin=0, vmax=vmax, extent=(x0, x1, len(nodes_t2), 0),
               interpolation="nearest")
ax.xaxis_date()
ax.grid(False)
ax.set_yticks(np.arange(len(nodes_t2)) + 0.5)
ax.set_yticklabels([node_label(v)[:30] for v in nodes_t2], fontsize=10)
time_axis(ax)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label(f"mean |{TJ}| per {rule_h}")
if z_lim <= vmax:
    cb.ax.axhline(z_lim, color=INK, lw=1.5)
ax.set_title(f"|{TJ}| for {len(nodes_t2)} nodes (rows) over time; colour capped at {vmax:.1f}", loc="left")
save(fig, "heatmap_abs_T.png", f"Heatmap of mean |{TJ}| (nodes x time); line on the colour bar = z limit",
     f"nodes={len(nodes_t2)}; bins={H.shape[0]}; colour_cap={vmax:.2f}; z_lim={z_lim:.2f}")
del H, Hm

# ============================================================ 5) top root-cause nodes per flare event and overall
INDEX.append({"filename": "tables/root_cause_by_event.csv", "what_it_shows": "Top nodes per flare event, with the "
              "top-loaded tags of each PC (table)",
              "key_numbers": f"events_with_alarm={ev_tab['event_id'].nunique()}; events_without_alarm={len(s5['events_without_ooc'])}"})
INDEX.append({"filename": "tables/flagged_pc_tags.csv", "what_it_shows": "Flagged PCs per event traced to their top-loaded "
              "plant tags (table)", "key_numbers": f"flagged_pcs={', '.join(s5['flagged_pcs']) or 'none'}"})
INDEX.append({"filename": "tables/root_cause_nodes.csv", "what_it_shows": "Root-cause statistics for every node (table)",
              "key_numbers": f"nodes={len(rc)}"})
ev_ids = list(ev_tab["event_id"].drop_duplicates())
if len(ev_ids) > 30:
    log(f"{len(ev_ids)} flare events with an alarm; bar charts drawn for the first 30, all are in the CSV")
for ev_id in ev_ids[:30]:
    d = ev_tab[ev_tab["event_id"] == ev_id].sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(d) + 2))
    for y, (node, v, sig) in enumerate(d[["node", "frac_significant", "significant_at_first_ooc"]].itertuples(index=False)):
        ax.barh(y, 100 * v, color=FLARE_COLOR if flare_of[node] else PC_COLOR, hatch="" if sig else "//",
                edgecolor=SURFACE, height=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([node_label(v) for v in d["node"]], fontsize=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    first = pd.to_datetime(d["first_ooc"].iloc[0])
    ax.set_xlabel(f"% of samples with |{TJ}| > {z_lim:.2f} in the 30 min after the first T² alarm")
    ax.set_title(f"Flare event {ev_id}: first T² alarm at {first:%Y-%m-%d %H:%M}, top {len(d)} nodes", loc="left")
    ax.legend(handles=[Patch(facecolor=PC_COLOR, label="PC"), Patch(facecolor=FLARE_COLOR, label="flare tag"),
                       Patch(facecolor=MUTED, hatch="//", edgecolor=SURFACE, label="not significant at first alarm")],
              loc="lower right", fontsize=11)
    top3 = ", ".join(d.sort_values("rank")["node"].head(3))
    save(fig, f"root_cause_event_{int(ev_id):02d}.png", f"Top root-cause nodes for flare event {ev_id}",
         f"first_alarm={first}; top3={top3}")

top = rc[~rc["is_flare"].astype(bool)].head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(12, 0.45 * len(top) + 2))
ax.barh(range(len(top)), top["pct_significant_when_T2_ooc"], color=PC_COLOR, edgecolor=SURFACE, height=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([node_label(v) for v in top["node"]], fontsize=12)
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"% of out-of-control samples with |{TJ}| > {z_lim:.2f}")
ax.set_title("PCs most often flagged as the origin of a T² alarm (all samples)", loc="left")
save(fig, "root_cause_overall.png", "PCs by share of out-of-control samples where their causal term is significant",
     "top3=" + ", ".join(top["node"].iloc[::-1].head(3)))

# ---- index and checks
idx = pd.DataFrame(INDEX, columns=["filename", "what_it_shows", "key_numbers"])
idx.to_csv(OUT / "figures_index.csv", index=False)
n_per_node = len(list((FIG / "per_node").glob("*.png")))
gate(n_per_node >= len(nodes_t2) and idx["filename"].str.startswith("figures/per_node/").sum() == len(nodes_t2),
     f"every T² node has a chart ({len(nodes_t2)} nodes, {n_per_node} PNGs in figures/per_node)")
gate(all((OUT / f).exists() for f in idx["filename"]), "every file in figures_index.csv exists")
seconds = time.time() - t0
log(f"figures_index.csv: {len(idx)} rows")
done(S, seconds)
