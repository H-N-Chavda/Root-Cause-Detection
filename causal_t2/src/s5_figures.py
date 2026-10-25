"""Stage 5 - figures and tables for the later slides (the PPT itself is not built here).

One style for every figure: 14-16 pt text, labelled axes, saved with a tight bounding box (no clipped text).
Colours: 8-hue categorical palette in fixed order; units beyond the 8th reuse the hues with a second marker
(graphs) or a hatch (bars), so no colour is generated. Time series with more than PLOT_MAX_POINTS samples are
binned; T_j·PA(j) charts show the min-max band of each bin so no spike is hidden.

Writes outputs/figures/** (PNG) and outputs/figures_index.csv (filename, what it shows, key numbers).
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

from common import (DATA, FIG, FLARE_FOLDER, GRAPH, OUT, PLOT_MAX_POINTS, SEED, TAB, UNIT_FOLDERS, ZOOM_N,  # noqa: E402
                    clear_notes, done, flags_to_events, gate, load_json, log, read_flare_events, read_registry,
                    safe_name, stage)

S = "s5_figures"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                    "#184f95", "#0d366b"])
FLARE_SHADE, REF_SHADE = SERIES[1], SERIES[2]
MARKERS, HATCHES = ["o", "s", "D", "v"], ["", "//", "\\\\", ".."]
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
stage("Stage 5 - figures and tables")
clear_notes(S)
reg = read_registry()
unit_of = dict(zip(reg["tag"], reg["unit"]))
flare_of = dict(zip(reg["tag"], reg["is_flare"]))
s4 = load_json("s4_summary.json")
z_lim, t2_lim = s4["z_lim"], s4["t2_lim"]
T = pd.read_parquet(DATA / "T_terms.parquet")
t2 = pd.read_parquet(DATA / "t2.parquet")
meta = pd.read_parquet(DATA / "meta.parquet")
events = read_flare_events()
rc = pd.read_csv(TAB / "root_cause_tags.csv", dtype={"tag": str})
ev_tab = pd.read_csv(TAB / "root_cause_by_event.csv", dtype={"tag": str})
parents = load_json("parents.json")
tags = list(T.columns)

# ---- unit styles: fixed hue order, then a second marker / hatch (never a generated colour)
present = set(reg["unit"])
units = [u for u in UNIT_FOLDERS if u in present] + sorted(present - set(UNIT_FOLDERS) - {FLARE_FOLDER})
STYLE = {u: (SERIES[i % 8], MARKERS[(i // 8) % 4], HATCHES[(i // 8) % 4]) for i, u in enumerate(units)}
STYLE[FLARE_FOLDER] = (INK, "^", "xx")


def unit_patches(us):
    return [Patch(facecolor=STYLE[u][0], hatch=STYLE[u][2], edgecolor=SURFACE, label=u) for u in us]


def unit_markers(us):
    return [Line2D([], [], ls="", marker=STYLE[u][1], color=STYLE[u][0], markersize=9, label=u) for u in us]


# ---- binning shared by the time-series figures
span = T.index.max() - T.index.min()
rule = max(pd.Timedelta(minutes=1), (span / PLOT_MAX_POINTS).ceil("min"))
flare_sp = spans(meta["flare"].astype(float).resample(rule).max().fillna(0), rule)
ref_sp = spans(meta["reference"].astype(float).resample(rule).max().fillna(0), rule)
log(f"time series binned to {rule} ({len(flare_sp)} flare spans, {len(ref_sp)} reference spans shaded)")
Tmin, Tmax = T.resample(rule).min(), T.resample(rule).max()
for old in (FIG / "per_tag").glob("*.png"):     # generated files from an earlier run
    old.unlink()

# ---- 1) one T_j·PA(j) chart per tag
rcx = rc.set_index("tag")
for i, t in enumerate(tags):
    lo, hi = Tmin[t], Tmax[t]
    big = float(np.nanmax(np.abs(np.r_[lo.to_numpy(), hi.to_numpy()])))
    ylim = max(1.5 * z_lim, float(np.nanpercentile(np.abs(np.r_[lo.dropna(), hi.dropna()]), 99.9)))
    fig, ax = plt.subplots(figsize=(12, 5))
    shade(ax, flare_sp, ref_sp)
    ax.fill_between(lo.index, lo.to_numpy(), hi.to_numpy(), color=SERIES[0], lw=0, step="post",
                    label=f"{TJ} (min-max per {int(rule.total_seconds() // 60)} min)")
    ax.axhline(z_lim, color=INK2, ls="--", lw=1.5, label=f"±z(α'/2) = ±{z_lim:.2f}")
    ax.axhline(-z_lim, color=INK2, ls="--", lw=1.5)
    ax.set_ylim(-ylim, ylim)
    pct = rcx.loc[t, "pct_significant"]
    clip = f"; y clipped at ±{ylim:.1f}, max |T| = {big:.1f}" if big > ylim else ""
    ax.set_title(f"{t}  ({unit_of[t]})\n{len(parents[t])} parents, significant in {pct:.1f}% of samples{clip}",
                 loc="left", fontsize=14)
    ax.set_ylabel(TJ)
    time_axis(ax)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=11)
    save(fig, f"per_tag/{i + 1:03d}_{safe_name(t)}.png", f"{TJ} over time for {t} ({unit_of[t]}) with ±z limits, "
         "flare events and reference period shaded",
         f"parents={len(parents[t])}; significant={pct:.2f}%; max|T|={big:.1f}")
log(f"per-tag charts: {len(tags)}")
del Tmin, Tmax

# ---- 2) overall T² chart
t2max, t2med = t2["T2"].resample(rule).max(), t2["T2"].resample(rule).median()
fig, ax = plt.subplots(figsize=(13, 6))
shade(ax, flare_sp, ref_sp)
ax.plot(t2max.index, t2max.to_numpy(), color=SERIES[0], lw=1.2, label="T² (max per bin)")
ax.plot(t2med.index, t2med.to_numpy(), color="#0d366b", lw=1.5, label="T² (median per bin)")
ax.axhline(t2_lim, color=INK2, ls="--", lw=1.8, label=f"χ² limit = {t2_lim:.1f} (df = {s4['df']}, α = {s4['alpha0']})")
ax.set_yscale("log")
ax.set_ylabel(r"$T^2 = Z^{T}\Sigma^{-1}Z$ (log scale)")
ax.set_title(f"Overall T²: above the limit in {s4['pct_ooc_all']:.1f}% of samples "
             f"({s4['pct_ooc_reference']:.1f}% in the reference period)", loc="left")
time_axis(ax)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
save(fig, "t2_overall.png", "Overall Hotelling T² (Eq. 7) with the χ² limit, flare events and reference period shaded",
     f"limit={t2_lim:.1f}; ooc_all={s4['pct_ooc_all']:.2f}%; ooc_reference={s4['pct_ooc_reference']:.2f}%")

# ---- 3) Proposition 1 discrepancy over time
dmed = t2["rel_discrepancy"].resample(rule).median()
d90 = t2["rel_discrepancy"].resample(rule).quantile(0.9)
fig, ax = plt.subplots(figsize=(13, 5.5))
shade(ax, flare_sp, ref_sp)
ax.fill_between(d90.index, 0, d90.to_numpy(), color="#9ec5f4", lw=0, step="post", label="90th percentile per bin")
ax.plot(dmed.index, dmed.to_numpy(), color=SERIES[0], lw=1.5, label="median per bin")
dis = s4["discrepancy"]
ax.set_ylabel(r"$|T^2 - \sum_j T^2_{j\cdot PA(j)}|\ /\ T^2$")
ax.set_title(f"Graph-quality diagnostic (Proposition 1): median relative discrepancy {dis['all']['median']:.2f} "
             f"(reference {dis['reference']['median']:.2f})", loc="left")
time_axis(ax)
ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
save(fig, "t2_discrepancy.png", "Relative gap between T² and the sum of the causal terms (0 for a correct graph)",
     f"median_all={dis['all']['median']:.3f}; p90_all={dis['all']['p90']:.3f}; "
     f"median_reference={dis['reference']['median']:.3f}")

# ---- 4) heatmap of |T_j·PA(j)|, tags grouped by unit
rule_h = max(pd.Timedelta(days=1), (span / 400).ceil("h"))
H = T.abs().resample(rule_h).mean()
order = sorted(range(len(tags)), key=lambda j: (([*units, FLARE_FOLDER]).index(unit_of[tags[j]]), j))
Hm = H.iloc[:, order].to_numpy().T
vmax = max(z_lim, float(np.nanpercentile(Hm, 99)))
fig, ax = plt.subplots(figsize=(14, 10))
x0, x1 = mdates.date2num(H.index[0]), mdates.date2num(H.index[-1] + rule_h)
im = ax.imshow(Hm, aspect="auto", cmap=BLUES, vmin=0, vmax=vmax, extent=(x0, x1, len(order), 0),
               interpolation="nearest")
ax.xaxis_date()
ax.grid(False)
ou = [unit_of[tags[j]] for j in order]
bounds = [k for k in range(1, len(ou)) if ou[k] != ou[k - 1]]
for b in bounds:
    ax.axhline(b, color=SURFACE, lw=1.5)
starts = [0] + bounds
ends = bounds + [len(ou)]
ax.set_yticks([(a + b) / 2 for a, b in zip(starts, ends)])
ax.set_yticklabels([f"{ou[a]} ({b - a})" for a, b in zip(starts, ends)], fontsize=11)
time_axis(ax)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label(f"mean |{TJ}| per {rule_h}")
if z_lim <= vmax:
    cb.ax.axhline(z_lim, color=INK, lw=1.5)
ax.set_title(f"|{TJ}| for {len(tags)} tags (rows) over time, grouped by unit; colour capped at {vmax:.1f}", loc="left")
save(fig, "heatmap_abs_T.png", f"Heatmap of mean |{TJ}| (tags x time), tags grouped by unit; line on the colour bar = z limit",
     f"tags={len(tags)}; bins={H.shape[0]}; colour_cap={vmax:.2f}; z_lim={z_lim:.2f}")
del H, Hm

# ---- 5) top root-cause tags per flare event (bars) and overall
INDEX.append({"filename": "tables/root_cause_by_event.csv", "what_it_shows": "Top tags per flare event (table)",
              "key_numbers": f"events_with_alarm={ev_tab['event_id'].nunique()}; events_without_alarm={len(s4['events_without_ooc'])}"})
INDEX.append({"filename": "tables/root_cause_tags.csv", "what_it_shows": "Root-cause statistics for every tag (table)",
              "key_numbers": f"tags={len(rc)}"})
ev_ids = list(ev_tab["event_id"].drop_duplicates())
if len(ev_ids) > 30:
    log(f"{len(ev_ids)} flare events with an alarm; bar charts drawn for the first 30, all are in the CSV")
for ev_id in ev_ids[:30]:
    d = ev_tab[ev_tab["event_id"] == ev_id].sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(d) + 2))
    for y, (tag, unit, v) in enumerate(d[["tag", "unit", "frac_significant"]].itertuples(index=False)):
        c, _, h = STYLE.get(unit, (MUTED, "o", ""))
        ax.barh(y, 100 * v, color=c, hatch=h, edgecolor=SURFACE, height=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["tag"], fontsize=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    first = pd.to_datetime(d["first_ooc"].iloc[0])
    ax.set_xlabel(f"% of samples with |{TJ}| > {z_lim:.2f} in the 30 min after the first T² alarm")
    ax.set_title(f"Flare event {ev_id}: first T² alarm at {first:%Y-%m-%d %H:%M}, top {len(d)} tags", loc="left")
    ax.legend(handles=unit_patches(list(dict.fromkeys(d["unit"][::-1]))), loc="lower right", fontsize=11)
    top3 = ", ".join(d.sort_values("rank")["tag"].head(3))
    save(fig, f"root_cause_event_{int(ev_id):02d}.png", f"Top root-cause tags for flare event {ev_id}",
         f"first_alarm={first}; top3={top3}")

top = rc[~rc["is_flare"]].head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(12, 0.45 * len(top) + 2))
for y, (unit, v) in enumerate(top[["unit", "pct_significant_when_T2_ooc"]].itertuples(index=False)):
    c, _, h = STYLE.get(unit, (MUTED, "o", ""))
    ax.barh(y, v, color=c, hatch=h, edgecolor=SURFACE, height=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["tag"], fontsize=12)
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"% of out-of-control samples with |{TJ}| > {z_lim:.2f}")
ax.set_title("Tags most often flagged as the origin of a T² alarm (all samples)", loc="left")
ax.legend(handles=unit_patches(list(dict.fromkeys(top["unit"][::-1]))), loc="lower right", fontsize=11)
save(fig, "root_cause_overall.png", "Top 20 process tags by share of out-of-control samples where their causal term is significant",
     "top3=" + ", ".join(top["tag"].iloc[::-1].head(3)))

# ---- 6) PC graph views
pc_edges = pd.read_csv(GRAPH / "pc_edges.csv", dtype={"source": str, "target": str})
pc_nodes = [str(t) for t in np.load(DATA / "pc_graph.npz", allow_pickle=True)["names"]]
g = nx.Graph()
g.add_nodes_from(pc_nodes)
g.add_edges_from(zip(pc_edges["source"], pc_edges["target"]))
pos = nx.spring_layout(g, seed=SEED, k=1.5 / np.sqrt(max(len(pc_nodes), 1)), iterations=100)
deg = dict(g.degree())
fig, ax = plt.subplots(figsize=(14, 11))
nx.draw_networkx_edges(g, pos, ax=ax, edge_color=MUTED, alpha=0.2, width=0.5)
us = [u for u in [*units, FLARE_FOLDER] if any(unit_of[t] == u for t in pc_nodes)]
for u in us:
    nl = [t for t in pc_nodes if unit_of[t] == u]
    nx.draw_networkx_nodes(g, pos, nodelist=nl, ax=ax, node_color=STYLE[u][0], node_shape=STYLE[u][1],
                           node_size=[25 + 4 * deg[t] for t in nl], edgecolors=SURFACE, linewidths=0.5)
ax.axis("off")
types = pc_edges["type"].value_counts().to_dict()
ax.set_title(f"PC graph: {len(pc_nodes)} tags, {len(pc_edges)} edges ({types.get('->', 0)} directed, "
             f"{types.get('--', 0)} undirected, {types.get('<->', 0)} bidirected); node size = degree", loc="left")
ax.legend(handles=unit_markers(us), loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, title="unit")
save(fig, "pc_graph_full.png", "Full PC graph (CPDAG skeleton), nodes coloured by unit",
     f"nodes={len(pc_nodes)}; edges={len(pc_edges)}; directed={types.get('->', 0)}; undirected={types.get('--', 0)}")

dag = nx.DiGraph()
dag.add_nodes_from(pc_nodes)
dag_edges = pd.read_csv(GRAPH / "dag_edges.csv", dtype={"source": str, "target": str})
dag.add_edges_from(zip(dag_edges["source"], dag_edges["target"]))
dv = np.array([deg[t] for t in pc_nodes])
pv = np.array([dag.in_degree(t) for t in pc_nodes])
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, v, lab in ((axes[0], dv, "degree in the PC graph"), (axes[1], pv, "number of parents in the DAG")):
    ax.hist(v, bins=range(0, int(v.max()) + 2), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel(lab)
    ax.set_ylabel("number of tags")
    ax.set_title(f"mean {v.mean():.1f}, max {v.max()}", loc="left")
save(fig, "pc_degree_distribution.png", "Degree distribution of the PC graph and parent counts in the DAG",
     f"mean_degree={dv.mean():.2f}; max_degree={dv.max()}; mean_parents={pv.mean():.2f}; max_parents={pv.max()}")

zoom = list(dict.fromkeys([t for t in ev_tab.sort_values(["rank", "event_id"])["tag"] if not flare_of.get(t, False)]
                          + list(rc.loc[~rc["is_flare"], "tag"])))[:ZOOM_N]
for k, c in enumerate(zoom, start=1):
    par, chi = list(dag.predecessors(c)), list(dag.successors(c))
    others = (par + sorted(chi, key=lambda t: -deg[t]))[:30]
    sub = dag.subgraph([c] + others)
    sp = nx.shell_layout(sub, [[c], others]) if others else {c: np.zeros(2)}
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(sub, sp, ax=ax, edge_color=INK2, width=1.3, arrows=True, arrowsize=16, node_size=900)
    for u in dict.fromkeys(unit_of[t] for t in sub.nodes):
        nl = [t for t in sub.nodes if unit_of[t] == u]
        nx.draw_networkx_nodes(sub, sp, nodelist=nl, ax=ax, node_color=STYLE[u][0], node_shape=STYLE[u][1],
                               node_size=[1600 if t == c else 700 for t in nl],
                               edgecolors=[INK if t == c else SURFACE for t in nl], linewidths=2)
    nx.draw_networkx_labels(sub, {t: (x, y - 0.09) for t, (x, y) in sp.items()}, ax=ax, font_size=10, font_color=INK)
    ax.axis("off")
    ax.margins(0.15)
    us_sub = list(dict.fromkeys(unit_of[t] for t in sub.nodes))
    ax.legend(handles=unit_markers(us_sub), loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, title="unit")
    pct = rcx.loc[c, "pct_significant_when_T2_ooc"]
    ax.set_title(f"{c} ({unit_of[c]}) in the DAG: {len(par)} parents, {len(chi)} children"
                 f"{' (30 shown)' if len(par) + len(chi) > 30 else ''}\nsignificant in {pct:.1f}% of T² alarms",
                 loc="left")
    save(fig, f"pc_zoom_{k}_{safe_name(c)}.png", f"DAG neighbourhood (parents and children) of root-cause tag {c}",
         f"parents={len(par)}; children={len(chi)}; pct_sig_when_ooc={pct:.2f}%")

# ---- index and checks
idx = pd.DataFrame(INDEX, columns=["filename", "what_it_shows", "key_numbers"])
idx.to_csv(OUT / "figures_index.csv", index=False)
n_per_tag = len(list((FIG / "per_tag").glob("*.png")))
gate(n_per_tag >= len(tags) and idx["filename"].str.startswith("figures/per_tag/").sum() == len(tags),
     f"every T² tag has a chart ({len(tags)} tags, {n_per_tag} PNGs in figures/per_tag)")
gate(all((OUT / f).exists() for f in idx["filename"]), "every file in figures_index.csv exists")
seconds = time.time() - t0
log(f"figures_index.csv: {len(idx)} rows")
done(S, seconds)
