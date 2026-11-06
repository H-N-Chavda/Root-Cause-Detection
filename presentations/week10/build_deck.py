"""Week 10 deck: unit-wise PCA -> PC -> causation-based T2 on Jul-Oct 2025 plant data (lab run 2f3c75a).

Built on the TA's reference deck (HPCL Project_08072026_external.pptx) so the master, theme, title bar and fonts
are the TA's own. Reads only the committed unit_pca_pc_t2/outputs/ (no plant data); the previous run (commit
eaebb1e) is read with `git show` for the loop slides. Run from the repo root:

    .venv/bin/python presentations/week10/build_deck.py
"""

import copy
import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "unit_pca_pc_t2" / "outputs"
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
REF = ROOT / "references" / "Previous Project Reports" / "HPCL Project_08072026_external.pptx"
DECK = HERE / "HPCL Project_05112026_flare_RCA.pptx"
OLD = "eaebb1e"  # previous lab run (14 units, NOC with 120-min buffer)
CASE_EVENTS = [144, 449]
PC_DEEP = ["DAK_PC1", "APS_NVPS_LEU_PC1"]
TITLE_PT = 28
SRC = "Lab run of unit_pca_pc_t2 (commit 2f3c75a), 1-min data, 1 Jul – 31 Oct 2025."
W_IN = 60  # minutes in the pre-flare window
FN_CUT = 8  # t/h: flare_net axis cut in the four-month chart

# ---------------------------------------------------------------- data (current run)
ev = pd.read_csv(OUT / "tables" / "event_outcomes.csv",
                 parse_dates=["onset", "end", "window_start", "first_above", "analysed_minute"])
bl = pd.read_csv(OUT / "tables" / "nonflare_blocks.csv", parse_dates=["start", "end", "first_above"])
fe = pd.read_csv(OUT / "tables" / "flare_events.csv", parse_dates=["start", "end"])
M = pd.read_csv(OUT / "tables" / "metrics.csv").set_index("metric")["value"]
lim = pd.read_csv(OUT / "t2" / "limits.csv").iloc[0]
tl = pd.read_csv(OUT / "t2" / "term_limits.csv").set_index("node")
dag = pd.read_csv(OUT / "graph" / "dag_edges.csv")
pce = pd.read_csv(OUT / "graph" / "pc_edges.csv")
pcu = pd.read_csv(OUT / "tables" / "pcs_per_unit.csv")
load = pd.read_csv(OUT / "tables" / "pca_loadings.csv", index_col=0)
scores = pd.read_parquet(OUT / "graph" / "pc_scores.parquet")
terms = pd.read_parquet(OUT / "t2" / "terms_all_rows.parquet")
t2 = pd.read_parquet(OUT / "t2" / "t2_all_rows.parquet")
fn = pd.read_parquet(OUT / "tables" / "flare_net_raw.parquet")["flare_net"]

TP, FN, FP = int(M["TP (flare events with T² above the limit in the window)"]), int(M["FN (flare events without)"]), \
    int(M["FP (non-flare blocks with T² above the limit)"])
F1 = float(M["F1 = 2TP / (2TP + FP + FN)"])
N_NS = int(M["flare events not scored (no T² in the window)"])
N_BL, N_BL_NS = int(M["non-flare blocks"]), int(M["non-flare blocks not scored"])
N_SC_BL = N_BL - N_BL_NS
N_ALREADY = int(M["TP: T² already above the limit when the window opened"])
LEAD_MED = float(M["median minutes from first T² above the limit to onset (TP)"])
assert (TP, FN, FP, len(fe)) == (519, 68, 250, 620), "outputs changed: re-check every number in this script"
PREC, REC = TP / (TP + FP), TP / (TP + FN)
assert abs(2 * PREC * REC / (PREC + REC) - F1) < 1e-9
F1_ALWAYS = 2 * (TP + FN) / (2 * (TP + FN) + N_SC_BL)  # T² above the limit at every scored minute
tp = ev[ev.outcome == "TP"]
fpb = bl[bl.outcome == "FP"]
rose = tp[~tp.above_at_window_start.astype(bool)]
top1_tp, top1_fp = tp.pc1.value_counts(), fpb.pc1.value_counts()
N_PCS, N_UNITS, N_TAGS = int(pcu.n_pcs.sum()), len(pcu), int(pcu.n_tags.sum())
VAR_ALL = 100 * float((pcu.variance_kept * pcu.n_tags).sum() / pcu.n_tags.sum())
pc_dag = dag[dag.target != "flare_net"]
N_EDGES, N_CROSS = len(pc_dag), int((pc_dag.source.str.rsplit("_PC", n=1).str[0]
                                     != pc_dag.target.str.rsplit("_PC", n=1).str[0]).sum())
N_DIR, N_UND = int((pce.type == "->").sum()), int((pce.type != "->").sum())
n_par = pc_dag.groupby("target").size().reindex(tl.index, fill_value=0)
scored = t2.scored.to_numpy()
nf = ~(t2.flare_event | t2.short_flare)
PCT_NOC_ABOVE = float(lim.pct_noc_above_t2_limit)
PCT_NF_ABOVE = 100 * float(t2.above_limit[t2.scored & nf].mean())
N_UNSCORED = int((~t2.scored).sum())
FIRST_SCORED = t2.index[scored.argmax()]
ratio = (terms.astype(np.float64) ** 2) / tl.t2_limit
assert abs(VAR_ALL - 46.7) < 0.05 and N_EDGES == 75 and N_PCS == 24

# share of minutes each PC is above its own limit, by period (whole four months)
in_pre = (t2.preflare_event_id > 0) & t2.scored
in_flare = t2.flare_event & t2.scored
normal = t2.scored & nf & ~(t2.preflare_event_id > 0)
PCAVG = pd.DataFrame({"pre": 100 * (ratio[in_pre] > 1).mean(), "flare": 100 * (ratio[in_flare] > 1).mean(),
                      "normal": 100 * (ratio[normal] > 1).mean()})
PCAVG["lift"] = PCAVG.pre / PCAVG.normal
LIFT = PCAVG[PCAVG.pre >= 5].sort_values("lift", ascending=False)
unit_tp = tp.pc1.str.rsplit("_PC", n=1).str[0].value_counts()
unit_fp = fpb.pc1.str.rsplit("_PC", n=1).str[0].value_counts()


def git_csv(path):
    return pd.read_csv(io.StringIO(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{OLD}:{path}"]).decode()))


def git_text(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{OLD}:{path}"]).decode()


old_pce = git_csv("unit_pca_pc_t2/outputs/graph/pc_edges.csv")
old_dag = git_csv("unit_pca_pc_t2/outputs/graph/dag_edges.csv")
old_cyc = git_csv("unit_pca_pc_t2/outputs/graph/cycles.csv")
old_stats = git_csv("unit_pca_pc_t2/outputs/graph/dag_stats.csv").iloc[0]
old_pcu = git_csv("unit_pca_pc_t2/outputs/tables/pcs_per_unit.csv")
old_dir = {(s, t) for s, t, ty in old_pce.itertuples(index=False) if ty == "->"}
old_dg = {(s, t) for s, t in old_dag.itertuples(index=False)}
OLD_REV = sorted(e for e in old_dir if (e[1], e[0]) in old_dg)
OLD_CYC = old_cyc[old_cyc.graph.str.startswith("PC output")]
assert len(OLD_CYC) == 3 and int(old_stats.n_reversed_edges) == 2
assert OLD_REV == [("ISOM_3_PC2", "unit_wise_control_data_PC1"), ("PDA_PC1", "LOUP_PC1")]
assert "NOC rows 12157" in git_text("unit_pca_pc_t2/outputs/logs/run.log")
assert len(pd.read_csv(OUT / "graph" / "cycles.csv")) == 0


def top_tags(pc, n=3):
    w = load.loc[pc] ** 2
    w = w[w > 0].sort_values(ascending=False) / w.sum()
    return [(t, 100 * v) for t, v in w.head(n).items()]


# ---------------------------------------------------------------- figures (TA's matplotlib look)
ASSETS.mkdir(exist_ok=True)
for old in ASSETS.glob("*.png"):
    old.unlink()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.titlelocation": "left"})
T0, T1 = fn.index[0], fn.index[-1]
DPI = 150
C_PC = ["#8e44ad", "#16a085", "#d35400"]  # colours of the 1st, 2nd and 3rd PC


def figure(rows, cols, h, w=12.9, **kw):
    return plt.subplots(rows, cols, figsize=(w, h), layout="constrained", **kw)


def leg(ax, ncol=4, handles=None, labels=None):
    """Legend in one row just above the axes, never on the data."""
    args = (handles, labels) if handles is not None else ()
    ax.legend(*args, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, borderaxespad=0.15,
              handlelength=1.6, columnspacing=1.4)


def title(fig, text):
    fig.suptitle(text, x=0.005, ha="left", fontsize=13)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 11, 21]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(T0, T1)


def minute_axis(ax, a, b, step=30):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=list(range(0, 60, step))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(a, b)


def save(fig, name):
    for ax in fig.axes:
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(ASSETS / name, dpi=DPI)
    plt.close(fig)
    return ASSETS / name


def shade_events(ax, alpha=0.18):
    for s, e in zip(fe.start, fe.end):
        ax.axvspan(s, e, color="orange", alpha=alpha, lw=0)


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b - 1) for a, b in zip(e[::2], e[1::2])]


def fig_counts():
    fig, ax = figure(1, 1, 4.6, 6.6)
    vals = [TP, FN, FP]
    ax.bar(["TP\nflare events caught", "FN\nflare events missed", "FP\ngaps with a false rise"], vals,
           color=["#2ca02c", "#c0392b", "#e67e22"], width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v + 8, str(v), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Precision {PREC:.3f}   ·   Recall {REC:.3f}   ·   F1 {F1:.3f}", fontweight="bold")
    return save(fig, "counts.png")


def fig_daily_above():
    d = 100 * t2[t2.scored & nf].above_limit.resample("1D").mean()
    fig, ax = figure(1, 1, 3.9)
    ax.bar(d.index, d.values, width=0.8, color="#1f5fa8", label="flare-free scored minutes above the T² limit (%)")
    ax.axhline(5, color="red", ls="--", lw=1.2, label="5 % expected by chance (α = 0.05)")
    ax.axhline(PCT_NF_ABOVE, color="black", ls=":", lw=1.2, label=f"four-month average {PCT_NF_ABOVE:.1f} %")
    ax.set_ylabel("% of the day's minutes")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    leg(ax, 3)
    date_axis(ax)
    title(fig, "Daily share of flare-free minutes with overall T² above its limit")
    return save(fig, "daily_above.png")


def fig_lead():
    bins = np.arange(0.5, W_IN + 1.5, 1)
    fig, ax = figure(1, 1, 4.9, 7.4)
    ax.hist([rose.lead_min, tp[tp.above_at_window_start.astype(bool)].lead_min], bins=bins, stacked=True,
            color=["#2ca02c", "#9e9e9e"], label=[f"T² crossed inside the window (n = {len(rose)})",
                                                  f"T² already above when the window opened (n = {N_ALREADY})"])
    ax.set_xlabel("minutes from first T² above the limit to flare onset")
    ax.set_ylabel("TP flare events")
    ax.set_xlim(0, W_IN + 1)
    ax.grid(alpha=0.3)
    leg(ax, 1)
    title(fig, f"Warning time of the {TP} TP events")
    return save(fig, "lead.png")


def fig_pca_var():
    p = pcu.sort_values("variance_kept")
    fig, ax = figure(1, 1, 5.4, 6.0)
    ax.barh(p.unit, 100 * p.variance_kept, color="#5B9BD5")
    for y, (v, k, n) in enumerate(zip(p.variance_kept, p.n_pcs, p.n_tags)):
        ax.text(100 * v + 1, y, f"{100 * v:.0f}%  ({k} PC{'s' if k > 1 else ''}, {n} tag{'s' if n > 1 else ''})",
                va="center", fontsize=10)
    ax.set_xlim(0, 140)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("variance of the unit kept by its PCs (%, NOC rows)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{N_PCS} PCs from {N_UNITS} units keep {VAR_ALL:.1f} % overall")
    return save(fig, "pca_var.png")


def fig_t2_overall():
    fig, axs = figure(2, 1, 5.0, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.25]))
    n_cut = int((fn > FN_CUT).sum())
    axs[0].plot(fn.index, fn.clip(upper=FN_CUT).values, color="#7FA7E0", lw=0.3, label="flare_net (1-min)")
    hm = fn.resample("1h").median()
    axs[0].plot(hm.index, hm.values, color="#0B2C8A", lw=0.9, label="hourly median")
    axs[0].axhline(1.5, color="red", ls="--", lw=1.2, label="threshold 1.5 t/h")
    axs[0].set_ylim(-1, FN_CUT)
    axs[0].set_ylabel("flare_net (t/h)")
    h, l_ = axs[0].get_legend_handles_labels()
    leg(axs[0], 4, h + [Line2D([], [], color="none")],
        l_ + [f"axis cut at {FN_CUT} t/h: {n_cut} min ({100 * n_cut / len(fn):.2f} %) above, max {fn.max():.1f} t/h"])
    axs[1].plot(t2.index, t2["T2"], color="green", lw=0.4, label="overall T² (moving-window baseline)")
    axs[1].axhline(lim.t2_limit, color="red", ls="--", lw=1.2, label=f"T² limit {lim.t2_limit:.1f}")
    for a, b in runs(~scored):
        axs[1].axvspan(t2.index[a], t2.index[b], color="#9e9e9e", alpha=0.35, lw=0)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("T² (log scale)")
    h, l_ = axs[1].get_legend_handles_labels()
    leg(axs[1], 3, h + [Patch(color="#9e9e9e", alpha=0.35)], l_ + ["no T² (too few flare-free minutes)"])
    for a in axs:
        a.grid(alpha=0.3)
    date_axis(axs[1])
    return save(fig, "t2_overall.png")


def fig_rootcause():
    d = pd.DataFrame({"tp": top1_tp.reindex(tl.index, fill_value=0), "fp": top1_fp.reindex(tl.index, fill_value=0)})
    d = d.sort_values(["tp", "fp"], ascending=False)
    x = np.arange(len(d))
    fig, ax = figure(1, 1, 4.6)
    ax.bar(x - 0.2, d.tp, 0.4, color="#1f5fa8", label=f"top-1 PC of a TP flare event ({TP} events)")
    ax.bar(x + 0.2, d.fp, 0.4, color="#c0392b", label=f"top-1 PC of an FP gap ({FP} gaps)")
    ax.set_xticks(x, d.index, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    leg(ax, 2)
    title(fig, "How often each PC has the largest T²_j / own limit at the first minute above the overall limit")
    return save(fig, "rootcause.png")


def fig_pc_avg():
    d = PCAVG.sort_values("pre", ascending=False)
    x = np.arange(len(d))
    fig, ax = figure(1, 1, 4.6)
    ax.bar(x - 0.27, d.pre, 0.27, color="#d62728", label="60 min before a flare")
    ax.bar(x, d.normal, 0.27, color="#1f5fa8", label="normal time (no flare, not before one)")
    ax.bar(x + 0.27, d.flare, 0.27, color="#e67e22", label="during a flare")
    ax.axhline(5, color="black", ls="--", lw=1.1, label="5 % (α = 0.05)")
    ax.set_xticks(x, d.index, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("% of minutes above own limit")
    ax.grid(axis="y", alpha=0.3)
    leg(ax, 4)
    title(fig, "Share of minutes each PC is above its own limit, Jul – Oct 2025 (scored minutes only)")
    return save(fig, "pc_avg.png")


def fig_top_raw(pcs):
    fig, axs = figure(len(pcs), 1, 4.7, sharex=True)
    for ax, pc, c in zip(axs, pcs, ["#1f5fa8", "#8e44ad", "#16a085"]):
        shade_events(ax, 0.12)
        s = scores[pc]
        ax.plot(s.index, s.values, color=c, lw=0.3, alpha=0.45)
        ax.plot(s.index, s.rolling("6h").median().values, color=c, lw=1.4)
        ax.set_ylabel(pc, fontsize=10)
        ax.grid(alpha=0.3)
    date_axis(axs[-1])
    title(fig, "Raw PC scores (standardised with NOC mean and std): light = 1-min, dark = 6-h median; "
               "orange = flare events")
    return save(fig, "top_raw.png")


def fig_pc_deep(pc):
    fig, axs = figure(2, 1, 4.4, sharex=True)
    shade_events(axs[0], 0.12)
    s = scores[pc]
    axs[0].plot(s.index, s.values, color="#1f5fa8", lw=0.5)
    axs[0].set_ylabel(pc, fontsize=11)
    r = ratio[pc].resample("1h").max()
    axs[1].plot(r.index, r.clip(lower=1e-3).values, color="#8e44ad", lw=0.7, label="T²_j / own limit (hourly max)")
    axs[1].axhline(1, color="red", ls="--", lw=1.2, label="own limit (ratio 1)")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(f"{pc}\nT²_j / limit", fontsize=11)
    leg(axs[1], 2)
    for a in axs:
        a.grid(alpha=0.3)
    date_axis(axs[1])
    title(fig, f"{pc}: raw score (top, orange = flare events) and causal T² term (bottom), four months")
    return save(fig, f"pc_{pc}.png")


def shade_window(ax, r, mark=True):
    ax.axvspan(r.onset - pd.Timedelta(minutes=W_IN), r.onset, color="#d62728", alpha=0.10, lw=0)
    ax.axvspan(r.onset, r.end + pd.Timedelta(minutes=1), color="orange", alpha=0.22, lw=0)
    if mark and pd.notna(r.first_above):
        ax.axvline(r.first_above, color="black", ls=":", lw=1.4)


def win_patches():
    return [Patch(color="#d62728", alpha=0.1), Patch(color="orange", alpha=0.22),
            Line2D([], [], color="black", ls=":", lw=1.4)], ["60 min before onset", "flare event",
                                                               "first minute above the T² limit"]


def fig_event_overview(r):
    a, b = r.onset - pd.Timedelta(minutes=180), r.onset + pd.Timedelta(minutes=150)
    pcs = [r.pc1, r.pc2, r.pc3]
    fig, axs = figure(3, 1, 5.05, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.3]))
    f = fn.loc[a:b]
    axs[0].plot(f.index, f.values, color="blue", lw=1.2, label="flare_net")
    axs[0].axhline(1.5, color="red", ls="--", lw=1.1, label="threshold 1.5 t/h")
    axs[0].set_ylabel("flare_net\n(t/h)")
    hw, lw_ = win_patches()
    h, l_ = axs[0].get_legend_handles_labels()
    leg(axs[0], 5, h + hw, l_ + lw_)
    t = t2["T2"].loc[a:b]
    axs[1].plot(t.index, t.values, color="green", lw=1.2, label="overall T²")
    axs[1].axhline(lim.t2_limit, color="red", ls="--", lw=1.1, label=f"T² limit {lim.t2_limit:.1f}")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("overall T²\n(log)")
    leg(axs[1], 2)
    for pc, c in zip(pcs, C_PC):
        y = ratio[pc].loc[a:b].clip(lower=1e-3)
        axs[2].plot(y.index, y.values, color=c, lw=1.2, label=pc)
    axs[2].axhline(1, color="red", ls="--", lw=1.1, label="own limit")
    axs[2].set_yscale("log")
    axs[2].set_ylabel("T²_j / own limit\n(log)")
    leg(axs[2], 4)
    for ax in axs:
        shade_window(ax, r)
        ax.grid(alpha=0.3)
    minute_axis(axs[-1], a, b)
    axs[-1].set_xlabel(f"{r.onset:%d %b %Y}")
    return save(fig, f"E{r.event_id}_overview.png")


def fig_event_zoom(r):
    a, b = r.onset - pd.Timedelta(minutes=110), r.onset + pd.Timedelta(minutes=20)
    pcs = [r.pc1, r.pc2, r.pc3]
    fig, axs = figure(3, 2, 5.05, sharex=True)
    for i, (pc, c) in enumerate(zip(pcs, C_PC)):
        y = ratio[pc].loc[a:b].clip(lower=1e-3)
        axs[i, 0].plot(y.index, y.values, color=c, lw=1.4)
        axs[i, 0].axhline(1, color="red", ls="--", lw=1.0)
        axs[i, 0].set_yscale("log")
        axs[i, 0].set_ylabel(f"{pc}\nT²_j / limit", fontsize=10)
        s = scores[pc].loc[a:b]
        axs[i, 1].plot(s.index, s.values, color=c, lw=1.4)
        axs[i, 1].set_ylabel(pc, fontsize=10)
        for ax in axs[i]:
            shade_window(ax, r)
            ax.grid(alpha=0.3)
    axs[0, 0].set_title("Causal T² term / own limit (log; red dashed = limit)", fontsize=12)
    axs[0, 1].set_title("Raw PC score", fontsize=12)
    for ax in axs[-1]:
        minute_axis(ax, a, b, step=20)
        ax.set_xlabel(f"{r.onset:%d %b %Y}")
    return save(fig, f"E{r.event_id}_zoom.png")


def fig_equation(name, heading, lines, width=5.6):
    h = 0.55 + 0.5 * len(lines)
    fig = plt.figure(figsize=(width, h), dpi=200)
    fig.patch.set_facecolor("#F8F8F8")
    fig.text(0.03, 1 - 0.3 / h, heading, fontsize=11, fontweight="bold", va="center")
    for i, (txt, size, x) in enumerate(lines):
        fig.text(x, 1 - (0.75 + 0.5 * i) / h, txt, fontsize=size, va="center")
    fig.savefig(ASSETS / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ASSETS / name


FIG = {"counts": fig_counts(), "daily": fig_daily_above(), "lead": fig_lead(), "pca": fig_pca_var(),
       "t2": fig_t2_overall(), "rc": fig_rootcause(), "avg": fig_pc_avg(), "raw": fig_top_raw(list(top1_tp.index[:3]))}
for pc in PC_DEEP:
    FIG[pc] = fig_pc_deep(pc)
CASES = {e: ev[ev.event_id == e].iloc[0] for e in CASE_EVENTS}
for e, r in CASES.items():
    assert r.outcome == "TP"
    FIG[f"E{e}_a"], FIG[f"E{e}_b"] = fig_event_overview(r), fig_event_zoom(r)
FIG["eq_pca"] = fig_equation("eq_pca.png", "PCA of one unit (fitted on NOC rows only)", [
    ("$z_i = (x_i - \\mu_i^{NOC}) \\,/\\, \\sigma_i^{NOC}$   for every tag $i$ of the unit", 13, 0.05),
    ("$t_k = \\mathbf{w}_k^{\\top}\\,\\mathbf{z}$,   $k = 1, 2$", 15, 0.25),
    ("$\\mathbf{w}_k$ = $k$-th eigenvector of the unit's NOC correlation matrix", 10.5, 0.05)])
FIG["eq_overall"] = fig_equation("eq_overall.png", f"1. Overall T² ({N_PCS} PCs)", [
    ("For the standardised PC scores $\\mathbf{z}$ at minute $t$:", 10.5, 0.05),
    ("$T^2 = \\mathbf{z}^{\\top} R^{-1}\\, \\mathbf{z}$", 16, 0.30),
    ("$R$ = correlation of the PCs in the baseline window", 10.5, 0.05)])
FIG["eq_term"] = fig_equation("eq_term.png", "2. Causal T² term of each PC", [
    ("$\\beta_j = R_{PP}^{-1}R_{Pj}$,   $\\sigma^2_{j\\cdot PA(j)} = 1 - \\beta_j^{\\top} R_{Pj}$", 13, 0.05),
    ("$T^2_{j\\cdot PA(j)} = \\left(\\frac{z_j - \\beta_j^{\\top} \\mathbf{z}_{PA(j)}}{\\sigma_{j\\cdot PA(j)}}\\right)^2$",
     16, 0.20),
    ("$PA(j)$ = parents of PC $j$ in the causal graph;  root PC: $T_j = z_j$", 10.5, 0.05)])
FIG["eq_limit"] = fig_equation("eq_limit.png", "3. Control limit (report Eq. 13), × 2", [
    ("$T^2_{lim} = 2 \\cdot \\frac{p\\,(m+1)(m-1)}{m\\,(m-k-p)}\\,F_{1-\\alpha;\\,p,\\,m-k-p}$", 15, 0.08),
    (f"overall: $p$ = {N_PCS}, $k$ = 0;   per PC: $p$ = 1, $k$ = number of parents", 10.5, 0.05),
    (f"$m$ = {int(lim.window_rows):,} baseline rows,  $\\alpha$ = 0.05", 10.5, 0.05)], width=6.2)

# ---------------------------------------------------------------- pptx helpers
prs = Presentation(REF)
slides = list(prs.slides)
TITLE_SP = copy.deepcopy([s for s in slides[7].shapes if s.name == "Title 1"][0]._element)  # TA title bar
sldIdLst = prs.slides._sldIdLst
COVER = list(sldIdLst)[0]
for sldId in list(sldIdLst)[1:]:
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
LAYOUT = prs.slide_layouts[1]
TNR_BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", TITLE_PT)
SW = 13.333


def new_slide(title_text):
    assert TNR_BOLD.getlength(title_text) < 930, f"title too long: {title_text}"
    s = prs.slides.add_slide(LAYOUT)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    sp = copy.deepcopy(TITLE_SP)
    s.shapes._spTree.append(sp)
    ts = sp.findall(".//" + qn("a:t"))
    ts[0].text = title_text
    for t in ts[1:]:
        t.getparent().getparent().remove(t.getparent())
    for rpr in sp.findall(".//" + qn("a:rPr")) + sp.findall(".//" + qn("a:endParaRPr")):
        rpr.set("sz", str(TITLE_PT * 100))
    return s


def _runs(p, text, size=None, color=None, font=None):
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        r = p.add_run()
        r.text = part
        r.font.bold = True if i % 2 else None
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        if font:
            r.font.name = font


def _bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "285750")
    pPr.set("indent", "-285750")
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


def text_box(s, x, y, w, h, items, size=16, space=4):
    """items: str = bullet; ('H', str) = bold sub-heading (TA slide 37 style); ('P', str) = plain paragraph."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for k, it in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if isinstance(it, tuple) and it[0] == "H":
            _runs(p, "**" + it[1] + "**", size=it[2] if len(it) > 2 else 22, color="1F3864")
            p.space_after = Pt(4)
            continue
        if isinstance(it, tuple) and it[0] == "P":
            _runs(p, it[1], size=size)
        else:
            _bullet(p)
            _runs(p, it, size=size)
        p.space_after = Pt(space)
    return tb


def caption(s, x, y, w, text, size=10):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    tb.text_frame.word_wrap = True
    _runs(tb.text_frame.paragraphs[0], text, size=size, color="404040")
    return tb


def picture(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def full_picture(s, path, y=1.0, w=12.9):
    pic = picture(s, path, (SW - w) / 2, y, w=w)
    return y + pic.height / 914400


def table(s, x, y, w, rows, col_w, size=13, row_h=0.36, bold_first_col=False):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tb = shp.table
    for j, cw in enumerate(col_w):
        tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tb.cell(i, j)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            txt = str(val)
            if bold_first_col and j == 0 and i > 0:
                txt = "**" + txt + "**"
            _runs(tf.paragraphs[0], txt, size=size)
            if j > 0:
                tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    return tb


def box(s, x, y, w, h, text, fill, size=12, bold=False, color="000000", line="7F7F7F", dash=False,
        align=PP_ALIGN.CENTER, shape=MSO_SHAPE.RECTANGLE, anchor=MSO_ANCHOR.MIDDLE):
    shp = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        shp.line.color.rgb = RGBColor.from_string(line)
        shp.line.width = Pt(0.75)
        if dash:
            shp.line.dash_style = 4
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _runs(p, ("**" + ln + "**") if (bold and i == 0) else ln, size=size, color=color)
    return shp


def arrow(s, x1, y1, x2, y2, color="404040", width=1.5, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")).set("type", "triangle")
    return c


def node_arrow(s, p, q, r=0.7, **kw):
    (x1, y1), (x2, y2) = p, q
    d = np.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    return arrow(s, x1 + ux * r, y1 + uy * r, x2 - ux * (r + 0.03), y2 - uy * (r + 0.03), **kw)


def node(s, xy, label, fill="DEEBF7", r=0.68, size=13):
    return box(s, xy[0] - r, xy[1] - r, 2 * r, 2 * r, label, fill, size=size, bold=True, shape=MSO_SHAPE.OVAL,
               line="1F3864")


def console(s, x, y, w, h, text, size=12):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string("1E1E1E")
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_top = Inches(0.15)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.name = "Consolas"
        r.font.color.rgb = RGBColor.from_string("E6E6E6")


def takeaway(s, y, text, h=0.55):
    """Blue-bordered conclusion line under a figure."""
    return box(s, 0.4, y, SW - 0.8, h, text, "EAF1FB", size=15, line="5B9BD5", align=PP_ALIGN.LEFT)


def fmt(n):
    return f"{int(n):,}"


def figure_slide(title_text, fig_path, source, bullets, size=15, w=12.9):
    """Full-width figure, its source line, then the explanation bullets."""
    s = new_slide(title_text)
    y = full_picture(s, fig_path, 1.0, w)
    caption(s, (SW - w) / 2, y + 0.02, w, source)
    text_box(s, 0.5, y + 0.4, 12.3, 7.4 - y - 0.4, bullets, size=size, space=3)
    return s


# ---------------------------------------------------------------- 1 cover (TA's slide 1)
cover = prs.slides.get(int(COVER.get("id")))
for shp in cover.shapes:
    if shp.name == "TextBox 3":
        p0 = shp.text_frame.paragraphs[0]
        p1 = copy.deepcopy(p0._p)
        p0._p.addnext(p1)
        for t in p1.findall(".//" + qn("a:t")):
            t.text = ""
        p1.findall(".//" + qn("a:t"))[0].text = "Results on July – October 2025 plant data"
        for r in p1.findall(".//" + qn("a:rPr")):
            r.set("sz", "1600")

# ---------------------------------------------------------------- 2 problem and data
s = new_slide("Goal: name the unit behind each flare and warn before it starts")
text_box(s, 0.5, 1.15, 6.4, 5.6, [
    ("H", "Problem"),
    "Flares are frequent: **620 events** of 30 min or more in four months, covering half of the time.",
    "We want to see **which unit moves first**, and raise a warning **before** the flare starts.",
    ("H", "What the method gives"),
    "A **T² statistic** per minute: one number for the whole plant.",
    "Its **causal split** per PC: which unit's PC is out of line once its graph parents are accounted for.",
    "Each PC traces back to its heaviest **tags**."], size=17, space=6)
rows = [["Data item", "Value"],
        ["Period", "1 Jul – 31 Oct 2025"],
        ["Sampling", "1 minute (177,120 rows)"],
        ["Units", f"{N_UNITS} unit folders"],
        ["Process tags used", f"{N_TAGS} (75 of 376 dropped)"],
        ["Flare signal", "flare_net (t/h), from 4 flare tags"],
        ["Flare event", "flare_net > 1.5 t/h for ≥ 30 min"],
        ["Flare events", "620 (88,590 min, 50 % of time)"],
        ["Normal rows (NOC)", f"{fmt(lim.n_noc_rows)} (first 60 %, no flares)"]]
table(s, 7.3, 1.25, 5.6, rows, [2.4, 3.2], size=14, row_h=0.5, bold_first_col=True)
caption(s, 7.3, 5.85, 5.6, "Source: outputs/SUMMARY.md and logs/run.log. Dropped tags: 49 with negative values, "
        "19 with > 10 % missing, 7 constant or near-constant.")

# ---------------------------------------------------------------- 3 approach in blocks
s = new_slide("Our approach in six steps, from raw tags to a ranked list of units")
STEPS = [
    ("1  DATA", "DEEBF7", "Join tags on a 1-min grid, drop bad tags, fill small gaps, standardise with NOC",
     f"{N_TAGS} tags\n177,120 rows"),
    ("2  UNIT-WISE PCA", "E2F0D9", "One PCA per unit on NOC rows; keep PC1 and PC2 of each unit",
     f"{N_PCS} PCs\n{VAR_ALL:.1f} % variance"),
    ("3  PC ALGORITHM", "EDE2F6", "Learn causal links among the PCs only (flare_net is not an input)",
     f"{N_EDGES} edges\n{N_CROSS} between units"),
    ("4  DAG + FLARE", "FBE5D6", "Orient to a graph without loops; add PC → flare_net for every PC",
     f"0 loops\n{N_PCS} edges to flare_net"),
    ("5  CAUSAL T²", "FFF2CC", "Moving 1,250-min baseline; overall T² and one T² term per PC",
     f"limit {lim.t2_limit:.1f}\nper PC ≈ 7.7"),
    ("6  DETECT + RANK", "FCE4EC", "Flag a flare if T² is above the limit in the 60 min before it; rank PCs",
     f"F1 = {F1:.3f}\ntop 3 PCs per event"),
]
bw, gap, x0 = 1.92, 0.24, 0.32
for i, (head, fill, what, out) in enumerate(STEPS):
    x = x0 + i * (bw + gap)
    box(s, x, 1.4, bw, 0.8, head, fill, size=13, bold=True, line="1F3864",
        shape=MSO_SHAPE.PENTAGON if i < 5 else MSO_SHAPE.RECTANGLE)
    box(s, x, 2.4, bw, 1.75, what, "FFFFFF", size=14, line="5B9BD5")
    box(s, x, 4.35, bw, 1.0, out, fill, size=14, bold=True, dash=True)
    if i < 5:
        arrow(s, x + bw + 0.01, 3.27, x + bw + gap - 0.01, 3.27)
caption(s, 0.32, 1.02, 12.7, "Top row: step; middle: what it does; bottom (dashed): result in this lab run.", size=11)
CHG = [("Warning window", "60 min before onset (was 30)"), ("Scoring", "TP/FN per flare event, FP per gap, F1"),
       ("T²", "inv(R), no pseudo-inverse"), ("Normal rows", "no 120-min buffer around flares"),
       ("Units", "13: unit_wise_control_data not in this run")]
box(s, 0.32, 5.65, 12.7, 0.38, "Changed since the last review", "1F3864", size=13, bold=True, color="FFFFFF",
    line=None, align=PP_ALIGN.LEFT)
cw = (12.7 - 4 * 0.12) / 5
for i, (k, v) in enumerate(CHG):
    box(s, 0.32 + i * (cw + 0.12), 6.1, cw, 0.95, f"**{k}**\n{v}", "F2F2F2", size=13, line="BFBFBF")

# ---------------------------------------------------------------- 4 metric definition (timeline)
s = new_slide("How we score: one count per flare event and one per gap between flares")
AX = 2.95
box(s, 0.5, AX - 0.02, 12.3, 0.04, "", "404040", line=None)
segs = [(0.5, 3.0, "gap"), (3.0, 4.2, "pre"), (4.2, 5.3, "flare"), (5.3, 7.2, "gap"), (7.2, 8.4, "pre"),
        (8.4, 9.4, "flare"), (9.4, 10.6, "gap"), (10.6, 11.8, "pre"), (11.8, 12.8, "flare")]
col = {"gap": "F2F2F2", "pre": "F4CCCC", "flare": "F8CBAD"}
for a, b, k in segs:
    box(s, a, AX - 0.45, b - a, 0.45, {"gap": "", "pre": "60 min", "flare": "flare"}[k], col[k], size=11,
        line="BFBFBF")
for a, b in [(0.9, 1.15), (1.9, 2.3), (3.5, 3.9), (9.8, 11.2)]:
    box(s, a, AX - 0.95, b - a, 0.28, "", "C0392B", line=None)
box(s, 0.5, 1.2, 0.35, 0.22, "", "C0392B", line=None)
box(s, 0.95, 1.12, 5.5, 0.38, "minutes with T² above its limit", "FFFFFF", size=13, line=None, align=PP_ALIGN.LEFT)
LAB = [(0.5, 3.0, "Gap: T² rose twice\n→ **1 FP**"), (3.0, 5.3, "T² rose in the 60 min\n→ **TP**"),
       (5.3, 7.2, "Gap: no rise\n→ not counted"), (7.2, 9.4, "No rise in the 60 min\n→ **FN**"),
       (9.4, 12.8, "One long rise: in the gap → **1 FP**;\nin the 60 min → **TP**")]
for a, b, t in LAB:
    box(s, a, AX + 0.12, b - a, 0.75, t, "FFFFFF", size=12, line=None)
text_box(s, 0.5, 4.2, 7.2, 3.0, [
    "**Flare block** = [onset − 60 min, flare end]. **Gap** = time between two flare blocks.",
    "**TP**: T² above its limit at least once in the 60 min before onset. **FN**: never above.",
    "**FP**: a gap with T² above its limit at least once, however long or how often.",
    "**TN is not counted.** No alarm delays: one minute above the limit is enough.",
    "Minutes with no T² value are **not scored**."], size=16, space=6)
box(s, 8.1, 4.35, 4.7, 1.25, "F1 = 2 TP / (2 TP + FP + FN)", "EAF1FB", size=22, bold=True, line="5B9BD5")
caption(s, 8.1, 5.75, 4.7, "Rules: unit_pca_pc_t2/evaluation.py, evaluate(); window FLARE_PRE_WINDOW_MIN = 60 "
        "(config.py). Only flare events ≥ 30 min count as flares.", size=11)

# ---------------------------------------------------------------- 5 overall result
s = new_slide(f"F1 = {F1:.2f}: T² rose before {TP} of {TP + FN} flares, but also in {FP} of {N_SC_BL} gaps")
console(s, 0.3, 1.1, 5.7, 5.0, "\n".join([
    "=====================================",
    " FLARE DETECTION, JUL – OCT 2025",
    "=====================================",
    f" Flare events (>= 30 min)  : {len(fe)}",
    f"   not scored (no T²)      : {N_NS}",
    f"   scored                  : {TP + FN}",
    f" TP  events caught         : {TP}",
    f" FN  events missed         : {FN}",
    "-------------------------------------",
    f" Gaps between flare blocks : {N_BL}",
    f"   not scored              : {N_BL_NS}",
    f" FP  gaps with T² rise     : {FP}",
    "-------------------------------------",
    f" Precision = TP/(TP+FP)    : {PREC:.3f}",
    f" Recall    = TP/(TP+FN)    : {REC:.3f}",
    f" F1 = 2TP/(2TP+FP+FN)      : {F1:.3f}",
    f" Median warning (TP)       : {LEAD_MED:.0f} min",
    "====================================="]), size=13)
picture(s, FIG["counts"], 6.3, 1.1, w=6.75)
caption(s, 0.3, 6.15, 5.7, "Source: outputs/tables/metrics.csv, event_outcomes.csv, nonflare_blocks.csv. " + SRC)
takeaway(s, 6.6, f"**Reading:** recall is high ({REC:.3f}): only {FN} of {TP + FN} flares were missed. Precision is "
                  f"lower ({PREC:.3f}): {FP} of the {N_SC_BL} scored gaps also had a T² rise, and these FPs pull F1 "
                  f"down to {F1:.3f}.", h=0.7)

# ---------------------------------------------------------------- 6 caveat: T² high too often
figure_slide(f"T² is high so often that an always-on alarm would score F1 = {F1_ALWAYS:.2f}", FIG["daily"],
             "Source: outputs/t2/t2_all_rows.parquet (overall T², above_limit, scored; flare-free = outside flare "
             "events and short flares). " + SRC, [
                 f"Overall T² is above its limit in **{PCT_NF_ABOVE:.0f} %** of flare-free minutes "
                 f"({PCT_NOC_ABOVE:.0f} % of NOC minutes). By chance it should be **about 5 %** or less.",
                 f"An alarm that is **always on** would catch every scored flare and flag every scored gap: "
                 f"F1 = 2·{TP + FN} / (2·{TP + FN} + {N_SC_BL}) = **{F1_ALWAYS:.3f}**, higher than our **{F1:.3f}**.",
                 "So the current F1 **does not yet show skill**: the T² limit is crossed too easily."], size=16)

# ---------------------------------------------------------------- 7 warning time
s = new_slide(f"In {N_ALREADY} of {TP} TP events T² was already high an hour before the flare")
picture(s, FIG["lead"], 0.3, 1.05, w=7.6)
text_box(s, 8.15, 1.15, 4.9, 4.9, [
    f"**Green ({len(rose)} events):** T² was below the limit when the window opened and crossed inside it: "
    f"a genuine rise before the flare. Median **{rose.lead_min.median():.0f} min** before onset.",
    f"**Grey ({N_ALREADY} events):** T² was already above the limit when the window opened. These count as TP "
    "under the rule, but the rise did not start in the hour before the flare.",
    "Grey bars sit at 60 min, or less when the window starts right after an earlier flare."], size=16, space=8)
caption(s, 0.3, 6.15, 7.6, "Source: outputs/tables/event_outcomes.csv (lead_min, above_at_window_start). " + SRC)
takeaway(s, 6.6, f"**Reading:** only {len(rose)} of {TP} TP events show T² rising inside the 60-min window; "
                 "this is the stronger evidence of an early warning.", h=0.7)

# ---------------------------------------------------------------- 8 PCA
s = new_slide("How PCs work: each unit's tags are compressed into two scores")
picture(s, FIG["eq_pca"], 0.4, 1.1, w=6.2)
ex = top_tags("SEU2_PC1")
text_box(s, 0.4, 3.3, 6.3, 4.0, [
    "Tags of one unit move together; PCA finds the **directions** in which they move most.",
    "**PC1** is the strongest common movement of the unit, **PC2** the next, uncorrelated with PC1.",
    "The **weights w** (loadings) tell which tags drive a PC, so a PC can be traced back to tags.",
    f"Example: SEU2_PC1 is driven mostly by {ex[0][0]} ({ex[0][1]:.0f} %) and {ex[1][0]} ({ex[1][1]:.0f} %).",
    "Fitted on **NOC rows only**, then applied to every minute.",
    "Why per unit: a PC then belongs to **one unit**, so a high T² term points to that unit."], size=15, space=4)
picture(s, FIG["pca"], 7.0, 1.05, w=6.0)
caption(s, 7.0, 6.5, 6.0, "Source: outputs/tables/pcs_per_unit.csv and pca_loadings.csv. DAK has rank 1 and "
        "ISOM_4 has one tag, so each gives 1 PC.")

# ---------------------------------------------------------------- 9 causal graph
s = new_slide(f"The PC algorithm links the {N_PCS} PCs with {N_EDGES} edges, {N_CROSS} between units")
picture(s, OUT / "figures" / "fig5_causal_graph.png", 0.25, 0.95, h=6.4)
text_box(s, 6.75, 1.1, 6.3, 4.8, [
    f"Input: the {N_PCS} PC scores, **every 10th minute** (17,712 rows). **flare_net is not an input.**",
    "Test: **Fisher-Z**, α = 0.01, stable PC, conditioning sets up to 3 PCs.",
    f"Output: **{N_EDGES} edges** ({N_DIR} directed, {N_UND} undirected); **{N_CROSS}** link different units.",
    f"Parents per PC: mean **{n_par.mean():.2f}**, max {n_par.max()} (DAK_PC1); {int((n_par == 0).sum())} PCs have none.",
    f"Then every PC gets an edge to flare_net (**{N_PCS} edges**, added by us, not learned).",
    "This run's PC output has **no loops** (next two slides)."], size=16, space=6)
caption(s, 6.75, 5.95, 6.3, "Figure: outputs/figures/fig5_causal_graph.png; edges: outputs/graph/pc_edges.csv and "
        "dag_edges.csv. Node colour = unit; (n) and node size = TP events where the PC is top-1.")

# ---------------------------------------------------------------- 10-11 loops
P_ = {"SEU1_PC1": (1.3, 2.0), "PDA_PC1": (5.2, 2.0), "LOUP_PC1": (5.2, 5.4), "CTRL_PC1": (1.3, 5.4)}
NM = {"unit_wise_control_data_PC1": "CTRL_PC1"}
loop_nodes = {"SEU1_PC1", "PDA_PC1", "LOUP_PC1", "unit_wise_control_data_PC1"}
old_edges = [(NM.get(a, a), NM.get(b, b)) for a, b in sorted(old_dir) if a in loop_nodes and b in loop_nodes]
assert len(old_edges) == 6


def loop_diagram(s, reversed_=False):
    for k, xy in P_.items():
        node(s, xy, k.replace("_PC1", "\nPC1"), fill="FFF2CC" if k == "CTRL_PC1" else "DEEBF7")
    for a, b in old_edges:
        if (a, b) == ("PDA_PC1", "LOUP_PC1"):
            if reversed_:
                node_arrow(s, P_[b], P_[a], color="C0392B", width=2.25, dash=True)
            else:
                node_arrow(s, P_[a], P_[b], color="C0392B", width=2.25)
        else:
            node_arrow(s, P_[a], P_[b], width=2)


s = new_slide("Loops: the previous run had three, all through PDA → LOUP")
loop_diagram(s)
caption(s, 0.4, 6.3, 6.2, "Directed edges among these four PCs in the previous run's PC output (commit eaebb1e). "
        "Red = PDA_PC1 → LOUP_PC1, on all three loops. CTRL_PC1 = unit_wise_control_data_PC1.", size=11)
rows = [["#", "Loop found by simple_cycles", "Length"]]
for i, r in enumerate(OLD_CYC.itertuples(index=False), 1):
    rows.append([i, r.cycle.replace("unit_wise_control_data_PC1", "CTRL_PC1"), r.length])
table(s, 6.75, 1.15, 6.3, rows, [0.4, 5.1, 0.8], size=12, row_h=0.55)
text_box(s, 6.75, 3.5, 6.3, 3.8, [
    "**Why loops matter:** each PC's T² term removes what its parents explain. In a loop a PC is its own "
    "ancestor, so no order puts every parent first and the T² split loses its meaning.",
    "**How we find them:** list every directed cycle of the PC output with networkx.simple_cycles "
    "(check_cycles.py, same check as the TA's Main.py).",
    f"**Previous run:** 14 units, 26 PCs, {len(old_pce)} edges ({len(old_dir)} directed); 3 loops, two of them "
    "through the control-data PC."], size=15, space=6)

s = new_slide("Loop removal reversed PDA → LOUP; the current run has no loops")
loop_diagram(s, reversed_=True)
caption(s, 0.4, 6.3, 6.2, "Same four PCs in the previous run's final graph (dag_edges.csv at eaebb1e). Red dashed = "
        "the reversed edge, now LOUP_PC1 → PDA_PC1. No loop is left.", size=11)
text_box(s, 6.75, 1.1, 6.3, 3.3, [
    "**Procedure (cpdag_to_dag):** repeatedly take a PC with no outgoing edge and point its edges into it. When none "
    "is left, every remaining PC leads into a loop: remove the one with the fewest outgoing edges and **reverse** them.",
    "Previous run: **PDA_PC1 → LOUP_PC1** reversed, breaking all three loops. **ISOM_3_PC2 → CTRL_PC1** was "
    "reversed too although it was **not on a loop** (tie-break by name order): a weakness of the rule."],
    size=14, space=6)
rows = [["", "Previous run", "Current run"],
        ["Unit folders", f"{len(old_pcu)}", f"{N_UNITS}"],
        ["PCs / NOC rows", f"{int(old_pcu.n_pcs.sum())} / 12,157", f"{N_PCS} / {fmt(lim.n_noc_rows)}"],
        ["Loops in PC output", len(OLD_CYC), 0],
        ["Edges reversed", int(old_stats.n_reversed_edges), 0]]
table(s, 6.75, 4.5, 6.3, rows, [2.5, 1.9, 1.9], size=13, row_h=0.4, bold_first_col=True)
caption(s, 6.75, 6.55, 6.3, "Two inputs changed between the runs (control-data unit left out, NOC without the "
        "buffer), so we cannot say which one removed the loops. Sources: graph/{pc_edges, dag_edges, dag_stats, "
        "cycles}.csv in both commits.", size=11)

# ---------------------------------------------------------------- 12 T² equations and baseline
s = new_slide("Each PC's T² term is measured after removing what its parents explain")
picture(s, FIG["eq_overall"], 0.3, 1.1, w=6.0)
picture(s, FIG["eq_term"], 0.3, 3.25, w=6.0)
picture(s, FIG["eq_limit"], 6.75, 1.1, w=6.3)
text_box(s, 6.75, 3.25, 6.3, 3.9, [
    f"Limits: overall **{lim.t2_limit:.2f}**, per PC **{tl.t2_limit.min():.2f} – {tl.t2_limit.max():.2f}**, "
    "both × 2 (TA's notebook scales by 1.2 – 1.5).",
    f"**Moving baseline:** flare-free minutes in the last **{int(lim.window_rows):,} min**, re-estimated every 10 min.",
    f"Fewer than 100 such minutes: no T². That is **{fmt(N_UNSCORED)} minutes**, "
    f"{100 * (~t2.scored & t2.flare_event).sum() / N_UNSCORED:.0f} % inside flares, incl. 1 – 4 Jul (first T² "
    f"{FIRST_SCORED:%d %b %H:%M}).",
    "Root cause = PCs ranked by **T²_j / own limit** at the first minute above the overall limit."],
    size=14, space=5)
caption(s, 0.3, 5.75, 6.0, "Equations: unit_pca_pc_t2/MATHS.md (report Eqs. 7, 9–13); R⁻¹ is computed with "
        "numpy.linalg.inv. Values: outputs/t2/limits.csv and term_limits.csv.")

# ---------------------------------------------------------------- 13 overall T²
figure_slide(f"Over four months T² is above its limit in {PCT_NF_ABOVE:.0f} % of flare-free minutes", FIG["t2"],
             "Source: outputs/tables/flare_net_raw.parquet and t2/t2_all_rows.parquet. " + SRC, [
                 f"flare_net is cut at {FN_CUT} t/h so the bulk (0 – 5 t/h) is visible; dark line = hourly median. T² is on a "
                 "log scale.",
                 f"T² stays near or above its limit **{lim.t2_limit:.1f}** for most of the period, with or "
                 "without a flare. The top 1 % of minutes exceed "
                 f"**{t2['T2'].quantile(0.99):,.0f}**; the maximum is about **{t2['T2'].max() / 1e6:.0f} million**.",
                 "Grey bands: no T² value (too few flare-free minutes in the baseline window)."])

# ---------------------------------------------------------------- 14 ranking TP vs FP
u4 = list(unit_tp.index[:4])
figure_slide(f"Which PCs rank first: DAK_PC1 leads {top1_tp.get('DAK_PC1', 0)} TP events and "
             f"{top1_fp.get('DAK_PC1', 0)} FP gaps", FIG["rc"],
             "Source: outputs/tables/event_outcomes.csv (pc1 of TP events) and nonflare_blocks.csv (pc1 of FP gaps). "
             + SRC, [
                 "By unit (top-1 counts, TP / FP): " + ", ".join(f"**{u}** {unit_tp[u]} / {unit_fp.get(u, 0)}"
                                                                 for u in u4) + ".",
                 "The PCs that lead the TP events also lead the FP gaps: ranking first is **not by itself** a cause.",
                 f"A median of **{tp.n_pcs_above.median():.0f} PCs** are above their own limit at that minute, both "
                 f"in TP events and in FP gaps; in {int((tp.ratio1 < 1).sum())} TP events even the top PC is below "
                 "its limit."])

# ---------------------------------------------------------------- 15 average over the whole window
l3 = list(LIFT.index[:3])
assert (PCAVG.pre > PCAVG.normal).all()
figure_slide("Every PC is above its limit more often before flares than in normal time", FIG["avg"],
             "Source: outputs/t2/terms_all_rows.parquet and t2_all_rows.parquet (scored minutes: "
             f"{fmt(in_pre.sum())} before flares, {fmt(normal.sum())} normal, {fmt(in_flare.sum())} during flares). "
             + SRC, [
                 "Before flares, " + ", ".join(f"**{p}** ({LIFT.lift[p]:.1f}×)" for p in l3) + " are above their "
                 "limit about twice as often as in normal time: the clearest pre-flare signal by PC.",
                 f"**DAK_PC1** is above its limit {PCAVG.pre['DAK_PC1']:.0f} % of the time before flares and "
                 f"{PCAVG.normal['DAK_PC1']:.0f} % in normal time: **no difference**, despite leading the ranking.",
                 f"In normal time {int((PCAVG.normal > 5).sum())} of {N_PCS} PCs are still above their limit more "
                 "than 5 % of the time; during flares almost all are far above."])

# ---------------------------------------------------------------- 16 raw scores of the top three
t3 = list(top1_tp.index[:3])
figure_slide("The three PCs ranked first most often, as raw scores over four months", FIG["raw"],
             "Source: outputs/graph/pc_scores.parquet. " + SRC, [
                 f"**{t3[0]}** is a straight line: it falls at a constant rate from 1 Jul to about 10 Oct, then stays "
                 "flat.",
                 f"**{t3[1]}** jumps between two levels every few weeks; **{t3[2]}** is mostly flat with sharp dips in "
                 "August, September and October.",
                 "A moving baseline treats each jump or dip as a large deviation, so these PCs get large T² terms."])

# ---------------------------------------------------------------- 17-18 deep-dive into the main TP contributors
dak_tags = ", ".join(t for t, _ in top_tags("DAK_PC1"))
figure_slide("Deep-dive on top TP contributors (1/2): DAK_PC1 is a straight line", FIG["DAK_PC1"],
             "Source: outputs/graph/pc_scores.parquet and t2/terms_all_rows.parquet (hourly max). Std-floor share "
             "recomputed from the same outputs. " + SRC, [
                 f"DAK's 3 tags ({dak_tags}) have **rank 1**: one signal, a straight line falling at a constant rate "
                 "until about 10 Oct, then flat. It does not look like measured process data (UNVERIFIED with the "
                 "plant).",
                 "In a 1,250-min window the line is almost flat: its window std is below 1 % of its overall std in "
                 "**99 % of window steps**, so the code floors it, and its 11 graph parents explain it almost fully.",
                 "Small changes then give large T² terms: **its ranking says nothing about DAK driving flares.**"])
aps = top_tags("APS_NVPS_LEU_PC1")
figure_slide("Deep-dive on top TP contributors (2/2): APS_NVPS_LEU_PC1", FIG["APS_NVPS_LEU_PC1"],
             "Source: outputs/graph/pc_scores.parquet and t2/terms_all_rows.parquet (hourly max). " + SRC, [
                 "Main tags: " + ", ".join(f"**{t}** ({v:.0f} %)" for t, v in aps) + ". The score jumps between two "
                 "levels every few weeks.",
                 f"Ranked first in **{top1_tp.get('APS_NVPS_LEU_PC1', 0)} TP events** and "
                 f"{top1_fp.get('APS_NVPS_LEU_PC1', 0)} FP gaps; **{n_par['APS_NVPS_LEU_PC1']} parents** in the graph.",
                 f"Above its limit {PCAVG.pre['APS_NVPS_LEU_PC1']:.0f} % of the time before flares vs "
                 f"{PCAVG.normal['APS_NVPS_LEU_PC1']:.0f} % in normal time: a real but weak pre-flare signal."])

# ---------------------------------------------------------------- 19-22 flare event examples
EV_TEXT = {  # (title overview, note overview, title zoom, note zoom); notes read from the event figures
    144: ("E144: SEU2 spiked 21 minutes before a flare on 1 August",
          "The warning is a short spike at 09:08; most of the 18 minutes above the limit come after it.",
          "E144 zoom: SEU1_PC2 ranks first because its parent SEU2_PC1 spiked",
          "SEU1_PC2 itself changes smoothly. Its T² term jumps at 09:08 because its parent SEU2_PC1 spikes and the "
          "regression expects SEU1_PC2 to follow. The real move is in SEU2_PC1."),
    449: ("E449: T² was already high an hour before a 5.6 t/h flare on 16 Sep",
          "T² was above the limit when the window opened and again from about 13:30: counted as TP, but the change "
          "began before the window (next slide).",
          "E449 zoom: NSU_NHT_CCR and SEU3 shifted 100 minutes before onset",
          "NSU_NHT_CCR_PC2 steps up and SEU3_PC2 steps down around 12:05 – 12:10, about 100 min before onset and "
          "outside the 60-min window. Both stay at the new level through the flare."),
}
for e, r in CASES.items():
    tA, nA, tB, nB = EV_TEXT[e]
    pa = sorted(dag[dag.target == r.pc1].source)
    figure_slide(tA, FIG[f"E{e}_a"],
                 f"flare_net, overall T² and the T² terms of the top 3 PCs, 3 h before to 2.5 h after onset. Source: "
                 f"outputs/tables/flare_net_raw.parquet, t2/t2_all_rows.parquet, terms_all_rows.parquet, "
                 f"event_outcomes.csv (E{e}). " + SRC, [
                     f"Onset **{r.onset:%d %b %Y %H:%M}**, {int(r.duration_min)} min, peak **{r.peak_flare_net:.2f} "
                     f"t/h**. T² above its limit for **{int(r.window_minutes_above)} of {int(r.window_rows_scored)} "
                     f"min** of the window, first at **{r.first_above:%H:%M}** ({int(r.lead_min)} min before onset).",
                     f"Top 3 at {r.first_above:%H:%M}: **{r.pc1} {r.ratio1:.1f}×**, {r.pc2} {r.ratio2:.1f}×, "
                     f"{r.pc3} {r.ratio3:.1f}× their own limits. " + nA], size=14)
    t1 = top_tags(r.pc1, 2)
    figure_slide(tB, FIG[f"E{e}_b"],
                 "Zoom: 110 min before to 20 min after onset; each row = one of the top 3 PCs, its T² term (left) and "
                 "raw score (right). Source: outputs/t2/terms_all_rows.parquet, graph/pc_scores.parquet. " + SRC, [
                     f"**{r.pc1}**: tags {', '.join(f'{t} ({v:.0f} %)' for t, v in t1)}; {len(pa)} graph parent"
                     f"{'s' if len(pa) != 1 else ''}{' incl. ' + r.pc2 if r.pc2 in pa else ''}.",
                     nB], size=14)

# ---------------------------------------------------------------- 23 done and next
s = new_slide("What we have done, and what we will investigate next")
DONE = [f"Pipeline on 4 months of 1-min data: {N_TAGS} tags → {N_PCS} PCs → {N_EDGES}-edge graph → causal T².",
        f"New scoring: 60-min window, one count per flare event or gap; **F1 = {F1:.3f}**.",
        "Loops found (3 in the previous run) and removed; the current graph has none.",
        "Per-event Excel: outputs/RCA_report.xlsx, sheet events (620 rows, 32 columns), and a figure per event.",
        f"Found: T² above its limit in {PCT_NF_ABOVE:.0f} % of flare-free minutes, so an always-on alarm scores "
        f"higher ({F1_ALWAYS:.3f}); DAK_PC1 is a straight-line signal."]
NEXT = [("Moving window", "explore applying a moving-window approach beyond the T² baseline."),
        ("Direct PC / PCMCI+", "test how much data PC or PCMCI+ can take directly, without transformations."),
        ("Tag-level PC", "run the PC algorithm on tags instead of PCs."),
        ("Edge reduction", "heuristics or observations to drop PC → flare_net edges that are not needed."),
        ("Nominal-only graph", "learn the PC graph from normal operation only; flares are outliers.")]
box(s, 0.35, 1.1, 6.15, 0.45, "Done", "2E75B6", size=16, bold=True, color="FFFFFF", line=None)
box(s, 6.85, 1.1, 6.15, 0.45, "Next: proposed investigations", "C55A11", size=16, bold=True, color="FFFFFF",
    line=None)
for i, t in enumerate(DONE):
    box(s, 0.35, 1.7 + i * 1.1, 6.15, 0.98, t, "DEEBF7", size=14, line="9DC3E6", align=PP_ALIGN.LEFT)
for i, (k, t) in enumerate(NEXT):
    box(s, 6.85, 1.7 + i * 1.1, 1.85, 0.98, f"{i + 1}. {k}", "FBE5D6", size=14, bold=True, line="F4B183")
    box(s, 8.75, 1.7 + i * 1.1, 4.25, 0.98, t, "FFFFFF", size=14, line="F4B183", align=PP_ALIGN.LEFT)

prs.save(DECK)
print("saved", DECK, "slides:", len(prs.slides))
