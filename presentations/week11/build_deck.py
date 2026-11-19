"""Week 11 deck: drum-level PCA -> plant-wide PC graph -> drum T², and the vectorized PC that made the graph feasible.

Built like presentations/week10/build_deck.py: on the TA's reference deck (HPCL Project_08072026_external.pptx), so
the master, theme, title bar and fonts are the TA's own. Reads only committed code, the drum_pca_pc outputs and the
logs in presentations/week11/evidence/ (no plant data). Every number on a slide is computed here from those files or
checked against them with an assert. Run from the repo root (needs python-pptx, pandas, pyarrow, matplotlib):

    python presentations/week11/build_deck.py
"""

import copy
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib.patches import Patch
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "drum_pca_pc" / "outputs"
GROUP = ROOT / "unit_drum_grouping"
HERE = Path(__file__).resolve().parent
EVID = HERE / "evidence"
ASSETS = HERE / "assets"
REF = ROOT / "references" / "Previous Project Reports" / "HPCL Project_08072026_external.pptx"
DECK = HERE / "HPCL Project_19112026_drum_PC.pptx"
TITLE_PT = 28
SRC = "drum_pca_pc run, 1-min data, 22 May 2025 – 21 May 2026."
EXAMPLE_DRUM = ("SEU1", "D503")


# ---------------------------------------------------------------- notebook logs (numbers printed by the run)
def nb_text(path):
    nb = json.loads(Path(path).read_text())
    parts = []
    for c in nb["cells"]:
        parts.append("".join(c["source"]))
        for o in c.get("outputs", []):
            parts.append("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


NB = nb_text(ROOT / "drum_pca_pc" / "drum_pca_pc.ipynb")
GRP = nb_text(GROUP / "unit_drum_grouping.ipynb")


def grab(pattern, text=NB, cast=lambda s: int(s.replace(",", ""))):
    m = re.search(pattern, text)
    assert m, f"not found in the notebook output: {pattern}"
    return cast(m.group(1))


N_ROWS = grab(r"BEFORE cleaning: ([\d,]+) rows x 376 tags")
N_TAGS_RAW = grab(r"process\s+(\d+) tags -> \d+ kept")
N_TAGS_CLEAN1 = grab(r"process\s+\d+ tags -> (\d+) kept")
N_TAGS = grab(r"AFTER cleaning: [\d,]+ rows x (\d+) tags")
N_EVENTS = grab(r"flare events: (\d+) in the flare record")
N_FLARE_ROWS = grab(r"removed, flare\s+([\d,]+)")
N_PRE_ROWS = grab(r"removed, pre-window\s+([\d,]+)")
N_NORMAL = grab(r"rows remaining\s+([\d,]+)")
N_DRUMS_MOD = grab(r"drums modelled: (\d+) of 301")
N_PCS = grab(r"PCs in total: (\d+)")
N_ONE_TAG = grab(r"modelled drums with only 1 tag: (\d+)")
N_SHEETS_UNITS = grab(r"Units: (\d+) \| drums: \d+", GRP)
N_DRUMS = grab(r"Units: \d+ \| drums: (\d+)", GRP)
N_UNIQUE_TAGS = grab(r"unique tags: (\d+)", GRP)
N_DRUMS_NO_DATA = grab(r"Drums left out \(no tag present in parquet\): (\d+)", GRP)
assert "Units left out (no parquet data): ['CPP', 'LPG', 'PDU']" in GRP
assert (N_ROWS, N_TAGS, N_EVENTS, N_NORMAL, N_DRUMS, N_PCS) == (523084, 282, 1294, 310439, 301, 329), \
    "notebook outputs changed: re-check every number in this script"
assert "largest |r| 0.008" in NB

# ---------------------------------------------------------------- output tables
pca = pd.read_csv(OUT / "drum_pca_summary.csv")
gs = pd.read_csv(OUT / "drum_pc_graph_summary.csv")
edges = pd.read_csv(OUT / "pc_edges.csv")
drops = pd.read_csv(OUT / "pc_dropped_nodes.csv")
t2i = pd.read_csv(OUT / "t2_index.csv")
sweep = pd.read_csv(OUT / "autocorr_sweep" / "metrics.csv")
drum_tags = json.loads((GROUP / "drum_tags.json").read_text())
UNITS = list(json.loads((GROUP / "unit_drums.json").read_text()))

N_PC_ROWS = N_NORMAL // 10 + (N_NORMAL % 10 > 0)
assert N_PC_ROWS == 31044
N_EDGES = len(edges)
N_CROSS = int((edges.scope == "cross unit").sum())
NODES = sorted(set(edges.source) | set(edges.target))
N_NODES = len(NODES) + 0
N_DROP = len(drops)
N_DUP = int(drops.reason.str.startswith("collinear").sum())
N_NEARDEP = N_DROP - N_DUP
assert N_NODES + N_DROP == N_PCS and N_NODES == 254 and N_EDGES == 10155
DENS = N_EDGES / (N_NODES * (N_NODES - 1) / 2)

# drums that share tags
tag_count = pd.Series([t for v in drum_tags.values() for t in v["process_tags"]]).value_counts()
N_SHARED_TAGS, MAX_SHARE = int((tag_count > 1).sum()), int(tag_count.max())
tags_per_drum = pd.Series([len(v["process_tags"]) for v in drum_tags.values()]).value_counts()
dup_share = 0
for r in drops[drops.reason.str.startswith("collinear")].itertuples():
    a, b = r.node.rsplit("_PC", 1)[0], r.partner.rsplit("_PC", 1)[0]
    dup_share += bool(set(drum_tags[a]["process_tags"]) & set(drum_tags[b]["process_tags"]))

# links, parents, children per drum
dire = edges[edges.type == "->"]
gs["links"] = gs.edges_to_other_drums
gs["parents"] = gs.drum.map(dire.groupby("target_drum").size()).fillna(0).astype(int)
gs["children"] = gs.drum.map(dire.groupby("source_drum").size()).fillna(0).astype(int)
assert int(gs.n_edges.sum()) == 0                       # no link inside a drum (its PCs are uncorrelated)
N_ZERO = int((gs.links == 0).sum())
assert (gs[gs.links == 0].left_out_of_pc == gs[gs.links == 0].n_PCs).all()
LINK_MED = float(gs.links.median())
unit_pc = pd.DataFrame({"parents": dire.groupby("target_unit").size(), "children": dire.groupby("source_unit").size()}
                       ).reindex(UNITS).fillna(0).astype(int)

# order test and timing logs
ORDER = [dict(re.findall(r"(\w+)=([\w.]+)", ln)) | {"name": ln.split(":")[0]}
         for ln in (EVID / "order_test.log").read_text().splitlines() if "flipped=" in ln]
assert len(ORDER) == 5 and all(o["skeleton_identical"] == "True" for o in ORDER)
REV_PCT = float(ORDER[0]["pct"])
RND = [float(o["pct"]) for o in ORDER[1:]]
TLOG = (EVID / "vectorized_pc_real_data.log").read_text()
STAGE = {k: float(v) for k, v in re.findall(r"(corr|l=0|l=1|v-structures|meek) ([\d.]+)s", TLOG)}
T_TOTAL = float(re.search(r"TOTAL ([\d.]+) s", TLOG).group(1))
assert "edges 10155 directed 10154" in TLOG
CLOG = (EVID / "compare_small_medium.log").read_text()
N_RUNS_OK, N_RUNS = map(int, re.search(r"(\d+)/(\d+) runs identical", CLOG).groups())
SPEED = [float(x) for x in re.findall(r"speedup ([\d.]+)x", CLOG)]
assert N_RUNS_OK == N_RUNS == 21 and len(SPEED) == 21

# T² ratios per drum
t2i["r_flare"] = t2i.T2_median_flare / t2i.T2_median_normal
t2i["r_pre"] = t2i.T2_median_prewindow / t2i.T2_median_normal
R_FLARE_MED, R_PRE_MED = float(t2i.r_flare.median()), float(t2i.r_pre.median())
N_FLARE2, N_PRE2 = int((t2i.r_flare >= 2).sum()), int((t2i.r_pre >= 2).sum())
N_PRE_UP = int((t2i.r_pre > 1.05).sum())

# causal-learn list sizes on our skeleton (GraphClass.find_unshielded_triples / find_triangles / find_kites)
idx = {n: i for i, n in enumerate(NODES)}
A = np.zeros((N_NODES, N_NODES), bool)
A[edges.source.map(idx), edges.target.map(idx)] = True
A |= A.T
deg = A.sum(1)
ADJ = int(A.sum())                                       # causal-learn find_adj(): both directions
PATHS2 = int((deg * (deg - 1)).sum())
TRI = int(np.trace(np.linalg.matrix_power(A.astype(np.int64), 3)))
UT = PATHS2 - TRI
KITES = 0
for i, l in zip(*np.nonzero(A)):
    nb = np.flatnonzero(A[i] & A[l])
    if len(nb) > 1:
        KITES += len(nb) * (len(nb) - 1) // 2 - int(A[np.ix_(nb, nb)].sum()) // 2
PAIRS_ADJ = ADJ * (ADJ - 1)
PAIRS_TRI = TRI * (TRI - 1)
LOOKUPS_TRI = PATHS2 * ADJ // 2
assert ADJ == 2 * N_EDGES and abs(PAIRS_TRI - 4.146e11) / 4.146e11 < 0.01

# autocorrelation sweep (row step 10 = the run)
SW10 = sweep[sweep.step_min == 10].iloc[0]


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}×10{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] if c.isdigit() else '⁻' for c in str(int(e)))}"


def fmt(n):
    return f"{int(n):,}"


# ---------------------------------------------------------------- figures (TA's matplotlib look, as week 10)
ASSETS.mkdir(exist_ok=True)
for old in ASSETS.glob("*.png"):
    old.unlink()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.titlelocation": "left"})
DPI = 150
UCOL = dict(zip(UNITS, ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
                        "#9467bd", "#c5b0d5", "#8c564b", "#e377c2", "#7f7f7f"]))


def figure(rows, cols, h, w=12.9, **kw):
    return plt.subplots(rows, cols, figsize=(w, h), layout="constrained", **kw)


def leg(ax, ncol=4, handles=None, labels=None, **kw):
    args = (handles, labels) if handles is not None else ()
    opts = dict(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, borderaxespad=0.15,
                handlelength=1.4, columnspacing=1.2) | kw
    ax.legend(*args, **opts)


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=DPI)
    plt.close(fig)
    return ASSETS / name


def fig_pca_units():
    t = pca.assign(k=pca.n_PCs.clip(upper=3)).groupby(["unit", "k"]).size().unstack(fill_value=0)
    t = t.reindex(UNITS[::-1]).fillna(0)
    fig, ax = figure(1, 1, 4.7, 6.4)
    left = np.zeros(len(t))
    for k, col, lab in [(0, "#BFBFBF", "skipped (no usable tag)"), (1, "#9DC3E6", "1 PC"), (2, "#2E75B6", "2 PCs"),
                        (3, "#1F3864", "3 PCs")]:
        v = t.get(k, pd.Series(0, index=t.index)).to_numpy()
        ax.barh(t.index, v, left=left, color=col, label=lab, height=0.7)
        left += v
    for y, tot in enumerate(left):
        ax.text(tot + 0.6, y, str(int(tot)), va="center", fontsize=10)
    ax.set_xlabel("drums")
    ax.set_xlim(0, left.max() * 1.1)
    ax.grid(axis="x", alpha=0.3)
    leg(ax, ncol=2)
    return save(fig, "pca_units.png")


def fig_unit_heatmap():
    c = pd.crosstab(edges.source_unit, edges.target_unit).reindex(index=UNITS, columns=UNITS, fill_value=0)
    m = c.to_numpy()
    sym = m + m.T - np.diag(np.diag(m))
    assert sym[np.triu_indices(len(UNITS))].sum() == N_EDGES
    fig, ax = figure(1, 1, 5.6, 6.6)
    im = ax.imshow(sym, cmap="Blues")
    ax.set_xticks(range(len(UNITS)), UNITS, rotation=45, ha="right")
    ax.set_yticks(range(len(UNITS)), UNITS)
    for (i, j), v in np.ndenumerate(sym):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=8, color="white" if v > sym.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="links (direction ignored)")
    return save(fig, "unit_heatmap.png")


def fig_links_per_drum():
    d = gs.sort_values("links", ascending=False).reset_index(drop=True)
    fig, ax = figure(1, 1, 3.3)
    ax.bar(d.index, d.links, color=d.unit.map(UCOL), width=1.0)
    ax.axhline(LINK_MED, color="#404040", ls="--", lw=1)
    ax.text(len(d) * 0.62, LINK_MED + 6, f"median {LINK_MED:.0f} links", fontsize=11)
    z0 = int((d.links > 0).sum())
    ax.annotate(f"{N_ZERO} drums with 0 links:\nall their PCs duplicate another drum", xy=(z0 + N_ZERO / 2, 2),
                xytext=(z0 - 75, 175), fontsize=11, arrowprops=dict(arrowstyle="->", color="#404040"))
    ax.set_xlim(-1, len(d))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} modelled drums, sorted by number of links")
    ax.set_ylabel("links to other drums")
    ax.grid(axis="y", alpha=0.3)
    leg(ax, ncol=13, handles=[Patch(color=UCOL[u]) for u in UNITS], labels=UNITS, fontsize=10, columnspacing=0.8)
    return save(fig, "links_per_drum.png")


def fig_parent_child():
    fig, ax = figure(1, 1, 4.4, 7.4)
    x = np.arange(len(UNITS))
    ax.bar(x - 0.2, unit_pc.parents, 0.4, color="#2E75B6", label="parent links (arrows into the unit)")
    ax.bar(x + 0.2, unit_pc.children, 0.4, color="#ED7D31", label="child links (arrows out of the unit)")
    ax.set_xticks(x, [f"{i + 1}. {u}" for i, u in enumerate(UNITS)], rotation=45, ha="right")
    ax.set_xlabel("units in the order they enter the PC algorithm (column order)")
    ax.set_ylabel("directed links")
    ax.grid(axis="y", alpha=0.3)
    leg(ax, ncol=1)
    return save(fig, "parent_child.png")


def fig_t2_ratios():
    fig, ax = figure(1, 1, 3.9, 6.2)
    bins = np.arange(0.0, 5.01, 0.2)
    ax.hist(t2i.r_pre.clip(upper=4.99), bins=bins, color="#ED7D31", alpha=0.75, label="hour before a flare")
    ax.hist(t2i.r_flare.clip(upper=4.99), bins=bins, histtype="step", color="#C00000", lw=2, label="during a flare")
    ax.axvline(1, color="#404040", lw=1)
    ax.axvline(2, color="#404040", lw=1, ls="--")
    ax.text(2.05, ax.get_ylim()[1] * 0.85, "2× normal", fontsize=10)
    ax.set_xlabel("median drum T² ÷ median drum T² in normal operation")
    ax.set_ylabel("drums")
    ax.grid(alpha=0.3)
    leg(ax, ncol=2)
    return save(fig, "t2_ratios.png")


def fig_t2_example():
    unit, drum = EXAMPLE_DRUM
    t2 = pd.read_parquet(OUT / "t2" / unit / f"{unit}__{drum}.parquet", columns=["T2"])["T2"].astype(float)
    mask = pd.read_parquet(OUT / "flare_mask.parquet")
    h = t2.resample("60min").max()
    fl = mask.in_flare.resample("60min").max()
    fig, ax = figure(1, 1, 2.05)
    tr = ax.get_xaxis_transform()
    ax.fill_between(fl.index, 0, 1, where=fl.to_numpy(), step="post", transform=tr, color="red", alpha=0.22, lw=0)
    ax.plot(h.index, h.values, lw=0.6, color="#1F3864")
    ax.set_yscale("log")
    ax.set_ylabel(f"T² {drum}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(h.index[0], h.index[-1])
    ax.grid(alpha=0.3)
    leg(ax, ncol=2, handles=[plt.Line2D([], [], color="#1F3864"), Patch(color="red", alpha=0.3)],
        labels=[f"{unit} {drum}: T² (max per hour)", "flare event"])
    return save(fig, "t2_example.png")


def fig_complexity():
    rows = [("causal-learn: pairs of triangles\nsearched for kites (Meek R3)", PAIRS_TRI, "#C00000"),
            ("causal-learn: list look-ups\nto build the triangle list", LOOKUPS_TRI, "#C00000"),
            ("causal-learn: pairs of adjacency\nentries scanned for triples", PAIRS_ADJ, "#C00000"),
            ("vectorized PC: kites\nenumerated directly", KITES, "#2E75B6"),
            ("vectorized PC: one 254 × 254\nmatrix product", N_NODES ** 3, "#2E75B6")]
    fig, ax = figure(1, 1, 4.3, 6.6)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for yy, r in zip(y, rows):
        ax.text(r[1] * 1.6, yy, sci(r[1]), va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e14)
    ax.set_xlabel("elementary steps on our graph (log scale)")
    ax.grid(axis="x", alpha=0.3)
    return save(fig, "complexity.png")


def fig_stage_times():
    lab = {"corr": "correlation matrix", "l=0": "skeleton, depth 0", "l=1": "skeleton, depth 1",
           "v-structures": "colliders", "meek": "Meek rules"}
    fig, ax = figure(1, 1, 2.6, 5.6)
    k = list(lab)[::-1]
    ax.barh([lab[s] for s in k], [STAGE[s] for s in k], color="#2E75B6", height=0.6)
    for i, s in enumerate(k):
        ax.text(STAGE[s] + 0.01, i, f"{STAGE[s]:.2f} s" if STAGE[s] >= 0.01 else f"{STAGE[s] * 1000:.0f} ms",
                va="center", fontsize=10)
    ax.set_xlim(0, max(STAGE.values()) * 1.3)
    ax.set_xlabel("seconds")
    ax.grid(axis="x", alpha=0.3)
    return save(fig, "stage_times.png")


def fig_equation(name, heading, lines, width=6.2):
    h = 0.55 + 0.5 * len(lines)
    fig = plt.figure(figsize=(width, h), dpi=200)
    fig.patch.set_facecolor("#F8F8F8")
    fig.text(0.03, 1 - 0.3 / h, heading, fontsize=11, fontweight="bold", va="center")
    for i, (txt, size, x) in enumerate(lines):
        fig.text(x, 1 - (0.75 + 0.5 * i) / h, txt, fontsize=size, va="center")
    fig.savefig(ASSETS / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ASSETS / name


FIG = {"pca": fig_pca_units(), "heat": fig_unit_heatmap(), "links": fig_links_per_drum(), "pc": fig_parent_child(),
       "t2r": fig_t2_ratios(), "t2x": fig_t2_example(), "cx": fig_complexity(), "st": fig_stage_times()}
FIG["eq_skel"] = fig_equation("eq_skel.png", "1. Skeleton, depth 1: every test of the level in one array", [
    ("$\\rho_{ij\\cdot k} = \\frac{r_{ij} - r_{ik}\\,r_{jk}}{\\sqrt{(1-r_{ik}^2)(1-r_{jk}^2)}}$"
     "   for all edges $(i,j)$ × all $k$ at once", 13, 0.04),
    ("$i \\perp j \\mid k \\Leftrightarrow |\\rho_{ij\\cdot k}| < \\tau$,   "
     "$\\tau = \\tanh\\!\\left(z_{1-\\alpha/2}/\\sqrt{N-4}\\right)$", 13, 0.04),
    ("$R$ = correlation matrix of the 254 PCs; adjacency frozen per level (PC-stable)", 10.5, 0.04)])
FIG["eq_meek"] = fig_equation("eq_meek.png", "2. Meek rules as boolean matrix products", [
    ("$D_{ij}=1$: $i\\to j$     $U_{ij}=1$: $i - j$     $N_{ij}=1$: $i, j$ not adjacent", 11.5, 0.04),
    ("R1:  $U \\wedge (D^{\\top} N > 0)$        $i\\to j - k$, $i, k$ not adj.  ⇒  $j \\to k$", 12.5, 0.04),
    ("R2:  $U \\wedge (D\\, D > 0)$             $i\\to j\\to k$, $i - k$  ⇒  $i \\to k$", 12.5, 0.04),
    ("R3:  $U_{il} \\wedge \\sum_{j,k} U_{ij}U_{ik}D_{jl}D_{kl}N_{jk} > 0$   ⇒  $i \\to l$", 12.5, 0.04),
    ("Repeat until no new arrow; skip an arrow that would close a directed cycle", 10.5, 0.04)])

# ---------------------------------------------------------------- pptx helpers (as week 10)
prs = Presentation(REF)
slides = list(prs.slides)
TITLE_SP = copy.deepcopy([s for s in slides[7].shapes if s.name == "Title 1"][0]._element)
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
    """items: str = bullet; ('H', str) = bold sub-heading; ('P', str) = plain paragraph."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for k, it in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if isinstance(it, tuple) and it[0] == "H":
            _runs(p, "**" + it[1] + "**", size=it[2] if len(it) > 2 else 20, color="1F3864")
            p.space_after = Pt(3)
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


def table(s, x, y, w, rows, col_w, size=13, row_h=0.36, bold_first_col=False, align_first=PP_ALIGN.LEFT):
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
            tf.paragraphs[0].alignment = align_first if j == 0 else PP_ALIGN.CENTER
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
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
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


def console(s, x, y, w, h, text, size=12):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string("1E1E1E")
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_top = Inches(0.12)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.name = "Consolas"
        r.font.color.rgb = RGBColor.from_string("E6E6E6")


def takeaway(s, y, text, h=0.55, x=0.4, w=None):
    """Blue-bordered conclusion line under a figure."""
    return box(s, x, y, w or SW - 2 * x, h, text, "EAF1FB", size=15, line="5B9BD5", align=PP_ALIGN.LEFT)


def kpi(s, x, y, w, h, value, label, fill="DEEBF7", line="9DC3E6", vsize=26):
    shp = box(s, x, y, w, h, "", fill, line=line)
    tf = shp.text_frame
    _runs(tf.paragraphs[0], "**" + value + "**", size=vsize, color="1F3864")
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    _runs(p, label, size=12, color="404040")
    return shp


def header(s, x, y, w, text, fill="2E75B6", h=0.42, size=16):
    return box(s, x, y, w, h, text, fill, size=size, bold=True, color="FFFFFF", line=None)


# ---------------------------------------------------------------- 1 cover (TA's slide 1)
cover = prs.slides.get(int(COVER.get("id")))
for shp in cover.shapes:
    if shp.name == "TextBox 3":
        p0 = shp.text_frame.paragraphs[0]
        p1 = copy.deepcopy(p0._p)
        p0._p.addnext(p1)
        for t in p1.findall(".//" + qn("a:t")):
            t.text = ""
        p1.findall(".//" + qn("a:t"))[0].text = "Drum-level causal discovery, May 2025 – May 2026"
        for r in p1.findall(".//" + qn("a:rPr")):
            r.set("sz", "1600")

# ---------------------------------------------------------------- 2 executive summary
s = new_slide("Summary: one causal graph across 254 drum PCs, built in 1.4 s")
BLOCKS = [
    ("What we built", "2E75B6", "DEEBF7", [
        f"A drum-level pipeline on **one year of 1-min data**: {N_DRUMS} drums from the PSV sheets → "
        f"{N_DRUMS_MOD} PCA models → **one causal graph of {N_NODES} PCs** → T² per drum.",
        "All outputs per drum and per unit: graph files, T² series, figures."]),
    ("What we found", "1F3864", "E2E8F3", [
        f"**{fmt(N_EDGES)} links**, {100 * N_CROSS / N_EDGES:.0f} % between units; median **{LINK_MED:.0f} links per "
        f"drum**.",
        f"{N_ZERO} drums duplicate another drum's tags and add no information.",
        f"Drum T² rises only slightly at flares: median **{R_FLARE_MED:.2f}×** normal; ≥ 2× in {N_FLARE2} drums."]),
    ("How far we can trust it", "C55A11", "FBE5D6", [
        f"**Same graph as causal-learn** ({N_RUNS_OK}/{N_RUNS} tests identical; plant skeleton identical).",
        "**Which drums are linked** is reproducible in any column order.",
        "**Which way** a link points is **not yet reliable**, and autocorrelation inflates the link count."]),
    ("Engineering gain", "548235", "E2F0D9", [
        f"causal-learn **did not finish** on this graph; vectorized PC builds it in **{T_TOTAL:.1f} s**.",
        f"On synthetic tests: identical output, **{min(SPEED):.0f}–{max(SPEED):.0f}× faster**.",
        "Key idea: replace pair-by-pair loops with **matrix operations** that use every core."])]
bw = (SW - 0.7 - 3 * 0.2) / 4
for i, (head, col, fill, items) in enumerate(BLOCKS):
    x = 0.35 + i * (bw + 0.2)
    header(s, x, 1.15, bw, head, fill=col, h=0.5, size=17)
    box(s, x, 1.65, bw, 3.75, "", fill, line=None)
    text_box(s, x + 0.3, 1.8, bw - 0.4, 3.55, items, size=15, space=9)
KPIS = [(fmt(N_EDGES), "links between drum PCs"), (f"{100 * N_CROSS / N_EDGES:.0f} %", "of links cross units"),
        (f"{N_RUNS_OK}/{N_RUNS}", "tests identical to causal-learn"), (f"{T_TOTAL:.1f} s", "to build the graph"),
        (fmt(N_EVENTS), "flare events in the year")]
kw = (SW - 0.7 - 4 * 0.2) / 5
for i, (v, l) in enumerate(KPIS):
    kpi(s, 0.35 + i * (kw + 0.2), 5.6, kw, 1.05, v, l, fill="F2F2F2", line="BFBFBF", vsize=24)
caption(s, 0.35, 6.8, 12.6, "Sources: drum_pca_pc/outputs (pc_edges.csv, drum_pc_graph_summary.csv, t2_index.csv), "
        "drum_pca_pc.ipynb run log, presentations/week11/evidence/*.log. " + SRC)

# ---------------------------------------------------------------- 3 approach in six steps
s = new_slide("Our approach in six steps, from PSV sheets to drum T²")
STEPS = [
    ("1  DRUMS", "DEEBF7", "Each PSV's protected equipment becomes a drum; its P / T instruments are its tags",
     f"{N_DRUMS} drums\n{N_UNIQUE_TAGS} tags"),
    ("2  CLEAN + MASK", "E2F0D9", "1-min grid, TA cleaning rules; remove flares and the hour before each",
     f"{N_TAGS} tags\n{fmt(N_NORMAL)} normal rows"),
    ("3  DRUM PCA", "EDE2F6", "One PCA per drum on normal rows; keep 80 % of the variance",
     f"{N_DRUMS_MOD} drums\n{N_PCS} PCs"),
    ("4  PC SKELETON", "FBE5D6", "All drum PCs together; Fisher-z, α = 0.05, stable PC, depth ≤ 1",
     f"{N_NODES} PCs\n{fmt(N_EDGES)} links"),
    ("5  ORIENT", "FFF2CC", "Colliders and Meek rules, computed with matrix operations (vectorized PC)",
     f"{T_TOTAL:.1f} s for\nsteps 4 + 5"),
    ("6  DRUM T²", "FCE4EC", "T² of every drum and every PC at every minute; compare flare vs normal",
     f"{N_DRUMS_MOD} T² series\n{fmt(N_EVENTS)} flares"),
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
caption(s, 0.32, 1.02, 12.7, "Top row: step; middle: what it does; bottom (dashed): result in this run.", size=11)
ASSUME = [("Flare event", "flow ≥ 1.5 t/h for ≥ 30 min (TA rule)"), ("Normal operation", "outside flares and the 60 "
          "min before them"), ("Flare signal", "one plant-wide flow, so the mask applies to every unit"),
          ("PC input", "every 10th normal row (31,044 rows)"), ("Tags", "PSV pressure / temperature only; no "
                                                                   "control valves in the data")]
box(s, 0.32, 5.65, 12.7, 0.38, "Key assumptions", "1F3864", size=13, bold=True, color="FFFFFF", line=None,
    align=PP_ALIGN.LEFT)
cw = (12.7 - 4 * 0.12) / 5
for i, (k, v) in enumerate(ASSUME):
    box(s, 0.32 + i * (cw + 0.12), 6.1, cw, 0.95, f"**{k}**\n{v}", "F2F2F2", size=13, line="BFBFBF")

# ---------------------------------------------------------------- 4 drum-level information
s = new_slide(f"Drums come from the plant's PSV list: {N_DRUMS} drums, {N_UNIQUE_TAGS} tags")
header(s, 0.35, 1.15, 6.2, "From one PSV row to one drum")
table(s, 0.35, 1.75, 6.2, [["PSV tag", "Set pressure\n(kg/cm²g)", "Description", "Pressure\ninstrument"],
                           ["4202", "3.86", "D4201", "42PC4215.PV"]], [1.1, 1.5, 1.6, 2.0], size=13, row_h=0.5)
arrow(s, 3.45, 2.85, 3.45, 3.35, width=2)
box(s, 1.6, 3.4, 3.7, 0.95, "**Drum SEU3 · D4201**\ntag: 42PC4215.PV", "DEEBF7", size=14, line="1F3864")
text_box(s, 0.35, 4.5, 6.2, 2.3, [
    "Source: the tag workbook **FINAL_SHEETS.xlsx**, one sheet per unit, one row per **PSV**.",
    "**Drum** = the equipment named under Description; **tags** = its pressure and temperature instruments.",
    "Instrument cells such as 14PT416XA/XB.PV are split into tags; names differing only in case are merged."],
    size=14, space=5)
header(s, 6.85, 1.15, 6.15, "What the sheets give us", fill="1F3864")
FUN = [(f"16 sheets → **{N_SHEETS_UNITS} units** with data", "CPP, LPG and PDU have no process data"),
       (f"**{N_DRUMS} drums** with at least one recorded tag", f"{N_DRUMS_NO_DATA} more drums have no tag in the data"),
       (f"**{N_UNIQUE_TAGS} tags**, mostly 1 – 2 per drum",
        f"{tags_per_drum.get(1, 0)} drums with 1 tag, {tags_per_drum.get(2, 0)} with 2, "
        f"{int(tags_per_drum[tags_per_drum.index > 2].sum())} with 3 – 5"),
       (f"**{N_SHARED_TAGS} tags shared** by several drums", f"up to {MAX_SHARE} drums list the same tag")]
for i, (a, b) in enumerate(FUN):
    y = 1.75 + i * 0.95
    box(s, 6.85 + i * 0.18, y, 6.15 - i * 0.36, 0.82, f"{a}\n{b}", ["DEEBF7", "D6E4F5", "C9DCF1", "FBE5D6"][i],
        size=13, line="9DC3E6" if i < 3 else "F4B183")
takeaway(s, 5.75, f"**Assumption to confirm with the plant:** a drum is the equipment a PSV protects, so its P / T "
                  f"tags are the closest measured signals to a relief to flare. Shared tags mean some drums are the "
                  f"same signal under two names.", h=0.95, x=6.85, w=6.15)
caption(s, 0.35, 6.95, 12.6, "Source: unit_drum_grouping/unit_drum_grouping.ipynb (example row = SEU3 sheet, "
        "S.No 4), drum_tags.json, unit_drums.json.")

# ---------------------------------------------------------------- 5 data and normal operation
s = new_slide("Normal operation = no flare, and not in the hour before one")
header(s, 0.35, 1.15, 7.3, "Rows: one year of 1-minute data")
FLOW = [(f"**{fmt(N_ROWS)} rows**  ·  22 May 2025 – 21 May 2026", 7.3, "DEEBF7"),
        (f"− {fmt(N_FLARE_ROWS)} rows inside **{fmt(N_EVENTS)} flare events**", 6.6, "FBE5D6"),
        (f"− {fmt(N_PRE_ROWS)} rows in the **60 min before** a flare", 5.9, "FBE5D6"),
        (f"= **{fmt(N_NORMAL)} normal rows** ({100 * N_NORMAL / N_ROWS:.1f} %): PCA fitted here", 5.2, "E2F0D9"),
        (f"every 10th row → **{fmt(N_PC_ROWS)} rows** for the PC algorithm", 4.5, "E2F0D9")]
for i, (t, w, f) in enumerate(FLOW):
    box(s, 0.35 + (7.3 - w) / 2, 1.75 + i * 0.78, w, 0.62, t, f, size=14, line="9DC3E6")
header(s, 8.0, 1.15, 5.0, "Tags", fill="1F3864")
table(s, 8.0, 1.75, 5.0, [["Step", "Tags"], ["In the drum sheets", str(N_TAGS_RAW)],
                          ["After TA cleaning*", str(N_TAGS_CLEAN1)], ["After screening (flat)", str(N_TAGS)]],
      [3.4, 1.6], size=14, row_h=0.45, bold_first_col=True)
caption(s, 8.0, 3.65, 5.0, "* TA clean_and_impute: > 10 % missing, constant, constant slope or any negative value.",
        size=11)
text_box(s, 8.0, 4.15, 5.0, 2.2, [
    ("H", "Why remove the hour before a flare", 16),
    "The model must learn **normal** behaviour; minutes leading into a flare are already abnormal.",
    "The T² of step 6 is still computed on **every** minute, flares included."], size=14, space=4)
takeaway(s, 6.1, "**Reading:** the PCA and the causal graph are learned on the 59 % of the year that is clearly "
                 "normal; flares are only used to compare T² afterwards.", h=0.6)
caption(s, 0.35, 6.85, 12.6, "Source: drum_pca_pc.ipynb, Steps 1 – 2 run log; flare_mask.parquet. " + SRC)

# ---------------------------------------------------------------- 6 drum-wise PCA
s = new_slide(f"Drum-wise PCA: {N_DRUMS_MOD} drums modelled with {N_PCS} PCs")
picture(s, FIG["pca"], 0.3, 1.05, w=6.4)
text_box(s, 7.0, 1.15, 6.0, 4.6, [
    f"One PCA per drum on its standardised tags; keep the fewest PCs that explain **80 %** of the variance.",
    f"**{N_DRUMS - N_DRUMS_MOD} drums skipped**: all their tags were removed by cleaning.",
    f"**{N_ONE_TAG} drums have one tag**: their PC is the tag itself; most others keep 1 – 2 PCs.",
    "**Why one plant-wide PC run:** PCs of the same drum are uncorrelated by construction (largest |r| 0.008), "
    "so a run per drum always gives zero links. The links are **between** drums.",
    "Each PC keeps its drum and unit name, so every link traces back to equipment."], size=15, space=8)
caption(s, 0.3, 6.0, 6.4, "Source: drum_pca_pc/outputs/drum_pca_summary.csv (one row per drum).")
takeaway(s, 6.5, f"**Reading:** most drums are 1 – 2 instruments, so a drum PC is close to the raw pressure or "
                  f"temperature signal of that equipment.", h=0.55)

# ---------------------------------------------------------------- 7 plant-wide graph
s = new_slide(f"One plant-wide graph: {fmt(N_EDGES)} links, {100 * N_CROSS / N_EDGES:.0f} % between units")
picture(s, FIG["heat"], 0.25, 1.0, w=6.55)
KP = [(str(N_NODES), f"PC nodes ({N_DUP} duplicates, {N_NEARDEP} near-dependent left out)"),
      (fmt(N_EDGES), "links found"), (fmt(N_CROSS), "links between units"),
      (f"{deg.mean():.0f}", "links per PC on average"), (f"{100 * DENS:.0f} %", "of all PC pairs are linked")]
for i, (v, l) in enumerate(KP):
    kpi(s, 7.05 + (i % 2) * 3.0, 1.1 + (i // 2) * 1.25, 2.85, 1.1, v, l, vsize=24)
box(s, 10.05, 3.6, 2.85, 1.1, "Direction ignored: these counts do not depend on column order.", "FFFFFF",
    size=13, line="BFBFBF", dash=True)
caption(s, 0.3, 6.6, 6.5, "Cell = links between two units (diagonal: inside one unit). Source: drum_pca_pc/outputs/"
        "pc_edges.csv. " + SRC)
takeaway(s, 5.0, "**Reading:** the graph is dense; CCR, LOUP and PG carry the most links and most links cross unit "
                 "boundaries. Part of this density comes from autocorrelation (slide 11), so counts are an upper bound.",
         h=1.15, x=7.05, w=5.95)

# ---------------------------------------------------------------- 8 links per drum
s = new_slide(f"Links per drum: median {LINK_MED:.0f}; {N_ZERO} drums add no new information")
picture(s, FIG["links"], 0.2, 0.98, w=12.9)
top = gs.sort_values("links", ascending=False).head(6)
rows = [["Unit", "Most-linked drums", "PCs", "Links"]] + [[r.unit, r.drum, r.n_PCs, r.links] for r in top.itertuples()]
table(s, 0.35, 4.4, 7.2, rows, [0.9, 4.6, 0.7, 1.0], size=12, row_h=0.34, bold_first_col=True)
text_box(s, 8.05, 4.35, 4.95, 2.4, [
    f"**{N_ZERO} drums with 0 links**: every PC of theirs equals another drum's PC (|r| ≥ 0.999) and was left out; "
    f"{dup_share} of the {N_DUP} duplicates share a tag with their twin.",
    "No link inside a drum: its PCs are uncorrelated by construction.",
    "Per-drum and per-unit graph files: outputs/graphs_json, graphs_png."], size=13, space=5)
caption(s, 0.35, 6.85, 12.6, "Links = graph edges from a drum's PCs to PCs of other drums. Source: "
        "drum_pca_pc/outputs/drum_pc_graph_summary.csv, pc_dropped_nodes.csv. " + SRC)

# ---------------------------------------------------------------- 9 parents vs children
s = new_slide("Directions follow column order, so we report links, not causes")
picture(s, FIG["pc"], 0.25, 1.0, w=7.0)
header(s, 7.6, 1.15, 5.4, "Same data, different column order", fill="1F3864")
table(s, 7.6, 1.65, 5.4, [["Column order", "Same links?", "Directions flipped"]] +
      [[o["name"].capitalize(), "yes" if o["skeleton_identical"] == "True" else "no", f"{float(o['pct']):.1f} %"]
       for o in ORDER], [1.9, 1.5, 2.0], size=13, row_h=0.4, bold_first_col=True)
text_box(s, 7.6, 4.2, 5.4, 1.6, [
    f"Units entered **first** are almost only parents (SEU3: {unit_pc.parents['SEU3']} in, "
    f"{unit_pc.children['SEU3']} out); units entered **last** are almost only children (CDU: "
    f"{unit_pc.parents['CDU']} in, {unit_pc.children['CDU']} out).",
    "Cause: collider orientations conflict with each other; under causal-learn's rule the first one written "
    "wins, so the column order decides."], size=13, space=5)
caption(s, 0.25, 5.6, 7.0, "Source: drum_pca_pc/outputs/pc_edges.csv; presentations/week11/evidence/order_test.log "
        "(vectorized PC on the same input, reversed and 4 random column orders).")
takeaway(s, 6.15, f"**Reading:** which drums are linked is stable (identical in every order). Which way a link "
                  f"points is not: reversing the order flips {REV_PCT:.1f} % of directions, random orders about half. "
                  f"Parent / child counts are therefore not yet a root-cause ranking.", h=0.85)

# ---------------------------------------------------------------- 10 first T² results
s = new_slide("First T² results: drum signals shift only slightly at flares")
picture(s, FIG["t2r"], 0.25, 1.0, w=5.3)


def short(name, n=30):
    name = re.sub(r"\s*\(\s*", " (", re.sub(r"\s*\)", ")", name))
    return name if len(name) <= n else name[:n].rsplit(" ", 1)[0] + " …"


tops = t2i.sort_values("r_pre", ascending=False)
tops = tops[~tops.drum.isin(["D4208"])].head(5)       # D4208 lists the same tag as D4207
rows = [["Unit · drum", "Hour before", "During"]] + [[f"{r.unit} · {short(r.drum)}", f"{r.r_pre:.1f}×",
                                                      f"{r.r_flare:.1f}×"] for r in tops.itertuples()]
table(s, 5.85, 1.1, 7.15, rows, [4.55, 1.3, 1.3], size=12, row_h=0.34, bold_first_col=True)
caption(s, 5.85, 3.2, 7.15, "Top drums by T² in the hour before a flare (median ÷ normal median). SEU3 D4208 is the "
        "same tag as D4207 and is not repeated.", size=10)
text_box(s, 5.85, 3.62, 7.15, 0.9, [
    f"Median drum: **{R_FLARE_MED:.2f}×** normal during flares, **{R_PRE_MED:.2f}×** in the hour before.",
    f"**{N_FLARE2} drums** reach 2× during flares; **{N_PRE2}** in the hour before (two are one tag)."],
    size=13, space=2)
caption(s, 0.3, 4.42, 5.3, f"Source: drum_pca_pc/outputs/t2_index.csv ({N_DRUMS_MOD} drums).")
picture(s, FIG["t2x"], 0.25, 4.78, w=12.85)
caption(s, 0.3, 6.95, 12.7, f"Example: {EXAMPLE_DRUM[0]} {EXAMPLE_DRUM[1]} (t2/{EXAMPLE_DRUM[0]}/{EXAMPLE_DRUM[0]}__"
        f"{EXAMPLE_DRUM[1]}.parquet, flare_mask.parquet). Not yet available: drum-level T² limits, detection scores "
        "(TP / FN / F1) and the causal split of T² by parent drums.", size=10)

# ---------------------------------------------------------------- 11 confidence
s = new_slide("How far we can trust the results today")
SOLID = [("Same answer as the standard library",
          f"{N_RUNS_OK}/{N_RUNS} synthetic tests identical to causal-learn (links, separating sets, directions); on "
          f"plant data the skeleton matches causal-learn's link for link ({fmt(N_EDGES)})."),
         ("Links do not depend on column order",
          "Stable PC (Colombo & Maathuis, 2014), via causal-learn's stable=True: 5 column orders give the same "
          f"{fmt(N_EDGES)} links."),
         ("T² matches the TA's code", "Drum T² equals the TA's T²_slow + T²_fast to a relative 1.6×10⁻¹³ on every "
                                     "drum.")]
OPEN = [("Directions", f"Set by column order ({REV_PCT:.1f} % flip when reversed). Use links, not arrows."),
        ("Too many links", f"Autocorrelation: {100 * SW10.null_fpr_mean:.1f} % of truly unrelated pairs test as "
                           f"linked (5 % expected); {fmt(N_PC_ROWS)} rows are worth about "
                           f"{SW10.n_eff_median:.0f} independent ones."),
        ("Duplicate drums", f"{N_SHARED_TAGS} tags are listed under several drums; {N_ZERO} drums are copies."),
        ("No process check yet", "Links not yet compared with P&IDs or known flow paths.")]
header(s, 0.35, 1.15, 6.2, "Solid", fill="548235")
header(s, 6.8, 1.15, 6.2, "Not yet: use with care", fill="C55A11")
for i, (k, v) in enumerate(SOLID):
    box(s, 0.35, 1.7 + i * 1.6, 1.95, 1.45, k, "E2F0D9", size=14, bold=True, line="A9D18E")
    box(s, 2.35, 1.7 + i * 1.6, 4.2, 1.45, v, "FFFFFF", size=13, line="A9D18E", align=PP_ALIGN.LEFT)
for i, (k, v) in enumerate(OPEN):
    box(s, 6.8, 1.7 + i * 1.2, 1.95, 1.08, k, "FBE5D6", size=14, bold=True, line="F4B183")
    box(s, 8.8, 1.7 + i * 1.2, 4.2, 1.08, v, "FFFFFF", size=13, line="F4B183", align=PP_ALIGN.LEFT)
caption(s, 0.35, 6.6, 12.6, "Sources: evidence/compare_small_medium.log, evidence/order_test.log, drum_pca_pc.ipynb "
        "Step 6 check, drum_pca_pc/outputs/autocorr_sweep/metrics.csv (row step 10 min; details in Appendix A3).")

# ---------------------------------------------------------------- 12 why the standard tool could not run
s = new_slide("Why the standard library could not finish this graph")
picture(s, FIG["cx"], 0.25, 1.0, w=6.6)
caption(s, 0.3, 5.45, 6.5, f"Counts on our skeleton ({N_NODES} PCs, {fmt(N_EDGES)} links). causal-learn 0.1.4.8 "
        "builds its lists by comparing every pair of list entries (GraphClass.find_triangles / find_kites).")
GAP = [("Bottleneck is orientation, not skeleton", f"With conditioning sets of size ≤ 1 the skeleton is cheap "
        f"(causal-learn's finished; ours takes {STAGE['l=1']:.1f} s). Listing the kites alone needs "
        f"{sci(PAIRS_TRI)} comparisons."),
       ("Literature speeds up the skeleton only", "Parallel PC (Le et al.), cuPC (GPU) and pcalg stable.fast "
        "parallelise the skeleton; orientation stays sequential."),
       ("causal-learn is serial", "No parallel code in its PC search; lists built in pure Python."),
       ("Benchmarks are small or sparse", f"causal-learn's bundled data: 6 – 14 variables; TA's R script: 14. Ours: "
        f"{N_NODES} PCs, {deg.mean():.0f} links each.")]
for i, (k, v) in enumerate(GAP):
    box(s, 7.15, 1.1 + i * 1.32, 5.85, 1.2, f"**{k}**\n{v}", ["FBE5D6", "F2F2F2", "F2F2F2", "F2F2F2"][i],
        size=13, line="BFBFBF", align=PP_ALIGN.LEFT)
takeaway(s, 6.35, "**Reading:** existing tools solve a different bottleneck. Our case, few conditioning variables "
                  "but a dense graph, moves the cost into orientation, which none of the tools we reviewed "
                  "vectorizes.", h=0.65)

# ---------------------------------------------------------------- 13 vectorized PC
s = new_slide(f"Vectorized PC: the same graph as causal-learn in {T_TOTAL:.1f} s")
kpi(s, 0.35, 1.15, 2.9, 1.45, "did not finish", "causal-learn pc() on the lab machine", fill="FBE5D6",
    line="F4B183", vsize=22)
arrow(s, 3.3, 1.88, 3.75, 1.88, width=3)
kpi(s, 3.8, 1.15, 2.9, 1.45, f"{T_TOTAL:.1f} s", "vectorized PC, same input and settings", fill="E2F0D9",
    line="A9D18E", vsize=30)
HOW = [("1  Matrix, not pairs", "The graph is a 254 × 254 matrix. A whole level of independence tests is one "
                                "array formula on the correlation matrix, the idea behind PC-stable and Parallel PC."),
       ("2  Rules as matrix products", "Meek's orientation rules become boolean matrix products (D·D, Dᵀ·N): "
                                       "multi-core BLAS today, and the same form runs on a GPU."),
       ("3  Exact where it matters", "When collider orientations conflict, causal-learn's own order is replayed in "
                                     "compiled code, with triangles and kites generated on the fly, not from lists.")]
for i, (k, v) in enumerate(HOW):
    box(s, 0.35, 2.85 + i * 1.12, 6.35, 1.0, f"**{k}**\n{v}", "DEEBF7", size=13, line="9DC3E6", align=PP_ALIGN.LEFT)
picture(s, FIG["st"], 7.0, 1.05, w=6.0)
caption(s, 7.0, 3.85, 6.0, "Stage times on the plant input (31,044 × 254), Apple M1 8 cores, numba compile excluded. "
        "Source: evidence/vectorized_pc_real_data.log.")
box(s, 7.0, 4.45, 6.0, 1.5, f"**Validated against causal-learn**\n{N_RUNS_OK}/{N_RUNS} synthetic runs identical "
    f"(20 and 80 variables, α = 0.01 / 0.05 / 0.1): {min(SPEED):.0f}–{max(SPEED):.0f}× faster.\nPlant data: same "
    f"{fmt(N_EDGES)}-link skeleton as causal-learn.", "E2F0D9", size=13, line="A9D18E", align=PP_ALIGN.LEFT)
takeaway(s, 6.3, "**Impact:** the whole-plant drum graph can be rebuilt in seconds for every period, setting or "
                 "sensitivity check, instead of being out of reach. Mathematics: Appendix A1.", h=0.65)

# ---------------------------------------------------------------- 14 done and next
s = new_slide("What we have done, and what we will investigate next")
DONE = [f"Drum grouping from the PSV sheets: {N_DRUMS} drums, {N_UNIQUE_TAGS} tags, {N_SHEETS_UNITS} units.",
        f"Drum-wise PCA on one year of 1-min data: {N_DRUMS_MOD} drums, {N_PCS} PCs.",
        f"One plant-wide PC graph over {N_NODES} PCs: {fmt(N_EDGES)} links, files per drum and per unit.",
        f"Vectorized PC: identical to causal-learn; {T_TOTAL:.1f} s where causal-learn did not finish.",
        f"T² per drum and per PC for all {fmt(N_ROWS)} minutes; first flare vs normal comparison."]
NEXT = [("Remove duplicates", f"merge drums that share tags ({N_ZERO} drums carry no new information)."),
        ("Order-free directions", "conservative / majority-rule PC, or orient links with known process flow."),
        ("Autocorrelation", "thin rows by effective sample size, or a lag-aware test (PCMCI+ style)."),
        ("Drum causal T²", "T² split by parent drums, control limits and TP / FN / F1 as in week 10."),
        ("Process check", "review the strongest links with plant engineers against P&IDs.")]
header(s, 0.35, 1.1, 6.15, "Done", fill="2E75B6", h=0.45)
header(s, 6.85, 1.1, 6.15, "Next: proposed investigations", fill="C55A11", h=0.45)
for i, t in enumerate(DONE):
    box(s, 0.35, 1.7 + i * 1.1, 6.15, 0.98, t, "DEEBF7", size=14, line="9DC3E6", align=PP_ALIGN.LEFT)
for i, (k, t) in enumerate(NEXT):
    box(s, 6.85, 1.7 + i * 1.1, 1.95, 0.98, f"{i + 1}. {k}", "FBE5D6", size=14, bold=True, line="F4B183")
    box(s, 8.85, 1.7 + i * 1.1, 4.15, 0.98, t, "FFFFFF", size=14, line="F4B183", align=PP_ALIGN.LEFT)

# ---------------------------------------------------------------- 15 references
s = new_slide("References")
REFS = [
    "Spirtes, P., Glymour, C. and Scheines, R. *Causation, Prediction, and Search*, 2nd ed. MIT Press, 2000. "
    "(PC algorithm)",
    "Meek, C. Causal inference and causal explanation with background knowledge. *UAI*, 1995. (orientation rules)",
    "Kalisch, M. and Bühlmann, P. Estimating high-dimensional directed acyclic graphs with the PC-algorithm. "
    "*JMLR* 8: 613–636, 2007.",
    "Colombo, D. and Maathuis, M. H. Order-independent constraint-based causal structure learning. *JMLR* 15: "
    "3921–3962, 2014. (PC-stable; conservative and majority-rule variants)",
    "Le, T. D., Hoang, T., Li, J., Liu, L., Liu, H. and Hu, S. A fast PC algorithm for high dimensional causal "
    "discovery with multi-core PCs. *IEEE/ACM TCBB* 16(5), 2019. (Parallel PC)",
    "Zarebavani, B., Jafarinejad, F., Hashemi, M. and Salehkaleybar, S. cuPC: CUDA-based parallel PC algorithm for "
    "causal structure learning on GPU. *IEEE TPDS* 31(3), 2020.",
    "Zheng, Y. et al. Causal-learn: causal discovery in Python. *JMLR* 25(60), 2024. (version 0.1.4.8 used)",
    "Runge, J. et al. Detecting and quantifying causal associations in large nonlinear time series datasets. "
    "*Science Advances* 5(11): eaau4996, 2019. (autocorrelation, PCMCI)"]
text_box(s, 0.5, 1.2, 12.3, 5.8, [r.replace("*", "") for r in REFS], size=15, space=9)
caption(s, 0.5, 6.9, 12.3, "Copies of the PC, PC-stable and Parallel PC papers: references/Literature/. Review notes: "
        "references/Literature/Parallel_PC_Review.html.")

# ---------------------------------------------------------------- A1 vector mathematics
s = new_slide("Appendix A1 · The vector mathematics behind the speed-up")
picture(s, FIG["eq_skel"], 0.3, 1.05, w=6.35)
picture(s, FIG["eq_meek"], 0.3, 3.2, w=6.35)
header(s, 6.9, 1.1, 6.1, "Before: one pair at a time (causal-learn)", fill="C55A11", h=0.38, size=14)
console(s, 6.9, 1.5, 6.1, 1.95, "Adj = list of 20,310 (i, j) entries\n"
                                "Tri = [(a,b,c) for p,q in permutations(Adj,2)\n"
                                "       if p[1]==q[0] and (p[0],q[1]) in Adj]\n"
                                "Kite = [... for p,q in permutations(Tri,2) ...]\n"
                                "for (i,j,k,l) in Kite: test + orient i -> l", size=11)
header(s, 6.9, 3.6, 6.1, "After: whole matrices (vectorized PC)", fill="2E75B6", h=0.38, size=14)
console(s, 6.9, 4.0, 6.1, 1.65, "D = A & ~A.T ; U = A & A.T ; N = ~(A | A.T)\n"
                                "new  = U & (D.T @ N > 0)        # R1\n"
                                "new |= U & (D @ D > 0)          # R2\n"
                                "new |= R3 per node with >= 2 parents\n"
                                "A &= ~new.T   # repeat until no change", size=11)
text_box(s, 6.9, 5.8, 6.1, 1.2, [
    "Same decisions as causal-learn: tests near the threshold are re-run with its exact formula.",
    "If colliders conflict (pattern not extendable, Dor–Tarsi check), causal-learn's order is replayed in numba."],
    size=12, space=3)
caption(s, 0.3, 6.45, 6.3, "Code: parallel_pc_experiment/vectorized_pc.py (_level1, _meek_matrix, "
        "_meek_sequential). On the plant input the conflict path ran.", size=10)

# ---------------------------------------------------------------- A2 cost and benchmark conditions
s = new_slide("Appendix A2 · Cost of each stage and benchmark conditions")
rows = [["Stage", "causal-learn 0.1.4.8", "Size on our graph", "Vectorized PC"],
        ["Skeleton, depth 0", "test every pair", f"{fmt(N_NODES * (N_NODES - 1) // 2)} pairs",
         f"one threshold on R ({STAGE['l=0'] * 1000:.0f} ms)"],
        ["Skeleton, depth 1", "test each edge given each neighbour", "edges × neighbours",
         f"array formula in blocks ({STAGE['l=1']:.2f} s)"],
        ["Unshielded triples", "pairs of adjacency entries", f"{sci(PAIRS_ADJ)} pairs → {fmt(UT)} triples",
         "read from the matrix"],
        ["Colliders", "loop over triples", f"{fmt(UT // 2)} triples (i < k)", f"compiled loop ({STAGE['v-structures']:.2f} s)"],
        ["Triangles", "pairs + list look-ups", f"{sci(LOOKUPS_TRI)} look-ups → {fmt(TRI)}", "on the fly"],
        ["Kites (Meek R3)", "pairs of triangles", f"{sci(PAIRS_TRI)} pairs → {fmt(KITES)} kites",
         f"on the fly (Meek {STAGE['meek']:.2f} s)"]]
table(s, 0.35, 1.15, 12.65, rows, [2.3, 3.4, 3.75, 3.2], size=13, row_h=0.48, bold_first_col=True)
COND = [["Condition", "Value"], ["Input", f"{fmt(N_PC_ROWS)} rows × {N_NODES} drum PCs (drum_pca_pc/outputs/Dk.npy)"],
        ["Settings", "Fisher-z, α = 0.05, stable PC, depth ≤ 1, uc_priority 2, Meek R1–R3"],
        ["Machine", "Apple M1, 8 cores, 8 GB; numba 0.68.1, numpy 2.5.5; compile time excluded"],
        ["causal-learn", "same input: run did not finish (lab machine, drum_pca_pc.ipynb Step 4 note)"]]
table(s, 0.35, 4.75, 12.65, COND, [2.3, 10.35], size=13, row_h=0.4, bold_first_col=True)
caption(s, 0.35, 6.85, 12.6, "Counts computed from drum_pca_pc/outputs/pc_edges.csv with causal-learn's list "
        "definitions; times from evidence/vectorized_pc_real_data.log.")

# ---------------------------------------------------------------- A3 autocorrelation sweep
s = new_slide("Appendix A3 · Autocorrelation inflates the number of links")
picture(s, OUT / "autocorr_sweep" / "all_plots.png", 0.85, 0.98, w=11.6)
caption(s, 0.85, 6.95, 11.6, "Source: drum_pca_pc/autocorr_sweep.py → outputs/autocorr_sweep/ (null pairs = one series "
        "circularly shifted, so real dependence is removed but autocorrelation kept).")
takeaway(s, 4.75, f"**Reading:** at the 10-min step we use, {100 * SW10.null_fpr_mean:.1f} % of unrelated pairs test "
                  f"as linked (5 % expected); lag-1 autocorrelation is {SW10.rho_median:.2f}. Wider steps help but "
                  f"do not fix it.", h=1.75, x=9.05, w=3.75)

# ---------------------------------------------------------------- A4 approaches explored
s = new_slide("Appendix A4 · Approaches we tried before the final method")
TRIED = [("causal-learn pc()", "All 254 PCs at once (commit d01956d)", "Skeleton fine; run hung while listing "
          "triangles and kites", "FBE5D6"),
         ("One run per drum", "PC inside each drum", "Zero links: a drum's PCs are uncorrelated by construction",
          "FBE5D6"),
         ("Parallel PC", "Le et al. re-implemented (0389f27)", "Same graph as PC-stable, but speeds up only the "
          "skeleton; overhead dominates small runs", "F2F2F2"),
         ("pc_fast", "numpy version of causal-learn's rules (df65579)", "Same graph in about 1 s, but depth ≤ 1 only",
          "F2F2F2"),
         ("vectorized PC", "matrix + numba, any depth (7fddb3a, 3c97a54)", f"Identical to causal-learn; "
          f"{T_TOTAL:.1f} s; now used in the notebook", "E2F0D9")]
for x, w, t in [(0.35, 2.4, "Approach"), (2.85, 4.2, "What it was"), (7.15, 5.85, "Outcome")]:
    header(s, x, 1.1, w, t, fill="1F3864", h=0.38, size=14)
for i, (k, a, b, f) in enumerate(TRIED):
    y = 1.6 + i * 1.04
    box(s, 0.35, y, 2.4, 0.92, f"{i + 1}. {k}", f, size=15, bold=True, line="7F7F7F")
    box(s, 2.85, y, 4.2, 0.92, a, "FFFFFF", size=13, line="BFBFBF", align=PP_ALIGN.LEFT)
    box(s, 7.15, y, 5.85, 0.92, b, "FFFFFF", size=13, line="BFBFBF", align=PP_ALIGN.LEFT)
caption(s, 0.35, 6.85, 12.6, "Sources: git history of drum_pca_pc/ and parallel_pc_experiment/ (18 – 19 Nov 2026); "
        "references/Literature/Parallel_PC_Review.html §8.")

prs.save(DECK)
print("saved", DECK, "slides:", len(prs.slides))
