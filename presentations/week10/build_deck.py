"""Week 10 deck: unit-wise PCA -> PC -> causation-based T2 on Jul-Oct 2025 plant data.

Built on the TA's reference deck (HPCL Project_08072026_external.pptx) so the
master, theme, title bar and fonts are the TA's own. Reads only the committed
unit_pca_pc_t2/outputs/ (no plant data). Run from the repo root:

    .venv/bin/python presentations/week10/build_deck.py
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
OUT = ROOT / "unit_pca_pc_t2" / "outputs"
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
REF = ROOT / "references" / "Previous Project Reports" / "HPCL Project_08072026_external.pptx"
DECK = HERE / "HPCL Project_05112026_flare_RCA.pptx"
CASE_EVENTS = [10, 144, 449]
TITLE_PT = 28  # descriptive titles are full sentences; 44 pt (TA master) fits only short labels
SRC = "Lab run of unit_pca_pc_t2, 1-min data, 1 Jul – 31 Oct 2025."

# ---------------------------------------------------------------- data
ev = pd.read_csv(OUT / "tables" / "event_outcomes.csv")
alarms = pd.read_csv(OUT / "tables" / "alarms.csv", parse_dates=["activation", "end"])
flare_events = pd.read_csv(OUT / "tables" / "flare_events.csv", parse_dates=["start", "end"])
metrics = pd.read_csv(OUT / "tables" / "metrics.csv").set_index("metric")["value"]
limits = pd.read_csv(OUT / "t2" / "limits.csv").iloc[0]
dag = pd.read_csv(OUT / "graph" / "dag_edges.csv")
registry = pd.read_csv(OUT / "tag_registry.csv")
M = {k: metrics[k] for k in metrics.index}

tp_ev = ev[ev.outcome == "TP"]
fp_al = alarms[alarms["class"] == "FP"]
tp_al = alarms[alarms["class"] == "TP"]
top1_tp = tp_ev.pc1.value_counts()
top1_fp = fp_al.pc1.value_counts()
n_tp_below = int((tp_ev.ratio1 < 1).sum())
lead_min, lead_max = int(tp_al.lead_min.min()), int(tp_al.lead_min.max())
pct_noc_above = float(limits.pct_noc_above_t2_limit)
assert int(M["TP (events detected)"]) == 74 and len(flare_events) == 620


def parents(pc):
    return sorted(dag[dag.target == pc].source)


def cat(status):
    for key, name in [("used", "kept"), ("missing", "missing"), ("near", "near"),
                      ("constant", "constant"), ("negative", "negative")]:
        if key in status:
            return name
    return "other"


registry["c"] = registry.status.map(cat)
reg_units = registry[registry.unit != "flare_data"]
by_unit = pd.crosstab(reg_units.unit, reg_units.c)
pca_units = [  # (unit, tags, PCs kept, variance captured) from SUMMARY.md / run.log
    ("APS_NVPS_LEU", 12, 2, 61.4), ("DAK", 3, 1, 100.0), ("DIU", 17, 2, 38.6),
    ("ISOM", 41, 2, 38.1), ("ISOM_3", 14, 2, 54.7), ("ISOM_4", 1, 1, 100.0),
    ("LOUP", 52, 2, 69.3), ("NSU_NHT_CCR", 53, 2, 42.4), ("PDA", 28, 2, 45.7),
    ("PG", 34, 2, 51.6), ("SEU1", 19, 2, 60.3), ("SEU2", 14, 2, 52.8),
    ("SEU3", 13, 2, 43.8), ("unit_wise_control_data", 5, 2, 65.2)]
for u, tags, _, _ in pca_units:
    assert by_unit.loc[u, "kept"] == tags, u

# ---------------------------------------------------------------- figures (TA's matplotlib look)
ASSETS.mkdir(exist_ok=True)
fn = pd.read_parquet(OUT / "tables" / "flare_net_raw.parquet")["flare_net"]
t2 = pd.read_parquet(OUT / "t2" / "t2_all_rows.parquet")
T0, T1 = fn.index[0], fn.index[-1]
SPAN = f"{T0:%d %b %Y %H:%M} – {T1:%d %b %Y %H:%M}"


def date_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 11, 21]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y\n%H:%M"))


def fig_flare_events(path):
    fig, ax = plt.subplots(figsize=(16, 4.9), dpi=120)
    for s, e in zip(flare_events.start, flare_events.end):
        ax.axvspan(s, e, color="orange", alpha=0.25, lw=0)
    ax.plot(fn.index, fn.values, color="blue", lw=0.5, label="flare_net (1-min)")
    ax.axhline(1.5, color="red", ls="--", lw=1.2, label="Threshold 1.5 t/h")
    ax.set_title(f"Flare flowrate (flare_net) and flare events, {SPAN}")
    ax.set_ylabel("flare_net (t/h)")
    ax.set_xlabel("Date and time")
    ax.grid(alpha=0.3)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color="orange", alpha=0.25)], lab + ["Flare event (≥ 30 min above threshold)"],
              loc="upper left")
    ax.text(0.995, 0.97, f"Threshold : 1.5 t/h\nMin Duration : 30 min\nFlare Events : {len(flare_events)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black"))
    date_time_axis(ax)
    ax.set_xlim(T0, T1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_results(path):
    fig, axs = plt.subplots(3, 1, figsize=(16, 7.2), dpi=120, sharex=True,
                            gridspec_kw=dict(height_ratios=[1.2, 1.2, 0.6]))
    axs[0].plot(fn.index, fn.values, color="blue", lw=0.5, label="flare_net (1-min)")
    axs[0].axhline(1.5, color="red", ls="--", lw=1.2, label="Threshold 1.5 t/h")
    axs[0].set_ylabel("flare_net (t/h)")
    axs[0].legend(loc="upper left")
    axs[0].text(0.995, 0.96,
                f"Flare Events : {len(flare_events)}\nScored : {int(M['actual flare events (scored)'])}\n"
                f"T² Alarms : {int(M['alarms (activations)'])}\nDetected : {int(M['TP (events detected)'])}\n"
                f"Missed : {int(M['FN (events missed)'])}\nFalse Alarms : {int(M['incorrect predictions (FP alarms)'])}\n"
                f"Recall : {100 * M['recall TP events/(TP+FN events)']:.1f}%",
                transform=axs[0].transAxes, ha="right", va="top", fontsize=8.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor="black"))
    axs[1].plot(t2.index, t2["T2"], color="green", lw=0.5, label="Overall T² (moving-window baseline)")
    axs[1].axhline(limits.t2_limit, color="red", ls="--", lw=1.2, label=f"T² limit {limits.t2_limit:.2f} (× 2)")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("T² (log scale)")
    axs[1].legend(loc="upper left")
    on = pd.Series(0, index=t2.index)
    for s, e in zip(alarms.activation, alarms.end):
        on[s:e] = 1
    axs[2].plot(on.index, on.values, color="black", lw=0.5, label=f"Alarm on ({len(alarms)} alarms)")
    axs[2].plot(tp_al.activation, [1] * len(tp_al), "o", color="red", ms=3,
                label=f"Alarm followed by a flare within 30 min (TP, n = {len(tp_al)})")
    axs[2].set_ylabel("Alarm (1 = on)")
    axs[2].set_yticks([0, 1])
    axs[2].legend(loc="upper left", fontsize=8)
    axs[2].set_xlabel("Date and time")
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title(f"flare_net, overall T² and alarms, {SPAN}", fontweight="bold")
    date_time_axis(axs[2])
    axs[0].set_xlim(T0, T1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_equation(path, heading, lines, width=5.2):
    """Equation card in the look of the TA's slide 48 images."""
    fig = plt.figure(figsize=(width, 0.55 + 0.5 * len(lines)), dpi=200)
    fig.patch.set_facecolor("#F8F8F8")
    fig.text(0.03, 1 - 0.3 / (0.55 + 0.5 * len(lines)), heading, fontsize=11, fontweight="bold", va="center")
    for i, (txt, size, x) in enumerate(lines):
        y = 1 - (0.75 + 0.5 * i) / (0.55 + 0.5 * len(lines))
        fig.text(x, y, txt, fontsize=size, va="center")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


fig_flare_events(ASSETS / "flare_events.png")
fig_results(ASSETS / "t2_overall.png")
fig_equation(ASSETS / "eq_overall.png", "1. Overall T² (26 PCs)", [
    ("For the standardised PC scores $\\mathbf{z}$ at minute $t$,", 9.5, 0.05),
    ("$T^2 = \\mathbf{z}^{\\top}\\,\\mathrm{pinv}(R)\\,\\mathbf{z}$", 15, 0.30),
    ("$R$ = correlation of the PCs in the baseline window", 9.5, 0.05)])
fig_equation(ASSETS / "eq_term.png", "2. Causal term of each PC", [
    ("$\\beta_j = R_{PP}^{-1}R_{Pj}$,   $\\mu_{j\\cdot PA(j)} = \\sum_k \\beta_{jk}\\, z_{pa_k}$", 12, 0.05),
    ("$\\sigma^2_{j\\cdot PA(j)} = 1 - \\sum_k \\beta_{jk}\\,\\rho(z_j, z_{pa_k})$", 12, 0.05),
    ("$T^2_{j\\cdot PA(j)} = \\left(\\frac{z_j - \\mu_{j\\cdot PA(j)}}{\\sigma_{j\\cdot PA(j)}}\\right)^2$", 15, 0.25),
    ("$PA(j)$ = parents of PC $j$ in the causal graph; root PC: $T_j = z_j$", 9.5, 0.05)])
fig_equation(ASSETS / "eq_limit.png", "3. Control Limit (report Eq. 13)", [
    ("$T^2_{lim} = \\frac{p\\,(m+1)(m-1)}{m\\,(m-k-p)}\\,F_{1-\\alpha;\\,p,\\,m-k-p}$", 15, 0.12),
    ("overall: $p$ = 26, $k$ = 0;   per PC: $p$ = 1, $k$ = number of parents", 9.5, 0.05),
    ("$m$ = baseline rows, $\\alpha$ = 0.05; every limit is multiplied by 2", 9.5, 0.05)], width=5.9)
fig_equation(ASSETS / "eq_rule.png", "4. Decision Rule", [
    ("$T^2 > T^2_{lim}$ for 3 min  $\\Rightarrow$  Alarm on", 13, 0.12),
    ("$T^2 \\leq T^2_{lim}$ for 10 min  $\\Rightarrow$  Alarm off", 13, 0.12)], width=5.9)

# ---------------------------------------------------------------- pptx helpers
prs = Presentation(REF)
slides = list(prs.slides)
TITLE_SP = copy.deepcopy([s for s in slides[7].shapes if s.name == "Title 1"][0]._element)  # TA title bar
keep = {0: "cover", 1: "outline", 39: "future"}
sldIdLst = prs.slides._sldIdLst
kept = {}
for i, sldId in enumerate(list(sldIdLst)):
    if i in keep:
        kept[keep[i]] = sldId
    else:
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)
LAYOUT = prs.slide_layouts[1]  # Title and Content, as in the TA's deck
TNR_BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", TITLE_PT)


def new_slide(title):
    # one line in the 13.33 in title bar (0.1 in insets): at most about 945 pt wide
    assert TNR_BOLD.getlength(title) < 930, f"title too long: {title}"
    s = prs.slides.add_slide(LAYOUT)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    sp = copy.deepcopy(TITLE_SP)
    s.shapes._spTree.append(sp)
    ts = sp.findall(".//" + qn("a:t"))
    ts[0].text = title
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
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


def text_box(s, x, y, w, h, items, size=None, space=0):
    """items: str = bullet; ('H', str) = 36 pt bold sub-heading (TA slide 37)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if isinstance(it, tuple):
            _runs(p, "**" + it[1] + "**", size=36)
            p.space_after = Pt(6)
            continue
        _bullet(p)
        _runs(p, it, size=size)
        if space:
            p.space_after = Pt(space)
    return tb


def caption(s, x, y, w, text, size=11):
    """Data-source / reading note under a chart or table."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    _runs(tf.paragraphs[0], text, size=size, color="404040")
    return tb


def picture(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y),
                                Inches(w) if w else None, Inches(h) if h else None)


def table(s, x, y, w, rows, col_w, size=12, row_h=0.34, bold_last=False):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                             Inches(row_h * len(rows)))
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
            p = tf.paragraphs[0]
            txt = str(val)
            if bold_last and i == len(rows) - 1:
                txt = "**" + txt + "**"
            _runs(p, txt, size=size)
            if j > 0 and isinstance(val, (int, float)):
                p.alignment = PP_ALIGN.CENTER
    return tb


def box(s, x, y, w, h, text, fill, size=10, bold=False, color="000000", line="7F7F7F", dash=False,
        align=PP_ALIGN.CENTER):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    shp.line.color.rgb = RGBColor.from_string(line)
    shp.line.width = Pt(0.75)
    if dash:
        shp.line.dash_style = 4  # dash
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.05))
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _runs(p, ("**" + ln + "**") if (bold and i == 0) else ln, size=size, color=color)
    return shp


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string("404040")
    c.line.width = Pt(1.25)
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


def replace_list(shape, items, keep_first_empty=False, size=None):
    """Refill a TA bullet box: paragraph 1 is the template (bullet format kept)."""
    ps = shape.text_frame.paragraphs
    tmpl = ps[1]._p
    anchor = ps[0]._p
    for para in ps[1:]:
        para._p.getparent().remove(para._p)
    prev = anchor
    for _ in items:
        np_ = copy.deepcopy(tmpl)
        prev.addnext(np_)
        prev = np_
    if not keep_first_empty:
        anchor.getparent().remove(anchor)
    paras = shape.text_frame.paragraphs[1 if keep_first_empty else 0:]
    for p, item in zip(paras, items):
        for r in p._p.findall(qn("a:r")):
            p._p.remove(r)
        _runs(p, item, size=size)


def fmt(n):
    return f"{int(n):,}"


# ---------------------------------------------------------------- slide 1: cover (TA's slide 1)
cover = prs.slides.get(int(kept["cover"].get("id")))
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

# ---------------------------------------------------------------- slide 2: outline (TA's slide 2)
outline = prs.slides.get(int(kept["outline"].get("id")))
OUTLINE = ["Data and preprocessing (July – October 2025)",
           "Final pipeline: unit-wise PCA – PC algorithm – causation-based T²",
           "Detection results over four months",
           "Flare event case studies and recurring contributors",
           "Limitations, conclusions and future work"]
for shp in outline.shapes:
    if shp.name == "Rectangle 2":
        replace_list(shp, OUTLINE)

# ---------------------------------------------------------------- 3 objective and scope
s = new_slide("We want to find which unit drives each flare, and warn early")
text_box(s, 0.52, 1.05, 12.3, 2.5, [
    ("H", "Objective"),
    "Find which **unit and process variables** move before a flare starts.",
    "Raise a **T² alarm** early enough for the operator to act, and name the unit behind it."], size=20, space=8)
text_box(s, 0.52, 3.55, 12.3, 3.7, [
    ("H", "Data and approach"),
    "HPCL historian data, **1 July – 31 October 2025** (4 months, 1-minute samples, 177,120 rows).",
    "**416 tags**: 412 process tags in 14 unit folders + 4 flare tags.",
    "Approach: **unit-wise PCA → PC algorithm (causal graph) → causation-based T² → alarm and root-cause ranking.**",
    "All numbers in this deck come from the lab-computer run of the final pipeline."], size=20, space=8)

# ---------------------------------------------------------------- 4 raw dataset structure
s = new_slide("Four months of 1-minute data from 14 units: 306 of 412 tags kept")
rows = [["Unit", "Tags loaded", "Tags kept", "Tags dropped"]]
for u, tags, _, _ in pca_units:
    tot = int(by_unit.loc[u].sum())
    rows.append([u, tot, tags, tot - tags])
rows.append(["Total (process tags)", 412, 306, 106])
table(s, 0.4, 1.05, 7.4, rows, [3.2, 1.4, 1.4, 1.4], size=12, row_h=0.36, bold_last=True)
text_box(s, 8.1, 1.1, 5.0, 4.9, [
    "One parquet file per tag, grouped in **14 unit folders** + flare_data.",
    "**4 flare tags** are used only to build flare_net (next slide).",
    "**376 duplicate copies** in All_unit_tags were skipped.",
    "Window: 2025-07-01 00:00 to 2025-11-01 00:00 (end excluded).",
    "Sampling: **1 minute** in the files and on the grid.",
    "unit_wise_control_data keeps only 5 of 36 tags (21 have >10% missing)."], size=18, space=8)
caption(s, 8.1, 6.2, 5.0, "Table source: outputs/tag_registry.csv (status of every tag in the lab run). "
        "Loaded = tag files in the unit folder; kept = used in PCA.")

# ---------------------------------------------------------------- 5 flare flowrate and events
s = new_slide("A 1.5 t/h threshold gives 620 flare events in four months")
text_box(s, 0.52, 0.98, 12.3, 2.2, [
    "**flare_net** = (LRflare + LREflare + 99FI4601)/1000 + 509FI5201/1000 − (44FT2505 × 509MW5101 / 22.414)/1000, in t/h.",
    "Flare flowrate threshold is set to **1.5 t/h** (TA's value).",
    "A flare event is a run above the threshold lasting **at least 30 consecutive minutes**.",
    f"**{len(flare_events)} flare events** found. 1,723 shorter runs are not events but are kept out of the baseline.",
    "flare_net median is 1.64 t/h, so **56.8%** of all minutes are above the threshold."], size=16)
picture(s, ASSETS / "flare_events.png", 0.17, 2.55, w=13.0)
caption(s, 0.47, 2.55 + 13.0 * 588 / 1920 + 0.05, 12.4, "Source: outputs/tables/flare_net_raw.parquet (flare_net, 1-min) and "
        "outputs/tables/flare_events.csv (620 events). " + SRC)

# ---------------------------------------------------------------- 6 tags removed
s = new_slide("106 tags were removed, mostly for negative values or missing data")
rows = [["Rule", "Tags dropped", "Example from the log", "Why"],
        ["Any negative value", 55, "98PIC1601.DACA.PV (min −877)", "TA notebook rule (reason pending, Q1)"],
        ["More than 10% missing", 40, "508pc5101.pida.op (91.1% missing)", "Too many gaps to fill reliably"],
        ["Constant", 7, "99PI1542.DACA.PV", "No variance: carries no information"],
        ["Near-constant (>99% one value)", 4, "102PI5001.DACA.PV (100%)", "Almost no variance"],
        ["Total dropped", 106, "306 process tags kept", ""]]
table(s, 0.4, 1.1, 12.5, rows, [3.1, 1.5, 3.9, 4.0], size=16, row_h=0.72, bold_last=True)
caption(s, 0.4, 5.47, 12.5, "Source: outputs/logs/run.log (one line per dropped tag) and outputs/tag_registry.csv. "
        "Rules and thresholds from unit_pca_pc_t2/config.py.")
text_box(s, 0.52, 5.85, 12.3, 1.6, [
    "Most negative-value drops are in **LOUP (24 tags)**; most missing-value drops are in **unit_wise_control_data (21)** and **NSU_NHT_CCR (16)**.",
    "flare_net is never dropped; its 4 source tags have 0% missing on the grid.",
    "No unit lost all its tags; ISOM_4 keeps only 1 of its 3 tags."], size=16, space=3)

# ---------------------------------------------------------------- 7 gap filling, baseline, scaling
s = new_slide("Gaps were tiny (0.06%), and the baseline has 12,157 normal rows")
text_box(s, 0.52, 1.0, 12.3, 3.0, [
    ("H", "Time grid and missing data"),
    "All tags joined on a complete **1-minute grid** (outer join, no resampling): **177,120 rows**.",
    "Sentinel values (−999, −9999, −99999, 99999) set to missing: **none found**.",
    "Gaps filled by **cubic spline**, then forward / back fill (TA's method): 31,680 values = **0.06%** of the matrix.",
    "Rows with less than 90% of tags observed are dropped: **none**."], size=20, space=8)
text_box(s, 0.52, 4.05, 12.3, 2.9, [
    ("H", "Baseline (NOC) and scaling"),
    "Baseline = first **60%** of the window (before 12 Sep 2025 19:11), minus flare events ±120 min, short flares and pre-flare windows: **12,157 rows**.",
    "Every tag is standardised with the **NOC mean and standard deviation**.",
    "Final model input: **177,120 rows × 306 process tags**; flare_net is kept aside as the target."], size=20, space=8)
caption(s, 0.52, 7.05, 12.3, "Source: outputs/logs/run.log (lab run). Gap-fill and baseline rules from unit_pca_pc_t2/config.py.")

# ---------------------------------------------------------------- 8 approaches tried before
s = new_slide("Six earlier design choices were replaced before the final pipeline")
rows = [["Approach", "What it did", "What we saw", "Decision"],
        ["PC on raw tags (causal_t2)", "Causal graph directly on hundreds of tags",
         "PC step failed (stack limit) on one month of data", "Dropped"],
        ["Global PCA (pca_pc_t2)", "26 PCs over all tags + flare tags as graph nodes",
         "A global PC mixes tags of all units, so it cannot point to one unit", "Replaced by unit-wise PCA"],
        ["flare_net inside the PC graph", "flare_net as a node of the PC algorithm",
         "—", "PC on the 26 PCs only; PC → flare_net edges added after"],
        ["Fixed baseline T²", "One Jul–Sep baseline for the whole window",
         "T² drifts above the limit from mid-September and stays there (next slide)", "Moving-window baseline"],
        ["Median + 3×MAD threshold", "Data-driven flare threshold", "—", "Manual 1.5 t/h (TA's value)"],
        ["10-minute flare events (TA)", "Shorter minimum event length", "—", "30-minute events"]]
table(s, 0.4, 1.1, 12.5, rows, [2.9, 3.2, 3.8, 2.6], size=14, row_h=0.72)
caption(s, 0.4, 6.25, 12.5, "Earlier pipelines are in causal_t2/ and pca_pc_t2/; the final one is unit_pca_pc_t2/. "
        "“—” = changed by a team decision, no separate result was recorded.")

# ---------------------------------------------------------------- 8b fixed vs moving baseline (full width)
s = new_slide("A fixed baseline drifts above the limit, so we use a moving window")
H8 = 5.55
W8 = H8 * 1624 / 818
picture(s, OUT / "figures" / "fig8_baseline_drift.png", (13.333 - W8) / 2, 1.05, w=W8)
caption(s, (13.333 - W8) / 2, 1.05 + H8 + 0.1, W8,
        "Top: daily median overall T² over flare-free minutes (log scale) with each limit. Bottom: share of flare-free "
        "minutes above the limit (%). Grey = fixed Jul–Sep baseline, blue = moving-window baseline. "
        "Source: outputs/figures/fig8_baseline_drift.png. " + SRC)

# ---------------------------------------------------------------- 9 final flow diagram (TA divider style)
s = prs.slides.add_slide(prs.slide_layouts[6])  # Blank, like the TA's flow-diagram slides
box(s, 2.15, 0.12, 9.0, 0.5, "Final Flow Diagram – Unit-wise PCA – PC – Causation-based T² Flare Root Cause Identification",
    "1F3864", size=13, bold=True, color="FFFFFF", line="1F3864")
STAGES = [
    ("1", "DATA\nPREPARATION", "DEEBF7", "Load 416 tags · 1-min grid · drop 106 tags\nspline gap fill · NOC baseline · standardise",
     "306 tags × 177,120 rows\nNOC = 12,157 rows"),
    ("2", "UNIT-WISE PCA", "E2F0D9", "For each of the 14 units: PCA on NOC rows\nkeep PC1 and PC2 of the unit",
     "26 PCs\n51.6% of variance"),
    ("3", "CAUSAL\nDISCOVERY", "EDE2F6", "PC algorithm on the 26 PCs\nFisher-Z, α = 0.01, every 10th row",
     "87 edges\n78 between units"),
    ("4", "DAG + FLARE\nNODE", "FBE5D6", "Orient to a DAG (2 edges reversed)\nadd PC → flare_net for every PC",
     "acyclic graph\n26 PC → flare_net edges"),
    ("5", "MONITORING\n(T²)", "FFF2CC", "Moving-window baseline (1,350 rows, step 10)\noverall T² and per-PC causal T²_j, limits × 2",
     f"overall limit {limits.t2_limit:.2f}\nper PC 7.70 – 7.76"),
    ("6", "ALARM & ROOT\nCAUSE", "FCE4EC", "Alarm on after 3 min, off after 10 min\nrank PCs by T²_j / own limit → unit → tags",
     "top 3 PCs per alarm\nRCA_report.xlsx"),
]
y0, step, h = 0.85, 1.1, 0.8
for i, (n, label, fill, center, note) in enumerate(STAGES):
    y = y0 + i * step
    box(s, 0.9, y, 2.0, h, f"{n}  {label}", fill, size=10, bold=True)
    box(s, 3.4, y, 5.6, h, center, fill, size=11, line="5B9BD5")
    box(s, 9.5, y, 2.9, h, note, "FFFFFF", size=10, dash=True)
    arrow(s, 2.9, y + h / 2, 3.4, y + h / 2)
    arrow(s, 9.0, y + h / 2, 9.5, y + h / 2)
    if i < len(STAGES) - 1:
        arrow(s, 6.2, y + h, 6.2, y + step)
caption(s, 9.5, 7.1, 3.6, "Right column: results of the lab run (run.log).", size=10)

# ---------------------------------------------------------------- 10 unit-wise PCA
s = new_slide("Two PCs per unit give 26 PCs that keep 51.6% of the variance")
rows = [["Unit", "Tags", "PCs kept", "Variance captured"]]
rows += [[u, t, k, f"{v:.1f}%"] for u, t, k, v in pca_units]
rows.append(["All units", 306, 26, "51.6%"])
table(s, 0.4, 1.05, 7.4, rows, [3.2, 1.2, 1.4, 1.6], size=12, row_h=0.37, bold_last=True)
text_box(s, 8.1, 1.1, 5.0, 4.9, [
    "PCA is fitted **per unit** on the NOC rows only.",
    "**2 PCs per unit**; DAK (rank 1) and ISOM_4 (1 tag) give 1 PC each.",
    "**26 PCs** keep 51.6% of the process variance.",
    "PCs of the same unit are uncorrelated over NOC (checked).",
    "Each PC is labelled with its heaviest tags, so an alarm can be traced from PC to unit to tag."], size=18, space=10)
caption(s, 8.1, 6.2, 5.0, "Table source: outputs/logs/run.log and SUMMARY.md. Variance captured = share of the "
        "unit's NOC variance (12,157 rows) kept by its PCs.")

# ---------------------------------------------------------------- 11 causal graph
s = new_slide("The causal graph links the 26 PCs with 87 edges, 78 between units")
picture(s, OUT / "figures" / "fig5_causal_graph.png", 0.2, 0.95, h=6.45)
text_box(s, 6.6, 1.05, 6.5, 4.9, [
    "Input: the 26 PC scores, **every 10th row** (17,712 rows). flare_net is not an input.",
    "PC algorithm: **Fisher-Z test, α = 0.01**, stable version, conditioning sets up to 3.",
    "Output: **87 edges** (85 directed, 2 undirected); **78** link different units.",
    "2 edges were reversed to remove cycles; the final graph is **acyclic**.",
    "Each PC has **3.35 parents** on average (max 10); 5 PCs have none.",
    "Every PC is then linked to flare_net (**26 edges**)."], size=18, space=8)
caption(s, 6.6, 6.0, 6.5, "Figure: outputs/figures/fig5_causal_graph.png; edges in outputs/graph/dag_edges.csv. "
        "Node colour = unit; node size and (n) = number of detected events where the PC is the top-1 contributor; "
        "red dashed = edge reversed to break a cycle; light grey = PC → flare_net.")

# ---------------------------------------------------------------- 12 T² statistic
s = new_slide("Each PC's T² term is measured after removing what its parents explain")
picture(s, ASSETS / "eq_overall.png", 0.11, 1.15, w=5.6)
picture(s, ASSETS / "eq_term.png", 0.11, 3.25, w=5.6)
picture(s, ASSETS / "eq_limit.png", 6.4, 1.15, w=6.3)
picture(s, ASSETS / "eq_rule.png", 6.4, 3.25, w=6.3)
text_box(s, 6.4, 5.05, 6.6, 1.8, [
    "Baseline window: **1,350 rows** (50 × 27 variables), moved every **10 rows**.",
    "Window with fewer than 100 flare-free rows: minute **not scored**.",
    f"Limits after × 2: overall **{limits.t2_limit:.2f}**, per PC **7.70 – 7.76**."], size=16)
caption(s, 0.11, 6.95, 12.8, "Equations: unit_pca_pc_t2/MATHS.md rows 14–21 (report Eqs. 9–13). "
        "Limit values: outputs/t2/limits.csv (lab run).")

# ---------------------------------------------------------------- 13 detection strategy
s = new_slide("An alarm counts as correct only if a flare starts within 30 minutes")
text_box(s, 0.52, 1.05, 12.3, 2.7, [
    "The **causation-based T²** on the 26 PCs is used for flare detection.",
    "An alarm switches **on** after T² stays above its limit for **3 minutes**, and **off** after **10 minutes** below it.",
    "An alarm is a **correct prediction** if a flare event starts within **30 minutes** after it switches on.",
    "Root cause: at the alarm's first minute, PCs are ranked by **T²_j / own limit**; the top 3 and their heaviest tags are reported."],
    size=18, space=6)
text_box(s, 0.38, 3.85, 12.3, 3.4, [
    ("H", "Performance Evaluation"),
    "**True Positive (TP):** flare event with an alarm switching on in the 30 min before it.",
    "**False Negative (FN):** flare event with no alarm before it, or one that starts while an alarm is already on.",
    "**False Positive (FP):** alarm outside any flare with no flare in the next 30 min.",
    "**True Negative (TN):** flare-free 30-minute block with no alarm."], size=18, space=6)
caption(s, 0.52, 7.05, 12.3, "Rule settings from unit_pca_pc_t2/config.py "
        "(ALARM_ON_DELAY_MIN = 3, ALARM_OFF_DELAY_MIN = 10, PRED_HORIZON_MIN = 30, TOP_N_ROOT = 3).")

# ---------------------------------------------------------------- 14 T² over four months
s = new_slide("Over four months, T² crossed its limit often and raised 792 alarms")
W14 = 12.9
picture(s, ASSETS / "t2_overall.png", (13.333 - W14) / 2, 0.95, w=W14)
caption(s, (13.333 - W14) / 2, 0.95 + W14 * 864 / 1920 + 0.05, W14,
        "Top: flare_net (t/h) with the 1.5 t/h threshold. Middle: overall T² with the moving-window limit "
        f"{limits.t2_limit:.2f}. Bottom: alarm state (1 = on); red dots = alarms followed by a flare within 30 min. "
        "Source: outputs/t2/t2_all_rows.parquet, outputs/tables/alarms.csv, metrics.csv. " + SRC)

# ---------------------------------------------------------------- 15 detection results (TA slide 35 pairing)
s = new_slide("The model caught 74 of 591 flare events, with a median 13.5 min warning")
console(s, 0.0, 0.9, 6.2, 6.6, "\n".join([
    "==========================================",
    "OVERALL FLARE DETECTION SUMMARY (JUL-OCT 2025)",
    "==========================================",
    f"Total samples (1-min)         : {fmt(len(t2))}",
    "Flare threshold (t/h)         : 1.50",
    f"T2 limit (overall, x2)        : {limits.t2_limit:.2f}",
    "------------------------------------------",
    f"Flare events (>= 30 min)      : {len(flare_events)}",
    f"Events scored                 : {fmt(M['actual flare events (scored)'])}",
    f"Alarms (activations)          : {fmt(M['alarms (activations)'])}",
    f"  during a flare              : {fmt(M['alarms during a flare (not predictions)'])}",
    f"  outside a flare             : {fmt(M['predicted flare events (alarms outside flares)'])}",
    "------------------------------------------",
    f"Detected events (TP)          : {fmt(M['TP (events detected)'])}",
    f"Missed events (FN)            : {fmt(M['FN (events missed)'])}",
    f"  no alarm                    : {fmt(M['FN: no alarm'])}",
    f"  already in alarm            : {fmt(M['FN: already in alarm'])}",
    f"False alarms (FP)             : {fmt(M['incorrect predictions (FP alarms)'])}",
    f"Flare-free blocks (TN)        : {fmt(M['TN (30-min flare-free blocks, no alarm)'])}",
    "------------------------------------------",
    f"Recall                        : {100 * M['recall TP events/(TP+FN events)']:.1f} %",
    f"Precision                     : {100 * M['precision TP alarms/(TP+FP alarms)']:.1f} %",
    f"Accuracy                      : {100 * M['base accuracy (TP+TN)/(TP+FN+FP+TN)']:.1f} %",
    f"Median lead time (TP)         : {M['median lead time TP (min)']:.1f} min",
    "=========================================="]), size=11)
picture(s, OUT / "figures" / "fig7_metrics.png", 6.4, 1.0, w=6.85)
text_box(s, 6.4, 5.6, 6.85, 0.8, [
    f"Correct alarms gave **{lead_min} – {lead_max} min** of warning.",
    "Most missed events started while an alarm was **already on**."], size=14)
caption(s, 6.4, 6.4, 6.85, "Left: outputs/tables/metrics.csv. TP and FN count flare events, FP counts alarms, "
        "TN counts flare-free 30-min blocks, so the four cells use different units. Right: fig7_metrics.png "
        "(confusion matrix, TP lead time, FP alarm duration, events by month).", size=10)

# ---------------------------------------------------------------- 16-18 case studies
CASE = {10: ("E10: SEU2 alarmed 8 minutes before a small flare on 6 July",
             "A short, sharp SEU2 disturbance just before a small flare."),
        144: ("E144: the same SEU2 signal came back 19 minutes before a flare on 1 August",
              "Same PC and the same two tags as E10, 26 days later."),
        449: ("E449: LOUP alarmed 15 minutes before a 5.6 t/h flare on 16 September",
              "The largest and longest of the three; LOUP is first, APS_NVPS_LEU second.")}


def rx(v):
    return f"{v:.2f}" if 0.95 <= v < 1.05 else f"{v:.1f}"


def tags(s_):
    return s_.split(";")[0].split(" ")[0] + ", " + s_.split(";")[1].strip().split(" ")[0]


for eid in CASE_EVENTS:
    r = ev[ev.event_id == eid].iloc[0]
    on = pd.Timestamp(r.onset)
    act = pd.Timestamp(r.activation)
    s = new_slide(CASE[eid][0])
    picture(s, OUT / "figures" / "events" / f"E{eid:03d}_TP.png", 0.1, 0.95, h=6.45)
    pa = parents(r.pc1)
    text_box(s, 7.6, 1.0, 5.6, 4.9, [
        f"Onset **{on:%d %b %Y %H:%M}**, lasted {int(r.duration_min)} min, peak flare_net **{r.peak_flare_net:.2f} t/h**.",
        f"Alarm on at **{act:%H:%M}**: **{int(r.lead_min)} min before** the flare.",
        f"Top contributor: **{r.pc1}** at **{r.ratio1:.1f}×** its own limit ({tags(r.tags1)}).",
        f"Next: {r.pc2} at {rx(r.ratio2)}× ({tags(r.tags2)}); {r.pc3} at {rx(r.ratio3)}×.",
        f"Graph parents of {r.pc1}: {', '.join(pa) if pa else 'none (root PC)'}.",
        f"{r.pc1} is top-1 in {int(top1_tp.get(r.pc1, 0))} detected events and {int(top1_fp.get(r.pc1, 0))} false alarms.",
        CASE[eid][1] + " A contributor, not a confirmed cause."], size=15, space=3)
    caption(s, 7.6, 5.95, 5.6, f"Figure: outputs/figures/events/E{eid:03d}_TP.png; values from "
            "outputs/tables/event_outcomes.csv. Panels: flare_net (t/h) with the 1.5 t/h threshold; overall T² with "
            "limit 79.7; T²_j of the top 3 PCs with their own limits (about 7.7). Pink = 30 min before onset, "
            "orange = flare, yellow = alarm on. × = T²_j / own limit at the alarm's first minute.", size=10)

# ---------------------------------------------------------------- 19 recurring contributors
s = new_slide("DAK, SEU3 and DIU lead the detected events, and the false alarms too")
t3 = ", ".join(f"{k} ({v})" for k, v in top1_tp.head(3).items())
f3 = ", ".join(f"{k} ({v})" for k, v in top1_fp.head(2).items())
text_box(s, 0.52, 0.95, 12.3, 1.3, [
    f"Most frequent top-1 PC in the 74 detected events: **{t3}**.",
    f"The same units lead the false alarms: **{f3}**. A PC that alarms often is not by itself a cause."], size=16)
picture(s, OUT / "figures" / "fig6_root_cause_frequency.png", 0.2, 2.2, w=6.4)
picture(s, OUT / "figures" / "fig9_root_cause_by_week.png", 6.75, 2.2, w=6.4)
caption(s, 0.2, 2.2 + 6.4 * 1252 / 1756 + 0.05, 12.95,
        "Left: fig6_root_cause_frequency.png (top-1 and top-3 PC counts over the 74 detected events, top-1 PC of the "
        "339 false alarms, by unit). Right: fig9_root_cause_by_week.png (top-1 PC of detected events per week; colour = "
        "number of events). Source: outputs/tables/event_outcomes.csv and alarms.csv. " + SRC, size=10)

# ---------------------------------------------------------------- 20 what the model shows
s = new_slide("The model points to contributing units, not confirmed root causes")
rows = [["Level", "Meaning", "What we have"],
        ["Association", "T² is above its limit before a flare",
         f"{int(M['TP (events detected)'])} of {int(M['actual flare events (scored)'])} events; "
         f"{int(M['correct predictions (TP alarms)'])} of {int(M['predicted flare events (alarms outside flares)'])} alarms outside flares"],
        ["Contributing PC / tag", "PC with the largest T²_j / limit, after removing what its graph parents explain",
         "Top 3 PCs and their tags for every alarm and event (RCA_report.xlsx)"],
        ["Confirmed root cause", "Physical cause checked against plant operations",
         "Not established yet; needs review by the plant team"]]
table(s, 0.5, 1.2, 12.3, rows, [2.6, 4.9, 4.8], size=16, row_h=0.95)
caption(s, 0.5, 5.05, 12.3, "Counts from outputs/tables/metrics.csv; contributor lists in outputs/RCA_report.xlsx (lab run).")
text_box(s, 0.52, 5.5, 12.3, 1.8, [
    "The causal graph is learned from data on PC scores; its edges are statistical, not process-flow links.",
    f"In {n_tp_below} of the 74 detected events the top PC is still **below** its own limit (ratio < 1), so the ranking there is weak."],
    size=20, space=8)

# ---------------------------------------------------------------- 21 limitations
s = new_slide("Detection is held back by the flare definition and standing alarms")
text_box(s, 0.52, 1.05, 12.3, 5.8, [
    "The 1.5 t/h threshold is **below the flare_net median (1.64 t/h)**: 56.8% of minutes are above it and events cover half the window.",
    f"**{int(M['FN: already in alarm'])} of {int(M['FN (events missed)'])}** missed events started while an alarm was already on (standing alarms).",
    f"**{pct_noc_above:.1f}%** of baseline minutes are above the overall limit, which drives {int(M['incorrect predictions (FP alarms)'])} false alarms ({fmt(M['FP alarm minutes'])} alarm minutes).",
    "ISOM_4 is a single tag and DAK has rank 1; their PC standard deviation had to be floored in 15,237 window steps.",
    f"{int(M['flare events not scored'])} events were not scored (too few flare-free rows in the window).",
    "26 PCs keep only 51.6% of the variance; 2 graph edges were reversed to break cycles.",
    "Open questions to the TA: reason for the negative-value rule (Q1) and the form of Eq. 13 (Q2)."], size=22, space=12)
caption(s, 0.52, 7.05, 12.3, "Sources: outputs/logs/run.log, tables/metrics.csv, t2/limits.csv (lab run).")

# ---------------------------------------------------------------- 22 conclusions
s = new_slide("The pipeline runs end to end, but detection is not yet reliable")
text_box(s, 0.52, 1.0, 12.3, 3.1, [
    ("H", "What we achieved"),
    "The full pipeline ran on **4 months of 1-minute plant data** (177,120 rows, 306 tags, 14 units).",
    "Every alarm is traced from **T² → PC → unit → tag**, through a 26-node causal graph.",
    f"**74 flare events** were flagged **{lead_min} – {lead_max} min** before onset (median 13.5 min)."], size=20, space=8)
text_box(s, 0.52, 4.15, 12.3, 3.1, [
    ("H", "What the results tell us"),
    "Detection is **not yet reliable**: recall 12.5%, precision 18.3%.",
    "The main issues found so far are the **flare event definition** and **standing alarms**.",
    "The contributor lists are **leads for the plant team**, not confirmed causes."], size=20, space=8)

# ---------------------------------------------------------------- 23 future work (TA's slide 40)
future = prs.slides.get(int(kept["future"].get("id")))
FUTURE = ["**Principal component selection:** study the trade-off between model complexity and explained variance, "
          "and pick the number of PCs per unit. One PC may be enough for some units, others may need three or more.",
          "**PC interpretability:** map each principal component back to its raw input tags.",
          "**Alternative causal discovery:** implement PCMCI+ and VARLiNGAM, which may suit our time-series data better.",
          "**Flare definition and standing alarms:** re-check the 1.5 t/h threshold with the plant team, and reset long "
          "alarms so a new flare inside them can be scored.",
          "**Plant confirmation:** plant team to confirm the contributors found for E10, E144 and E449."]
for shp in future.shapes:
    if shp.name == "TextBox 2":
        replace_list(shp, FUTURE, keep_first_empty=True, size=16)
        shp.top = Inches(2.2)
        shp.height = Inches(5.0)
    if shp.name == "Title 1":
        shp.text_frame.paragraphs[0].runs[0].text = "Future work: unit-specific PCs, PC-to-tag mapping, new causal methods"
        for r in shp.text_frame.paragraphs[0].runs:
            r.font.size = Pt(28)

# ---------------------------------------------------------------- order and save
new_ids = [sid for sid in sldIdLst if sid not in kept.values()]
for sid in list(sldIdLst):
    sldIdLst.remove(sid)
for sid in [kept["cover"], kept["outline"]] + new_ids + [kept["future"]]:
    sldIdLst.append(sid)
prs.save(DECK)
print("saved", DECK, "slides:", len(prs.slides))
