"""Builds presentations/qtank_copula_benchmark.pptx from the result files.

    .venv/bin/python presentations/build_copula_deck.py

Everything is drawn natively (shapes, tables, charts); the only images are
the EDA figures from results/eda-20261002-*/figures and two tigramite plots.
"""

from __future__ import annotations

import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "presentations" / "qtank_copula_benchmark.pptx"

# Palette: charcoal text, burgundy accent, warm greys. No blue, no orange.
INK = RGBColor(0x26, 0x26, 0x26)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
ACCENT = RGBColor(0x7A, 0x1F, 0x3D)
ACCENT_LIGHT = RGBColor(0xC9, 0x9A, 0xAC)
PANEL = RGBColor(0xF3, 0xEE, 0xF0)
RULE = RGBColor(0xBF, 0xBF, 0xBF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
# Edge status colours: chosen to be far apart.
C_FOUND = RGBColor(0x1B, 0x9E, 0x4B)
C_INVENTED = RGBColor(0xE0, 0x31, 0x31)
C_REVERSED = RGBColor(0x8E, 0x24, 0xAA)
C_MISSED = RGBColor(0x9E, 0x9E, 0x9E)
FONT = "Helvetica"

SW, SH = Inches(13.333), Inches(7.5)
ML = Inches(0.55)
CW = SW - 2 * ML  # content width
TOP = Inches(1.45)  # content top
BOTTOM = Inches(6.8)  # content bottom

L = {"pc": "PC", "pcmci_plus": "PCMCI+", "var_lingam": "VAR-LiNGAM", "lste": "LSTE"}
RUNS = {
    "raw": {"960": "run-parcorr-tau20-s3", "2880": "run-parcorr-tau20-full"},
    "copula": {"960": "run-copula-tau20-s3", "2880": "run-copula-tau20-full"},
}
TRUTH = json.loads((REPO / "datasets/generated/ground_truth_edges.json").read_text())
TRUE_EDGES = {tuple(e) for e in TRUTH["edges"]}
NAMES = TRUTH["columns"]


# --------------------------------------------------------------------------- data
def scores(run: str) -> dict[tuple[str, str], dict]:
    rt = {(r["dataset"], r["algorithm"]): r["runtime_seconds"]
          for r in json.load(open(RES / "qtank" / run / "run_summary.json"))}
    out = {}
    for s in json.load(open(RES / "qtank" / run / "scores.json")):
        k = (s["dataset"], s["algorithm"])
        out[k] = {**s, "runtime": rt[k]}
    return out


def unique_edges(run: str, ds: str, algo: str) -> set[tuple[str, str]]:
    g = json.loads((RES / "qtank" / run / "graphs" / f"{ds}__{algo}.json").read_text())
    return {(e["cause"], e["effect"]) for e in g["directed_edges"] if e["cause"] != e["effect"]}


EDA = {d: json.load(open(RES / p / "eda_summary.json"))
       for d, p in [("P_minus", "eda-20261002-072603"), ("P_plus", "eda-20261002-072543")]}
EDA_FIG = RES / "eda-20261002-072603" / "figures"
TIG = RES / "qtank" / "tigramite-plots-run-copula-tau20-s3"


# ------------------------------------------------------------------------ helpers
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]
page = [0]


def text(slide, x, y, w, h, s, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def bullets(slide, x, y, w, h, items, size=13, gap=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "–  " + it
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.color.rgb = INK
    return tb


def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, lw=0.75):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    return s


def hline(slide, x, y, w, color=RULE, lw=0.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x, y, x + w, y)
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    return c


def arrow(slide, x1, y1, x2, y2, color, lw=2.0, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dashed:
        ln.append(etree.fromstring('<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="dash"/>'))
    ln.append(etree.fromstring('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'))
    return c


def panel_header(slide, x, y, w, title, sub=None, h=Inches(0.5)):
    rect(slide, x, y, w, h, fill=PANEL)
    text(slide, x + Inches(0.08), y, w - Inches(0.16), h, title, size=13, bold=True,
         color=ACCENT, anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(slide, x, y, w - Inches(0.1), h, sub, size=10, color=MUTED, align=PP_ALIGN.RIGHT,
             anchor=MSO_ANCHOR.MIDDLE)


def new_slide(kicker, title, source=None):
    page[0] += 1
    s = prs.slides.add_slide(BLANK)
    text(s, ML, Inches(0.3), CW, Inches(0.3), kicker.upper(), size=10, color=MUTED, bold=True)
    text(s, ML, Inches(0.55), CW, Inches(0.85), title, size=22, bold=True, color=INK,
         anchor=MSO_ANCHOR.TOP)
    hline(s, ML, Inches(7.0), CW)
    if source:
        text(s, ML, Inches(7.02), CW - Inches(1), Inches(0.35), "Source: " + source, size=8.5, color=MUTED)
    text(s, SW - ML - Inches(0.6), Inches(7.02), Inches(0.6), Inches(0.3), str(page[0]),
         size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def table(slide, x, y, w, rows, col_w=None, size=11, row_h=Inches(0.3), header_fill=PANEL):
    n_r, n_c = len(rows), len(rows[0])
    shp = slide.shapes.add_table(n_r, n_c, x, y, w, row_h * n_r)
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    # strip the default style id so no theme colours leak in
    for el in tblPr.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId"):
        tblPr.remove(el)
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = int(w * cw / tot)
    for r in range(n_r):
        tbl.rows[r].height = row_h
        for c in range(n_c):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(rows[r][c])
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = r == 0
            run.font.color.rgb = ACCENT if r == 0 else INK
            # bottom rule on every cell
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("lnL", "lnR", "lnT"):
                tcPr.append(etree.fromstring(
                    f'<a:{tag} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" w="0"><a:noFill/></a:{tag}>'))
            tcPr.append(etree.fromstring(
                '<a:lnB xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" w="6350">'
                '<a:solidFill><a:srgbClr val="BFBFBF"/></a:solidFill></a:lnB>'))
    return shp


def bar_chart(slide, x, y, w, h, categories, series: dict, colors, y_title=None,
              y_max=None, number_format="0.00", legend=True, labels=True, font=10):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series.items():
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, cd)
    ch = gf.chart
    ch.font.name = FONT
    ch.font.size = Pt(font)
    ch.font.color.rgb = INK
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(font)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(font)
    va.tick_labels.number_format = number_format
    va.tick_labels.number_format_is_linked = False
    if y_max is not None:
        va.maximum_scale = y_max
        va.minimum_scale = 0
    if y_title:
        va.has_title = True
        va.axis_title.text_frame.text = y_title
        va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(font)
        va.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(font)
    ca.format.line.color.rgb = RULE
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.font.size = Pt(font - 1)
        dl.number_format = number_format
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, col in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
        ser.invert_if_negative = False
    return ch


# ----------------------------------------------------------------- graph drawing
NODE_POS = {  # normalised panel coordinates
    "v1": (0.12, 0.08), "d": (0.50, 0.08), "v2": (0.88, 0.08),
    "h3": (0.27, 0.50), "h4": (0.73, 0.50),
    "h1": (0.27, 0.92), "h2": (0.73, 0.92),
}


def classify(edges: set) -> dict[tuple, str]:
    out = {}
    for e in edges:
        if e in TRUE_EDGES:
            out[e] = "found"
        elif (e[1], e[0]) in TRUE_EDGES:
            out[e] = "reversed"
        else:
            out[e] = "invented"
    for e in TRUE_EDGES - edges:
        out[e] = "missed"
    return out


STATUS_COLOR = {"found": C_FOUND, "invented": C_INVENTED, "reversed": C_REVERSED, "missed": C_MISSED}


def draw_graph(slide, x, y, w, h, edges: set | None, node_r=Inches(0.26), lw=2.5, label_size=12):
    """edges=None draws the ground truth in neutral colour."""
    pos = {k: (x + int(w * px), y + int(h * py)) for k, (px, py) in NODE_POS.items()}
    cls = classify(edges) if edges is not None else {e: "truth" for e in TRUE_EDGES}
    for (a, b), st in cls.items():
        (ax, ay), (bx, by) = pos[a], pos[b]
        dx, dy = bx - ax, by - ay
        d = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / d, dy / d
        off = 0
        if (b, a) in cls:  # both directions present: offset sideways
            off = Inches(0.07)
        px, py = -uy * off, ux * off
        x1, y1 = ax + ux * node_r + px, ay + uy * node_r + py
        x2, y2 = bx - ux * node_r + px, by - uy * node_r + py
        color = INK if st == "truth" else STATUS_COLOR[st]
        arrow(slide, int(x1), int(y1), int(x2), int(y2), color, lw=lw, dashed=(st == "missed"))
    for k, (cx, cy) in pos.items():
        n = slide.shapes.add_shape(MSO_SHAPE.OVAL, cx - node_r, cy - node_r, 2 * node_r, 2 * node_r)
        n.fill.solid()
        n.fill.fore_color.rgb = WHITE
        n.line.color.rgb = INK
        n.line.width = Pt(1.25)
        n.shadow.inherit = False
        tf = n.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = k
        r.font.name = FONT
        r.font.size = Pt(label_size)
        r.font.bold = True
        r.font.color.rgb = INK
    return cls


def edge_legend(slide, x, y, w):
    items = [("Found (true link recovered)", C_FOUND, False), ("Invented (not in ground truth)", C_INVENTED, False),
             ("Reversed (true link, wrong direction)", C_REVERSED, False), ("Missed (true link not found)", C_MISSED, True)]
    seg = w // len(items)
    for i, (lab, col, dashed) in enumerate(items):
        lx = x + seg * i
        arrow(slide, lx, y + Inches(0.14), lx + Inches(0.55), y + Inches(0.14), col, lw=2.5, dashed=dashed)
        text(slide, lx + Inches(0.62), y, seg - Inches(0.65), Inches(0.28), lab, size=10.5, color=INK,
             anchor=MSO_ANCHOR.MIDDLE)


# ================================================================== slides
# 1 cover
page[0] += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, Inches(0.35), SH, fill=ACCENT)
text(s, Inches(1.0), Inches(2.2), Inches(11), Inches(0.4), "ROOT-CAUSE DETECTION  ·  BENCHMARK RESULTS", size=11, color=MUTED, bold=True)
text(s, Inches(1.0), Inches(2.6), Inches(11), Inches(1.6),
     "Causal discovery on the quadruple-tank process: raw data versus Gaussian-copula-transformed data",
     size=32, bold=True, color=INK)
text(s, Inches(1.0), Inches(4.4), Inches(11), Inches(0.9),
     ["PC, PCMCI+, VAR-LiNGAM and LSTE on datasets P_minus and P_plus.",
      "Results produced on 2 October 2026."], size=15, color=INK)
text(s, Inches(1.0), Inches(6.6), Inches(11), Inches(0.4),
     "Runs: run-parcorr-tau20-s3, run-parcorr-tau20-full, run-copula-tau20-s3, run-copula-tau20-full, run-copula-lste-s3",
     size=9.5, color=MUTED)

# 2 system
s = new_slide("The system", "Two pumps, four tanks, one disturbance: eight directed links are the answer key for every result that follows",
              "datasets/QTank_Dynamics.m; Johansson, IEEE TCST 2000; datasets/generated/ground_truth_edges.json")
# left: rig schematic
lx, lw_ = ML, Inches(6.0)
panel_header(s, lx, TOP, lw_, "Physical rig (Johansson quadruple-tank process)")
gy = TOP + Inches(0.65)
gh = Inches(4.0)
def tank(x, y, label, sub):
    r = rect(s, x, y, Inches(1.5), Inches(0.95), fill=WHITE, line=INK, lw=1.25)
    text(s, x, y + Inches(0.12), Inches(1.5), Inches(0.4), label, size=16, bold=True, align=PP_ALIGN.CENTER)
    text(s, x, y + Inches(0.5), Inches(1.5), Inches(0.35), sub, size=10, color=MUTED, align=PP_ALIGN.CENTER)
TX = {"h3": lx + Inches(1.0), "h4": lx + Inches(3.6)}
tank(TX["h3"], gy, "h3", "upper left")
tank(TX["h4"], gy, "h4", "upper right")
tank(TX["h3"], gy + Inches(1.8), "h1", "lower left")
tank(TX["h4"], gy + Inches(1.8), "h2", "lower right")
# disturbance between upper tanks
rect(s, lx + Inches(2.75), gy + Inches(0.2), Inches(0.6), Inches(0.5), fill=RGBColor(0x59, 0x59, 0x59), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, lx + Inches(2.75), gy + Inches(0.2), Inches(0.6), Inches(0.5), "d", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, lx + Inches(2.75), gy + Inches(0.45), TX["h3"] + Inches(1.5), gy + Inches(0.45), MUTED, lw=1.75)
arrow(s, lx + Inches(3.35), gy + Inches(0.45), TX["h4"], gy + Inches(0.45), MUTED, lw=1.75)
# pumps
PY = gy + Inches(3.3)
rect(s, lx + Inches(0.2), PY, Inches(1.3), Inches(0.5), fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, lx + Inches(0.2), PY, Inches(1.3), Inches(0.5), "pump v1", size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, lx + Inches(4.5), PY, Inches(1.3), Inches(0.5), fill=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, lx + Inches(4.5), PY, Inches(1.3), Inches(0.5), "pump v2", size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
# v1 -> h1 (up), v1 -> h4 (diagonal); v2 -> h2, v2 -> h3
arrow(s, lx + Inches(0.85), PY, TX["h3"] + Inches(0.4), gy + Inches(1.8) + Inches(0.95), INK, lw=1.75)
arrow(s, lx + Inches(1.5), PY + Inches(0.1), TX["h4"] + Inches(0.2), gy + Inches(0.95), INK, lw=1.75)
arrow(s, lx + Inches(5.15), PY, TX["h4"] + Inches(1.1), gy + Inches(1.8) + Inches(0.95), INK, lw=1.75)
arrow(s, lx + Inches(4.5), PY + Inches(0.1), TX["h3"] + Inches(1.3), gy + Inches(0.95), INK, lw=1.75)
# drainage h3 -> h1, h4 -> h2 (accent)
arrow(s, TX["h3"] + Inches(0.75), gy + Inches(0.95), TX["h3"] + Inches(0.75), gy + Inches(1.8), ACCENT, lw=2.5)
arrow(s, TX["h4"] + Inches(0.75), gy + Inches(0.95), TX["h4"] + Inches(0.75), gy + Inches(1.8), ACCENT, lw=2.5)
text(s, lx, gy + Inches(4.0), lw_, Inches(0.7),
     "Each pump splits its flow between one lower tank and the diagonally opposite upper tank. "
     "Upper tanks drain into the tank below (burgundy). d feeds both upper tanks.", size=10.5, color=INK)
# right: causal graph + variable table
rx, rw = ML + Inches(6.4), CW - Inches(6.4)
panel_header(s, rx, TOP, rw, "Same system as a causal graph: 7 nodes, 8 directed links")
draw_graph(s, rx + Inches(0.5), TOP + Inches(0.95), rw - Inches(1.0), Inches(2.6), None, node_r=Inches(0.27))
table(s, rx, TOP + Inches(3.95), rw,
      [["Variable", "Meaning", "Type"],
       ["v1, v2", "pump voltages", "binary PRBS inputs"],
       ["d", "disturbance flow into upper tanks", "binary PRBS input"],
       ["h1, h2", "lower-tank levels", "continuous"],
       ["h3, h4", "upper-tank levels", "continuous"]],
      col_w=[1.1, 2.6, 1.6], size=10, row_h=Inches(0.27))

# 3 setup
s = new_slide("Data and setup", "Two operating points, two sampling rates, one window of 20 lags; every run is scored against the same 8 edges",
              "datasets/generated/ground_truth_edges.json; datasets/run_algorithms.py; docs/QTP_EXPECTATIONS.md")
lw_ = Inches(7.2)
panel_header(s, ML, TOP, lw_, "Benchmark settings")
table(s, ML, TOP + Inches(0.6), lw_,
      [["Item", "Value"],
       ["Datasets", "P_minus (minimum phase), P_plus (non-minimum phase); same simulator and inputs"],
       ["Variables", "v1, v2, d (binary PRBS inputs); h1, h2, h3, h4 (tank levels)"],
       ["Ground truth", "8 directed edges; no lag information recorded"],
       ["Sampling, full", "Ts = 5 s, 2880 rows (4 h)"],
       ["Sampling, stride 3", "Ts = 15 s, 960 rows (same 4 h)"],
       ["tau_max", "20 samples = 100 s (full) or 300 s (stride 3)"],
       ["Standardisation", "every column z-scored before each algorithm"],
       ["Scoring", "lags collapsed to unique directed edges; self-loops dropped; precision, recall, F1, SHD"]],
      col_w=[1.5, 5.0], size=11, row_h=Inches(0.42))
rx = ML + lw_ + Inches(0.4)
rw = CW - lw_ - Inches(0.4)
panel_header(s, rx, TOP, rw, "Ground-truth edges", "cause → effect")
rows = [["#", "Cause", "Effect", "Mechanism"]]
mech = {("v1", "h1"): "pump flow", ("v1", "h4"): "pump flow", ("v2", "h2"): "pump flow", ("v2", "h3"): "pump flow",
        ("d", "h3"): "disturbance", ("d", "h4"): "disturbance", ("h3", "h1"): "drainage", ("h4", "h2"): "drainage"}
for i, e in enumerate(TRUTH["edges"], 1):
    rows.append([i, e[0], e[1], mech[tuple(e)]])
table(s, rx, TOP + Inches(0.6), rw, rows, col_w=[0.5, 1, 1, 1.8], size=11, row_h=Inches(0.36))

# 4 algorithms
s = new_slide("Algorithms", "Four algorithms with identical settings on raw and transformed data; LSTE is restricted to 8 lags by cost",
              "datasets/run_algorithms.py (CONFIGS); run_summary.json of each run")
panel_header(s, ML, TOP, CW, "Configuration")
table(s, ML, TOP + Inches(0.6), CW,
      [["Algorithm", "Type", "Independence test / estimator", "Settings", "Runs in this deck"],
       ["PC (lagged, PC-stable)", "constraint-based", "ParCorr", "pc_alpha 0.01, tau_max 20", "raw + copula, 960 and 2880 rows"],
       ["PCMCI+", "constraint-based, autocorrelation-aware", "ParCorr", "pc_alpha 0.01, tau_max 20", "raw + copula, 960 and 2880 rows"],
       ["VAR-LiNGAM", "functional (non-Gaussian ICA)", "none; VAR + ICA", "lags 20, bootstrap 100, edge threshold 0.9", "raw + copula, 960 and 2880 rows"],
       ["LSTE", "information-theoretic", "CMIknn, shuffle test", "tau_max 8, 100 shuffles, alpha 0.05, FDR BH", "copula, 960 rows, P_minus only"]],
      col_w=[1.6, 2.0, 1.7, 2.4, 2.3], size=11, row_h=Inches(0.5))
bullets(s, ML, TOP + Inches(3.5), CW, Inches(1.8), [
    "ParCorr assumes linear dependence with Gaussian residuals; the copula transform targets the second assumption.",
    "LSTE at tau_max 20 was not run: 980 CMIknn tests per dataset at that window did not finish in earlier attempts. Cost at tau 20 is UNVERIFIED.",
    "The raw-data LSTE reference (run-lste-quick) returned 0 edges; its runtime was not recorded.",
], size=12)

# 5 EDA histograms
s = new_slide("Data preprocessing · 1 of 3", "The copula transform maps each tank level to a near-Gaussian marginal; the three binary inputs stay two-valued",
              "src/causal_bench/eda/copula.py; results/eda-20261002-072603/figures (P_minus, 2880 rows)")
half = (CW - Inches(0.3)) // 2
panel_header(s, ML, TOP, half, "Before: raw marginals", "P_minus, 2880 rows")
panel_header(s, ML + half + Inches(0.3), TOP, half, "After: per-column Gaussian copula", "same rows")
pic = s.shapes.add_picture(str(EDA_FIG / "step1_histograms.png"), ML, TOP + Inches(0.6), width=half)
pic2 = s.shapes.add_picture(str(EDA_FIG / "step3_histograms_transformed.png"), ML + half + Inches(0.3), TOP + Inches(0.6), width=half)
by = TOP + Inches(0.6) + max(pic.height, pic2.height) + Inches(0.15)
bullets(s, ML, by, CW, BOTTOM - by, [
    "Transform: QuantileTransformer(output_distribution=\"normal\"), one transformer per column, n_quantiles = sample size. Dependence between columns is left unchanged.",
    "v1, v2, d are PRBS inputs with two values; they map to two normal quantiles and remain two-valued. No marginal transform makes a two-valued column continuous.",
], size=11.5)

# 6 EDA normality tests (native chart + table)
s = new_slide("Data preprocessing · 2 of 3", "Formal normality tests still reject on most columns after the transform; only Anderson-Darling rejections fall, from 7 to 3",
              "results/eda-20261002-072603/eda_summary.json (P_minus); results/eda-20261002-072543/eda_summary.json (P_plus); alpha 0.05")
lw_ = Inches(7.0)
panel_header(s, ML, TOP, lw_, "Columns rejected by each normality test (of 7)", "P_minus, before vs after")
a = EDA["P_minus"]
tests = ["shapiro_wilk", "dagostino_k2", "jarque_bera", "anderson_darling"]
bar_chart(s, ML, TOP + Inches(0.55), lw_, Inches(4.0), ["Shapiro-Wilk", "D'Agostino K²", "Jarque-Bera", "Anderson-Darling"],
          {"before": [a["before_normality"]["rejections"][t] for t in tests],
           "after": [a["after_normality"]["rejections"][t] for t in tests]},
          [ACCENT_LIGHT, ACCENT], y_max=7, number_format="0", y_title="columns rejected")
rx = ML + lw_ + Inches(0.4)
rw = CW - lw_ - Inches(0.4)
panel_header(s, rx, TOP, rw, "Both datasets", "rejections of 7 columns")
b = EDA["P_plus"]
rows = [["Test", "P− before", "P− after", "P+ before", "P+ after"]]
for t, lab in zip(tests, ["Shapiro-Wilk", "D'Agostino K²", "Jarque-Bera", "Anderson-Darling"]):
    rows.append([lab, a["before_normality"]["rejections"][t], a["after_normality"]["rejections"][t],
                 b["before_normality"]["rejections"][t], b["after_normality"]["rejections"][t]])
rows.append(["Practically Gaussian*", a["before_normality"]["n_practically_gaussian"], a["after_normality"]["n_practically_gaussian"],
             b["before_normality"]["n_practically_gaussian"], b["after_normality"]["n_practically_gaussian"]])
table(s, rx, TOP + Inches(0.6), rw, rows, col_w=[2.0, 1, 1, 1, 1], size=10.5, row_h=Inches(0.36))
text(s, rx, TOP + Inches(2.9), rw, Inches(0.5), "* |skew| < 0.5 and |excess kurtosis| < 1. Counts the four tank levels before and after.", size=9.5, color=MUTED)
bullets(s, rx, TOP + Inches(3.4), rw, Inches(2.0), [
    "At n = 2880 the formal tests reject on departures too small to matter; the effect-size row is the operative one.",
    "The three binary inputs fail every test before and after, as expected.",
], size=11)

# 7 EDA outliers + T2
s = new_slide("Data preprocessing · 3 of 3", "No MAD outliers in either dataset; after the transform 3 to 4 percent of rows exceed the Hotelling T² control limit, against 5 percent expected",
              "results/eda-20261002-072603/figures/step4_t2_control_chart.png; eda_summary.json of both EDA runs")
lw_ = Inches(8.2)
panel_header(s, ML, TOP, lw_, "Hotelling T² per observation after the transform", "P_minus; UCL at alpha 0.05")
pic = s.shapes.add_picture(str(EDA_FIG / "step4_t2_control_chart.png"), ML, TOP + Inches(0.6), width=lw_)
rx = ML + lw_ + Inches(0.4)
rw = CW - lw_ - Inches(0.4)
panel_header(s, rx, TOP, rw, "Outliers and T² exceedance")
table(s, rx, TOP + Inches(0.6), rw,
      [["Measure", "P_minus", "P_plus"],
       ["MAD outliers (|z| > 3.5)", a["outliers"]["total_modified_z_outliers"], b["outliers"]["total_modified_z_outliers"]],
       ["IQR outliers (1.5 IQR)", a["outliers"]["total_iqr_outliers"], b["outliers"]["total_iqr_outliers"]],
       ["T² UCL", f"{a['t2']['ucl']:.2f}", f"{b['t2']['ucl']:.2f}"],
       ["Fraction above UCL", f"{a['t2']['fraction_exceeding']:.3f}", f"{b['t2']['fraction_exceeding']:.3f}"],
       ["Expected under null", "0.050", "0.050"]],
      col_w=[2.2, 1, 1], size=10.5, row_h=Inches(0.38))
bullets(s, rx, TOP + Inches(3.2), rw, Inches(2.2), [
    "Nothing is removed: outlier counts are diagnostic only, so the time index stays intact.",
    "The exceedance fraction below 0.05 is consistent with a successful marginal transform; it says nothing about causal structure.",
], size=11)

# 8 raw results
s = new_slide("Results on raw data", "PCMCI+ scores highest on both datasets at both row counts; stride 3 beats the full series for every algorithm",
              "results/qtank/run-parcorr-tau20-s3/scores.json; run-parcorr-tau20-full/scores.json")
r960, r2880 = scores(RUNS["raw"]["960"]), scores(RUNS["raw"]["2880"])
cats = ["PC", "PCMCI+", "VAR-LiNGAM"]
algos = ["pc", "pcmci_plus", "var_lingam"]
lw_ = Inches(6.6)
panel_header(s, ML, TOP, lw_, "F1 against the 8 true edges", "raw data")
bar_chart(s, ML, TOP + Inches(0.55), lw_, Inches(3.3), cats,
          {"P_minus, 960 rows": [r960[("P_minus", a)]["f1"] for a in algos],
           "P_plus, 960 rows": [r960[("P_plus", a)]["f1"] for a in algos],
           "P_minus, 2880 rows": [r2880[("P_minus", a)]["f1"] for a in algos],
           "P_plus, 2880 rows": [r2880[("P_plus", a)]["f1"] for a in algos]},
          [ACCENT, ACCENT_LIGHT, RGBColor(0x59, 0x59, 0x59), RGBColor(0xBF, 0xBF, 0xBF)], y_max=1.0)
rx = ML + lw_ + Inches(0.3)
rw = CW - lw_ - Inches(0.3)
panel_header(s, rx, TOP, rw, "All scores", "raw data")
rows = [["Rows", "Dataset", "Algorithm", "Prec.", "Recall", "F1", "SHD"]]
for n_rows, sc in [("960", r960), ("2880", r2880)]:
    for ds in ["P_minus", "P_plus"]:
        for a_ in algos:
            v = sc[(ds, a_)]
            rows.append([n_rows, ds, L[a_], f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v["shd"]])
table(s, rx, TOP + Inches(0.6), rw, rows, col_w=[0.7, 1.0, 1.3, 0.7, 0.8, 0.6, 0.6], size=9.5, row_h=Inches(0.3))
bullets(s, ML, TOP + Inches(4.0), lw_, Inches(1.4), [
    "PC has the lowest recall everywhere (0.38 to 0.62): it finds the pump links and misses the drainage links.",
    "The denser 5 s series lowers F1 for PC and VAR-LiNGAM; a 20-lag window then spans only 100 s against a 90 s time constant.",
], size=11)

# 9 copula results
s = new_slide("Results after the copula transform", "The transform moves F1 by at most 0.10 in either direction and leaves the ranking unchanged",
              "results/qtank/COPULA_RESULTS.md; run-copula-tau20-s3, run-copula-tau20-full, run-copula-lste-s3")
c960, c2880 = scores(RUNS["copula"]["960"]), scores(RUNS["copula"]["2880"])
lw_ = Inches(6.6)
panel_header(s, ML, TOP, lw_, "F1, raw vs copula", "960 rows (left pair) and 2880 rows (right pair) per algorithm")
cats2 = [f"{L[a_]}\n{ds} {n}" for n in ["960", "2880"] for ds in ["P_minus", "P_plus"] for a_ in algos]
bar_chart(s, ML, TOP + Inches(0.55), lw_, Inches(3.5), cats2,
          {"raw": [sc[(ds, a_)]["f1"] for sc in [r960, r2880] for ds in ["P_minus", "P_plus"] for a_ in algos],
           "copula": [sc[(ds, a_)]["f1"] for sc in [c960, c2880] for ds in ["P_minus", "P_plus"] for a_ in algos]},
          [RGBColor(0xBF, 0xBF, 0xBF), ACCENT], y_max=1.0, font=8)
rx = ML + lw_ + Inches(0.3)
rw = CW - lw_ - Inches(0.3)
panel_header(s, rx, TOP, rw, "Change raw → copula")
rows = [["Rows", "Dataset", "Algorithm", "F1 raw", "F1 cop.", "ΔF1", "SHD raw", "SHD cop."]]
for n_rows, sr, sc in [("960", r960, c960), ("2880", r2880, c2880)]:
    for ds in ["P_minus", "P_plus"]:
        for a_ in algos:
            x_, y_ = sr[(ds, a_)], sc[(ds, a_)]
            d = y_["f1"] - x_["f1"]
            rows.append([n_rows, ds, L[a_], f"{x_['f1']:.2f}", f"{y_['f1']:.2f}", f"{d:+.2f}", x_["shd"], y_["shd"]])
rows.append(["960", "P_minus", "LSTE", "0.00", "0.00", "+0.00", 8, 8])
table(s, rx, TOP + Inches(0.6), rw, rows, col_w=[0.6, 0.95, 1.2, 0.7, 0.7, 0.6, 0.7, 0.7], size=9, row_h=Inches(0.29))
bullets(s, ML, TOP + Inches(4.2), lw_, Inches(1.3), [
    "PC: F1 unchanged or higher in all four cases (+0.04, 0.00, +0.10, +0.07); false positives fall on P_minus.",
    "PCMCI+: within 0.07 of raw in every case; SHD unchanged or +1. VAR-LiNGAM: mixed, from −0.09 to +0.08.",
    "LSTE on copula P_minus: 0 edges in 1313 s. P_plus was not run.",
], size=10.5)

# 10-15 graph slides, one per algorithm per dataset
GRAPH_SRC = "directed_edges in results/qtank/{raw}/graphs and {cop}/graphs, lags collapsed; scores.json of both runs"
def graph_slide(algo, ds):
    raw_run, cop_run = RUNS["raw"]["960"], RUNS["copula"]["960"]
    er, ec = unique_edges(raw_run, ds, algo), unique_edges(cop_run, ds, algo)
    sr, sc = scores(raw_run)[(ds, algo)], scores(cop_run)[(ds, algo)]
    cr, cc = classify(er), classify(ec)
    def counts(c):
        return {k: sum(1 for v in c.values() if v == k) for k in ["found", "invented", "reversed", "missed"]}
    kr, kc = counts(cr), counts(cc)
    dsl = "P_minus (minimum phase)" if ds == "P_minus" else "P_plus (non-minimum phase)"
    def headline():
        d = sc["f1"] - sr["f1"]
        if abs(d) < 0.005:
            return f"{L[algo]} on {ds}: F1 unchanged at {sr['f1']:.2f} after the transform"
        return f"{L[algo]} on {ds}: F1 {'rises' if d > 0 else 'falls'} from {sr['f1']:.2f} to {sc['f1']:.2f} after the transform"
    s = new_slide(f"Recovered graphs · {L[algo]} · {dsl}", headline(),
                  GRAPH_SRC.format(raw=raw_run, cop=cop_run) + "; 960 rows, tau_max 20")
    edge_legend(s, ML, TOP - Inches(0.05), CW)
    gy = TOP + Inches(0.35)
    pw = (CW - 2 * Inches(0.3)) // 3
    gh = Inches(3.9)
    panels = [("Ground truth", "8 links", None, None),
              (f"{L[algo]}, raw data", f"F1 {sr['f1']:.2f} · SHD {sr['shd']}", er, kr),
              (f"{L[algo]}, copula-transformed", f"F1 {sc['f1']:.2f} · SHD {sc['shd']}", ec, kc)]
    for i, (ttl, sub, edges, k) in enumerate(panels):
        px = ML + i * (pw + Inches(0.3))
        panel_header(s, px, gy, pw, ttl, sub)
        draw_graph(s, px + Inches(0.45), gy + Inches(0.95), pw - Inches(0.9), gh - Inches(0.6), edges, node_r=Inches(0.3), lw=3.0, label_size=13)
        if k:
            text(s, px, gy + gh + Inches(0.5), pw, Inches(0.3),
                 f"found {k['found']}   ·   invented {k['invented']}   ·   reversed {k['reversed']}   ·   missed {k['missed']}",
                 size=11, bold=True, align=PP_ALIGN.CENTER)
    # one-line delta
    gained = (cc.keys() & {e for e, v in cc.items() if v == 'found'}) - {e for e, v in cr.items() if v == 'found'}
    lost = {e for e, v in cr.items() if v == 'found'} - {e for e, v in cc.items() if v == 'found'}
    new_fp = {e for e, v in cc.items() if v in ('invented', 'reversed')} - {e for e, v in cr.items() if v in ('invented', 'reversed')}
    gone_fp = {e for e, v in cr.items() if v in ('invented', 'reversed')} - {e for e, v in cc.items() if v in ('invented', 'reversed')}
    fmt = lambda S: ", ".join(f"{a}→{b}" for a, b in sorted(S)) or "none"
    text(s, ML, gy + gh + Inches(0.9), CW, Inches(0.5),
         f"Change raw → copula.  True links gained: {fmt(gained)}.  True links lost: {fmt(lost)}.  "
         f"Wrong links added: {fmt(new_fp)}.  Wrong links removed: {fmt(gone_fp)}.", size=11, color=INK)

for algo in algos:
    for ds in ["P_minus", "P_plus"]:
        graph_slide(algo, ds)

# 16 tigramite
s = new_slide("tigramite native plotting", "tigramite.plotting draws one graph adequately and does not cover comparison, scoring or long windows",
              "results/qtank/tigramite-plots-run-copula-tau20-s3/README.md and status.md; datasets/tigramite_plots.py")
lw_ = Inches(7.4)
panel_header(s, ML, TOP, lw_ // 2 - Inches(0.1), "plot_graph", "PCMCI+, copula P_minus")
panel_header(s, ML + lw_ // 2 + Inches(0.1), TOP, lw_ // 2 - Inches(0.1), "plot_time_series_graph", "same graph, 21 lag columns")
p1 = s.shapes.add_picture(str(TIG / "P_minus_pcmci_plus_graph.png"), ML, TOP + Inches(0.6), height=Inches(2.7))
p2 = s.shapes.add_picture(str(TIG / "P_minus_pcmci_plus_tsg.png"), ML + lw_ // 2 + Inches(0.1), TOP + Inches(0.6), width=lw_ // 2 - Inches(0.1))
text(s, ML, TOP + Inches(3.4), lw_ // 2 - Inches(0.1), Inches(0.5), "Edge colour MCI, node colour auto-MCI, lag labels on edges.", size=9.5, color=MUTED)
text(s, ML + lw_ // 2 + Inches(0.1), TOP + Inches(0.6) + p2.height + Inches(0.05), lw_ // 2 - Inches(0.1), Inches(0.5), "21 lag columns × 7 variables; every repeated lag drawn.", size=9.5, color=MUTED)
rx = ML + lw_ + Inches(0.4)
rw = CW - lw_ - Inches(0.4)
panel_header(s, rx, TOP, rw, "What was tried and what broke", "10 functions, 58 files")
bullets(s, rx, TOP + Inches(0.6), rw, Inches(4.8), [
    "All 10 plotting functions ran on the saved graphs after conversion to tigramite's (N, N, tau+1) string array.",
    "plot_time_series_graph and plot_tsg are unreadable at tau_max 20; usable only up to about 5 lags.",
    "Node colour in plot_graph is auto-MCI, which has no meaning for VAR-LiNGAM graphs.",
    "Mediation plots need a fitted LinearMediation model, drop contemporaneous links, and label the path np.int64(1) under NumPy 2.",
    "No multi-algorithm comparison, no ground-truth overlay, no precision/recall/SHD plots.",
    "plot_densityplots imports seaborn, which is not a declared dependency.",
    "Ground truth cannot be drawn at lag 0 without mirrored reverse links; it was drawn at lag 1.",
], size=11)
bullets(s, ML, TOP + Inches(4.3), lw_, Inches(1.0), [
    "Verdict: use tigramite plots for a single PCMCI+ result at short windows; keep the comparison figures in this deck native or matplotlib.",
], size=11)

# 17 runtime
s = new_slide("Runtime", "The copula transform roughly doubles VAR-LiNGAM runtime; PC and PCMCI+ stay under one minute in every configuration",
              "run_summary.json of run-parcorr-tau20-s3, run-parcorr-tau20-full, run-copula-tau20-s3, run-copula-tau20-full, run-copula-lste-s3")
lw_ = Inches(6.8)
panel_header(s, ML, TOP, lw_, "Wall-clock seconds per algorithm and dataset", "log scale")
cats3 = [f"{L[a_]}\n{ds} {n}" for n in ["960", "2880"] for ds in ["P_minus", "P_plus"] for a_ in algos]
ch = bar_chart(s, ML, TOP + Inches(0.55), lw_, Inches(4.4), cats3,
               {"raw": [sc[(ds, a_)]["runtime"] for sc in [r960, r2880] for ds in ["P_minus", "P_plus"] for a_ in algos],
                "copula": [sc[(ds, a_)]["runtime"] for sc in [c960, c2880] for ds in ["P_minus", "P_plus"] for a_ in algos]},
               [RGBColor(0xBF, 0xBF, 0xBF), ACCENT], number_format="0", font=8)
ch.value_axis.minimum_scale = 1
ch.value_axis.maximum_scale = 1000
sc_el = ch.value_axis._element.find("{http://schemas.openxmlformats.org/drawingml/2006/chart}scaling")
sc_el.insert(0, etree.fromstring('<c:logBase xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" val="10"/>'))
rx = ML + lw_ + Inches(0.3)
rw = CW - lw_ - Inches(0.3)
panel_header(s, rx, TOP, rw, "Copula / raw ratio")
rows = [["Rows", "Dataset", "Algorithm", "Raw (s)", "Copula (s)", "Ratio"]]
for n_rows, sr, sc in [("960", r960, c960), ("2880", r2880, c2880)]:
    for ds in ["P_minus", "P_plus"]:
        for a_ in algos:
            x_, y_ = sr[(ds, a_)]["runtime"], sc[(ds, a_)]["runtime"]
            rows.append([n_rows, ds, L[a_], f"{x_:.1f}", f"{y_:.1f}", f"{y_ / x_:.1f}"])
rows.append(["960", "P_minus", "LSTE", "UNVERIFIED", "1313.1", "n/a"])
table(s, rx, TOP + Inches(0.6), rw, rows, col_w=[0.6, 1.0, 1.2, 0.9, 1.0, 0.6], size=9.5, row_h=Inches(0.3))
text(s, rx, TOP + Inches(4.9), rw, Inches(0.6),
     "The copula runs shared the machine with the LSTE run; the share of the ratio due to contention is UNVERIFIED.", size=9.5, color=MUTED)

# 18 open items
s = new_slide("Open items", "Four measurements are missing from this benchmark",
              "Plan.md (2026-09-27); results/qtank/run-lste-quick; results/qtank/run-copula-lste-s3/run_summary.json")
panel_header(s, ML, TOP, CW, "Not run, not completed, or not recorded")
table(s, ML, TOP + Inches(0.6), CW,
      [["Item", "Status", "Reason", "Cost to close"],
       ["LSTE on P_plus (copula, tau 8, 100 shuffles)", "Not run", "Stopped after P_minus on request", "about 22 min (P_minus took 1313 s)"],
       ["LSTE at tau_max 20", "Not run", "980 CMIknn tests per dataset; earlier attempts did not finish", "UNVERIFIED"],
       ["CMIknn test for PC and PCMCI+ (Plan.md)", "Not completed", "Nonparametric test at 500 shuffles", "about 2.7 h for both datasets, UNVERIFIED"],
       ["LSTE raw-data runtime (run-lste-quick)", "Not recorded", "Run directory holds graphs only, no run_summary.json", "re-run, same cost as above"]],
      col_w=[3.0, 1.3, 3.6, 2.6], size=12, row_h=Inches(0.6))

prs.save(OUT)
print(f"wrote {OUT} ({page[0]} slides)")
