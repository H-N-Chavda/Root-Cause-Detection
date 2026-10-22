#!/usr/bin/env python3
"""Build the plant-data results deck (13 slides) with python-pptx.

Covers only the PCA -> PC algorithm -> T² run on the plant flare data
(pca_pc_t2/, lab computer run of 22 Oct 2026).

    .venv/bin/python presentations/build_findings_deck.py

Options
    --out PATH       output file (default: presentations/rcd_findings_deck.pptx)
    --force          write the deck even if a number check FAILs

What it does, in order
  1. Reads the result tables in pca_pc_t2/outputs/tables and RESULTS.md, and
     checks every number typed in the DATA section against them. Any FAIL stops
     the build unless --force is given.
  2. Draws the slides with plain shapes. Time-series charts are redrawn from the
     tables, so they use the deck colours.
  3. Checks that every shape sits inside the slide and estimates text overflow.

Fonts: Source Serif 4 / Source Sans 3 when installed, otherwise Georgia / Calibri.
Built against python-pptx 1.0.2.
"""

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "pca_pc_t2" / "outputs"
TAB = PT / "tables"
DEFAULT_OUT = ROOT / "presentations" / "rcd_findings_deck.pptx"
SRC = "pca_pc_t2/outputs"


# =============================================================================
# DATA - numbers typed on the slides. verify() checks each against the files.
# =============================================================================

DATA_START, DATA_END = "2025-04-12 00:00:00", "2026-05-21 00:00:00"
FLARE_WINDOWS = [("2025-04-12 12:59:00", "2025-07-03 19:01:00"),
                 ("2025-07-07 07:12:00", "2026-05-20 21:24:00")]       # flare_detection.csv
# TRAIN = before 2025-12-10 09:36 minus flare windows +/- 2 h (RESULTS.md), which leaves:
TRAIN_WINDOWS = [("2025-04-12 00:00:00", "2025-04-12 10:59:00"),
                 ("2025-07-03 21:01:00", "2025-07-07 05:12:00")]
T2_EVENT = ("2025-04-12 00:26:00", DATA_END)                           # t2_events.csv

P = {
    "params_expected": 418, "params_found": 376, "tags_kept": 355, "units": 13,
    "rows": 581761, "train_rows": 5469, "test_rows": 576292, "flare_rows_in_test": 575776,
    "n_pcs": 26, "cum_var": 0.8876, "pcs_80_90_95_99": (11, 30, 52, 104),
    "far_train_T2_99": 0.0922, "test_alarm_rate_T2_99": 0.9943, "test_alarm_rate_Q_99": 1.0,
    "sample_false_alarm_test_normal_T2_99": 0.6977, "auc_T2": 0.7497,
    "median_detection_delay_min": -149.0,
    "T2_99": 45.95, "T2_99_empirical_train": 89.28,
    "edges_test": 48, "edges_stable": 16, "bootstrap_B": 10, "jaccard_alpha": 0.62,
    "event_minutes": 581735, "event_peak": 9.278e7,
    "nsu_led_pcs": 13,
    "pc2_high_share": 0.376, "pc2_entries": 14, "pc2_first_high": "2025-07-15",
    "tag_peak": "99PI3219.DACA.PV", "tag_peak_contrib": 92829265.6,
    "tag_min": 0.09, "tag_max": 168.79, "tag_outlier_share": 0.399,
    "runtime_h": 1.5,
}
PC2_HIGH = 400.0        # PC2 score above this = the 'high' state (the two levels sit near 0 and 800)


def _ts(s):
    return datetime.fromisoformat(s)


def _days(a, b):
    return (_ts(b) - _ts(a)).total_seconds() / 86400


SPAN_DAYS = _days(DATA_START, DATA_END)
FLARE_DAYS = sum(_days(a, b) for a, b in FLARE_WINDOWS)
FLARE_PCT = 100 * FLARE_DAYS / SPAN_DAYS
TRAIN_DAYS = P["train_rows"] / 1440
NON_FLARE_TEST = P["test_rows"] - P["flare_rows_in_test"]


def table(name):
    p = TAB / f"{name}.csv"
    if not p.exists():
        sys.exit(f"missing input: {p}")
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pc2_states():
    """Share of 30-min bins with PC2 in the high state, number of entries into it, first entry."""
    rows = table("fig09_pc_timeseries")
    high = [float(r["PC2"]) > PC2_HIGH for r in rows]
    entries = [rows[i]["timestamp"] for i in range(1, len(rows)) if high[i] and not high[i - 1]]
    return sum(high) / len(high), len(entries), entries[0][:10]


# =============================================================================
# Verification
# =============================================================================

def verify():
    log = []

    def num(label, deck, found, tol):
        good = abs(float(deck) - float(found)) <= tol
        log.append(("PASS" if good else "FAIL", label, f"deck {deck} | file {found}"))

    def same(label, deck, found):
        log.append(("PASS" if deck == found else "FAIL", label, f"deck {deck} | file {found}"))

    txt = (ROOT / "pca_pc_t2" / "RESULTS.md").read_text(encoding="utf-8")
    m = re.search(r"Parameters audited: (\d+) vs (\d+)", txt)
    same("tags found / expected", (P["params_found"], P["params_expected"]), (int(m.group(1)), int(m.group(2))))
    m = re.search(r"Final matrix: (\d+) rows x (\d+) tags", txt)
    same("rows / tags kept", (P["rows"], P["tags_kept"]), (int(m.group(1)), int(m.group(2))))
    m = re.search(r"TRAIN (\d+) rows, TEST (\d+) rows, flare rows in TEST (\d+)", txt)
    same("TRAIN / TEST / flare rows", (P["train_rows"], P["test_rows"], P["flare_rows_in_test"]),
         tuple(int(g) for g in m.groups()))
    m = re.search(r"PCs for 80/90/95/99 %: (\d+)/(\d+)/(\d+)/(\d+)", txt)
    same("PCs for 80/90/95/99 %", P["pcs_80_90_95_99"], tuple(int(g) for g in m.groups()))
    m = re.search(r"stable \(>= 0\.7 of B = (\d+).*?\): (\d+)", txt)
    same("bootstrap B / stable links", (P["bootstrap_B"], P["edges_stable"]), (int(m.group(1)), int(m.group(2))))
    m = re.search(r"Jaccard (\d\.\d+)", txt)
    num("Jaccard alpha 0.05 vs 0.01", P["jaccard_alpha"], float(m.group(1)), 0.005)

    d = {r["metric"]: float(r["value"]) for r in table("detection_metrics")}
    for k in ("far_train_T2_99", "test_alarm_rate_T2_99", "test_alarm_rate_Q_99",
              "sample_false_alarm_test_normal_T2_99", "auc_T2"):
        num(k, P[k], d[k], 0.0005)
    num("median_detection_delay_min", P["median_detection_delay_min"], d["median_detection_delay_min"], 0.5)
    d = {r["limit"]: float(r["value"]) for r in table("control_limits")}
    num("T2_99", P["T2_99"], d["T2_99"], 0.01)
    num("T2_99_empirical_train", P["T2_99_empirical_train"], d["T2_99_empirical_train"], 0.01)
    same("flare windows", FLARE_WINDOWS, [(r["onset"], r["end"]) for r in table("flare_detection")])
    t = table("t2_events")
    same("number of T² events", 1, len(t))
    num("event length (min)", P["event_minutes"], float(t[0]["duration_min"]), 0.5)
    num("event peak T² (millions)", P["event_peak"] / 1e6, float(t[0]["peak_T2"]) / 1e6, 0.01)
    same("event start", T2_EVENT[0], t[0]["start"])
    e = {(r["variant"], r["alpha"]): int(r["edges"]) for r in table("pc_variants_summary")}
    same("links on TEST scores, alpha 0.05", P["edges_test"], e[("T_test_fisherz", "0.05")])
    row = [r for r in table("pca_variance") if int(r["PC"]) == P["n_pcs"]][0]
    num("cumulative variance at 26 PCs", P["cum_var"], float(row["cumulative"]), 0.0005)
    same("components led by NSU_NHT_CCR", P["nsu_led_pcs"],
         sum(r["dominant_unit"] == "NSU_NHT_CCR" for r in table("pca_dominant_units")))
    c = table("event_peak_tag_contrib")
    same("tag with largest contribution", P["tag_peak"], c[0]["column"])
    num("its contribution (millions)", P["tag_peak_contrib"] / 1e6, float(c[0]["contrib_signed"]) / 1e6, 0.01)
    a = [r for r in table("audit_tags") if r["tag"] == P["tag_peak"]][0]
    num("tag minimum", P["tag_min"], float(a["min"]), 0.005)
    num("tag maximum", P["tag_max"], float(a["max"]), 0.005)
    num("tag outlier share", P["tag_outlier_share"], int(a["outliers_robust_z_gt_10"]) / P["rows"], 0.0005)
    share, entries, first = pc2_states()
    num("PC2 high-state share", P["pc2_high_share"], share, 0.0005)
    same("PC2 entries into high state", P["pc2_entries"], entries)
    same("PC2 first high state", P["pc2_first_high"], first)
    train_min = sum(_days(a, b) for a, b in TRAIN_WINDOWS) * 1440
    num("derived TRAIN minutes vs TRAIN rows", P["train_rows"], train_min, 5)
    same("flare share rounds to 99 %", 99, round(FLARE_PCT))

    width = max(len(l) for _, l, _ in log)
    for status, label, detail in log:
        print(f"  {status:4}  {label:<{width}}  {detail}")
    n = {k: sum(s == k for s, _, _ in log) for k in ("PASS", "FAIL")}
    print(f"verify: {n['PASS']} pass, {n['FAIL']} fail")
    return n["FAIL"]


# =============================================================================
# Design system
# =============================================================================

FOREST, SAGE, IVORY, CHARCOAL, GOLD = "174A3A", "8DAA91", "F7F5EF", "292D32", "B79A5B"
GREY = "5F656B"
RULE = "D6D2C6"
PALE = "E9EEE7"
MISSED = "E2DED3"
SAGE_PALE = "C9D6C9"
GOLD_PALE = "EEE4CF"

W, H = 13.333, 7.5
ML = 0.65
CW = W - 2 * ML
CHAR_W = 0.5

TITLE_FONT, BODY_FONT = "Source Serif 4", "Source Sans 3"
FALLBACK_FONTS = ("Georgia", "Calibri")
SLIDE_NO = [0]
WARNINGS = []


def rgb(hex_):
    return RGBColor.from_string(hex_)


def I(x):
    return Inches(x)


def font_installed(prefix):
    key = prefix.lower().replace(" ", "")
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
            Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
            Path.home() / "Library" / "Fonts", Path("/Library/Fonts"),
            Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts", Path("/usr/share/fonts")]
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.name.lower().replace(" ", "").replace("-", "").replace("_", "").startswith(key):
                return True
    return False


def fill_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(color)


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, dash=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, I(x), I(y), I(w), I(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.15
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = dash
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def oval(slide, cx, cy, r, fill=None, line=None, lw=1.0):
    return rect(slide, cx - r, cy - r, 2 * r, 2 * r, fill=fill, line=line, lw=lw, shape=MSO_SHAPE.OVAL)


def line(slide, x1, y1, x2, y2, color=CHARCOAL, lw=1.0, dash=None, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    if dash:
        c.line.dash_style = dash
    if arrow:   # tailEnd must be the last child of <a:ln>
        ln = c.line._get_or_add_ln()
        etree.SubElement(ln, qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    return c


def polyline(slide, pts, color=CHARCOAL, lw=1.25):
    """One freeform shape through many points (inches); used for time series."""
    fb = slide.shapes.build_freeform(I(pts[0][0]), I(pts[0][1]), scale=1.0)
    fb.add_line_segments([(I(x), I(y)) for x, y in pts[1:]], close=False)
    s = fb.convert_to_shape()
    s.fill.background()
    s.line.color.rgb = rgb(color)
    s.line.width = Pt(lw)
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, size=14, color=CHARCOAL, bold=False, italic=False, font=None,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.08, after=0, shape=None):
    """paras: string (newlines split paragraphs) or list of paragraphs; a paragraph is a
    string, a (text, style) tuple, or a list of runs. style keys: size, color, bold, italic, font."""
    box = shape if shape is not None else slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = paras.split("\n")
    est = 0.0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if after and i < len(paras) - 1:
            p.space_after = Pt(after)
        runs = [para] if isinstance(para, (str, tuple)) else para
        n_chars, top = 0, 0.0
        for run in runs:
            t, st = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = st.get("font", font or BODY_FONT)
            f.size = Pt(st.get("size", size))
            f.bold = st.get("bold", bold)
            f.italic = st.get("italic", italic)
            f.color.rgb = rgb(st.get("color", color))
            n_chars += len(t)
            top = max(top, st.get("size", size))
        per_line = max(1.0, w * 72 / (CHAR_W * top) * 0.92)
        est += max(1, math.ceil(n_chars / per_line)) * top * 1.2 * spacing / 72
        est += after / 72 if i < len(paras) - 1 else 0
    if est > h + 0.04:
        first = paras[0] if isinstance(paras[0], str) else str(paras[0])
        WARNINGS.append(f"slide {SLIDE_NO[0]}: text may overflow ({est:.2f} in needed, {h:.2f} in given): "
                        f"{first[:60]!r}")
    return box


def new_slide(prs, kicker, title, source=None, notes=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    SLIDE_NO[0] += 1
    fill_bg(s, IVORY)
    text(s, ML, 0.42, CW, 0.28, kicker.upper(), size=11, color=FOREST, bold=True)
    text(s, ML, 0.74, CW, 1.05, title, size=26, font=TITLE_FONT, spacing=1.0)
    line(s, ML, 1.9, ML + CW, 1.9, RULE, 0.75)
    if source:
        text(s, ML, 7.02, CW - 0.8, 0.36, "Source: " + source, size=9, color=GREY)
    text(s, W - ML - 0.6, 7.02, 0.6, 0.3, str(SLIDE_NO[0]), size=9, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def stat(s, x, y, w, number, label, color=FOREST, nsize=26, lsize=11.5, lh=0.6):
    nh = nsize / 72 * 1.3
    text(s, x, y, w, nh, number, size=nsize, color=color, font=TITLE_FONT)
    text(s, x, y + nh + 0.02, w, lh, label, size=lsize, color=GREY)


def takeaway(s, x, y, w, h, msg, size=12):
    rect(s, x, y, 0.06, h, fill=GOLD)
    text(s, x + 0.22, y, w - 0.22, h, msg, size=size, anchor=MSO_ANCHOR.MIDDLE)


def centered(s, cx, cy, w, h, msg, **kw):
    kw.setdefault("align", PP_ALIGN.CENTER)
    kw.setdefault("anchor", MSO_ANCHOR.MIDDLE)
    return text(s, cx - w / 2, cy - h / 2, w, h, msg, **kw)


class TimeAxis:
    """Maps timestamps to x (inches) across the data span and draws month ticks."""

    def __init__(self, x0, w):
        self.x0, self.w, self.t0 = x0, w, _ts(DATA_START)

    def x(self, t):
        t = _ts(t) if isinstance(t, str) else t
        return self.x0 + self.w * (t - self.t0).total_seconds() / 86400 / SPAN_DAYS

    def ticks(self, s, y):
        for stamp, lab in (("2025-05-01", "May 2025"), ("2025-08-01", "Aug 2025"), ("2025-11-01", "Nov 2025"),
                           ("2026-02-01", "Feb 2026"), ("2026-05-01", "May 2026")):
            x = self.x(stamp + " 00:00:00")
            line(s, x, y, x, y + 0.07, GREY, 0.75)
            centered(s, x, y + 0.2, 1.0, 0.22, lab, size=9.5, color=GREY)

    def shade_train(self, s, y, h):
        for a, b in TRAIN_WINDOWS:
            rect(s, self.x(a), y, max(0.05, self.x(b) - self.x(a)), h, fill=GOLD_PALE)


def binned(rows, key, hours, agg):
    out = {}
    for r in rows:
        if r[key] == "":
            continue
        t = _ts(r["timestamp"])
        b = t.replace(minute=0, second=0) - timedelta(hours=t.hour % hours)
        out.setdefault(b, []).append(float(r[key]))
    return sorted((b, agg(v)) for b, v in out.items())


# =============================================================================
# Slides
# =============================================================================

def s01_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    SLIDE_NO[0] += 1
    fill_bg(s, FOREST)
    text(s, 0.9, 1.15, 11.0, 0.35, "ROOT-CAUSE DETECTION  ·  PLANT FLARE DATA", size=12, color=SAGE_PALE, bold=True)
    text(s, 0.9, 1.7, 10.8, 2.4, "PCA, PC algorithm and T² on the plant flare data: first results",
         size=40, color=IVORY, font=TITLE_FONT, spacing=1.0)
    line(s, 0.9, 4.2, 2.4, 4.2, GOLD, 2.5)
    text(s, 0.9, 4.45, 10.8, 0.9, "What the pipeline found on 13 months of data from 13 units, how well it worked, "
         "and what to fix before the next run.", size=18, color=SAGE_PALE)
    text(s, 0.9, 6.5, 10.0, 0.35, "Run on the lab computer, 22 October 2026", size=12, color=SAGE_PALE)
    s.notes_slide.notes_text_frame.text = (
        "This deck covers one run: PCA, then the PC algorithm, then T² monitoring, on the real plant data. "
        "Every number comes from the output folder of that run.")


def s02_summary(prs):
    s = new_slide(prs, "Summary",
                  "The pipeline works, but the results are not usable yet",
                  f"{SRC}/RESULTS.md, tables/ (lab computer run, 22 Oct 2026)",
                  "Three things stop us from reading the results today. All three can be fixed before the next run.")
    rows = [
        (f"{FLARE_PCT:.0f} %", GOLD, "of the time is labelled as flaring",
         f"So the 'normal' training data is only {TRAIN_DAYS:.1f} days long."),
        ("2 states", FOREST, "The plant switches between two operating states",
         "The training data contains only one of them. Everything else looks abnormal."),
        ("1 tag", FOREST, "One tag explains almost all of the biggest alarm",
         f"Tag {P['tag_peak'].split('.')[0]} in LOUP. It needs a check before we trust the alarm."),
    ]
    for i, (big, col, head, sub) in enumerate(rows):
        y = 2.15 + i * 1.55
        if i:
            line(s, ML, y - 0.15, 8.35, y - 0.15, RULE, 0.75)
        text(s, ML, y, 2.3, 0.75, big, size=36, color=col, font=TITLE_FONT)
        text(s, 3.0, y + 0.02, 5.3, 0.62, head, size=16, bold=True)
        text(s, 3.0, y + 0.66, 5.3, 0.7, sub, size=12.5, color=GREY)

    rect(s, 8.85, 2.1, 3.83, 4.65, fill=FOREST)
    text(s, 9.1, 2.35, 3.35, 0.3, "WHAT WE NEED NEXT", size=11, color=SAGE_PALE, bold=True)
    text(s, 9.1, 2.7, 3.35, 0.9, "Three fixes before the next run", size=19, color=IVORY,
         font=TITLE_FONT, spacing=1.0)
    fixes = ["Build the flare labels again, signal by signal",
             "Build a normal baseline that covers both operating states",
             f"Check tag {P['tag_peak'].split('.')[0]} for a sensor fault"]
    for i, f in enumerate(fixes):
        y = 3.65 + i * 1.0
        text(s, 9.1, y, 0.35, 0.4, str(i + 1), size=20, color=SAGE_PALE, font=TITLE_FONT)
        text(s, 9.5, y + 0.03, 2.95, 0.9, f, size=13, color=IVORY)


def s03_pipeline(prs):
    s = new_slide(prs, "What we ran",
                  "We ran six steps, from raw data to root-cause hypotheses",
                  f"prompts/CLAUDE_CLI_PROMPT_PCA_PC_T2.md; pca_pc_t2/src; {SRC}/run.log",
                  "Each step saves its results, so any step can be rerun alone. Each step also checks its own "
                  "output. All checks passed on the lab computer.")
    steps = [
        ("1  Audit", f"{P['params_found']} tags found in {P['units']} units ({P['params_expected']} expected)"),
        ("2  Clean", f"{P['tags_kept']} tags kept; {P['rows']:,} one-minute rows"),
        ("3  PCA", f"{P['n_pcs']} components keep {100 * P['cum_var']:.0f}% of the variation"),
        ("4  PC algorithm", "Finds links between the 26 components"),
        ("5  T² and Q", "Raises an alarm when the plant moves away from normal"),
        ("6  Trace", "Walks back from an alarm to a candidate unit"),
    ]
    bw, gap = 1.75, (CW - 6 * 1.75) / 5
    for i, (head, body) in enumerate(steps):
        x = ML + i * (bw + gap)
        rect(s, x, 2.3, bw, 2.0, fill=FOREST if i == 0 else PALE)
        ink = IVORY if i == 0 else CHARCOAL
        text(s, x + 0.15, 2.45, bw - 0.3, 0.35, head, size=13, bold=True, color=SAGE_PALE if i == 0 else FOREST)
        text(s, x + 0.15, 2.9, bw - 0.3, 1.3, body, size=12, color=ink)
        if i < 5:
            line(s, x + bw + 0.04, 3.3, x + bw + gap - 0.04, 3.3, FOREST, 1.5, arrow=True)
    text(s, ML, 4.75, 5.6, 0.3, "How the data was split", size=14, bold=True, color=FOREST)
    text(s, ML, 5.12, 5.6, 1.5,
         ["Training data: the first 60% of the time, minus the flare windows and 2 hours either side.",
          "Test data: everything else, including all flare windows."], size=12.5, after=6)
    text(s, 6.9, 4.75, 5.78, 0.3, "How the run went", size=14, bold=True, color=FOREST)
    text(s, 6.9, 5.12, 5.78, 1.5,
         [f"All self-checks passed. The run took about {P['runtime_h']:g} hours on the lab computer.",
          "Most of that time went to the PC algorithm and its stability check."], size=12.5, after=6)


def s04_data(prs):
    s = new_slide(prs, "The data",
                  "We used 13 months of one-minute data from 13 units",
                  f"{SRC}/00_data_audit.md; tables/missing_by_unit.csv, dropped_tags.csv",
                  "42 of the 418 expected tags were not in the unit folders. We have not found out why yet. "
                  "After cleaning, very little data is missing, except in two units.")
    bars = [(P["params_expected"], "tags expected", None),
            (P["params_found"], "tags found in the unit folders",
             f"{P['params_expected'] - P['params_found']} tags not found. Reason not known yet."),
            (P["tags_kept"], "tags kept after cleaning",
             f"{P['params_found'] - P['tags_kept']} dropped: 8 had over 30% missing, 13 never changed")]
    for i, (n, lab, note) in enumerate(bars):
        y = 2.15 + i * 0.7
        bw = 6.6 * n / P["params_expected"]
        b = rect(s, ML, y, bw, 0.56, fill=FOREST if i == 2 else SAGE)
        text(s, 0, 0, bw, 0.56, [[(f"{n}  ", {"bold": True, "size": 16}), (lab, {})]], size=13,
             color=IVORY if i == 2 else CHARCOAL, anchor=MSO_ANCHOR.MIDDLE, shape=b)
        b.text_frame.margin_left = I(0.18)
        if note:
            text(s, ML + bw + 0.15, y, 9.6 - (ML + bw + 0.15), 0.56, note, size=11, color=GREY,
                 anchor=MSO_ANCHOR.MIDDLE)

    miss = sorted((r for r in table("missing_by_unit") if float(r["missing_before_pct"]) > 0.1),
                  key=lambda r: -float(r["missing_before_pct"]))
    text(s, ML, 4.45, 8.9, 0.3, "Missing data by unit, before and after cleaning (units above 0.1% only)",
         size=12, bold=True)
    scale = 5.2 / 10.0      # inches per percentage point
    for i, r in enumerate(miss):
        y = 4.9 + i * 0.38
        text(s, ML, y, 1.7, 0.3, r["unit"], size=11)
        for j, (key, col) in enumerate((("missing_before_pct", SAGE), ("missing_after_pct", FOREST))):
            v = float(r[key])
            rect(s, 2.45, y + 0.02 + j * 0.14, max(0.02, v * scale), 0.12, fill=col)
        v0, v1 = float(r["missing_before_pct"]), float(r["missing_after_pct"])
        text(s, 2.55 + max(v0, v1) * scale, y, 1.6, 0.3, f"{v0:.1f}% → {v1:.1f}%", size=10.5, color=GREY)
    ly = 4.95 + len(miss) * 0.38
    for k, (lab, col) in enumerate((("before cleaning", SAGE), ("after cleaning", FOREST))):
        rect(s, 2.45 + k * 1.7, ly + 0.05, 0.18, 0.14, fill=col)
        text(s, 2.7 + k * 1.7, ly, 1.4, 0.26, lab, size=10, color=GREY)

    x, w = 9.9, 2.78
    stat(s, x, 2.1, w, f"{P['rows']:,}", "one-minute rows", nsize=24, lsize=11.5, lh=0.3)
    stat(s, x, 3.15, w, "13 months", "12 Apr 2025 to 21 May 2026", nsize=24, lsize=11.5, lh=0.3)
    stat(s, x, 4.2, w, "1.1 %", "of all values still missing after cleaning. These were filled with the mean.",
         nsize=24, lsize=11.5, lh=0.65)
    stat(s, x, 5.55, w, "4", "flare signals used to build the flare label", nsize=24, lsize=11.5, lh=0.5)


def s05_labels(prs):
    s = new_slide(prs, "The flare label",
                  f"The flare label marks {FLARE_PCT:.0f}% of the time as flaring",
                  f"{SRC}/00_data_audit.md; tables/flare_detection.csv, flare_thresholds.csv, t2_events.csv",
                  f"The label is 'flaring' whenever any of the 4 flare signals is above its median plus 3 MAD, "
                  f"with short gaps joined up. It ends up on for {FLARE_DAYS:.0f} of {SPAN_DAYS:.0f} days. That leaves "
                  "3.8 days to learn what normal looks like. We have not yet found why the short exceedances "
                  "join into windows that last months.")
    ax = TimeAxis(2.4, 10.0)
    rows = [("Flare label", 2.5), ("Training data", 3.45), ("T² alarms", 4.4)]
    for lab, y in rows:
        text(s, ML, y + 0.1, 1.7, 0.3, lab, size=12, bold=True)
        rect(s, ax.x0, y, ax.w, 0.5, fill=MISSED)
    text(s, ax.x0, 2.12, ax.w, 0.28, "Rule: flaring if any of 4 flare signals is above median + 3 × MAD. "
         "Gaps of up to 5 samples are joined.", size=10.5, color=GREY)
    for k, (a, b) in enumerate(FLARE_WINDOWS):
        seg = rect(s, ax.x(a), 2.5, ax.x(b) - ax.x(a), 0.5, fill=CHARCOAL)
        text(s, 0, 0, ax.x(b) - ax.x(a), 0.5, f"Flare window {k + 1}: {_days(a, b):.0f} days", size=11, color=IVORY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=seg)
    for a, b in TRAIN_WINDOWS:
        rect(s, ax.x(a), 3.45, max(0.05, ax.x(b) - ax.x(a)), 0.5, fill=GOLD)
    text(s, ax.x(TRAIN_WINDOWS[1][1]) + 0.15, 3.45, 5.0, 0.5,
         f"{TRAIN_DAYS:.1f} days in total ({100 * P['train_rows'] / P['rows']:.1f}% of the data)",
         size=12.5, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    ew = ax.x(T2_EVENT[1]) - ax.x(T2_EVENT[0])
    ev = rect(s, ax.x(T2_EVENT[0]), 4.4, ew, 0.5, fill=FOREST)
    text(s, 0, 0, ew, 0.5, f"One alarm that lasts {P['event_minutes'] / 1440:.0f} days", size=11, color=IVORY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=ev)
    ax.ticks(s, 4.92)
    takeaway(s, ML, 5.6, CW, 1.15,
             f"When almost every minute counts as flaring, the model learns 'normal' from only {TRAIN_DAYS:.1f} days. "
             "The label needs to be built again before the alarm numbers can mean anything. "
             "We do not know yet why short spikes join into windows that last months.", size=13)


def s06_pca(prs):
    s = new_slide(prs, "PCA",
                  "26 components keep 89% of the variation in the data",
                  f"{SRC}/tables/pca_variance.csv, pca_dominant_units.csv",
                  "We kept 26 components, as planned. They hold 89% of the variation. For each component we "
                  "checked which unit's tags carry most of its weight. NSU_NHT_CCR leads half of them.")
    var = table("pca_variance")[:60]
    x0, y0, pw, ph = 1.3, 2.35, 6.3, 3.7

    def X(k):
        return x0 + (k - 1) / 59 * pw

    def Y(v):
        return y0 + ph - v * ph

    line(s, x0, Y(0), x0 + pw, Y(0), GREY, 1.0)
    line(s, x0, y0, x0, Y(0), GREY, 1.0)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        text(s, x0 - 0.6, Y(v) - 0.11, 0.5, 0.22, f"{v:.0%}", size=10, color=GREY, align=PP_ALIGN.RIGHT)
    for k in (1, 10, 20, 30, 40, 50, 60):
        centered(s, X(k), Y(0) + 0.18, 0.5, 0.22, str(k), size=10, color=GREY)
    text(s, x0, Y(0) + 0.35, pw, 0.26, "Number of components", size=11, color=GREY, align=PP_ALIGN.CENTER)
    text(s, x0 - 0.6, y0 - 0.38, 4.0, 0.26, "Share of variation kept", size=11, color=GREY)
    for v in (0.8, 0.9):
        line(s, x0, Y(v), x0 + pw, Y(v), RULE, 0.75, MSO_LINE_DASH_STYLE.DASH)
    polyline(s, [(X(int(r["PC"])), Y(float(r["cumulative"]))) for r in var], FOREST, 2.25)
    line(s, X(26), y0, X(26), Y(0), GOLD, 2.0)
    oval(s, X(26), Y(P["cum_var"]), 0.08, fill=GOLD)
    text(s, X(26) + 0.12, Y(P["cum_var"]) + 0.12, 2.6, 0.3, f"26 components: {100 * P['cum_var']:.1f}%",
         size=12, bold=True)
    n80, n90, n95, n99 = P["pcs_80_90_95_99"]
    text(s, x0 + 3.2, Y(0.5), 3.0, 0.9,
         [f"80% needs {n80} components", f"90% needs {n90}", f"95% needs {n95}", f"99% needs {n99}"],
         size=11, color=GREY)

    counts = {}
    for r in table("pca_dominant_units"):
        counts[r["dominant_unit"]] = counts.get(r["dominant_unit"], 0) + 1
    x, w = 8.4, 4.28
    text(s, x, 2.15, w, 0.6, "Unit that carries the most weight in each of the 26 components", size=13, bold=True)
    for i, (unit, n) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
        y = 2.95 + i * 0.5
        text(s, x, y, 1.6, 0.3, unit, size=11.5)
        rect(s, x + 1.65, y + 0.04, 2.1 * n / 13, 0.26, fill=FOREST if i == 0 else SAGE)
        text(s, x + 1.75 + 2.1 * n / 13, y, 0.5, 0.3, str(n), size=12, bold=True)
    text(s, x, 6.1, w, 0.7, "Component 1 is led by LOUP, component 2 by NSU_NHT_CCR and component 3 by ISOM.",
         size=11, color=GREY)


def s07_states(prs):
    share, entries, first = pc2_states()
    s = new_slide(prs, "A new finding",
                  "The plant switches between two states, and the training data saw only one",
                  f"{SRC}/tables/fig09_pc_timeseries.csv (30-minute means; shown here as 6-hour means)",
                  "Component 2 sits near 0 or near 800, with almost nothing in between. That looks like two "
                  "operating states. The training data comes from April and early July 2025, before the first "
                  "switch, so it only knows the low state. Please confirm with operations what changes "
                  "between the two states.")
    ax = TimeAxis(1.4, 10.9)
    pts = binned(table("fig09_pc_timeseries"), "PC2", 6, lambda v: sum(v) / len(v))
    lo, hi = -200.0, 1400.0
    y0, ph = 2.35, 3.2

    def Y(v):
        return y0 + ph - (min(max(v, lo), hi) - lo) / (hi - lo) * ph

    ax.shade_train(s, y0, ph)
    for v in (0, 400, 800, 1200):
        line(s, ax.x0, Y(v), ax.x0 + ax.w, Y(v), RULE if v != 400 else GOLD, 0.75,
             MSO_LINE_DASH_STYLE.DASH if v == 400 else None)
        text(s, ax.x0 - 0.65, Y(v) - 0.11, 0.55, 0.22, str(v), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    polyline(s, [(ax.x(t), Y(v)) for t, v in pts], FOREST, 1.25)
    line(s, ax.x0, y0 + ph, ax.x0 + ax.w, y0 + ph, GREY, 1.0)
    ax.ticks(s, y0 + ph)
    text(s, ax.x0 - 0.65, 2.02, 4.0, 0.26, "Score of component 2", size=11, color=GREY)
    text(s, ax.x(TRAIN_WINDOWS[1][1]) + 0.1, 2.02, 3.0, 0.26, "Gold bands: training data", size=10.5,
         color=GREY)
    text(s, ax.x0 + ax.w - 2.6, Y(400) - 0.3, 2.6, 0.26, "High state: above 400", size=10.5, color=GREY,
         align=PP_ALIGN.RIGHT)

    stat(s, ML, 6.05, 3.8, f"{100 * share:.0f} %", "of the time is spent in the high state", nsize=22, lsize=11.5,
         lh=0.3)
    stat(s, ML + 4.08, 6.05, 3.8, str(entries), "times the plant moved into the high state", nsize=22,
         lsize=11.5, lh=0.3)
    stat(s, ML + 8.16, 6.05, 3.8, _ts(first + " 00:00:00").strftime("%d %b %Y").lstrip("0"),
         "first move, a week after the training data ends", nsize=22, lsize=11.5, lh=0.3)


def s08_t2(prs):
    s = new_slide(prs, "T² monitoring",
                  "T² stays above its limit almost all the time",
                  f"{SRC}/tables/fig15_t2_chart.csv, control_limits.csv, detection_metrics.csv "
                  "(shown here as 6-hour maxima)",
                  "T² measures how far the plant is from the training data. The limit is 46. On test data it is "
                  "above the limit 99.4% of the time. The highest plateaus, around 100 million, line up with the "
                  "high state on the previous slide. Even on the training data it crosses the limit 9% of the "
                  "time, where 1% is expected.")
    ax = TimeAxis(1.4, 10.9)
    pts = binned(table("fig15_t2_chart"), "T2", 6, max)
    y0, ph = 2.35, 3.2

    def Y(v):
        return y0 + ph - math.log10(max(v, 1.0)) / 8 * ph

    ax.shade_train(s, y0, ph)
    for k, lab in enumerate(("1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M")):
        line(s, ax.x0, Y(10 ** k), ax.x0 + ax.w, Y(10 ** k), RULE, 0.5)
        text(s, ax.x0 - 0.65, Y(10 ** k) - 0.11, 0.55, 0.22, lab, size=9.5, color=GREY, align=PP_ALIGN.RIGHT)
    polyline(s, [(ax.x(t), Y(v)) for t, v in pts], CHARCOAL, 1.0)
    line(s, ax.x0, Y(P["T2_99"]), ax.x0 + ax.w, Y(P["T2_99"]), GOLD, 2.0)
    text(s, ax.x0 + ax.w - 3.0, Y(P["T2_99"]) + 0.04, 3.0, 0.26, f"99% limit = {P['T2_99']:.0f}", size=11,
         bold=True, align=PP_ALIGN.RIGHT)
    line(s, ax.x0, y0 + ph, ax.x0 + ax.w, y0 + ph, GREY, 1.0)
    ax.ticks(s, y0 + ph)
    text(s, ax.x0 - 0.65, 2.02, 6.0, 0.26, "Hotelling T², log scale. Gold bands: training data", size=11,
         color=GREY)

    stat(s, ML, 6.05, 3.8, f"{100 * P['test_alarm_rate_T2_99']:.1f} %", "of test minutes are above the limit",
         nsize=22, lsize=11.5, lh=0.3)
    stat(s, ML + 4.08, 6.05, 3.8, f"{100 * P['far_train_T2_99']:.1f} %",
         "of training minutes are above it too (1% expected)", nsize=22, lsize=11.5, lh=0.3)
    stat(s, ML + 8.16, 6.05, 3.8, f"{100 * P['test_alarm_rate_Q_99']:.0f} %",
         "of test minutes are above the Q limit", nsize=22, lsize=11.5, lh=0.3)


def s09_tag(prs):
    s = new_slide(prs, "What drives the alarm",
                  "One tag explains almost all of the biggest alarm",
                  f"{SRC}/tables/event_peak_tag_contrib.csv, audit_tags.csv, t2_events.csv",
                  "We split the T² value at its peak into the share from each tag. One LOUP tag accounts for "
                  "almost all of it. The tag ranges from 0.09 to 169 and looks like an outlier in 40% of samples. "
                  "Our guess is that it stayed at one level in the training data, so any move looks huge. "
                  "That guess is not checked yet.")
    rows = table("event_peak_tag_contrib")[:10]
    vmax = float(rows[0]["contrib_signed"])
    text(s, ML, 2.1, 7.6, 0.3, f"Share of the peak T² value ({P['event_peak'] / 1e6:.1f} million) by tag, top 10",
         size=12, bold=True)
    for i, r in enumerate(rows):
        y = 2.6 + i * 0.4
        v = float(r["contrib_signed"])
        text(s, ML, y, 2.3, 0.3, r["column"].split(".")[0], size=11, bold=i == 0)
        text(s, ML + 2.3, y, 1.3, 0.3, r["unit"], size=10.5, color=GREY)
        rect(s, 4.3, y + 0.05, max(0.02, 4.0 * v / vmax), 0.22, fill=GOLD if i == 0 else SAGE)
        lab = f"{v / 1e6:.1f} million" if v >= 1e6 else f"{v:,.0f}"
        text(s, 4.4 + max(0.02, 4.0 * v / vmax), y, 1.5, 0.3, lab, size=10.5, bold=i == 0)
    x, w = 9.6, 3.08
    stat(s, x, 2.15, w, f"≈{100 * P['tag_peak_contrib'] / P['event_peak']:.0f} %",
         "of the peak comes from this one tag", color=GOLD, nsize=30, lsize=11.5, lh=0.3)
    stat(s, x, 3.35, w, f"{P['tag_min']:g} to {P['tag_max']:.0f}", "range of the tag over the year", nsize=24,
         lsize=11.5, lh=0.3)
    stat(s, x, 4.45, w, f"{100 * P['tag_outlier_share']:.0f} %", "of its samples look like outliers",
         nsize=24, lsize=11.5, lh=0.3)
    takeaway(s, x, 5.6, w, 1.2, "Check this tag with operations: a sensor fault or a real change.", size=12)


def s10_scores(prs):
    s = new_slide(prs, "Detection scores",
                  "The detection scores look good, but they do not mean much yet",
                  f"{SRC}/tables/detection_metrics.csv, flare_detection.csv; RESULTS.md",
                  "These are the scores the run reported. Each one is a side effect of the flare label covering "
                  "almost all the time, so none of them tells us how well T² detects flares.")
    lead_h = -P["median_detection_delay_min"] / 60
    pairs = [("2 of 2", "flares detected", "The 2 'flares' are windows of 82 and 318 days. Any alarm will hit them."),
             (f"{lead_h:.1f} h", "median warning before a flare starts",
              f"The alarm is on {100 * P['test_alarm_rate_T2_99']:.1f}% of the time, so it is always early."),
             (f"{P['auc_T2']:.2f}", "AUC for T²",
              f"Only {NON_FLARE_TEST} non-flare minutes are left to score against, "
              f"and {100 * P['sample_false_alarm_test_normal_T2_99']:.0f}% of them raise an alarm.")]
    text(s, ML, 2.15, 4.6, 0.3, "What the run reported", size=13, bold=True, color=GREY)
    text(s, 6.2, 2.15, 6.48, 0.3, "Why it does not hold", size=13, bold=True, color=FOREST)
    for i, (big, lab, why) in enumerate(pairs):
        y = 2.7 + i * 1.3
        rect(s, ML, y, 4.6, 1.1, fill=PALE)
        text(s, ML + 0.2, y + 0.08, 4.2, 0.55, big, size=28, color=FOREST, font=TITLE_FONT)
        text(s, ML + 0.2, y + 0.66, 4.2, 0.35, lab, size=12, color=GREY)
        line(s, 5.4, y + 0.55, 6.0, y + 0.55, GREY, 1.5, arrow=True)
        text(s, 6.2, y, 6.48, 1.1, why, size=14, anchor=MSO_ANCHOR.MIDDLE)


def s11_pc(prs):
    s = new_slide(prs, "PC algorithm",
                  "The PC algorithm found no links on the training data, as expected",
                  f"{SRC}/tables/pc_variants_summary.csv, fig14_pc_sensitivity.csv; RESULTS.md section 3",
                  "PCA makes its components uncorrelated on the data it was fitted to. So the PC algorithm sees "
                  "no links there, which the plan predicted. We then ran four other versions. The one on test "
                  "data became the final graph. Its links mostly reflect how far the test data has drifted from "
                  "training, not cause and effect.")
    rows = table("pc_variants_summary")
    edges = {(r["variant"], r["alpha"]): int(r["edges"]) for r in rows}
    groups = [("T_train_fisherz", "Training data (planned)"),
              ("T_test_fisherz", "Test data (final graph)"),
              ("flare_only_fisherz", "Flare rows only"),
              ("T_train_kci", "Nonlinear test, 300 training rows"),
              ("T_train_lagged_fisherz", "With time lags (check only)")]
    scale = 5.0 / 90
    for i, (key, lab) in enumerate(groups):
        y = 2.35 + i * 0.75
        text(s, ML, y + 0.05, 3.3, 0.5, lab, size=12, bold=key == "T_test_fisherz")
        for j, (alpha, col) in enumerate((("0.05", FOREST), ("0.01", SAGE))):
            n = edges.get((key, alpha))
            yy = y + j * 0.28
            if n is None:
                text(s, 4.1, yy, 2.0, 0.26, "not run", size=10, color=GREY)
                continue
            rect(s, 4.0, yy + 0.02, max(0.03, n * scale), 0.22, fill=col)
            text(s, 4.1 + max(0.03, n * scale), yy - 0.02, 0.6, 0.26, str(n), size=11, bold=True)
    for k, (lab, col) in enumerate((("α = 0.05", FOREST), ("α = 0.01", SAGE))):
        rect(s, 4.0 + k * 1.3, 6.2, 0.18, 0.14, fill=col)
        text(s, 4.25 + k * 1.3, 6.14, 1.0, 0.26, lab, size=10.5, color=GREY)
    text(s, ML, 6.14, 3.3, 0.26, "Number of links found", size=10.5, color=GREY)

    x, w = 9.6, 3.08
    text(s, x, 2.2, w, 1.4, "Training data gives 0 links because PCA makes the components uncorrelated there.",
         size=12.5)
    stat(s, x, 3.55, w, f"{P['edges_stable']} of {P['edges_test']}",
         f"links in the final graph appear in at least 70% of {P['bootstrap_B']} resamples", nsize=24,
         lsize=11.5, lh=0.55)
    stat(s, x, 4.9, w, f"{P['jaccard_alpha']:.2f}", "overlap between the graphs at α = 0.05 and α = 0.01",
         nsize=24, lsize=11.5, lh=0.55)


UNIT_ORDER = ["ISOM", "NSU_NHT_CCR", "LOUP", "PDA", "PG", "SEU1", "SEU2", "SEU3",
              "APS_NVPS_LEU", "DAK", "DIU", "ISOM_3", "ISOM_4"]


def s12_graph(prs):
    s = new_slide(prs, "Links between units",
                  "Most links connect NSU_NHT_CCR, ISOM and LOUP, but no root cause is found yet",
                  f"{SRC}/tables/fig13_unit_level_graph.csv, root_cause_hypotheses.csv, pc_edges_stable.csv",
                  "We turned the links between components into links between units, using how much each unit "
                  "weighs in each component. The strongest links run between NSU_NHT_CCR, ISOM and LOUP, in both "
                  "directions. The trace step found no upstream component for the alarm, so it names no "
                  "root-cause unit.")
    links = sorted((r for r in table("fig13_unit_level_graph") if r["from_unit"] != r["to_unit"]),
                   key=lambda r: -float(r["weight"]))[:8]
    used = {r["from_unit"] for r in links} | {r["to_unit"] for r in links}
    # hand layout for the linked units; any other linked unit falls back to the bottom row
    spots = {"NSU_NHT_CCR": (3.9, 4.35), "ISOM": (3.9, 2.55), "LOUP": (6.75, 3.45),
             "PDA": (1.35, 3.45), "PG": (3.9, 6.15)}
    spare = iter([(1.35, 5.6), (6.75, 5.6), (1.35, 6.4), (6.75, 6.4)])
    pos = {u: spots[u] if u in spots else next(spare) for u in UNIT_ORDER if u in used}
    hw, hh = 0.8, 0.21
    pairs = {(r["from_unit"], r["to_unit"]) for r in links}
    for r in links:
        a, b = r["from_unit"], r["to_unit"]
        (x1, y1), (x2, y2) = pos[a], pos[b]
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        off = 0.06 if (b, a) in pairs else 0.0
        px, py = -uy * off, ux * off
        cut = min(hw / max(abs(ux), 1e-6), hh / max(abs(uy), 1e-6)) + 0.05
        line(s, x1 + ux * cut + px, y1 + uy * cut + py, x2 - ux * cut + px, y2 - uy * cut + py,
             FOREST, 1.0 + 3.5 * float(r["weight"]), arrow=True)
    for u, (x, y) in pos.items():
        n = rect(s, x - hw, y - hh, 2 * hw, 2 * hh, fill=FOREST, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        text(s, 0, 0, 2 * hw, 2 * hh, u, size=11, bold=True, color=IVORY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=n)
    others = [u for u in UNIT_ORDER if u not in used]
    text(s, ML, 6.55, 7.0, 0.3, f"Top {len(links)} links shown, line width = weight. No strong links for: "
         + ", ".join(others) + ".", size=10, color=GREY)

    x, w = 8.0, 4.68
    text(s, x, 2.15, w, 0.3, "Strongest links (weight from 0 to 1)", size=13, bold=True)
    for i, r in enumerate(links[:5]):
        y = 2.6 + i * 0.36
        text(s, x, y, 3.8, 0.3, f"{r['from_unit']} → {r['to_unit']}", size=12)
        text(s, x + 3.8, y, 0.88, 0.3, f"{float(r['weight']):.2f}", size=12, bold=True, align=PP_ALIGN.RIGHT)
    hyp = table("root_cause_hypotheses")[0]
    text(s, x, 4.6, w, 0.3, "Root-cause trace for the alarm", size=13, bold=True)
    text(s, x, 4.98, w, 1.0,
         f"At the alarm peak, component {hyp['elevated_pc'][2:]} rises most. It is led by {hyp['elevated_pc_unit']}. "
         "No component points into it, so no root-cause unit can be named.", size=12)
    text(s, x, 6.05, w, 0.8, "These links join components, not tags, and reflect drift as much as cause.",
         size=11, color=GREY)


def s13_next(prs):
    s = new_slide(prs, "Next steps",
                  "We need three fixes before the next run, and two more after it",
                  f"{SRC}/RESULTS.md (limitations and suggested next step)",
                  "Fixes 1 to 3 decide whether the next run can be read at all. Fix 4 is a short talk with the data "
                  "owner. Fix 5 is the next step that RESULTS.md suggests, and it makes sense only after 1 to 3.")
    cards = [
        ("1", "Rebuild the flare labels", "One event list per signal, checked against the operator log"),
        ("2", "Rebuild the normal baseline", "Take months of non-flaring data that cover both operating states"),
        ("3", f"Check tag {P['tag_peak'].split('.')[0]}", "Find out if it is a sensor fault or a real change"),
        ("4", "Find the 42 missing tags", "Confirm 418 against 376 with the data owner"),
        ("5", "Find links between tags", "Use a method that handles time lags, such as PCMCI+, on flare windows"),
    ]
    w = (CW - 4 * 0.2) / 5
    for i, (n, title, detail) in enumerate(cards):
        x = ML + i * (w + 0.2)
        first = i < 3
        rect(s, x, 2.3, w, 3.2, fill=FOREST if first else PALE)
        ink, sub = (IVORY, SAGE_PALE) if first else (CHARCOAL, GREY)
        text(s, x + 0.2, 2.45, w - 0.4, 0.6, n, size=30, color=sub, font=TITLE_FONT)
        text(s, x + 0.2, 3.15, w - 0.4, 0.9, title, size=15, bold=True, color=ink)
        text(s, x + 0.2, 4.1, w - 0.4, 1.3, detail, size=12, color=ink)
    text(s, ML, 5.65, 3 * w + 0.4, 0.3, "Needed before the next run", size=11, bold=True, color=FOREST,
         align=PP_ALIGN.CENTER)
    text(s, ML + 3 * (w + 0.2), 5.65, 2 * w + 0.2, 0.3, "After that", size=11, bold=True, color=GREY,
         align=PP_ALIGN.CENTER)
    bar = rect(s, ML, 6.15, CW, 0.65, fill=FOREST)
    text(s, 0, 0, CW, 0.65, "Decision needed: approve fixes 1 to 3 so the next run gives results we can use.",
         size=15, color=IVORY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=bar)


SLIDES = [s01_title, s02_summary, s03_pipeline, s04_data, s05_labels, s06_pca, s07_states, s08_t2,
          s09_tag, s10_scores, s11_pc, s12_graph, s13_next]


def bounds_check(prs):
    sw, sh = prs.slide_width, prs.slide_height
    tol = Inches(0.01)
    for n, slide in enumerate(prs.slides, start=1):
        for shp in slide.shapes:
            if (shp.left < -tol or shp.top < -tol or shp.left + shp.width > sw + tol
                    or shp.top + shp.height > sh + tol):
                WARNINGS.append(f"slide {n}: shape '{shp.name}' extends past the slide edge")


def main():
    global TITLE_FONT, BODY_FONT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (font_installed("SourceSerif4") and font_installed("SourceSans3")):
        TITLE_FONT, BODY_FONT = FALLBACK_FONTS
        print(f"fonts: Source Serif 4 / Source Sans 3 not found -> using {TITLE_FONT} / {BODY_FONT}")

    print("verify: comparing the deck's numbers with the result files")
    fails = verify()
    if fails and not args.force:
        print(f"STOP: {fails} number(s) differ from the files. Fix the DATA section or rerun with --force.")
        return 1

    prs = Presentation()
    prs.slide_width, prs.slide_height = I(W), I(H)
    for build in SLIDES:
        build(prs)
    bounds_check(prs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"saved {args.out} ({len(prs.slides)} slides)")
    if WARNINGS:
        print(f"layout: {len(WARNINGS)} warning(s) - check these slides by eye:")
        for w in WARNINGS:
            print("  " + w)
    else:
        print("layout: no overflow or off-slide warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
