"""One function per slide.

Each slide picks its device from the shape of its own argument: small-multiple
cards for a comparison of kinds, a dot matrix for a cross-product, ranked bars
for magnitudes, a scatter against a frontier for a two-variable trade-off, a
log scale where the quantities span three orders. Only one slide uses a ruled
grid, and only because its data genuinely is a grid.

Every number rendered here comes from content.py, which carries its provenance.
"""

import math

from pptx.enum.text import PP_ALIGN

import content as C
import layout as L
import theme as T

TONE = {
    "degenerate": T.AMBER,
    "neutral": T.MUTED,
    "best": T.TEAL,
    "failed": T.RGBColor(0x9A, 0x3B, 0x3B),
    "slate": T.SLATE,
    "teal": T.TEAL,
}


def _notes(slide, number):
    if number in C.NOTES:
        slide.notes_slide.notes_text_frame.text = C.NOTES[number]


# ---------------------------------------------------------------------------
# 1 - title
# ---------------------------------------------------------------------------

def slide_01_title(prs):
    s = T.blank(prs, dark=True)
    T.rect(s, 0.95, 1.95, 0.045, 2.20, fill=T.TEAL)
    T.text(s, 1.25, 1.92, T.col_w(11), 1.40, C.TITLE.split("\n"), size=36,
           color=T.ON_DARK, bold=True, spacing=1.06)
    T.text(s, 1.25, 3.36, T.col_w(10), 0.40, C.SUBTITLE, size=17,
           color=T.RGBColor(0x9A, 0xA2, 0xAA), spacing=1.2)
    T.line(s, 1.25, 3.92, 5.6, 3.92, color=T.RGBColor(0x36, 0x3D, 0x45), pt=0.75)
    T.text(s, 1.25, 4.10, T.col_w(7), 0.26, C.TITLE_META, size=T.SZ_CAPTION,
           color=T.RGBColor(0x7E, 0x87, 0x90))
    T.text(s, 1.25, 4.64, T.col_w(6), 0.9, C.TITLE_CLAIM, size=T.SZ_SMALL,
           color=T.RGBColor(0xB6, 0xBD, 0xC4), spacing=1.34)

    # Chips sit under the claim, on the same left edge as everything else -
    # a right-aligned cluster reads as a second, unrelated column.
    x = 1.25
    for label in C.TITLE_CHIPS:
        w = 0.098 * len(label) + 0.40
        T.pill(s, x, 5.86, w, 0.36, label, fill=None,
               outline=T.RGBColor(0x3A, 0x44, 0x4C),
               color=T.RGBColor(0x8E, 0x97, 0xA0), size=T.SZ_MICRO, bold=False)
        x += w + 0.16
    _notes(s, 1)
    return s


# ---------------------------------------------------------------------------
# 2 - objective and scope
# ---------------------------------------------------------------------------

def slide_02_objective(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_FRAME, 2,
                  "What we set out to do, and the 16 runs we did it with",
                  C.OBJECTIVE)

    L.eyebrow(s, T.MARGIN_IN, y + 0.10, "How the work was sequenced")
    n = len(C.PIPELINE)
    gap = 0.46
    w = (T.col_w(12) - (n - 1) * gap) / n
    top, h = y + 0.42, 1.14
    for i, (head, body) in enumerate(C.PIPELINE):
        x = T.MARGIN_IN + i * (w + gap)
        accent = T.TEAL if i < 2 else T.SLATE
        T.rect(s, x, top, w, h, fill=T.PAPER_DIM if i < 2 else None,
               outline=T.RULE, pt=0.75)
        T.rect(s, x, top, 0.035, h, fill=accent)
        T.text(s, x + 0.22, top + 0.18, w - 0.38, 0.28, head, size=T.SZ_BODY,
               color=T.INK, bold=True)
        T.text(s, x + 0.22, top + 0.52, w - 0.38, 0.56, body.split("\n"),
               size=T.SZ_MICRO, color=T.MUTED, spacing=1.26)
        if i < n - 1:
            T.line(s, x + w + 0.10, top + h / 2, x + w + gap - 0.10, top + h / 2,
                   color=T.RULE, pt=1.0, arrow=True)

    sy = top + h + 0.46
    L.eyebrow(s, T.MARGIN_IN, sy, "The scope of this deck, in four numbers")
    sw = T.col_w(12) / 4
    for i, (value, label, gloss) in enumerate(C.SCOPE):
        x = T.MARGIN_IN + i * sw
        if i:
            T.line(s, x - 0.22, sy + 0.30, x - 0.22, sy + 1.42, color=T.RULE,
                   pt=0.5)
        L.stat(s, x, sy + 0.30, value, label, gloss, w=sw - 0.30,
               accent=T.TEAL if i < 2 else T.INK, value_size=27)

    L.takeaway(s, 5.82, C.SCORING_NOTE, accent=T.RULE)
    L.footer(s, [8], C.REFS)
    _notes(s, 2)
    return s


# ---------------------------------------------------------------------------
# 3 - the four methods, as small multiples
# ---------------------------------------------------------------------------

def slide_03_methods(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_SOTA, 3, accent=T.SLATE,
                  title="Four methods that disagree about what the data must look like",
                  kicker="Published assumptions, not our results. The four "
                         "Assumes lines do not overlap, which is why the "
                         "exploratory analysis has to come first.")

    gap_x, gap_y = 0.34, 0.22
    w = (T.col_w(12) - gap_x) / 2
    h = 2.16
    for i, (name, role, tone, rows) in enumerate(C.METHOD_CARDS):
        col, row = i % 2, i // 2
        x = T.MARGIN_IN + col * (w + gap_x)
        cy = y + 0.06 + row * (h + gap_y)
        accent = TONE[tone]
        ix, iw = L.card(s, x, cy, w, h, accent, name, role,
                        fill=T.PAPER_DIM if tone == "slate" else None)
        ry = cy + 0.86
        for label, value in rows:
            L.keyline(s, ix, ry, iw, label, value, accent=accent,
                      label_w=1.05)
            ry += 0.32

    L.footer(s, [1, 2, 3, 4, 5, 6, 7, 9], C.REFS, bottom=7.34)
    _notes(s, 3)
    return s


# ---------------------------------------------------------------------------
# 4 - the assumption matrix. The pivot slide, and the one real grid.
# ---------------------------------------------------------------------------

_MARK_X, _MARK_OK, _MARK_NA = "x", "ok", "-"


def _mark(slide, cx, cy, kind, d=0.22):
    """One cell of the assumption matrix.

    A filled amber disc is a violated assumption, an open teal ring a satisfied
    one, a hairline an assumption the method does not make. Fill weight carries
    the meaning, so the matrix reads as a texture before it is read as data.
    """
    if kind == _MARK_X:
        T.node(slide, cx, cy, d, "", fill=T.AMBER, outline=T.AMBER, pt=0.75)
    elif kind == _MARK_OK:
        T.node(slide, cx, cy, d, "", fill=T.PAPER, outline=T.TEAL, pt=1.75)
    else:
        T.line(slide, cx - 0.09, cy, cx + 0.09, cy, color=T.RULE, pt=1.5)


def slide_04_eda_matrix(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_EDA, 4, accent=T.AMBER,
                  title="What the EDA measured: every assumption but one is violated",
                  kicker=C.EDA_KICKER)

    name_x, name_w = T.MARGIN_IN, 2.55
    det_x, det_w = 3.35, 4.30
    col0, colw = 8.20, 1.12
    top, rh = y + 0.70, 0.46

    L.eyebrow(s, name_x, top - 0.34, "Assumption", w=name_w)
    L.eyebrow(s, det_x, top - 0.34, "What we measured on this dataset", w=det_w)
    for j, method in enumerate(C.EDA_METHODS):
        T.text(s, col0 + j * colw - colw / 2, top - 0.38, colw, 0.26, method,
               size=T.SZ_CAPTION, color=T.INK, bold=True, align=PP_ALIGN.CENTER)
    T.line(s, name_x, top - 0.08, T.SLIDE_W_IN - T.MARGIN_IN, top - 0.08,
           color=T.INK, pt=1.0)

    # Legend, on the title's own line so it does not crowd the column heads.
    lx = name_x
    for label, kind in (("violated", _MARK_X), ("satisfied", _MARK_OK),
                        ("not assumed", _MARK_NA)):
        _mark(s, lx + 0.10, y + 0.14, kind, d=0.17)
        T.text(s, lx + 0.28, y + 0.04, 1.30, 0.22, label, size=T.SZ_MICRO,
               color=T.MUTED)
        lx += 1.42

    for i, (name, detail, marks) in enumerate(C.EDA_MATRIX):
        ry = top + i * rh
        cy = ry + rh / 2 - 0.04
        satisfied = _MARK_X not in marks
        T.text(s, name_x, ry + 0.06, name_w, 0.28, name, size=T.SZ_SMALL,
               color=T.TEAL if satisfied else T.INK, bold=True)
        T.text(s, det_x, ry + 0.09, det_w, 0.28, detail, size=T.SZ_MICRO,
               color=T.MUTED)
        for j, kind in enumerate(marks):
            _mark(s, col0 + j * colw, cy, kind)
        if i < len(C.EDA_MATRIX) - 1:
            T.line(s, name_x, ry + rh, T.SLIDE_W_IN - T.MARGIN_IN, ry + rh,
                   color=T.PAPER_DIM, pt=0.5)

    vy = top + len(C.EDA_MATRIX) * rh + 0.06
    T.line(s, name_x, vy, T.SLIDE_W_IN - T.MARGIN_IN, vy, color=T.INK, pt=1.0)
    T.text(s, name_x, vy + 0.14, name_w, 0.28, "EDA verdict", size=T.SZ_SMALL,
           color=T.INK, bold=True)
    T.text(s, det_x, vy + 0.17, det_w, 0.28,
           "the EDA's own outlook, recorded before any run", size=T.SZ_MICRO,
           color=T.MUTED)
    for j, (_, verdict) in enumerate(C.EDA_VERDICT):
        good = verdict.startswith("works")
        T.text(s, col0 + j * colw - colw / 2, vy + 0.12, colw, 0.48,
               verdict.split("\n"), size=T.SZ_MICRO,
               color=T.TEAL if good else T.AMBER, bold=True,
               align=PP_ALIGN.CENTER, spacing=1.12)

    L.takeaway(s, 6.52, C.EDA_NOTE, accent=T.AMBER)
    _notes(s, 4)
    return s


# ---------------------------------------------------------------------------
# 5 - experimental setup
# ---------------------------------------------------------------------------

def slide_05_setup(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_SETUP, 5, accent=T.TEAL,
                  title="Experimental setup: every parameter read off the EDA, not chosen",
                  kicker="Where a value could have been a preference, the slide "
                         "names the measurement that fixed it instead")

    gap = 0.42
    w = (T.col_w(12) - 2 * gap) / 3
    bottom = y + 0.10
    for i, (group, rows) in enumerate(C.SETUP_GROUPS):
        x = T.MARGIN_IN + i * (w + gap)
        if i:
            T.line(s, x - gap / 2, y + 0.06, x - gap / 2, 5.86, color=T.RULE,
                   pt=0.5)
        L.eyebrow(s, x, y + 0.10, group, color=T.MUTED, w=w)
        ry = y + 0.46
        for label, value in rows:
            T.text(s, x, ry, w, 0.26, label, size=T.SZ_SMALL, color=T.INK,
                   bold=True)
            lines = value.split("\n")
            T.text(s, x, ry + 0.27, w, 0.26 * len(lines), lines,
                   size=T.SZ_MICRO, color=T.MUTED, spacing=1.28)
            ry += 0.26 + 0.215 * len(lines) + 0.16
        bottom = max(bottom, ry)

    L.takeaway(s, 6.14,
               "The prior-knowledge rows are the one place a result could "
               "flatter itself, so they are labelled as an oracle everywhere "
               "they appear - including on the results chart.",
               accent=T.RULE)
    L.footer(s, [5, 6, 8, 9, 10], C.REFS)
    _notes(s, 5)
    return s


# ---------------------------------------------------------------------------
# 6 - results, two panels
# ---------------------------------------------------------------------------

def slide_06_results(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_RESULT, 6, accent=T.TEAL,
                  title="Measured results: the SHD ranking and recovery ranking disagree",
                  kicker=C.RESULT_KICKER)

    # ---- left panel: SHD, ranked bars against the empty-graph baseline ------
    px, pw = T.MARGIN_IN, 7.05
    label_w = 1.72
    bx = px + label_w
    bw = pw - label_w - 0.68
    L.eyebrow(s, px, y + 0.10, "Structural Hamming distance  -  lower is better",
              w=4.9)

    top, rh, bar_h = y + 0.50, 0.80, 0.32
    scale = bw / C.RESULT_AXIS_MAX
    axis_y = top + len(C.RESULT_BARS) * rh - 0.18

    # No vertical gridlines. Every bar carries its own value, so a grid would
    # only put rules through the annotation under each bar.
    T.line(s, bx, axis_y, bx + bw, axis_y, color=T.RULE, pt=0.75)
    for tick in (0, 50, 100, 150, 200):
        tx = bx + tick * scale
        T.line(s, tx, axis_y, tx, axis_y + 0.06, color=T.RULE, pt=0.75)
        T.text(s, tx - 0.32, axis_y + 0.09, 0.64, 0.22, str(tick),
               size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.CENTER)

    # The baseline is a mark on the axis rather than a full-height rule, for
    # the same reason.
    ref_x = bx + C.EMPTY_GRAPH_SHD * scale
    T.line(s, ref_x, axis_y - 0.10, ref_x, axis_y + 0.10, color=T.AMBER, pt=1.5)
    T.text(s, ref_x + 0.10, axis_y + 0.32, 2.6, 0.22,
           "empty-graph baseline = 30.0", size=T.SZ_MICRO, color=T.AMBER,
           bold=True)

    for i, (label, shd, rng, tp, fp, fn, note, tone) in enumerate(C.RESULT_BARS):
        ry = top + i * rh
        colour = TONE[tone]
        head, _, sub = label.partition("  ")
        T.text(s, px, ry + 0.01, label_w, 0.26, head, size=T.SZ_SMALL,
               color=T.INK, bold=True)
        T.text(s, px, ry + 0.26, label_w, 0.22, sub, size=T.SZ_MICRO,
               color=T.MUTED)
        T.rect(s, bx, ry, max(shd * scale, 0.02), bar_h, fill=colour)
        if rng and (rng[1] - rng[0]) * scale > 0.10:
            lo, hi = rng
            wy = ry + bar_h / 2
            T.line(s, bx + lo * scale, wy, bx + hi * scale, wy, color=T.INK,
                   pt=1.0)
            for v in (lo, hi):
                T.line(s, bx + v * scale, wy - 0.08, bx + v * scale, wy + 0.08,
                       color=T.INK, pt=1.0)
        end = max(shd, rng[1] if rng else shd) * scale
        T.text(s, bx + end + 0.12, ry - 0.02, 1.1, 0.26, "%.1f" % shd,
               size=T.SZ_BODY, color=T.INK, bold=True)
        T.text(s, bx, ry + bar_h + 0.07, bw + 0.7, 0.24,
               "%s   |   TP %d  FP %d  FN %d" % (note, tp, fp, fn),
               size=T.SZ_MICRO, color=T.MUTED)

    T.text(s, px, axis_y + 0.62, 4.0, 0.22,
           "whiskers span tau = 1, 2, 3", size=T.SZ_MICRO, color=T.MUTED)

    # ---- right panel: recovery, the ordering that actually matters ----------
    qx = 8.30
    qw = T.SLIDE_W_IN - T.MARGIN_IN - qx
    T.line(s, qx - 0.42, y + 0.06, qx - 0.42, 6.10, color=T.RULE, pt=0.5)
    L.eyebrow(s, qx, y + 0.10, "True positives, of 30 recoverable edges", w=qw)

    dtop, drh = y + 0.68, 0.86
    dw = qw - 0.62
    for i, (name, tp, gloss) in enumerate(C.RECALL_DOTS):
        ry = dtop + i * drh
        T.line(s, qx, ry + 0.12, qx + dw, ry + 0.12, color=T.PAPER_DIM, pt=3.5)
        colour = T.TEAL if tp else TONE["failed"]
        if tp:
            T.line(s, qx, ry + 0.12, qx + dw * tp / C.RECALL_DENOM, ry + 0.12,
                   color=colour, pt=3.5)
            T.node(s, qx + dw * tp / C.RECALL_DENOM, ry + 0.12, 0.18, "",
                   fill=colour, outline=colour, pt=0.5)
        else:
            T.node(s, qx, ry + 0.12, 0.18, "", fill=T.PAPER, outline=colour,
                   pt=1.5)
        T.text(s, qx, ry - 0.26, dw, 0.24, name, size=T.SZ_SMALL, color=T.INK,
               bold=True)
        T.text(s, qx + dw + 0.12, ry - 0.02, 0.6, 0.26, str(tp),
               size=T.SZ_BODY, color=colour, bold=True)
        T.text(s, qx, ry + 0.28, dw + 0.55, 0.24, gloss, size=T.SZ_MICRO,
               color=T.MUTED)
    T.text(s, qx, dtop + len(C.RECALL_DOTS) * drh - 0.06, dw + 0.55, 0.26,
           "full bar = all 30 edges recovered", size=T.SZ_MICRO, color=T.MUTED)

    L.takeaway(s, 6.34, C.RESULT_NOTE, accent=T.TEAL)
    _notes(s, 6)
    return s


# ---------------------------------------------------------------------------
# 7 - predictions against outcomes, as a scatter against the frontier
# ---------------------------------------------------------------------------

def slide_07_analysis(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_ANALYSIS, 7, accent=T.AMBER,
                  title="Predicting more edges is not recovering more edges",
                  kicker=C.ANALYSIS_KICKER)

    # ---- the plot -----------------------------------------------------------
    ax, ay = 1.55, y + 0.66
    aw, ah = 6.20, 3.34
    L.eyebrow(s, T.MARGIN_IN, y + 0.10,
              "Edges predicted against edges recovered", w=5.0)

    def px_(e):
        return ax + aw * e / C.FRONTIER_MAX_EDGES

    def py_(tp):
        return ay + ah - ah * tp / C.FRONTIER_MAX_TP

    T.line(s, ax, ay, ax, ay + ah, color=T.RULE, pt=0.75)
    T.line(s, ax, ay + ah, ax + aw, ay + ah, color=T.RULE, pt=0.75)
    for tp in (0, 10, 20, 30):
        gy = py_(tp)
        if tp:
            T.line(s, ax, gy, ax + aw, gy, color=T.PAPER_DIM, pt=0.5)
        T.text(s, ax - 0.52, gy - 0.11, 0.42, 0.22, str(tp), size=T.SZ_MICRO,
               color=T.MUTED, align=PP_ALIGN.RIGHT)
    for e in (0, 50, 100, 150, 200):
        T.text(s, px_(e) - 0.32, ay + ah + 0.08, 0.64, 0.22, str(e),
               size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.CENTER)
    T.text(s, ax - 0.52, ay - 0.26, 2.6, 0.22, "true positives",
           size=T.SZ_MICRO, color=T.MUTED, bold=True)
    T.text(s, ax + aw - 2.6, ay + ah + 0.32, 2.6, 0.22, "edges predicted",
           size=T.SZ_MICRO, color=T.MUTED, bold=True, align=PP_ALIGN.RIGHT)

    # The ceiling: a method predicting e edges cannot score more than
    # min(e, 30) true positives. Nothing can sit above this line.
    T.line(s, px_(0), py_(0), px_(30), py_(30), color=T.SLATE, pt=1.25,
           dash=T.DASH)
    T.line(s, px_(30), py_(30), px_(C.FRONTIER_MAX_EDGES), py_(30),
           color=T.SLATE, pt=1.25, dash=T.DASH)
    T.text(s, px_(58), py_(30) - 0.34, 3.4, 0.24,
           "ceiling: true positives <= min(edges, 30)",
           size=T.SZ_MICRO, color=T.SLATE, bold=True)

    label, e_ideal, tp_ideal = C.SCATTER_IDEAL
    T.node(s, px_(e_ideal), py_(tp_ideal), 0.20, "", fill=T.PAPER,
           outline=T.SLATE, pt=2.0)
    T.text(s, px_(e_ideal) + 0.18, py_(tp_ideal) - 0.02, 1.9, 0.24,
           "%s: 30 of 30" % label, size=T.SZ_MICRO, color=T.SLATE, bold=True)

    for name, edges, tp, tone, (dx, dy) in C.SCATTER_POINTS:
        colour = TONE[tone]
        cx, cy = px_(edges), py_(tp)
        T.node(s, cx, cy, 0.20, "", fill=colour, outline=colour, pt=0.5)
        T.text(s, cx + dx, cy + dy, 1.86, 0.24,
               "%s  (%d, %d)" % (name, edges, tp), size=T.SZ_MICRO,
               color=colour, bold=True,
               align=PP_ALIGN.RIGHT if dx < 0 else PP_ALIGN.LEFT)

    # ---- the readings, one short block per point ----------------------------
    nx = 8.42
    nw = T.SLIDE_W_IN - T.MARGIN_IN - nx
    T.line(s, nx - 0.42, y + 0.06, nx - 0.42, 6.10, color=T.RULE, pt=0.5)
    L.eyebrow(s, nx, y + 0.10, "What the EDA said, and what we found", w=nw)
    ny = y + 0.50
    for name, chip, tone, body in C.ANALYSIS_NOTES:
        colour = TONE[tone]
        T.rect(s, nx - 0.16, ny, 0.03, 0.90, fill=colour)
        T.text(s, nx, ny, nw, 0.24, name, size=T.SZ_SMALL, color=T.INK,
               bold=True)
        T.text(s, nx, ny + 0.24, nw, 0.22, chip, size=T.SZ_MICRO,
               color=colour, bold=True)
        T.text(s, nx, ny + 0.48, nw, 0.6, body, size=T.SZ_MICRO, color=T.MUTED,
               spacing=1.26)
        ny += 1.02

    L.takeaway(s, 6.52, C.PK_NOTE, accent=T.AMBER)
    _notes(s, 7)
    return s


# ---------------------------------------------------------------------------
# 8 - cost and coverage
# ---------------------------------------------------------------------------

def slide_08_cost(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_RESULT, 8, accent=T.TEAL,
                  title="Why LSTE and VAR-LiNGAM were not run exhaustively: runtime",
                  kicker=C.COST_KICKER)

    # ---- left: runtime on a log scale --------------------------------------
    px, pw = T.MARGIN_IN, 5.55
    L.eyebrow(s, px, y + 0.10, "Runtime per configuration, log scale", w=4.4)
    bx = px + 1.55
    bw = pw - 1.55 - 0.78
    lo, hi = 1.0, 2000.0

    def _lx(v):
        return bx + bw * (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo))

    top, rh, bar_h = y + 0.56, 0.74, 0.28
    base = top + len(C.RUNTIME_BARS) * rh - 0.28
    for tick in (1, 10, 100, 1000):
        tx = _lx(tick)
        T.line(s, tx, top - 0.16, tx, base, color=T.PAPER_DIM, pt=0.5)
        T.text(s, tx - 0.42, base + 0.06, 0.84, 0.22, "%d s" % tick,
               size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.CENTER)

    for i, (name, secs, gloss) in enumerate(C.RUNTIME_BARS):
        ry = top + i * rh
        colour = T.AMBER if secs > 100 else T.SLATE
        T.text(s, px, ry - 0.01, 1.42, 0.26, name, size=T.SZ_SMALL,
               color=T.INK, bold=True)
        T.rect(s, bx, ry, max(_lx(secs) - bx, 0.03), bar_h, fill=colour)
        T.text(s, _lx(secs) + 0.10, ry - 0.02, 0.95, 0.26, "%.1f" % secs,
               size=T.SZ_SMALL, color=T.INK, bold=True)
        T.text(s, bx, ry + bar_h + 0.05, bw + 0.7, 0.22, gloss,
               size=T.SZ_MICRO, color=T.MUTED)

    hy = base + 0.52
    T.line(s, px, hy - 0.16, px + pw, hy - 0.16, color=T.RULE, pt=0.5)
    sw = pw / 3
    for i, (value, label) in enumerate(C.COST_HEADLINE):
        T.text(s, px + i * sw, hy, sw - 0.20, 0.30, value, size=17,
               color=T.AMBER, bold=True)
        T.text(s, px + i * sw, hy + 0.32, sw - 0.18, 0.5, label,
               size=T.SZ_MICRO, color=T.MUTED, spacing=1.24)

    # ---- right: which cells of the grid were run ---------------------------
    qx = 6.90
    T.line(s, qx - 0.40, y + 0.06, qx - 0.40, 6.20, color=T.RULE, pt=0.5)
    L.eyebrow(s, qx, y + 0.10, "Which cells of the grid were run", w=5.4)

    name_w, cell, cgap = 1.62, 0.50, 0.12
    grid_x = qx + name_w
    gtop, grh = y + 0.86, 0.76
    for j, head in enumerate(C.COVERAGE_COLS):
        T.text(s, grid_x + j * (cell + cgap), gtop - 0.30, cell, 0.22, head,
               size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.CENTER, bold=True)
    T.line(s, qx, gtop - 0.08, grid_x + 6 * (cell + cgap) - cgap, gtop - 0.08,
           color=T.INK, pt=1.0)

    for i, (name, cells, note) in enumerate(C.COVERAGE_ROWS):
        ry = gtop + i * grh
        T.text(s, qx, ry + 0.04, name_w - 0.16, 0.26, name, size=T.SZ_SMALL,
               color=T.INK, bold=True)
        T.text(s, qx, ry + 0.32, 5.4, 0.24, note, size=T.SZ_MICRO,
               color=T.AMBER if name == "LSTE" else T.MUTED,
               bold=name == "LSTE")
        for j, state in enumerate(cells):
            cx = grid_x + j * (cell + cgap)
            if state in ("run", "boot"):
                T.rect(s, cx, ry, cell, 0.28, fill=T.TEAL)
                if state == "boot":
                    T.rect(s, cx, ry + 0.30, cell, 0.05, fill=T.AMBER)
            elif state == "skip":
                T.rect(s, cx, ry, cell, 0.28, fill=None, outline=T.RULE,
                       pt=0.75, dash=T.DASH)
            else:
                T.rect(s, cx, ry, cell, 0.28, fill=T.PAPER_DIM)

    ly = gtop + len(C.COVERAGE_ROWS) * grh - 0.12
    lx = qx
    for state, label in (("run", "run and scored"), ("skip", "not run"),
                         ("n/a", "does not apply"), ("boot", "+ bootstrap")):
        if state in ("run", "boot"):
            T.rect(s, lx, ly + 0.03, 0.22, 0.14, fill=T.TEAL)
            if state == "boot":
                T.rect(s, lx, ly + 0.18, 0.22, 0.04, fill=T.AMBER)
        elif state == "skip":
            T.rect(s, lx, ly + 0.03, 0.22, 0.14, fill=None, outline=T.RULE,
                   pt=0.75, dash=T.DASH)
        else:
            T.rect(s, lx, ly + 0.03, 0.22, 0.14, fill=T.PAPER_DIM)
        T.text(s, lx + 0.30, ly, 1.30, 0.22, label, size=T.SZ_MICRO,
               color=T.MUTED)
        lx += 1.48

    L.takeaway(s, 6.50, C.COST_NOTE, accent=T.AMBER)
    _notes(s, 8)
    return s


# ---------------------------------------------------------------------------
# 9 - learnings
# ---------------------------------------------------------------------------

def slide_09_learnings(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_LEARN, 9, accent=T.TEAL,
                  title="Key learnings from the experiments", kicker=C.LEARN_KICKER)

    gap = 0.64
    w = (T.col_w(12) - gap) / 2
    top, rh = y + 0.10, 1.50
    for i, (head, body) in enumerate(C.LEARNINGS):
        col, row = i % 2, i // 2
        x = T.MARGIN_IN + col * (w + gap)
        ry = top + row * rh
        T.text(s, x, ry + 0.02, 0.40, 0.32, "%02d" % (i + 1), size=T.SZ_BODY,
               color=T.TEAL, bold=True)
        T.text(s, x + 0.46, ry, w - 0.46, 0.54, head, size=T.SZ_BODY,
               color=T.INK, bold=True, spacing=1.16)
        T.text(s, x + 0.46, ry + 0.52, w - 0.46, 0.92, body, size=T.SZ_SMALL,
               color=T.MUTED, spacing=1.26)
        if row < 2:
            T.line(s, x, ry + rh - 0.18, x + w, ry + rh - 0.18, color=T.RULE,
                   pt=0.5)

    L.takeaway(s, 6.56,
               "The first four transfer to any dataset. The last two are about "
               "this codebase, and both are already fixed in the working tree.",
               accent=T.TEAL)
    _notes(s, 9)
    return s


# ---------------------------------------------------------------------------
# 10 - conclusion
# ---------------------------------------------------------------------------

def slide_10_conclusion(prs):
    s = T.blank(prs)
    y = T.top_bar(s, C.ACT_CLOSE, 10, accent=T.TEAL,
                  title="Conclusion, and the one next step that matters",
                  kicker="Two of the three next steps are cheap. The third is "
                         "the only one that changes an outcome.")

    gap = 0.46
    w = (T.col_w(12) - 2 * gap) / 3
    for i, (head, points) in enumerate(C.CONCLUSION_COLS):
        x = T.MARGIN_IN + i * (w + gap)
        T.rect(s, x, y + 0.08, w, 0.034, fill=T.TEAL if i < 2 else T.AMBER)
        T.text(s, x, y + 0.24, w, 0.28, head, size=T.SZ_BODY, color=T.INK,
               bold=True)
        ry = y + 0.60
        for point in points:
            T.text(s, x, ry, w, 0.92, point, size=T.SZ_SMALL, color=T.MUTED,
                   spacing=1.26)
            ry += 0.98

    ry = y + 2.56
    T.line(s, T.MARGIN_IN, ry, T.SLIDE_W_IN - T.MARGIN_IN, ry, color=T.RULE,
           pt=0.75)
    L.eyebrow(s, T.MARGIN_IN, ry + 0.16, "Next", color=T.TEAL)
    ry += 0.46
    for head, chip, body in C.NEXT_STEPS:
        T.text(s, T.MARGIN_IN, ry, 3.15, 0.26, head, size=T.SZ_SMALL,
               color=T.INK, bold=True)
        T.text(s, T.MARGIN_IN + 3.25, ry + 0.02, 1.55, 0.24, chip,
               size=T.SZ_MICRO, color=T.AMBER, bold=True)
        T.text(s, T.MARGIN_IN + 4.90, ry, T.col_w(12) - 4.90, 0.30, body,
               size=T.SZ_SMALL, color=T.MUTED, spacing=1.2)
        ry += 0.46

    T.rect(s, T.MARGIN_IN, ry + 0.06, T.col_w(12), 0.36, fill=T.AMBER_SOFT)
    T.text(s, T.MARGIN_IN + 0.16, ry + 0.14, T.col_w(12) - 0.32, 0.24,
           C.UNVERIFIED, size=T.SZ_MICRO, color=T.INK, bold=True)

    T.text(s, T.MARGIN_IN, ry + 0.58, T.col_w(12), 0.46, C.CLOSING,
           size=T.SZ_BODY, color=T.INK, bold=True, spacing=1.24)
    _notes(s, 10)
    return s


SLIDES = [
    slide_01_title,
    slide_02_objective,
    slide_03_methods,
    slide_04_eda_matrix,
    slide_05_setup,
    slide_06_results,
    slide_07_analysis,
    slide_08_cost,
    slide_09_learnings,
    slide_10_conclusion,
]
