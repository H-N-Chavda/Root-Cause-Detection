"""One function per slide.

Each function takes the presentation and adds exactly one slide. Layouts are
deliberately different from one another — the device used on each slide is
chosen from the shape of that slide's argument, so there is no shared
"title + bullets" template to fall back on.
"""

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import analysis
import content as C
import theme as T


def _notes(slide, number):
    if number in C.NOTES:
        slide.notes_slide.notes_text_frame.text = C.NOTES[number]


# ---------------------------------------------------------------------------
# 1 — Title
# ---------------------------------------------------------------------------

def slide_01_title(prs, data):
    s = T.blank(prs, dark=True)
    T.rect(s, 0.95, 2.22, 0.042, 2.36, fill=T.TEAL)
    T.text(s, 1.22, 2.18, T.col_w(9), 1.70, C.TITLE, size=44,
           color=T.ON_DARK, bold=True, spacing=0.95)
    T.text(s, 1.22, 4.02, T.col_w(7), 0.5, C.SUBTITLE, size=17,
           color=T.RGBColor(0x9A, 0xA2, 0xAA), spacing=1.2)
    T.line(s, 1.22, 4.58, 4.6, 4.58, color=T.RGBColor(0x36, 0x3D, 0x45), pt=0.75)

    x = T.SLIDE_W_IN - T.MARGIN_IN
    for label in reversed(C.TITLE_CHIPS):
        w = 0.13 * len(label) + 0.34
        T.pill(s, x - w, 5.85, w, 0.34, label, fill=None,
               outline=T.RGBColor(0x3A, 0x44, 0x4C), color=T.RGBColor(0x8E, 0x97, 0xA0),
               size=T.SZ_CAPTION, bold=False)
        x -= w + 0.16
    _notes(s, 1)
    return s


# ---------------------------------------------------------------------------
# 2 / 20 — the pipeline
# ---------------------------------------------------------------------------

def _pipeline(s, y, filled, outlined_next=(), box_h=0.95, labels_below=True):
    """Shared five-stage strip. `filled` is the set of solid stages."""
    n = len(C.PIPELINE)
    gap = 0.34
    total = T.SLIDE_W_IN - 2 * T.MARGIN_IN
    box_w = (total - gap * (n - 1)) / n
    x = T.MARGIN_IN
    centres = []
    for i, (name, sub) in enumerate(C.PIPELINE):
        done = i in filled
        ahead = i in outlined_next
        fill = T.TEAL if done else (T.PAPER_DIM if not ahead else None)
        outline = T.TEAL if done else (T.RULE if not ahead else T.MUTED)
        box = T.rect(s, x, y, box_w, box_h, fill=fill, outline=outline, pt=1.0,
                     radius=0.10, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tf = box.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.08)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        T._fill_paragraph(p, name, T.SZ_BODY,
                          T.PAPER if done else T.INK, True, T.FONT, False, None)
        if labels_below:
            T.text(s, x, y + box_h + 0.10, box_w, 0.34, sub, size=T.SZ_MICRO,
                   color=T.MUTED, align=PP_ALIGN.CENTER, spacing=1.15)
        centres.append(x + box_w / 2)
        if i < n - 1:
            T.line(s, x + box_w + 0.06, y + box_h / 2,
                   x + box_w + gap - 0.06, y + box_h / 2,
                   color=T.RULE, pt=1.25, arrow=True)
        x += box_w + gap
    return centres, box_w


def slide_02_goal(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_FRAME, 2)
    T.slide_title(s, "The graph is a means, not the end",
                  "Locating the origin of a fault is the objective; the causal "
                  "graph is what makes that traversal possible")
    centres, box_w = _pipeline(s, 3.30, filled={C.PIPELINE_DELIVERS})

    cx = centres[C.PIPELINE_DELIVERS]
    T.line(s, cx, 4.90, cx, 5.30, color=T.TEAL, pt=1.0)
    T.text(s, cx - 1.5, 5.32, 3.0, 0.5, "what this work delivers",
           size=T.SZ_SMALL, color=T.TEAL, bold=True, align=PP_ALIGN.CENTER)
    T.caption(s, T.MARGIN_IN, 6.30, T.col_w(11),
              "A fault is visible in the data long before its cause is. Reading "
              "the graph upstream from a faulting variable is what converts a "
              "detection into a diagnosis.")
    _notes(s, 2)
    return s


# ---------------------------------------------------------------------------
# 3 — why correlation is not enough
# ---------------------------------------------------------------------------

def _triad(s, kind, cx, cy, spread=0.62, d=0.42, accent=T.INK, size=T.SZ_SMALL):
    """Draws one of the three canonical 3-node structures, centred on (cx, cy).

    Returns the node centres so callers can annotate them.
    """
    r = d / 2
    if kind == "chain":
        pts = {"X": (cx - spread, cy), "Y": (cx, cy), "Z": (cx + spread, cy)}
        arcs = [("X", "Y"), ("Y", "Z")]
    elif kind == "fork":
        pts = {"Y": (cx, cy - spread * 0.62), "X": (cx - spread, cy + spread * 0.42),
               "Z": (cx + spread, cy + spread * 0.42)}
        arcs = [("Y", "X"), ("Y", "Z")]
    else:  # collider
        pts = {"X": (cx - spread, cy - spread * 0.42), "Y": (cx + spread, cy - spread * 0.42),
               "Z": (cx, cy + spread * 0.62)}
        arcs = [("X", "Z"), ("Y", "Z")]

    for a, b in arcs:
        T.edge(s, pts[a], pts[b], r + 0.03, r + 0.03, color=accent, pt=1.5, arrow=True)
    for label, p in pts.items():
        T.node(s, p[0], p[1], d, label, fill=T.PAPER, outline=accent,
               label_color=accent, label_size=size, bold=True)
    return pts


def slide_03_correlation(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 3)
    T.slide_title(s, "Correlation cannot see direction",
                  "Three different causal structures, one identical correlation matrix")

    panel_w = T.col_w(4)
    centres = []
    for i, spec in enumerate(C.STRUCTURES):
        x = T.MARGIN_IN + i * (panel_w + T.GUTTER_IN * 2)
        cx = x + panel_w / 2
        T.text(s, x, 2.42, panel_w, 0.26, spec["name"], size=T.SZ_SUB,
               color=T.INK, bold=True, align=PP_ALIGN.CENTER)
        _triad(s, spec["name"].lower(), cx, 3.42)
        centres.append(cx)

    # one shared correlation matrix, fanning up to all three
    mx, my, mw = 5.30, 5.05, 2.72
    rows = C.CORRELATION_MATRIX
    cell_w, cell_h = mw / 4, 0.28
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            is_head = r == 0 or c == 0
            T.text(s, mx + c * cell_w, my + r * cell_h, cell_w, cell_h, val,
                   size=T.SZ_SMALL, color=T.MUTED if is_head else T.INK,
                   bold=is_head, align=PP_ALIGN.CENTER, font=T.MONO)
    T.rect(s, mx + cell_w * 0.98, my + cell_h * 0.94, cell_w * 3.04, cell_h * 3.12,
           fill=None, outline=T.RULE, pt=0.75)
    for cx in centres:
        T.line(s, mx + mw / 2, my - 0.06, cx, 4.28, color=T.RULE, pt=0.75)

    T.text(s, T.MARGIN_IN, 6.42, T.col_w(12),
           0.32, "Identical correlations. Different causes. Direction has to come "
           "from somewhere else.", size=T.SZ_SUB, color=T.INK, bold=True,
           align=PP_ALIGN.CENTER)
    _notes(s, 3)
    return s


# ---------------------------------------------------------------------------
# 4 — conditional independence
# ---------------------------------------------------------------------------

def slide_04_independence(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 4)
    T.slide_title(s, "Conditioning reveals the one useful asymmetry",
                  "It separates chains and forks — but connects colliders")

    panel_w = T.col_w(4) - 0.06
    for i, spec in enumerate(C.STRUCTURES):
        x = T.MARGIN_IN + i * (panel_w + T.GUTTER_IN * 2 + 0.06)
        promoted = spec["name"] == "Collider"
        accent = T.TEAL if promoted else T.INK
        if promoted:
            T.rect(s, x - 0.10, 2.34, panel_w + 0.20, 3.62, fill=T.TEAL_SOFT,
                   outline=T.TEAL, pt=1.0, radius=0.04,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        T.text(s, x, 2.50, panel_w, 0.26, spec["name"], size=T.SZ_SUB,
               color=accent, bold=True, align=PP_ALIGN.CENTER)
        T.text(s, x, 2.80, panel_w, 0.22, spec["form"], size=T.SZ_CAPTION,
               color=T.MUTED, align=PP_ALIGN.CENTER, font=T.MONO)
        _triad(s, spec["name"].lower(), x + panel_w / 2, 3.72,
               spread=0.60 if not promoted else 0.66,
               d=0.42 if not promoted else 0.46, accent=accent)

        for k, key in enumerate(("marginal", "conditional")):
            stmt, verdict = spec[key]
            yy = 4.85 + k * 0.52
            true = verdict == "true"
            T.text(s, x + 0.12, yy, panel_w * 0.60, 0.26, stmt,
                   size=T.SZ_BODY, color=T.INK, font=T.MONO)
            T.text(s, x + panel_w * 0.60, yy, panel_w * 0.40 - 0.12, 0.26,
                   verdict, size=T.SZ_BODY,
                   color=T.TEAL if true else T.MUTED, bold=true,
                   align=PP_ALIGN.RIGHT)

    T.text(s, 8.55, 6.10, T.col_w(4), 0.3, "← the only column that yields an arrow",
           size=T.SZ_SMALL, color=T.TEAL, bold=True)
    T.caption(s, T.MARGIN_IN, 6.52, T.col_w(12),
              "Every arrow PC draws traces back to the bottom-right cell: "
              "conditioning on a common effect creates dependence instead of "
              "removing it.")
    _notes(s, 4)
    return s


# ---------------------------------------------------------------------------
# 5 — PC in four frames
# ---------------------------------------------------------------------------

_PENTA = {  # 5-node worked example, shared across the four frames
    "A": (-0.62, -0.46), "B": (0.62, -0.46), "C": (0.0, -0.02),
    "D": (0.52, 0.52), "E": (-0.52, 0.52),
}
_SKELETON = [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "E")]
_COLLIDERS = [("A", "C"), ("B", "C")]
_PROPAGATED = [("C", "D"), ("D", "E")]


def _frame(s, cx, cy, stage, scale=1.0):
    d = 0.34
    r = d / 2
    pts = {k: (cx + x * scale, cy + y * scale) for k, (x, y) in _PENTA.items()}
    keys = list(pts)

    if stage == 0:
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                T.edge(s, pts[a], pts[b], r, r, color=T.RULE, pt=1.0)
    else:
        for a, b in _SKELETON:
            directed = (stage >= 2 and (a, b) in _COLLIDERS) or \
                       (stage >= 3 and (a, b) in _PROPAGATED)
            T.edge(s, pts[a], pts[b], r, r,
                   color=T.TEAL if directed else T.INK,
                   pt=1.5 if directed else 1.1, arrow=directed)
    for k, p in pts.items():
        T.node(s, p[0], p[1], d, k, fill=T.PAPER, outline=T.INK,
               label_size=T.SZ_CAPTION, bold=True)


def slide_05_four_frames(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 5)
    T.slide_title(s, "The algorithm in four frames",
                  "One graph, thinned and then oriented — the same five variables throughout")

    panel_w = (T.SLIDE_W_IN - 2 * T.MARGIN_IN - 3 * 0.30) / 4
    rail_y = 5.32
    for i, (name, sub, count) in enumerate(C.FRAMES):
        x = T.MARGIN_IN + i * (panel_w + 0.30)
        cx = x + panel_w / 2
        T.rect(s, x, 2.44, panel_w, 2.44, fill=None, outline=T.RULE, pt=0.75)
        _frame(s, cx, 3.58, i, scale=1.22)
        T.text(s, x, 2.56, panel_w, 0.24, name, size=T.SZ_BODY, color=T.INK,
               bold=True, align=PP_ALIGN.CENTER)
        T.text(s, x, 4.54, panel_w, 0.22, count, size=T.SZ_CAPTION,
               color=T.TEAL if i >= 2 else T.MUTED, bold=True,
               align=PP_ALIGN.CENTER, font=T.MONO)
        T.text(s, x, rail_y + 0.12, panel_w, 0.44, sub, size=T.SZ_SMALL,
               color=T.MUTED, align=PP_ALIGN.CENTER, spacing=1.15)

    T.line(s, T.MARGIN_IN, rail_y, T.SLIDE_W_IN - T.MARGIN_IN, rail_y,
           color=T.RULE, pt=0.75)
    mid = T.MARGIN_IN + 2 * panel_w + 1.5 * 0.30
    T.text(s, T.MARGIN_IN, rail_y - 0.30, mid - T.MARGIN_IN - 0.12, 0.24,
           "PHASE 1 — WHICH VARIABLES ARE RELATED", size=T.SZ_MICRO,
           color=T.INK, bold=True, space=1.2, align=PP_ALIGN.CENTER)
    T.text(s, mid + 0.12, rail_y - 0.30, T.SLIDE_W_IN - T.MARGIN_IN - mid - 0.12,
           0.24, "PHASE 2 — WHICH WAY THE INFLUENCE RUNS", size=T.SZ_MICRO,
           color=T.TEAL, bold=True, space=1.2, align=PP_ALIGN.CENTER)
    T.line(s, mid, rail_y - 0.34, mid, rail_y + 0.62, color=T.RULE, pt=0.75)
    _notes(s, 5)
    return s


# ---------------------------------------------------------------------------
# 6 — partial correlation
# ---------------------------------------------------------------------------

def slide_06_partial_correlation(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 6)
    T.slide_title(s, "What is actually tested: partial correlation",
                  "Under joint normality, conditional independence is exactly a "
                  "partial correlation of zero")

    X, Y, Z = T.TEAL, T.SLATE, T.AMBER
    T.rich(s, T.MARGIN_IN, 2.62, T.col_w(9), 0.5, [
        ("ρ(", {}), ("X", {"color": X, "bold": True}), (",", {}),
        ("Y", {"color": Y, "bold": True}), (" | Z ∪ {", {}),
        ("z", {"color": Z, "bold": True}), ("})", {}),
        ("   =   ", {"color": T.MUTED}),
    ], size=25)

    T.rich(s, T.MARGIN_IN + 0.22, 3.22, T.col_w(10), 0.44, [
        ("ρ(", {}), ("X", {"color": X, "bold": True}), (",", {}),
        ("Y", {"color": Y, "bold": True}), ("|Z)", {}),
        ("  −  ", {"color": T.MUTED}),
        ("ρ(", {}), ("X", {"color": X, "bold": True}), (",", {}),
        ("z", {"color": Z, "bold": True}), ("|Z)", {}),
        (" · ", {"color": T.MUTED}),
        ("ρ(", {}), ("Y", {"color": Y, "bold": True}), (",", {}),
        ("z", {"color": Z, "bold": True}), ("|Z)", {}),
    ], size=19)
    T.line(s, T.MARGIN_IN + 0.22, 3.74, T.MARGIN_IN + 6.15, 3.74,
           color=T.INK, pt=1.0)
    T.rich(s, T.MARGIN_IN + 0.22, 3.84, T.col_w(10), 0.44, [
        ("√", {"size": 22}), (" ( 1 − ρ(", {}), ("X", {"color": X, "bold": True}),
        (",", {}), ("z", {"color": Z, "bold": True}), ("|Z)² ) ( 1 − ρ(", {}),
        ("Y", {"color": Y, "bold": True}), (",", {}),
        ("z", {"color": Z, "bold": True}), ("|Z)² )", {}),
    ], size=19)

    T.rich(s, T.MARGIN_IN, 4.72, T.col_w(9), 0.34, [
        ("X", {"color": X, "bold": True}), (" ⫫ ", {"size": 17}),
        ("Y", {"color": Y, "bold": True}), (" | Z", {}),
        ("     ⟺     ", {"color": T.MUTED}),
        ("ρ(", {}), ("X", {"color": X, "bold": True}), (",", {}),
        ("Y", {"color": Y, "bold": True}), ("|Z) = 0", {}),
    ], size=17)

    T.caption(s, T.MARGIN_IN, 5.34, T.col_w(8),
              "The definition is recursive: each step peels one variable off the "
              "conditioning set, reducing a hard quantity to three easier ones "
              "of the same kind. Ordinary correlation is the base case.")

    # the peel step, drawn
    px = 9.35
    T.text(s, px, 2.52, T.col_w(3), 0.24, "ONE PEEL STEP", size=T.SZ_MICRO,
           color=T.MUTED, bold=True, space=1.2)
    _triad(s, "collider", px + 1.20, 3.50, spread=0.66, d=0.44)
    T.text(s, px, 4.30, T.col_w(3), 0.6,
           "the tinted symbols above are these three nodes", size=T.SZ_CAPTION,
           color=T.MUTED, align=PP_ALIGN.CENTER, spacing=1.2)
    for label, colour, yy in (("X", X, 4.80), ("Y", Y, 5.14), ("z", Z, 5.48)):
        T.rect(s, px + 0.62, yy + 0.05, 0.14, 0.14, fill=colour)
        T.text(s, px + 0.86, yy, 1.6, 0.24,
               {"X": "first variable", "Y": "second variable",
                "z": "conditioned on"}[label], size=T.SZ_CAPTION, color=T.MUTED)
    _notes(s, 6)
    return s


# ---------------------------------------------------------------------------
# 7 — Fisher's z
# ---------------------------------------------------------------------------

def slide_07_fisher(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 7)
    T.slide_title(s, "The decision rule",
                  "Every edge in the graph is the outcome of a hypothesis test")

    T.rich(s, T.MARGIN_IN, 2.72, T.col_w(5), 0.44, [
        ("z", {"italic": True, "bold": True}), ("  =  ½ ", {}),
        ("√", {"size": 20}), (" ( n − |Z| − 3 )  ·  ln", {}),
        ("  1 + ρ̂ ", {"size": 13}), ("⁄", {"size": 16, "color": T.MUTED}),
        (" 1 − ρ̂", {"size": 13}),
    ], size=17)
    T.rich(s, T.MARGIN_IN, 3.36, T.col_w(5), 0.4, [
        ("p", {"italic": True, "bold": True}),
        ("  =  2 ( 1 − Φ( |", {}), ("z", {"italic": True, "bold": True}),
        ("| ) )", {}),
    ], size=17)

    T.rect(s, T.MARGIN_IN, 3.98, T.col_w(5), 0.72, fill=T.PAPER_DIM,
           outline=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    T.rich(s, T.MARGIN_IN + 0.20, 4.20, T.col_w(5) - 0.4, 0.3, [
        ("Remove the edge when  ", {}),
        ("p ≥ α", {"bold": True, "color": T.TEAL, "size": 15}),
        ("      with α = 0.01", {"color": T.MUTED}),
    ], size=13.5)

    T.rect(s, T.MARGIN_IN, 5.06, 0.036, 0.92, fill=T.AMBER)
    T.text(s, T.MARGIN_IN + 0.20, 5.06, T.col_w(5) - 0.2, 0.9,
           [[("n − |Z| − 3", {"font": T.MONO, "bold": True, "color": T.AMBER}),
             (" assumes n independent observations.", {})],
            [("We return to this on slide 17 — it is the one assumption these "
              "data genuinely stretch.", {"color": T.MUTED})]],
           size=T.SZ_SMALL, spacing=1.25)

    # null distribution with two-sided rejection region
    gx, gy, gw, gh = 7.30, 2.76, 5.05, 2.30
    T.mini_axes(s, gx, gy, gw, gh)
    curve = T.normal_curve((gx, gx + gw), gh)
    crit = 2.576  # two-sided 1% critical value
    T.polyline(s, [(x, gy + gh - h) for x, h, _ in curve], color=T.INK, pt=1.5)
    for sign in (-1, 1):
        tail = [(x, gy + gh - h) for x, h, t in curve if sign * t >= crit]
        if len(tail) < 2:
            continue
        poly = tail + [(tail[-1][0], gy + gh), (tail[0][0], gy + gh)]
        T.polyline(s, poly, color=T.AMBER, pt=0.75, fill=T.AMBER_SOFT, close=True)
        edge_x = tail[0][0] if sign > 0 else tail[-1][0]
        T.line(s, edge_x, gy + 0.10, edge_x, gy + gh, color=T.AMBER, pt=0.75,
               dash=T.DASH)
    T.text(s, gx + gw / 2 - 0.9, gy + gh + 0.12, 1.8, 0.24,
           "z = 0   (edge is independent)", size=T.SZ_CAPTION, color=T.MUTED,
           align=PP_ALIGN.CENTER)
    T.text(s, gx + gw - 1.95, gy + 0.34, 1.9, 0.44,
           "α ⁄ 2 in each tail\nreject independence, keep the edge",
           size=T.SZ_CAPTION, color=T.AMBER, align=PP_ALIGN.RIGHT, spacing=1.2)
    T.caption(s, gx, gy + gh + 0.46, gw,
              "A small p-value means the correlation is too large to be an "
              "accident, so the two variables stay connected.")
    _notes(s, 7)
    return s


# ---------------------------------------------------------------------------
# 8 — orientation rules
# ---------------------------------------------------------------------------

def _rule_card(s, x, y, w, h, index, spec):
    name, precondition, reason = spec
    T.rect(s, x, y, w, h, fill=None, outline=T.RULE, pt=0.75)
    T.text(s, x + 0.22, y + 0.18, w - 0.44, 0.26,
           [[("%d " % (index + 1), {"color": T.TEAL, "bold": True}),
             (name, {"bold": True})]], size=T.SZ_BODY)
    T.text(s, x + 0.22, y + 0.50, w * 0.56, 0.44, precondition,
           size=T.SZ_CAPTION, color=T.MUTED, spacing=1.2)
    T.text(s, x + 0.22, y + h - 0.52, w - 0.44, 0.42, reason,
           size=T.SZ_SMALL, color=T.INK, spacing=1.2)

    # the transformation, drawn small on the right
    d, r = 0.28, 0.14
    cx, cy = x + w - 1.62, y + 0.62
    if index == 0:
        pts = {"a": (cx - 0.52, cy), "b": (cx, cy + 0.34), "c": (cx + 0.52, cy)}
        after = [("a", "b"), ("c", "b")]
        plain = []
    elif index == 1:
        pts = {"a": (cx - 0.52, cy), "b": (cx, cy + 0.34), "c": (cx + 0.52, cy)}
        after = [("b", "c")]
        plain = [("a", "b")]
    elif index == 2:
        pts = {"a": (cx - 0.52, cy + 0.34), "c": (cx, cy),
               "b": (cx + 0.52, cy + 0.34)}
        after = [("a", "b")]
        plain = [("a", "c"), ("c", "b")]
    else:
        pts = {"a": (cx - 0.46, cy), "b": (cx + 0.46, cy),
               "c": (cx, cy - 0.30), "d": (cx, cy + 0.40)}
        after = [("a", "b")]
        plain = [("c", "b"), ("d", "b")]

    for a, b in plain:
        T.edge(s, pts[a], pts[b], r, r, color=T.INK, pt=1.25,
               arrow=index in (1, 2, 3))
    for a, b in after:
        T.edge(s, pts[a], pts[b], r, r, color=T.TEAL, pt=1.75, arrow=True)
    for label, p in pts.items():
        T.node(s, p[0], p[1], d, label, fill=T.PAPER, outline=T.INK,
               label_size=T.SZ_MICRO, bold=True)


def slide_08_orientation_rules(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_METHOD, 8)
    T.slide_title(s, "Four rules decide every arrow",
                  "Each one is forced: the alternative would invent a collider "
                  "already ruled out, or close a cycle")

    w = (T.SLIDE_W_IN - 2 * T.MARGIN_IN - 0.30) / 2
    h = 1.86
    for i, spec in enumerate(C.ORIENTATION_RULES):
        x = T.MARGIN_IN + (i % 2) * (w + 0.30)
        y = 2.46 + (i // 2) * (h + 0.26)
        _rule_card(s, x, y, w, h, i, spec)

    T.text(s, T.MARGIN_IN, 6.52, T.col_w(12), 0.3,
           "What remains undirected afterwards is genuinely undetermined by the "
           "data — not an omission.", size=T.SZ_BODY, color=T.INK, bold=True,
           align=PP_ALIGN.CENTER)
    _notes(s, 8)
    return s


# ---------------------------------------------------------------------------
# 9 — the process
# ---------------------------------------------------------------------------

def slide_09_process(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_DATA, 9)
    T.slide_title(s, "The Tennessee Eastman process",
                  "A benchmark simulation of a real chemical plant — so a causal "
                  "structure genuinely exists to be found")

    y = 2.72
    bw, bh, gap = 1.72, 0.82, 0.46
    x = T.MARGIN_IN + 0.30
    T.rect(s, x, y, 1.32, bh, fill=T.PAPER_DIM, outline=T.RULE, radius=0.08,
           shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    T.text(s, x, y + 0.28, 1.32, 0.3, C.PROCESS_FEEDS, size=T.SZ_CAPTION,
           color=T.MUTED, align=PP_ALIGN.CENTER)
    x += 1.32 + gap
    unit_centres = []
    for i, unit in enumerate(C.PROCESS_UNITS):
        T.line(s, x - gap + 0.06, y + bh / 2, x - 0.06, y + bh / 2,
               color=T.RULE, pt=1.25, arrow=True)
        box = T.rect(s, x, y, bw, bh, fill=None, outline=T.INK, pt=1.25,
                     radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tf = box.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        T._fill_paragraph(p, unit, T.SZ_BODY, T.INK, True, T.FONT, False, None)
        unit_centres.append(x + bw / 2)
        x += bw + gap
    T.line(s, x - gap + 0.06, y + bh / 2, x - 0.06, y + bh / 2,
           color=T.RULE, pt=1.25, arrow=True)
    T.rect(s, x, y, 1.20, bh, fill=T.TEAL_SOFT, outline=T.TEAL, radius=0.08,
           shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    T.text(s, x, y + 0.28, 1.20, 0.3, C.PROCESS_PRODUCT, size=T.SZ_CAPTION,
           color=T.TEAL, bold=True, align=PP_ALIGN.CENTER)

    # recycle loop back to the reactor
    ry = y + bh + 0.52
    T.line(s, unit_centres[2], y + bh, unit_centres[2], ry, color=T.SLATE, pt=1.25)
    T.line(s, unit_centres[2], ry, unit_centres[0], ry, color=T.SLATE, pt=1.25)
    T.line(s, unit_centres[0], ry, unit_centres[0], y + bh + 0.06,
           color=T.SLATE, pt=1.25, arrow=True)
    T.text(s, unit_centres[0] + 0.16, ry + 0.06, 2.4, 0.24, C.PROCESS_RECYCLE,
           size=T.SZ_CAPTION, color=T.SLATE)

    T.line(s, T.MARGIN_IN, 4.60, T.SLIDE_W_IN - T.MARGIN_IN, 4.60,
           color=T.RULE, pt=0.75)
    sx = T.MARGIN_IN
    for value, label in C.DATA_STATS:
        T.text(s, sx, 4.80, 2.4, 0.5, value, size=34, color=T.INK, bold=True)
        T.text(s, sx, 5.32, 2.4, 0.3, label, size=T.SZ_SMALL, color=T.MUTED)
        sx += 2.9

    T.rect(s, 9.45, 4.76, 0.032, 1.32, fill=T.RULE)
    T.text(s, 9.66, 4.76, T.SLIDE_W_IN - T.MARGIN_IN - 9.66, 1.42,
           C.PROCESS_NOTE, size=T.SZ_CAPTION, color=T.MUTED, spacing=1.28)
    _notes(s, 9)
    return s


# ---------------------------------------------------------------------------
# 10 — preparation
# ---------------------------------------------------------------------------

def slide_10_preparation(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_DATA, 10)
    T.slide_title(s, "Preparing the data",
                  "Three decisions, each following from the mathematics rather "
                  "than from convention")

    pw = 2.55
    lx, rx = T.MARGIN_IN, T.SLIDE_W_IN - T.MARGIN_IN - pw
    for x, heading, rows, accent in (
        (lx, "AS PROVIDED", [("32", "columns"), ("1,499", "rows"),
                             ("mixed", "scales")], T.MUTED),
        (rx, "AS ANALYSED", [("31", "process variables"), ("1,499", "samples"),
                             ("unchanged", "scales")], T.TEAL),
    ):
        T.text(s, x, 2.50, pw, 0.24, heading, size=T.SZ_MICRO, color=accent,
               bold=True, space=1.2)
        T.rect(s, x, 2.80, pw, 1.86, fill=T.PAPER_DIM if accent is T.MUTED
               else T.TEAL_SOFT, outline=None, radius=0.05,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        for i, (value, label) in enumerate(rows):
            T.text(s, x + 0.24, 3.00 + i * 0.56, pw - 0.48, 0.32, value,
                   size=19, color=T.INK if accent is T.MUTED else T.TEAL,
                   bold=True)
            T.text(s, x + 0.24, 3.32 + i * 0.56, pw - 0.48, 0.24, label,
                   size=T.SZ_CAPTION, color=T.MUTED)

    mid_x = lx + pw + 0.55
    mid_w = rx - mid_x - 0.55
    for i, (headline, detail) in enumerate(C.PREPARATION):
        y = 2.52 + i * 0.72
        T.text(s, mid_x, y, mid_w, 0.24,
               [[("→  ", {"color": T.TEAL, "bold": True}),
                 (headline, {"bold": True})]], size=T.SZ_BODY)
        T.text(s, mid_x + 0.26, y + 0.26, mid_w - 0.26, 0.42, detail,
               size=T.SZ_CAPTION, color=T.MUTED, spacing=1.24)
    T.line(s, lx + pw + 0.12, 3.72, mid_x - 0.12, 3.72, color=T.RULE, pt=0.75)
    T.line(s, mid_x + mid_w + 0.12, 3.72, rx - 0.12, 3.72, color=T.RULE,
           pt=0.75, arrow=True)

    T.rect(s, T.MARGIN_IN, 5.30, T.col_w(12), 0.86, fill=None, outline=T.AMBER,
           pt=1.0, radius=0.04, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    T.text(s, T.MARGIN_IN + 0.24, 5.48, T.col_w(12) - 0.48, 0.6,
           C.PREPARATION_REFERENCE, size=T.SZ_BODY, color=T.INK, spacing=1.3)
    _notes(s, 10)
    return s


# ---------------------------------------------------------------------------
# 11 — parameters
# ---------------------------------------------------------------------------

def slide_11_parameters(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_DATA, 11)
    T.slide_title(s, "Two parameters, both chosen for a reason",
                  "One barely matters; the other turns out not to be a compromise at all")

    for i, spec in enumerate(C.PARAMETERS):
        y = 2.70 + i * 1.94
        T.text(s, T.MARGIN_IN, y, T.col_w(5), 0.28, spec["name"],
               size=T.SZ_SUB, color=T.INK, bold=True)

        tx, tw = T.MARGIN_IN, T.col_w(7)
        ty = y + 0.78
        T.line(s, tx, ty, tx + tw, ty, color=T.RULE, pt=2.0)
        ticks = spec["ticks"]
        step = tw / (len(ticks) - 1)
        for k, label in enumerate(ticks):
            cx = tx + k * step
            pinned = k == spec["pin"]
            T.line(s, cx, ty - 0.09, cx, ty + 0.09,
                   color=T.TEAL if pinned else T.RULE, pt=2.0 if pinned else 1.0)
            T.text(s, cx - 0.42, ty + 0.16, 0.84, 0.24, label,
                   size=T.SZ_CAPTION, color=T.TEAL if pinned else T.MUTED,
                   bold=pinned, align=PP_ALIGN.CENTER, font=T.MONO)
        px = tx + spec["pin"] * step
        T.node(s, px, ty, 0.22, "", fill=T.TEAL, outline=T.TEAL)
        T.text(s, px - 0.9, ty - 0.52, 1.8, 0.28, spec["chosen"], size=20,
               color=T.TEAL, bold=True, align=PP_ALIGN.CENTER)

        T.rect(s, T.col_x(8), y + 0.10, T.col_w(4), 1.16,
               fill=T.PAPER_DIM if i == 0 else T.TEAL_SOFT, outline=None,
               radius=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        T.text(s, T.col_x(8) + 0.22, y + 0.30, T.col_w(4) - 0.44, 0.9,
               spec["note"], size=T.SZ_SMALL,
               color=T.INK, spacing=1.28)
    _notes(s, 11)
    return s


# ---------------------------------------------------------------------------
# 12 — the skeleton search
# ---------------------------------------------------------------------------

def slide_12_funnel(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_BUILD, 12)
    T.slide_title(s, "From every possible pair to 35 relationships",
                  "Each stage conditions on a larger set — and most pairs fall early")

    max_w = 7.10
    top = 2.60
    row_h, gap = 0.62, 0.34
    start = C.FUNNEL[0][1]
    for i, (label, remaining, tests) in enumerate(C.FUNNEL):
        y = top + i * (row_h + gap)
        w = max(0.52, max_w * remaining / start)
        final = i == len(C.FUNNEL) - 1
        T.rect(s, T.MARGIN_IN, y, w, row_h,
               fill=T.TEAL if final else T.SLATE_SOFT,
               outline=T.TEAL if final else T.SLATE, pt=1.0)
        T.text(s, T.MARGIN_IN + w + 0.20, y + 0.02, 2.4, 0.34, str(remaining),
               size=21, color=T.TEAL if final else T.INK, bold=True)
        T.text(s, T.MARGIN_IN + w + 0.20, y + 0.36, 3.6, 0.24, label,
               size=T.SZ_CAPTION, color=T.MUTED)
        if tests:
            T.text(s, 10.30, y + 0.10, 2.0, 0.3,
                   "%s tests" % format(tests, ","), size=T.SZ_SMALL,
                   color=T.MUTED, align=PP_ALIGN.RIGHT, font=T.MONO)
        if i:
            T.line(s, T.MARGIN_IN + 0.30, y - gap + 0.04, T.MARGIN_IN + 0.30,
                   y - 0.04, color=T.RULE, pt=0.75, arrow=True)

    T.line(s, 10.30, top - 0.16, 12.30, top - 0.16, color=T.RULE, pt=0.75)
    T.text(s, 10.30, top - 0.44, 2.0, 0.24, "COST", size=T.SZ_MICRO,
           color=T.MUTED, bold=True, space=1.2, align=PP_ALIGN.RIGHT)
    T.line(s, 10.30, top + 4 * (row_h + gap) - gap, 12.30,
           top + 4 * (row_h + gap) - gap, color=T.RULE, pt=0.75)
    T.text(s, 10.30, top + 4 * (row_h + gap) - gap + 0.08, 2.0, 0.3,
           "%s total" % format(C.FUNNEL_TOTAL_TESTS, ","), size=T.SZ_BODY,
           color=T.INK, bold=True, align=PP_ALIGN.RIGHT, font=T.MONO)

    T.caption(s, T.MARGIN_IN, 6.42, T.col_w(9), C.FUNNEL_CAPTION,
              size=T.SZ_SMALL)
    _notes(s, 12)
    return s


# ---------------------------------------------------------------------------
# 13 — orientation outcome
# ---------------------------------------------------------------------------

def slide_13_orientation(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_BUILD, 13)
    T.slide_title(s, "How much direction the data supports",
                  "Of the 35 relationships found, the data fixes the direction of 20")

    total = sum(count for _, count, _ in C.ORIENTATION_SPLIT)
    bar_x, bar_y, bar_w, bar_h = T.MARGIN_IN, 2.86, 7.35, 0.78
    palette = [(T.TEAL, T.PAPER), (T.SLATE_SOFT, T.INK), (T.AMBER_SOFT, T.INK)]
    x = bar_x
    for (label, count, _), (fill, ink) in zip(C.ORIENTATION_SPLIT, palette):
        w = bar_w * count / total
        T.rect(s, x, bar_y, w, bar_h, fill=fill,
               outline=T.AMBER if fill is T.AMBER_SOFT else
               (T.SLATE if fill is T.SLATE_SOFT else T.TEAL), pt=1.0)
        T.text(s, x, bar_y + 0.20, w, 0.4, str(count), size=22, color=ink,
               bold=True, align=PP_ALIGN.CENTER)
        x += w

    # Segment widths differ too much to label in place, so the breakdown is a
    # stacked key beneath the bar instead of captions under each segment.
    for i, ((label, count, detail), (fill, _)) in enumerate(
            zip(C.ORIENTATION_SPLIT, palette)):
        y = bar_y + bar_h + 0.34 + i * 0.62
        T.rect(s, bar_x, y + 0.04, 0.17, 0.17, fill=fill,
               outline=T.AMBER if fill is T.AMBER_SOFT else
               (T.SLATE if fill is T.SLATE_SOFT else T.TEAL), pt=1.0)
        T.text(s, bar_x + 0.32, y - 0.02, 1.7, 0.26,
               "%s  %d" % (label, count), size=T.SZ_BODY, color=T.INK,
               bold=True)
        T.text(s, bar_x + 2.10, y - 0.02, bar_w - 2.10, 0.5, detail,
               size=T.SZ_CAPTION, color=T.MUTED, spacing=1.22)

    T.rect(s, 8.45, 2.54, 0.032, 3.1, fill=T.RULE)
    T.text(s, 8.80, 2.54, T.col_w(4), 0.24, "WHY 12 STAY UNRESOLVED",
           size=T.SZ_MICRO, color=T.AMBER, bold=True, space=1.2)
    cx, cy, d, r = 10.35, 3.52, 0.40, 0.20
    pts = {"a": (cx - 0.78, cy - 0.28), "b": (cx, cy + 0.30),
           "c": (cx + 0.78, cy - 0.28)}
    T.edge(s, pts["a"], pts["b"], r, r, color=T.AMBER, pt=1.75, arrow=True)
    T.edge(s, pts["c"], pts["b"], r, r, color=T.AMBER, pt=1.75, arrow=True)
    T.edge(s, pts["b"], pts["a"], r, r, color=T.AMBER, pt=1.0, arrow=True,
           dash=T.DASH)
    for label, p in pts.items():
        T.node(s, p[0], p[1], d, label, fill=T.PAPER, outline=T.INK,
               label_size=T.SZ_CAPTION, bold=True)
    T.text(s, 8.80, 4.20, T.col_w(4), 1.1,
           "Two different triples can each imply an arrowhead, pointing in "
           "opposite directions on the same edge. Rather than pick one, we "
           "report the relationship and leave its direction open.",
           size=T.SZ_CAPTION, color=T.MUTED, spacing=1.28)

    T.text(s, T.MARGIN_IN, 6.36, T.col_w(8), 0.4, C.ORIENTATION_CAPTION,
           size=T.SZ_BODY, color=T.INK, bold=True, spacing=1.25)
    _notes(s, 13)
    return s


# ---------------------------------------------------------------------------
# 14 — the recovered graph
# ---------------------------------------------------------------------------

def slide_14_graph(prs, data):
    s = T.blank(prs, dark=True)
    T.header(s, C.ACT_RESULT, 14, accent=T.TEAL, dark=True)
    T.text(s, T.MARGIN_IN, T.MARGIN_IN + 0.62, T.col_w(8), 0.44,
           "The recovered causal graph", size=T.SZ_TITLE, color=T.ON_DARK,
           bold=True)

    names = data["names"]
    d, r = 0.30, 0.15
    inset = r + 0.06
    panels = [
        (1.05 + inset, 2.32 + inset, 6.55 - 2 * inset, 3.85 - 2 * inset),
        (8.55 + inset, 2.32 + inset, 3.55 - 2 * inset, 1.95 - 2 * inset),
        (9.10 + inset, 4.95 + inset, 1.55 - 2 * inset, 0.30),
        (11.55, 5.10, 0.0, 0.0),
    ]
    placed = analysis.place_components(
        data["components"], data["skeleton"], panels)

    agreeing = set(data["agreeing"])
    for a, b in data["skeleton"]:
        if a not in placed or b not in placed:
            continue
        confirmed = (a, b) in agreeing
        directed = (a, b) in data["directed"] or (b, a) in data["directed"]
        forward = (a, b) in data["directed"]
        p1, p2 = (placed[a], placed[b]) if forward or not directed else \
                 (placed[b], placed[a])
        T.edge(s, p1, p2, r + 0.02, r + 0.02,
               color=T.TEAL if confirmed else T.SLATE,
               pt=1.9 if confirmed else 1.3, arrow=directed)
    for v, p in placed.items():
        T.node(s, p[0], p[1], d, names[v].replace("X", ""),
               fill=T.INK_DARK, outline=T.RGBColor(0x8A, 0x93, 0x9C),
               pt=1.0, label_color=T.ON_DARK, label_size=T.SZ_MICRO, bold=True)

    T.text(s, 8.55, 4.44, T.col_w(4), 0.24, "SEPARATE GROUPS",
           size=T.SZ_MICRO, color=T.RGBColor(0x7A, 0x82, 0x8A), bold=True,
           space=1.2)
    T.text(s, 11.05, 5.02, 1.3, 0.3, "isolated", size=T.SZ_MICRO,
           color=T.RGBColor(0x7A, 0x82, 0x8A))

    # Legend sits with the title so the four counts get the full bottom rail.
    for i, (colour, label) in enumerate((
            (T.TEAL, "confirmed by the reference graph"),
            (T.SLATE, "additional candidate relationship"))):
        y = 1.34 + i * 0.30
        T.rect(s, 9.30, y + 0.07, 0.24, 0.055, fill=colour)
        T.text(s, 9.66, y, 3.02, 0.24, label, size=T.SZ_MICRO,
               color=T.RGBColor(0x9A, 0xA2, 0xAA))

    lx = T.MARGIN_IN
    for value, label in C.GRAPH_COUNTS:
        T.text(s, lx, 6.28, 2.8, 0.36, value, size=26, color=T.ON_DARK,
               bold=True)
        T.text(s, lx + 0.02, 6.66, 2.8, 0.24, label, size=T.SZ_MICRO,
               color=T.RGBColor(0x8A, 0x92, 0x9A))
        lx += 3.0
    _notes(s, 14)
    return s


# ---------------------------------------------------------------------------
# 15 — assessment
# ---------------------------------------------------------------------------

def slide_15_assessment(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_RESULT, 15)
    T.slide_title(s, "Measured against the reference graph",
                  "Where our graph and the plant's documented topology agree, "
                  "and where they do not")

    names = data["names"]
    n = data["n_vars"]
    grid_x, grid_y, grid = 0.98, 2.56, 3.92
    cell = grid / n

    T.rect(s, grid_x, grid_y, grid, grid, fill=T.PAPER, outline=T.RULE, pt=0.75)
    for (a, b), colour in list(
            [(e, T.RULE) for e in data["missed"]] +
            [(e, T.SLATE) for e in data["additional"]] +
            [(e, T.TEAL) for e in data["agreeing"]]):
        for i, j in ((a, b), (b, a)):
            T.rect(s, grid_x + j * cell, grid_y + i * cell, cell, cell,
                   fill=colour)
    for k in range(0, n, 5):
        T.text(s, grid_x + k * cell - 0.16, grid_y - 0.22, 0.42, 0.2,
               names[k].replace("X", ""), size=T.SZ_MICRO, color=T.MUTED,
               align=PP_ALIGN.CENTER)
        T.text(s, grid_x - 0.42, grid_y + k * cell - 0.02, 0.32, 0.2,
               names[k].replace("X", ""), size=T.SZ_MICRO, color=T.MUTED,
               align=PP_ALIGN.RIGHT)
    T.text(s, grid_x, grid_y + grid + 0.14, grid, 0.24,
           "each square is one pair of variables", size=T.SZ_MICRO,
           color=T.MUTED, align=PP_ALIGN.CENTER)

    # Legend stacks to the right of the grid, which keeps the slide off the
    # bottom margin that a horizontal legend would push it past.
    for i, (colour, label) in enumerate((
            (T.TEAL, "both agree  (11)"),
            (T.SLATE, "ours only  (24)"),
            (T.RULE, "reference only  (17)"))):
        y = 2.60 + i * 0.34
        T.rect(s, grid_x + grid + 0.30, y + 0.04, 0.16, 0.16, fill=colour,
               outline=T.RULE if colour is T.RULE else None, pt=0.5)
        T.text(s, grid_x + grid + 0.56, y, 1.70, 0.24, label,
               size=T.SZ_MICRO, color=T.MUTED)

    mx = 7.42
    for i, (name, value, gloss) in enumerate(C.METRICS):
        y = 2.56 + i * 0.80
        T.text(s, mx, y, 1.4, 0.4, value, size=26, color=T.INK, bold=True)
        T.text(s, mx + 1.18, y + 0.06, 1.6, 0.26, name, size=T.SZ_BODY,
               color=T.INK, bold=True)
        T.text(s, mx + 1.18, y + 0.32, T.SLIDE_W_IN - T.MARGIN_IN - mx - 1.18,
               0.42, gloss, size=T.SZ_CAPTION, color=T.MUTED, spacing=1.22)
        if i:
            T.line(s, mx, y - 0.12, T.SLIDE_W_IN - T.MARGIN_IN, y - 0.12,
                   color=T.RULE, pt=0.5)

    T.rect(s, mx, 5.86, T.SLIDE_W_IN - T.MARGIN_IN - mx, 0.036, fill=T.AMBER)
    T.text(s, mx, 6.00, T.SLIDE_W_IN - T.MARGIN_IN - mx, 0.86,
           C.METRICS_READING, size=T.SZ_CAPTION, color=T.INK, spacing=1.3)
    _notes(s, 15)
    return s


# ---------------------------------------------------------------------------
# 16 — sensitivity
# ---------------------------------------------------------------------------

def slide_16_sensitivity(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_RESULT, 16)
    T.slide_title(s, "What the answer depends on",
                  "Same vertical scale on all three, so the slopes can be compared directly")

    pw = (T.SLIDE_W_IN - 2 * T.MARGIN_IN - 2 * 0.42) / 3
    ph = 2.28
    lo, hi = C.SENSITIVITY_AXIS
    for i, spec in enumerate(C.SENSITIVITY):
        x = T.MARGIN_IN + i * (pw + 0.42)
        y = 2.86
        T.text(s, x, 2.52, pw, 0.26, spec["title"], size=T.SZ_BODY,
               color=T.INK, bold=True)
        T.mini_axes(s, x, y, pw, ph)
        for tick in (0.10, 0.20, 0.30, 0.40):
            ty = y + ph - ph * (tick - lo) / (hi - lo)
            T.line(s, x, ty, x + pw, ty, color=T.PAPER_DIM, pt=0.5)
            T.text(s, x - 0.44, ty - 0.09, 0.36, 0.2, "%.2f" % tick,
                   size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.RIGHT)

        values = spec["values"]
        step = pw / (len(values) - 1)
        pts = [(x + k * step, y + ph - ph * (v - lo) / (hi - lo))
               for k, v in enumerate(values)]
        T.polyline(s, pts, color=T.SLATE, pt=2.0)
        for k, (px, py) in enumerate(pts):
            T.node(s, px, py, 0.11, "", fill=T.SLATE, outline=T.SLATE, pt=0.5)
            T.text(s, px - 0.34, y + ph + 0.10, 0.68, 0.2, spec["x_labels"][k],
                   size=T.SZ_MICRO, color=T.MUTED, align=PP_ALIGN.CENTER)
        T.text(s, x, y + ph + 0.40, pw, 0.44, spec["verdict"],
               size=T.SZ_SMALL, color=T.TEAL if i == 1 else T.MUTED,
               bold=i == 1, spacing=1.2)

    T.text(s, T.MARGIN_IN - 0.48, 2.62, 0.4, 0.24, "F1", size=T.SZ_MICRO,
           color=T.MUTED, bold=True)
    T.rect(s, T.MARGIN_IN, 6.22, T.col_w(12), 0.036, fill=T.RULE)
    T.text(s, T.MARGIN_IN, 6.38, T.col_w(11), 0.4, C.SENSITIVITY_CAPTION,
           size=T.SZ_BODY, color=T.INK, spacing=1.25)
    _notes(s, 16)
    return s


# ---------------------------------------------------------------------------
# 17 — the independence assumption
# ---------------------------------------------------------------------------

def slide_17_autocorrelation(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_RESULT, 17, accent=T.AMBER)
    T.slide_title(s, "1,499 samples are not 1,499 observations",
                  "The one assumption behind every test on the previous slides")

    series = data["series"]
    gx, gy, gw, gh = T.MARGIN_IN, 3.00, 8.35, 1.72
    values = series["values"]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = [(gx + gw * i / (len(values) - 1), gy + gh - gh * (v - lo) / span)
           for i, v in enumerate(values)]
    T.rect(s, gx, gy, gw, gh, fill=T.PAPER_DIM, outline=None)
    T.polyline(s, pts, color=T.SLATE, pt=1.25)
    T.text(s, gx, gy + gh + 0.12, gw, 0.24,
           "one process variable, 400 consecutive samples",
           size=T.SZ_MICRO, color=T.MUTED)

    T.text(s, 9.45, 2.78, 3.2, 1.45, C.AUTOCORR_HEADLINE, size=76,
           color=T.AMBER, bold=True)
    T.text(s, 9.52, 4.26, 3.2, 0.5, "strongest lag-one\nautocorrelation",
           size=T.SZ_SMALL, color=T.AMBER, spacing=1.2)

    T.text(s, T.MARGIN_IN, 5.24, T.col_w(6), 0.6, C.AUTOCORR_BODY,
           size=T.SZ_BODY, color=T.INK, spacing=1.3)
    T.rect(s, T.col_x(6) + 0.14, 5.18, 0.036, 1.32, fill=T.AMBER)
    T.text(s, T.col_x(6) + 0.40, 5.18, T.col_w(6) - 0.4, 1.3,
           C.AUTOCORR_CONSEQUENCE, size=T.SZ_BODY, color=T.INK, spacing=1.3)
    _notes(s, 17)
    return s


# ---------------------------------------------------------------------------
# 18 — the other two cases
# ---------------------------------------------------------------------------

def slide_18_other_cases(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_RESULT, 18)
    T.slide_title(s, "The same procedure, two further cases",
                  "Nothing in the method is tuned to one dataset")

    spec = C.OTHER_CASES
    widths = [3.75, 1.35, 1.45, 1.20, 1.55, 1.35, 1.38]
    x0, y0, rh = T.MARGIN_IN, 2.66, 0.30
    x = x0
    for label, w in zip(spec["header"], widths):
        align = PP_ALIGN.LEFT if label == "Case" else PP_ALIGN.RIGHT
        T.text(s, x, y0, w, 0.24, label.upper(), size=T.SZ_MICRO,
               color=T.MUTED, bold=True, space=1.1, align=align)
        x += w
    T.line(s, x0, y0 + rh, x0 + sum(widths), y0 + rh, color=T.INK, pt=1.0)

    for r, row in enumerate(spec["rows"]):
        y = y0 + rh + 0.14 + r * 0.66
        highlight = r == spec["highlight_row"]
        if highlight:
            T.rect(s, x0 - 0.14, y - 0.10, sum(widths) + 0.28, 0.56,
                   fill=T.TEAL_SOFT, outline=None)
        x = x0
        for c, (value, w) in enumerate(zip(row, widths)):
            first = c == 0
            T.text(s, x, y, w, 0.34, value,
                   size=T.SZ_SUB if first else 15,
                   color=T.TEAL if highlight and first else T.INK,
                   bold=first or c in (4,),
                   align=PP_ALIGN.LEFT if first else PP_ALIGN.RIGHT,
                   font=T.FONT if first else T.MONO)
            x += w
        if not highlight:
            T.line(s, x0, y + 0.50, x0 + sum(widths), y + 0.50, color=T.RULE,
                   pt=0.5)
    T.text(s, x0 + 0.06, y0 + rh + 0.44, 3.6, 0.24, "walked through here",
           size=T.SZ_MICRO, color=T.TEAL)

    T.rect(s, T.MARGIN_IN, 5.24, 0.036, 0.9, fill=T.AMBER)
    T.text(s, T.MARGIN_IN + 0.24, 5.24, T.col_w(8), 0.9, spec["note"],
           size=T.SZ_BODY, color=T.INK, spacing=1.3)
    T.caption(s, T.MARGIN_IN, 6.42, T.col_w(11),
              "Precision differs across cases mainly because the reference "
              "graphs differ in density — a sparser reference leaves more room "
              "for a reported relationship to fall outside it.")
    _notes(s, 18)
    return s


# ---------------------------------------------------------------------------
# 19 — claims
# ---------------------------------------------------------------------------

def slide_19_claims(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_CLOSE, 19)
    T.slide_title(s, "What this supports, and what it does not",
                  "Equal space, because the second column is part of the result")

    cw = (T.SLIDE_W_IN - 2 * T.MARGIN_IN - 0.60) / 2
    lx = T.MARGIN_IN
    rx = lx + cw + 0.60

    for x, heading, accent in ((lx, "SUPPORTED BY THIS WORK", T.TEAL),
                              (rx, "NOT YET SUPPORTED", T.AMBER)):
        T.rect(s, x, 2.48, cw, 0.042, fill=accent)
        T.text(s, x, 2.60, cw, 0.24, heading, size=T.SZ_MICRO, color=accent,
               bold=True, space=1.2)

    for i, item in enumerate(C.SUPPORTED):
        y = 3.02 + i * 0.86
        T.text(s, lx, y, 0.26, 0.26, "✓", size=T.SZ_BODY, color=T.TEAL,
               bold=True)
        T.text(s, lx + 0.30, y, cw - 0.30, 0.66, item, size=T.SZ_BODY,
               color=T.INK, spacing=1.3)

    for i, (claim, remedy) in enumerate(C.NOT_SUPPORTED):
        y = 3.02 + i * 0.86
        T.text(s, rx, y, 0.26, 0.26, "○", size=T.SZ_BODY, color=T.AMBER,
               bold=True)
        T.text(s, rx + 0.30, y, cw - 0.30, 0.3, claim, size=T.SZ_BODY,
               color=T.INK, spacing=1.3)
        T.text(s, rx + 0.30, y + 0.30, cw - 0.30, 0.3,
               [[("needs  ", {"color": T.MUTED}),
                 (remedy, {"color": T.AMBER})]], size=T.SZ_CAPTION)

    T.line(s, rx - 0.30, 2.48, rx - 0.30, 6.52, color=T.RULE, pt=0.75)
    _notes(s, 19)
    return s


# ---------------------------------------------------------------------------
# 20 — next
# ---------------------------------------------------------------------------

def slide_20_next(prs, data):
    s = T.blank(prs)
    T.header(s, C.ACT_CLOSE, 20)
    T.slide_title(s, "Where this goes next",
                  "The graph now exists, which is what the rest of the objective needs")

    centres, box_w = _pipeline(s, 2.86, filled={0, C.PIPELINE_DELIVERS},
                               outlined_next=set(C.PIPELINE_NEXT),
                               labels_below=False)
    T.text(s, centres[1] - box_w / 2, 3.92, box_w, 0.26, "delivered",
           size=T.SZ_MICRO, color=T.TEAL, bold=True, align=PP_ALIGN.CENTER)

    for i, (headline, detail) in enumerate(C.NEXT_STEPS):
        cx = centres[C.PIPELINE_NEXT[i]] if i < len(C.PIPELINE_NEXT) else 0
        y = 4.62 + i * 0.72
        T.text(s, T.MARGIN_IN, y, 0.34, 0.26, str(i + 1), size=T.SZ_BODY,
               color=T.MUTED, bold=True)
        T.text(s, T.MARGIN_IN + 0.38, y, T.col_w(5), 0.28, headline,
               size=T.SZ_SUB, color=T.INK, bold=True)
        T.text(s, T.MARGIN_IN + 0.38, y + 0.30, T.col_w(5), 0.24, detail,
               size=T.SZ_CAPTION, color=T.MUTED)

    T.rect(s, T.col_x(7), 4.54, T.col_w(5), 1.96, fill=T.TEAL_SOFT,
           outline=None, radius=0.04, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    T.text(s, T.col_x(7) + 0.34, 4.82, T.col_w(5) - 0.68, 1.4,
           "A fault shows up as an anomaly in one variable. Reading the graph "
           "upstream from that variable is what turns “something is wrong” "
           "into “this is where it started.”",
           size=T.SZ_SUB, color=T.TEAL, spacing=1.34)
    _notes(s, 20)
    return s


SLIDES = [
    slide_01_title, slide_02_goal, slide_03_correlation, slide_04_independence,
    slide_05_four_frames, slide_06_partial_correlation, slide_07_fisher,
    slide_08_orientation_rules, slide_09_process, slide_10_preparation,
    slide_11_parameters, slide_12_funnel, slide_13_orientation,
    slide_14_graph, slide_15_assessment, slide_16_sensitivity,
    slide_17_autocorrelation, slide_18_other_cases, slide_19_claims,
    slide_20_next,
]
