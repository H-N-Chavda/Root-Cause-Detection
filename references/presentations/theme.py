"""Visual system for the presentation: palette, type, grid and drawing helpers.

Everything here is plain functions over python-pptx objects. Slides in slides.py
compose these; no slide should need to touch python-pptx internals directly.
"""

import math

from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
MARGIN_IN = 0.65

SLIDE_W = Inches(SLIDE_W_IN)
SLIDE_H = Inches(SLIDE_H_IN)

# 12-column grid across the usable width
_USABLE = SLIDE_W_IN - 2 * MARGIN_IN
GUTTER_IN = 0.17
COL_IN = (_USABLE - 11 * GUTTER_IN) / 12


def col_x(index):
    """Left edge of grid column `index` (0-11), in inches."""
    return MARGIN_IN + index * (COL_IN + GUTTER_IN)


def col_w(span):
    """Width of `span` grid columns, in inches."""
    return span * COL_IN + (span - 1) * GUTTER_IN


# ---------------------------------------------------------------------------
# Palette — colours carry meaning, never decoration
# ---------------------------------------------------------------------------

PAPER = RGBColor(0xFA, 0xFA, 0xF7)
PAPER_DIM = RGBColor(0xEE, 0xEC, 0xE6)
INK = RGBColor(0x1A, 0x1D, 0x21)
INK_DARK = RGBColor(0x14, 0x18, 0x1D)
ON_DARK = RGBColor(0xEC, 0xEA, 0xE4)

TEAL = RGBColor(0x1A, 0x7F, 0x72)
TEAL_SOFT = RGBColor(0xD5, 0xE8, 0xE4)
SLATE = RGBColor(0x3D, 0x6B, 0x99)
SLATE_SOFT = RGBColor(0xDA, 0xE4, 0xEF)
AMBER = RGBColor(0xB8, 0x86, 0x0B)
AMBER_SOFT = RGBColor(0xF5, 0xEA, 0xCE)

RULE = RGBColor(0xC8, 0xC6, 0xC0)
MUTED = RGBColor(0x6B, 0x6E, 0x73)

FONT = "Aptos"
MONO = "Consolas"

# Type scale (pt)
SZ_TITLE = 30
SZ_SUB = 16
SZ_BODY = 13.5
SZ_SMALL = 11.5
SZ_CAPTION = 10
SZ_MICRO = 8.5


# ---------------------------------------------------------------------------
# Slide scaffolding
# ---------------------------------------------------------------------------

def blank(prs, dark=False):
    """Adds a slide with no placeholders and a solid background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = INK_DARK if dark else PAPER
    return slide


def header(slide, section, number, accent=TEAL, dark=False):
    """Hairline plus section name and slide number — the deck's only furniture.

    The section name's colour shifts per act; that is what replaces divider
    slides as the navigation cue.
    """
    y = MARGIN_IN + 0.42
    hair = RGBColor(0x33, 0x39, 0x40) if dark else RULE
    line(slide, MARGIN_IN, y, SLIDE_W_IN - MARGIN_IN, y, color=hair, pt=0.75)
    text(slide, MARGIN_IN, y - 0.24, col_w(8), 0.2, section,
         size=SZ_MICRO, color=accent, bold=True, space=1.4)
    text(slide, SLIDE_W_IN - MARGIN_IN - 0.6, y - 0.24, 0.6, 0.2, str(number),
         size=SZ_MICRO, color=MUTED if not dark else RGBColor(0x7A, 0x80, 0x88),
         align=PP_ALIGN.RIGHT)


def slide_title(slide, title, kicker=None, dark=False, width=None, y=None):
    """Standard title block. `kicker` is the one-line thesis under the title."""
    ink = ON_DARK if dark else INK
    top = y if y is not None else MARGIN_IN + 0.62
    width = width or col_w(10)
    text(slide, MARGIN_IN, top, width, 0.5, title,
         size=SZ_TITLE, color=ink, bold=True)
    if kicker:
        text(slide, MARGIN_IN, top + 0.52, width, 0.34, kicker,
             size=SZ_SUB, color=MUTED if not dark else RGBColor(0x9A, 0xA0, 0xA8))
    return top + (0.98 if kicker else 0.58)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def text(slide, x, y, w, h, body, size=SZ_BODY, color=INK, bold=False,
         align=PP_ALIGN.LEFT, font=FONT, italic=False, spacing=None,
         anchor=MSO_ANCHOR.TOP, space=None, wrap=True):
    """Adds a textbox. `body` may be a string, or a list of paragraphs where
    each paragraph is a string or a list of (text, overrides) runs.
    """
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    paragraphs = body if isinstance(body, list) else [body]
    for i, item in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        _fill_paragraph(p, item, size, color, bold, font, italic, space)
    return box


def _fill_paragraph(p, item, size, color, bold, font, italic, space):
    runs = item if isinstance(item, list) else [(item, {})]
    for run_text, override in runs:
        r = p.add_run()
        r.text = run_text
        f = r.font
        f.name = override.get("font", font)
        f.size = Pt(override.get("size", size))
        f.bold = override.get("bold", bold)
        f.italic = override.get("italic", italic)
        f.color.rgb = override.get("color", color)
        spc = override.get("space", space)
        if spc:
            _set_char_spacing(r, spc)


def _set_char_spacing(run, points):
    """Letter-spacing, used only for the small caps-style furniture labels."""
    run.font._rPr.set("spc", str(int(points * 100)))


def rich(slide, x, y, w, h, runs, size=SZ_BODY, color=INK, align=PP_ALIGN.LEFT,
         font=FONT, bold=False, spacing=None):
    """Single paragraph built from (text, overrides) runs — used for formulas
    whose individual terms are tinted to match an adjacent diagram."""
    return text(slide, x, y, w, h, [runs], size=size, color=color, align=align,
                font=font, bold=bold, spacing=spacing)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def line(slide, x1, y1, x2, y2, color=RULE, pt=1.0, dash=None,
         arrow=False, back_arrow=False):
    """Straight connector. Set colour and dash before arrowheads so the
    resulting <a:ln> child order matches the DrawingML schema."""
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(pt)
    if dash:
        conn.line.dash_style = dash
    if arrow:
        _arrowhead(conn.line, "tailEnd")
    if back_arrow:
        _arrowhead(conn.line, "headEnd")
    conn.shadow.inherit = False
    return conn


def _arrowhead(line_format, tag_name, size="med"):
    ln = line_format._get_or_add_ln()
    tag = qn("a:%s" % tag_name)
    for existing in ln.findall(tag):
        ln.remove(existing)
    ln.append(ln.makeelement(tag, {"type": "triangle", "w": size, "len": size}))


def rect(slide, x, y, w, h, fill=None, outline=None, pt=1.0, radius=None,
         shape=MSO_SHAPE.RECTANGLE, dash=None):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    if outline is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = outline
        box.line.width = Pt(pt)
        if dash:
            box.line.dash_style = dash
    box.shadow.inherit = False
    if radius is not None and box.adjustments:
        box.adjustments[0] = radius
    box.text_frame.word_wrap = True
    return box


def pill(slide, x, y, w, h, label, fill=None, outline=TEAL, color=TEAL,
         size=SZ_CAPTION, bold=True):
    box = rect(slide, x, y, w, h, fill=fill, outline=outline, pt=1.0,
               radius=0.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = box.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _fill_paragraph(p, label, size, color, bold, FONT, False, None)
    return box


def node(slide, cx, cy, d, label, fill=PAPER, outline=INK, pt=1.25,
         label_color=INK, label_size=SZ_SMALL, bold=False):
    """Graph node: a circle centred on (cx, cy) with a centred label."""
    box = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2), Inches(d), Inches(d)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = outline
    box.line.width = Pt(pt)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _fill_paragraph(p, label, label_size, label_color, bold, FONT, False, None)
    return box


def edge(slide, a, b, r_a, r_b, color=INK, pt=1.25, arrow=False,
         back_arrow=False, dash=None):
    """Edge between two node centres, trimmed so it stops at the circle rims."""
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6:
        return None
    ux, uy = dx / dist, dy / dist
    return line(slide,
                x1 + ux * r_a, y1 + uy * r_a,
                x2 - ux * r_b, y2 - uy * r_b,
                color=color, pt=pt, arrow=arrow, back_arrow=back_arrow, dash=dash)


def caption(slide, x, y, w, body, color=MUTED, size=SZ_CAPTION, h=0.3,
            italic=False):
    return text(slide, x, y, w, h, body, size=size, color=color, italic=italic,
                spacing=1.15)


def legend(slide, x, y, items, size=SZ_CAPTION, gap=1.9, swatch=0.11):
    """Horizontal legend: items is a list of (colour, label, dash_or_None)."""
    cx = x
    for color, label, dash in items:
        if dash is None:
            rect(slide, cx, y + 0.045, swatch, swatch, fill=color)
        else:
            line(slide, cx, y + 0.1, cx + 0.24, y + 0.1, color=color, pt=1.5,
                 dash=dash)
        text(slide, cx + (0.32 if dash else swatch + 0.09), y - 0.01,
             gap - 0.34, 0.2, label, size=size, color=MUTED)
        cx += gap
    return cx


def polyline(slide, points, color=INK, pt=1.5, fill=None, close=False):
    """Freeform path through `points` [(x, y), ...] in inches.

    Used for the curves and small charts, so they stay vector and match the
    deck's palette exactly rather than carrying a chart engine's own styling.
    """
    builder = slide.shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]))
    builder.add_line_segments(
        [(Inches(x), Inches(y)) for x, y in points[1:]], close=close
    )
    shape = builder.convert_to_shape()
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if pt:
        shape.line.color.rgb = color
        shape.line.width = Pt(pt)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def normal_curve(x_span, height, sigma=1.0, samples=61):
    """Points of a standard normal bell, scaled into a box of `height`.

    Returns unit-x offsets paired with heights so callers can position freely.
    """
    x0, x1 = x_span
    points = []
    for i in range(samples):
        t = -3.2 + 6.4 * i / (samples - 1)
        y = math.exp(-0.5 * (t / sigma) ** 2)
        points.append((x0 + (x1 - x0) * i / (samples - 1), y * height, t))
    return points


def mini_axes(slide, x, y, w, h, color=RULE):
    """L-shaped axes for the small charts."""
    line(slide, x, y + h, x + w, y + h, color=color, pt=0.75)
    line(slide, x, y, x, y + h, color=color, pt=0.75)


DASH = MSO_LINE_DASH_STYLE.DASH
