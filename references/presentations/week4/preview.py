"""Rasterises the built deck to one PNG per slide, for visual review.

    python preview.py [out_dir]

Why not LibreOffice: on this machine the LibreOffice download runs at ~7 KB/s
and direct TLS to download.documentfoundation.org fails, so there is no
converter to call. This walks the shape tree python-pptx already gives us and
draws it with Pillow using the real font file, which is what matters here --
it lays text out with true glyph advances instead of a characters-per-line
estimate, so the picture shows the wrapping and the collisions that a reader
would actually see.

It is a proof reader, not a renderer: gradients, arrowheads, shadows and
PowerPoint's own hyphenation are not reproduced. Geometry and text flow are.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
DECK = HERE / "Week4_PCMCIplus_VARLiNGAM_LSTE.pptx"
DPI = 120

_FONT_DIR = Path("/System/Library/Fonts/Supplemental")
_FACES = {
    (False, False): _FONT_DIR / "Helvetica Neue.ttc",
    (True, False): _FONT_DIR / "Helvetica Neue.ttc",
}
_FALLBACK = {
    (False, False): _FONT_DIR / "Arial.ttf",
    (True, False): _FONT_DIR / "Arial Bold.ttf",
    (False, True): _FONT_DIR / "Arial Italic.ttf",
    (True, True): _FONT_DIR / "Arial Bold Italic.ttf",
}
_cache = {}


def font(size_pt, bold=False, italic=False):
    key = (round(size_pt, 1), bold, italic)
    if key not in _cache:
        path = _FALLBACK[(bold, italic)]
        _cache[key] = ImageFont.truetype(str(path), int(round(size_pt * DPI / 72)))
    return _cache[key]


def px(emu):
    return Emu(emu).inches * DPI


def rgb(color, default=(0, 0, 0)):
    try:
        v = color.rgb
        return (v[0], v[1], v[2])
    except Exception:
        return default


def draw_shape(img, drw, shape):
    x, y = px(shape.left), px(shape.top)
    w, h = px(shape.width), px(shape.height)
    box = [x, y, x + w, y + h]

    if shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE or shape.has_text_frame:
        fill_rgb = line_rgb = None
        try:
            if shape.fill.type is not None and shape.fill.type == 1:
                fill_rgb = rgb(shape.fill.fore_color)
        except Exception:
            pass
        try:
            if shape.line.fill.type == 1:
                line_rgb = rgb(shape.line.color)
        except Exception:
            pass
        width = 1
        try:
            if shape.line.width:
                width = max(1, int(round(px(shape.line.width))))
        except Exception:
            pass

        name = getattr(shape, "shape_type", None)
        is_oval = False
        try:
            is_oval = shape._element.spPr.prstGeom.get("prst") == "ellipse"
        except Exception:
            pass

        if fill_rgb or line_rgb:
            if is_oval:
                drw.ellipse(box, fill=fill_rgb, outline=line_rgb, width=width)
            elif w > 0 and h > 0:
                drw.rectangle(box, fill=fill_rgb, outline=line_rgb, width=width)

    if shape.shape_type == MSO_SHAPE_TYPE.LINE or shape._element.tag.endswith(
            "}cxnSp"):
        colour = (200, 200, 200)
        try:
            colour = rgb(shape.line.color, (200, 200, 200))
        except Exception:
            pass
        width = 1
        try:
            if shape.line.width:
                width = max(1, int(round(px(shape.line.width))))
        except Exception:
            pass
        flip_h = shape._element.xfrm.get("flipH") == "1"
        flip_v = shape._element.xfrm.get("flipV") == "1"
        x1, x2 = (x + w, x) if flip_h else (x, x + w)
        y1, y2 = (y + h, y) if flip_v else (y, y + h)
        drw.line([x1, y1, x2, y2], fill=colour, width=width)
        return

    if shape.has_text_frame:
        draw_text(drw, shape, x, y, w, h)


def wrap(runs, fnts, max_w):
    """Greedy word wrap over a list of (text, font) runs. Returns lines, each
    a list of (text, font) pieces."""
    lines, cur, cur_w = [], [], 0.0
    for text, fnt in zip([r[0] for r in runs], fnts):
        for i, word in enumerate(text.split(" ")):
            piece = word if (i == 0 and not cur) else " " + word
            wpx = fnt.getlength(piece)
            if cur and cur_w + wpx > max_w:
                lines.append(cur)
                piece = word
                wpx = fnt.getlength(piece)
                cur, cur_w = [], 0.0
            cur.append((piece, fnt))
            cur_w += wpx
    if cur:
        lines.append(cur)
    return lines or [[("", fnts[0])]]


def draw_text(drw, shape, x, y, w, h):
    tf = shape.text_frame
    ml = px(tf.margin_left or 0)
    mr = px(tf.margin_right or 0)
    mt = px(tf.margin_top or 0)
    inner_w = max(w - ml - mr, 1)

    blocks = []
    for p in tf.paragraphs:
        runs = [(r.text, r) for r in p.runs if r.text]
        if not runs:
            blocks.append((None, 0.0, p))
            continue
        fnts = [font(r.font.size.pt if r.font.size else 12,
                     bool(r.font.bold), bool(r.font.italic)) for _, r in runs]
        sizes = [r.font.size.pt if r.font.size else 12 for _, r in runs]
        lines = wrap(runs, fnts, inner_w) if tf.word_wrap is not False else [
            [(t, f) for (t, _), f in zip(runs, fnts)]]
        lh = max(sizes) * (p.line_spacing or 1.15) * DPI / 72
        colour = rgb(runs[0][1].font.color, (26, 29, 33))
        blocks.append((lines, lh, p, colour))

    total = sum(len(b[0]) * b[1] for b in blocks if b[0])
    cy = y + mt
    if tf.vertical_anchor == MSO_ANCHOR.MIDDLE:
        cy = y + (h - total) / 2

    for block in blocks:
        if not block[0]:
            continue
        lines, lh, p, colour = block
        for line in lines:
            line_w = sum(f.getlength(t) for t, f in line)
            if p.alignment == PP_ALIGN.CENTER:
                cx = x + ml + (inner_w - line_w) / 2
            elif p.alignment == PP_ALIGN.RIGHT:
                cx = x + ml + inner_w - line_w
            else:
                cx = x + ml
            for text, fnt in line:
                drw.text((cx, cy), text, font=fnt, fill=colour)
                cx += fnt.getlength(text)
            cy += lh


def render(deck=DECK, out_dir=None):
    out_dir = Path(out_dir or HERE / "_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    prs = Presentation(str(deck))
    W = int(round(Emu(prs.slide_width).inches * DPI))
    H = int(round(Emu(prs.slide_height).inches * DPI))

    paths = []
    for n, slide in enumerate(prs.slides, start=1):
        bg = (250, 250, 247)
        try:
            bg = rgb(slide.background.fill.fore_color, bg)
        except Exception:
            pass
        img = Image.new("RGB", (W, H), bg)
        drw = ImageDraw.Draw(img)
        for shape in slide.shapes:
            try:
                draw_shape(img, drw, shape)
            except Exception as exc:                      # keep going
                print("  slide %d: skipped %s (%s)" % (n, shape.shape_id, exc))
        path = out_dir / ("slide-%02d.png" % n)
        img.save(path)
        paths.append(path)
        print(path)
    return paths


if __name__ == "__main__":
    render(out_dir=sys.argv[1] if len(sys.argv) > 1 else None)
