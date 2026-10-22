"""Composition helpers over theme.py.

There is deliberately no table helper here. A ruled table is the right device
for exactly one thing in this deck -- the assumption matrix, where the data
really is a cross-product -- and everywhere else it flattens an argument into
rows and makes the deck look generated. The helpers below are the alternatives:
cards, ranked bars, dot plots, scatter marks.
"""

import theme as T


def footer(slide, numbers, refs, bottom=7.26, dark=False):
    """Numbered references, resolved at the bottom of the slide that cites them.

    Placed from `bottom` upwards, so a slide citing eight sources grows into
    the margin instead of off the canvas.
    """
    if not numbers:
        return None
    colour = T.RGBColor(0x76, 0x7A, 0x80) if not dark else T.RGBColor(0x6E, 0x76, 0x7E)
    body = "   ".join("[%d] %s" % (n, refs[n]) for n in numbers)
    lines = max(1, -(-len(body) // 172))          # ~172 chars per line at 10pt
    height = lines * 0.175
    y = bottom - height
    T.line(slide, T.MARGIN_IN, y - 0.14, T.SLIDE_W_IN - T.MARGIN_IN, y - 0.14,
           color=T.RULE if not dark else T.RGBColor(0x2E, 0x34, 0x3A), pt=0.5)
    T.text(slide, T.MARGIN_IN, y, T.col_w(12), height, body, size=T.SZ_MICRO,
           color=colour, spacing=1.2)
    return y - 0.14


def takeaway(slide, y, body, accent=T.INK, width=None, size=None):
    """The one-line reading of the slide, on a rule, at the foot of the canvas.

    Every content slide ends with one. If a slide cannot produce a sentence
    here it has no argument and should not exist.
    """
    width = width or T.col_w(12)
    T.rect(slide, T.MARGIN_IN, y, width, 0.034, fill=accent)
    return T.text(slide, T.MARGIN_IN, y + 0.16, width, 0.72, body,
                  size=size or T.SZ_SMALL, color=T.INK, spacing=1.26)


def stat(slide, x, y, value, label, gloss=None, w=2.6, accent=T.INK,
         value_size=26):
    """A single measured number, its name, and an optional one-line gloss."""
    T.text(slide, x, y, w, 0.46, value, size=value_size, color=accent, bold=True)
    T.text(slide, x, y + value_size / 72 + 0.12, w, 0.26, label,
           size=T.SZ_SMALL, color=T.INK, bold=True)
    if gloss:
        T.text(slide, x, y + value_size / 72 + 0.42, w, 0.5, gloss,
               size=T.SZ_MICRO, color=T.MUTED, spacing=1.22)


def eyebrow(slide, x, y, label, color=T.MUTED, w=3.4):
    """Small caps-style label naming a region of the slide, so each panel's
    purpose is readable without the speaker."""
    return T.text(slide, x, y, w, 0.2, label.upper(), size=T.SZ_MICRO,
                  color=color, bold=True, space=1.3)


def card(slide, x, y, w, h, accent, title, subtitle=None, fill=None):
    """A bordered panel with an accent edge, the unit of the small-multiple
    layouts. Returns the inner (x, width) so callers can fill it."""
    T.rect(slide, x, y, w, h, fill=fill, outline=T.RULE, pt=0.75)
    T.rect(slide, x, y, w, 0.035, fill=accent)
    T.text(slide, x + 0.18, y + 0.18, w - 0.36, 0.28, title, size=T.SZ_BODY,
           color=T.INK, bold=True)
    if subtitle:
        T.text(slide, x + 0.18, y + 0.50, w - 0.36, 0.24, subtitle,
               size=T.SZ_MICRO, color=accent, bold=True)
    return x + 0.18, w - 0.36


def keyline(slide, x, y, w, label, value, accent=T.MUTED, label_w=1.15,
            size=None):
    """A label and its value on one line, the label set small in the accent.
    Used where a card needs two or three facts rather than a paragraph."""
    T.text(slide, x, y + 0.03, label_w, 0.24, label.upper(), size=T.SZ_MICRO,
           color=accent, bold=True, space=0.8)
    return T.text(slide, x + label_w, y, w - label_w, 0.5, value,
                  size=size or T.SZ_SMALL, color=T.INK, spacing=1.2)
