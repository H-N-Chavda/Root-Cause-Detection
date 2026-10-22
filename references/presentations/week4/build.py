"""Builds Week4_PCMCIplus_VARLiNGAM_LSTE.pptx.

    python build.py

Every number on every slide lives in content.py with the file it came from
named above it, so the deck cannot quietly drift away from the runs it reports.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # theme.py lives one level up
sys.path.insert(0, str(HERE))

from pptx import Presentation      # noqa: E402

import slides                      # noqa: E402
import theme as T                  # noqa: E402

OUTPUT = HERE / "Week4_PCMCIplus_VARLiNGAM_LSTE.pptx"


def build():
    prs = Presentation()
    prs.slide_width = T.SLIDE_W
    prs.slide_height = T.SLIDE_H

    for number, make_slide in enumerate(slides.SLIDES, start=1):
        make_slide(prs)
        print("  %2d  %s" % (number, make_slide.__name__))

    prs.save(OUTPUT)
    print("\nsaved %s" % OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
