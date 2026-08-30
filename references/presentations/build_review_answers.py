"""Builds Review_Answers.pptx.

    python build_review_answers.py

A short deck answering the five questions raised in the last review, one slide
each. Reuses the visual system in theme.py so it sits alongside the main deck.
"""

from pathlib import Path

from pptx import Presentation

import review_answers
import theme as T

OUTPUT = Path(__file__).resolve().parent / "Review_Answers.pptx"


def build():
    prs = Presentation()
    prs.slide_width = T.SLIDE_W
    prs.slide_height = T.SLIDE_H

    print("building %d slides ..." % len(review_answers.SLIDES))
    for number, make_slide in enumerate(review_answers.SLIDES, start=1):
        make_slide(prs)
        print("  %d  %s" % (number, make_slide.__name__))

    prs.save(OUTPUT)
    print("\nsaved %s" % OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
