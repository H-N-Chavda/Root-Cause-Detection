"""Builds PC_Algorithm.pptx.

    python build_presentation.py

Runs the analysis on the Tennessee Eastman dataset, then adds the twenty slides
defined in slides.py. Every figure on every slide comes either from that run or
from content.py, so the deck cannot drift out of step with the results.
"""

from pathlib import Path

from pptx import Presentation

import analysis
import slides
import theme as T

OUTPUT = Path(__file__).resolve().parent / "PC_Algorithm.pptx"


def gather_data():
    """Collects everything the slides need from the actual analysis."""
    print("running the analysis on the Tennessee Eastman dataset ...")
    data = analysis.tennessee_result()
    data["series"] = analysis.series_slice(variable_index=0, count=400)
    print("  %d variables, %d samples, %d relationships found"
          % (data["n_vars"], data["n_samples"], len(data["skeleton"])))
    print("  %d confirmed by the reference graph, %d additional, %d missed"
          % (len(data["agreeing"]), len(data["additional"]), len(data["missed"])))
    return data


def build():
    data = gather_data()

    prs = Presentation()
    prs.slide_width = T.SLIDE_W
    prs.slide_height = T.SLIDE_H

    print("building %d slides ..." % len(slides.SLIDES))
    for number, make_slide in enumerate(slides.SLIDES, start=1):
        make_slide(prs, data)
        print("  %2d  %s" % (number, make_slide.__name__))

    prs.save(OUTPUT)
    print("\nsaved %s" % OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
