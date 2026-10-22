"""Renders the built deck to one PNG per slide, for visual review.

    python render.py [out_dir]

Goes through LibreOffice: pptx -> pdf -> png, because LibreOffice's direct png
filter only ever emits the first slide. Requires LibreOffice and pdftoppm
(the latter ships with poppler; macOS also has `sips` but it cannot page a PDF).
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "Week4_PCMCIplus_VARLiNGAM_LSTE.pptx"
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def render(out_dir, dpi=110):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    subprocess.run(
        [SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir),
         str(DECK)],
        check=True, capture_output=True,
    )
    pdf = out_dir / (DECK.stem + ".pdf")
    if not pdf.exists():
        raise SystemExit("LibreOffice produced no PDF")

    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(out_dir / "slide")],
        check=True,
    )
    pages = sorted(out_dir.glob("slide-*.png"))
    for page in pages:
        print(page)
    return pages


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else HERE / "_preview")
