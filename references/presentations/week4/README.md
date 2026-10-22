# Week 4 deck — PCMCI+, VAR-LiNGAM and LSTE on Tennessee Eastman

```bash
../../../.venv/bin/python build.py      # writes the .pptx
../../../.venv/bin/python lint.py       # geometry check, must print 0 warnings
../../../.venv/bin/python preview.py    # one PNG per slide in _preview/
```

| File | What it holds |
|---|---|
| `content.py` | every fact on every slide, each with the repo file it came from |
| `slides.py` | one function per slide; the drawing |
| `layout.py` | the repeated devices: footer, takeaway, card, stat |
| `theme.py` | palette, type scale, primitives. Forked from `../theme.py`, two steps larger, McKinsey-style top bar |
| `lint.py` | estimates rendered text height; flags overflow and collisions |
| `preview.py` | rasterises the shape tree with Pillow |
| `render.py` | LibreOffice route, unused — see below |

## Notes

- `preview.py`, not LibreOffice: the LibreOffice download runs at ~7 KB/s on
  this network and direct TLS to `download.documentfoundation.org` fails.
  `render.py` is the LibreOffice path if that host ever becomes reachable.
- Fonts: the deck asks for Aptos, which is **not installed on this machine**,
  so PowerPoint here will substitute. `preview.py` measures with Arial, which
  is close. Check the deck on the presenting machine before the meeting.
- Numbers come from `results/runs/run-20260831-025032`,
  `results/lste_only/run-20260831-025653` and `docs/EDA_REPORT.md`. Nothing
  comes from the uncommitted `lste.py` / `var_lingam.py` fixes; slide 10 says so.
