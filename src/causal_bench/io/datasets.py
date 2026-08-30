"""Dataset loaders.

`load_dataset` still lives in `causal_bench.algorithms.pc_manual` (the former
`utils.py`), which the restructure was required to move verbatim. This module
is the import site the layout calls for and re-exports it unchanged; when the
freeze on `pc.py` lifts, the function body moves here and the re-export is
deleted, with no change to any caller.
"""

from __future__ import annotations

from ..algorithms.pc_manual import load_dataset

__all__ = ["load_dataset"]
