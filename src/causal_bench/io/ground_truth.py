"""Ground-truth adjacency-matrix loaders.

As with `datasets`, these are re-exports from `causal_bench.algorithms.pc`
pending the lift of the no-refactor freeze on that module.
"""

from __future__ import annotations

from ..algorithms.pc import format_ground_truth_info, load_ground_truth

__all__ = ["load_ground_truth", "format_ground_truth_info"]
