"""Scoring: structural Hamming distance and edge-level detection scores.

The Markov checker, the out-of-sample predictive score and the bootstrap
stability metric belong to a later phase and are deliberately absent.
"""

from .edges import EdgeScore, score_edges
from .shd import (
    DEFAULT_UNDIRECTED_WEIGHT,
    SHDResult,
    align_to,
    structural_hamming_distance,
)

__all__ = [
    "structural_hamming_distance",
    "SHDResult",
    "align_to",
    "DEFAULT_UNDIRECTED_WEIGHT",
    "score_edges",
    "EdgeScore",
]
