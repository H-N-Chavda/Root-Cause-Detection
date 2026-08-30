"""Causal discovery algorithms behind one common interface.

`pc_manual` is the repository's own Phase 1 implementation, kept as a
correctness oracle. The other four are thin wrappers over `tigramite` and
`lingam`; the algorithms themselves are the libraries'.
"""

from .base import CausalDiscoveryAlgorithm
from .lste import LSTEAlgorithm
from .pc import PCAlgorithm
from .pc_manual import PCManualAlgorithm, PCResult, compute_metrics, pc_algorithm
from .pcmci_plus import PCMCIPlusAlgorithm
from .registry import (
    DEFAULT_ALGORITHMS,
    REGISTRY,
    available,
    build,
    get_algorithm_class,
)
from .var_lingam import VARLiNGAMAlgorithm

__all__ = [
    "CausalDiscoveryAlgorithm",
    "PCAlgorithm",
    "PCManualAlgorithm",
    "PCMCIPlusAlgorithm",
    "VARLiNGAMAlgorithm",
    "LSTEAlgorithm",
    "REGISTRY",
    "DEFAULT_ALGORITHMS",
    "available",
    "build",
    "get_algorithm_class",
    # Phase 1 surface, re-exported so existing imports keep working.
    "PCResult",
    "pc_algorithm",
    "compute_metrics",
]
