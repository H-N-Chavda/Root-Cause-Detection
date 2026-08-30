"""Running the algorithms and scoring the results.

`targets` builds the three ground-truth graphs, `runner` drives the algorithm
registry, `report` renders the outputs and `figures` draws them.
"""

from . import figures, report, runner, targets
from .targets import ScoringTargets, build_targets

__all__ = ["runner", "report", "figures", "targets", "build_targets", "ScoringTargets"]
