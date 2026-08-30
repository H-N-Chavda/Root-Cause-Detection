"""One source of truth for prior knowledge, converted two ways.

The YAML file lists known edges, each flagged `forced` or `forbidden`. From that
single list this module emits:

* a ``link_assumptions`` dictionary for tigramite, accepted by both
  ``run_pcmciplus`` and ``run_pcalg``;
* a ``prior_knowledge`` matrix for ``lingam.DirectLiNGAM``.

The two libraries encode "no edge" and "edge exists, direction unknown"
differently, which is exactly the kind of mismatch that produces a plausible but
wrong graph, so each converter is tested against a hand-built example.

**tigramite encoding.** ``link_assumptions[j][(i, -tau)]`` is a mark describing
the link from ``(i, t - tau)`` to ``(j, t)``. An *absent key* forbids the link
outright; a mark constrains the orientation *if* the link survives the
independence tests. ``PCMCI.build_link_assumptions`` fills in the permissive
default (``"o?o"`` contemporaneous, ``"o?>"`` lagged) for every pair not
mentioned, so the sparse dict written here says only what is actually known.

A contemporaneous link occupies **two** entries and tigramite validates that
they agree -- its docstring requires ``graph[i,j,0] = "-->"`` to be paired with
``graph[j,i,0] = "<--"``. Writing only one side is a silent no-op: the algorithm
finds the edge from the other direction and the constraint never bites.

**What "forbidden" means here.** Forbidding ``A -> B`` rules out that
*direction*, not the adjacency: it is encoded as "if A and B are adjacent, the
edge runs B -> A". To forbid the adjacency itself, declare both directions
forbidden, which removes the keys on both sides.

**lingam encoding.** ``prior_knowledge[effect, cause]`` -- the same
row = effect, column = cause transpose that `var_lingam` documents -- taking:

===  =========================================
 0   no directed path from cause to effect
 1   a directed path exists
-1   unknown (the default for everything else)
===  =========================================

**Scope limit.** The lingam matrix constrains the contemporaneous ``B0`` only.
A lagged entry in the YAML cannot be expressed there and is dropped, with a
warning, rather than silently mapped onto the contemporaneous matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ..utils.logging import get_logger

log = get_logger(__name__)

EdgeKind = Literal["forced", "forbidden"]

#: tigramite marks. "-->" asserts the direction; "" (an absent key) forbids the
#: link; "o?o" / "o?>" say "may exist, direction unknown".
TIGRAMITE_FORCED = "-->"
TIGRAMITE_UNKNOWN_CONTEMPORANEOUS = "o?o"
TIGRAMITE_UNKNOWN_LAGGED = "o?>"

#: lingam prior-knowledge codes.
LINGAM_NO_PATH = 0
LINGAM_PATH = 1
LINGAM_UNKNOWN = -1


@dataclass(frozen=True)
class PriorEdge:
    """One known constraint: `cause -> effect` at `lag`, forced or forbidden."""

    cause: str
    effect: str
    kind: EdgeKind
    lag: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("forced", "forbidden"):
            raise ValueError(f"kind must be forced or forbidden, got {self.kind!r}")
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {self.lag}")
        if self.cause == self.effect:
            raise ValueError(f"self-loop in prior knowledge: {self.cause}")


@dataclass
class PriorKnowledge:
    """A validated list of prior edges over a fixed variable set."""

    edges: list[PriorEdge] = field(default_factory=list)
    source: Path | None = None
    description: str = ""

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriorKnowledge:
        path = Path(path)
        payload = yaml.safe_load(path.read_text()) or {}
        edges = []
        for i, entry in enumerate(payload.get("edges", [])):
            missing = {"cause", "effect", "kind"} - set(entry)
            if missing:
                raise ValueError(f"edges[{i}] is missing {sorted(missing)}")
            edges.append(
                PriorEdge(
                    cause=str(entry["cause"]),
                    effect=str(entry["effect"]),
                    kind=entry["kind"],
                    lag=int(entry.get("lag", 0)),
                    note=str(entry.get("note", "")),
                )
            )
        return cls(
            edges=edges,
            source=path.resolve(),
            description=str(payload.get("description", "")),
        )

    def validate_against(self, var_names: Sequence[str]) -> dict[str, Any]:
        """Checks every named variable exists, and reports what does not.

        Unknown names are reported rather than raising: a controller-loop list
        written for the full plant will legitimately name variables that have no
        column in the data, and dropping those is the right behaviour as long as
        it is visible.
        """
        known = set(var_names)
        applicable = [e for e in self.edges if e.cause in known and e.effect in known]
        dropped = [e for e in self.edges if e not in applicable]
        missing_names = sorted(
            {e.cause for e in dropped if e.cause not in known}
            | {e.effect for e in dropped if e.effect not in known}
        )
        if dropped:
            log.warning(
                "prior knowledge: %d of %d edges name variables absent from the "
                "data (%s) and were dropped",
                len(dropped),
                len(self.edges),
                ", ".join(missing_names),
            )
        return {
            "source": str(self.source) if self.source else None,
            "description": self.description,
            "n_edges_declared": len(self.edges),
            "n_edges_applicable": len(applicable),
            "n_edges_dropped": len(dropped),
            "variables_not_in_data": missing_names,
            "dropped_edges": [
                {"cause": e.cause, "effect": e.effect, "lag": e.lag, "kind": e.kind}
                for e in dropped
            ],
            "applicable_edges": [
                {"cause": e.cause, "effect": e.effect, "lag": e.lag, "kind": e.kind}
                for e in applicable
            ],
        }

    def applicable(self, var_names: Sequence[str]) -> list[PriorEdge]:
        known = set(var_names)
        return [e for e in self.edges if e.cause in known and e.effect in known]

    # -- converters -------------------------------------------------------

    def to_tigramite(
        self, var_names: Sequence[str], tau_max: int, tau_min: int = 0
    ) -> dict[int, dict[tuple[int, int], str]]:
        """A `link_assumptions` dictionary.

        Built by handing a sparse "what is known" dict to
        `PCMCI.build_link_assumptions`, which fills the permissive default for
        every unmentioned pair. That helper is used rather than writing the full
        dictionary by hand because it is the library's own definition of the
        default, and duplicating it would be a second place to get wrong.
        """
        from tigramite.pcmci import PCMCI

        var_names = list(var_names)
        index = {name: i for i, name in enumerate(var_names)}
        known: dict[int, dict[tuple[int, int], str]] = {
            j: {} for j in range(len(var_names))
        }

        applicable = [
            e for e in self.applicable(var_names) if _in_range(e, tau_min, tau_max)
        ]
        forbidden_pairs = {
            (index[e.cause], index[e.effect], e.lag)
            for e in applicable
            if e.kind == "forbidden"
        }

        for edge in applicable:
            i, j = index[edge.cause], index[edge.effect]

            if edge.lag > 0:
                # Time is directional, so a lagged link has exactly one entry.
                key = (i, -edge.lag)
                known[j][key] = "" if edge.kind == "forbidden" else TIGRAMITE_FORCED
                continue

            # A contemporaneous link has TWO entries and tigramite validates
            # that they agree: its own docstring requires graph[i,j,0] = "-->"
            # to be paired with graph[j,i,0] = "<--". Writing only one side
            # leaves the constraint unenforced -- the algorithm simply finds the
            # edge from the other direction -- which is a silent no-op, not an
            # error.
            if edge.kind == "forced":
                known[j][(i, 0)] = TIGRAMITE_FORCED
                known[i][(j, 0)] = "<--"
            elif (j, i, 0) in forbidden_pairs:
                # Both directions ruled out: no adjacency at all.
                known[j][(i, 0)] = ""
                known[i][(j, 0)] = ""
            else:
                # Only this direction is ruled out. A mark constrains the
                # orientation if the link exists; it does not force the link to
                # exist, so this says "if i and j are adjacent, it runs j -> i"
                # without asserting an adjacency nobody claimed.
                known[j][(i, 0)] = "<--"
                known[i][(j, 0)] = TIGRAMITE_FORCED

        return PCMCI.build_link_assumptions(
            link_assumptions_absent_link_means_no_knowledge=known,
            n_component_time_series=len(var_names),
            tau_max=tau_max,
            tau_min=tau_min,
        )

    def to_lingam(self, var_names: Sequence[str]) -> np.ndarray:
        """A `prior_knowledge` matrix for `DirectLiNGAM`.

        Shape ``(n, n)`` with ``matrix[effect, cause]``, filled with -1
        (unknown) and overwritten only where the YAML says something. Lagged
        entries are dropped with a warning: `DirectLiNGAM` sees only the
        contemporaneous residuals, so there is nowhere to put them.
        """
        var_names = list(var_names)
        index = {name: i for i, name in enumerate(var_names)}
        n = len(var_names)
        matrix = np.full((n, n), LINGAM_UNKNOWN, dtype=int)
        # A variable is never its own contemporaneous cause; B0 has a zero
        # diagonal by construction, so stating it removes n useless unknowns.
        np.fill_diagonal(matrix, LINGAM_NO_PATH)

        n_lagged_dropped = 0
        for edge in self.applicable(var_names):
            if edge.lag != 0:
                n_lagged_dropped += 1
                continue
            i, j = index[edge.cause], index[edge.effect]
            matrix[j, i] = LINGAM_PATH if edge.kind == "forced" else LINGAM_NO_PATH

        if n_lagged_dropped:
            log.warning(
                "prior knowledge: %d lagged edge(s) dropped for lingam; "
                "DirectLiNGAM's prior_knowledge constrains the contemporaneous "
                "B0 only and cannot express a lagged constraint",
                n_lagged_dropped,
            )
        return matrix


def _in_range(edge: PriorEdge, tau_min: int, tau_max: int) -> bool:
    if tau_min <= edge.lag <= tau_max:
        return True
    log.warning(
        "prior knowledge: %s -> %s at lag %d is outside the searched range "
        "[%d, %d] and was dropped",
        edge.cause,
        edge.effect,
        edge.lag,
        tau_min,
        tau_max,
    )
    return False


def load_prior_knowledge(path: str | Path | None) -> PriorKnowledge | None:
    """Loads a prior-knowledge YAML, or returns None when no path is given."""
    if path is None:
        return None
    return PriorKnowledge.from_yaml(path)
