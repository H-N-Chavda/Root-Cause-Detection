"""Conversions between graph forms: lagged to summary, DAG to CPDAG, latent projection.

These are the parts most likely to hold a silent bug -- a transposed matrix or a
reversed edge still produces a plausible-looking SHD -- so every function here
has a test against a hand-built example whose expected output is written out by
hand rather than computed the same way twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import networkx as nx
import numpy as np

from .representation import DIRECTED_MARKS, UNDIRECTED_MARKS, CausalGraph


def lagged_to_summary(graph: CausalGraph) -> CausalGraph:
    """Collapses a lagged graph to a summary graph over the same variables.

    An edge X -> Y is drawn if *any* lag carries it. The ground truth has no lag
    information, so a lag-resolved SHD is not defined against it; the full
    lagged graph is kept in the output for the later propagation-delay check.

    Self-loops are dropped. A variable's own past is autocorrelation, and the
    ground truth has no self-loops, so keeping them would only inflate the false
    positive count with a quantity nothing scores against.

    When one lag says X -> Y and another says Y -> X, the summary edge is
    marked ``x-x`` (conflicting) rather than silently picking one. A cycle
    between two variables cannot be represented in an acyclic summary, and
    hiding the conflict would be a fabricated orientation.
    """
    n = graph.n_vars
    summary = CausalGraph.empty(list(graph.var_names), tau_max=0, meta=dict(graph.meta))

    forward: set[tuple[int, int]] = set()
    adjacent: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(graph.tau_max + 1):
                mark = str(graph.links[i, j, tau])
                if mark in DIRECTED_MARKS:
                    forward.add((i, j))
                    adjacent.add((min(i, j), max(i, j)))
                elif mark in UNDIRECTED_MARKS:
                    adjacent.add((min(i, j), max(i, j)))

    for i, j in sorted(adjacent):
        both_ways = (i, j) in forward and (j, i) in forward
        if both_ways:
            summary.links[i, j, 0] = "x-x"
            summary.links[j, i, 0] = "x-x"
        elif (i, j) in forward:
            summary.links[i, j, 0] = "-->"
            summary.links[j, i, 0] = "<--"
        elif (j, i) in forward:
            summary.links[j, i, 0] = "-->"
            summary.links[i, j, 0] = "<--"
        else:
            summary.links[i, j, 0] = "o-o"
            summary.links[j, i, 0] = "o-o"

    summary.meta = {**graph.meta, "derived_from": "lagged_to_summary"}
    return summary


# ---------------------------------------------------------------------------
# CausalGraph <-> networkx
# ---------------------------------------------------------------------------


def summary_to_digraph(graph: CausalGraph) -> nx.DiGraph:
    """Directed edges of a summary graph as a networkx DiGraph.

    Undirected marks are dropped: a DiGraph cannot express "adjacent, direction
    unknown", and rendering one as a pair of arcs would invent two claims the
    algorithm never made. Use `adjacency_pairs` when the skeleton is wanted.
    """
    if graph.tau_max != 0:
        raise ValueError("summary_to_digraph expects a summary graph (tau_max = 0)")
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.var_names)
    for cause, effect, _ in graph.directed_edges():
        digraph.add_edge(cause, effect)
    return digraph


def digraph_to_summary(digraph: nx.DiGraph, var_names: Sequence[str]) -> CausalGraph:
    """A DiGraph back into a tau_max = 0 CausalGraph over `var_names`."""
    index = {name: i for i, name in enumerate(var_names)}
    graph = CausalGraph.empty(list(var_names), tau_max=0)
    for cause, effect in digraph.edges():
        if cause == effect:
            continue
        i, j = index[cause], index[effect]
        graph.links[i, j, 0] = "-->"
        # Only mirror when the reverse arc is absent; a 2-cycle would otherwise
        # overwrite one direction's mark with the other's mirror.
        if not digraph.has_edge(effect, cause):
            graph.links[j, i, 0] = "<--"
        else:
            graph.links[j, i, 0] = "-->"
    return graph


def adjacency_matrix_to_graph(matrix: np.ndarray, var_names: Sequence[str]) -> CausalGraph:
    """A binary directed adjacency matrix, ``matrix[i, j] == 1`` meaning i -> j.

    This is the ground-truth file convention used throughout the project, and it
    is the *opposite* of lingam's coefficient matrices; see `var_lingam`.
    """
    matrix = (np.asarray(matrix) != 0).astype(int)
    graph = CausalGraph.empty(list(var_names), tau_max=0)
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j]:
                graph.links[i, j, 0] = "-->"
                if not matrix[j, i]:
                    graph.links[j, i, 0] = "<--"
                else:
                    graph.links[j, i, 0] = "-->"
    return graph


# ---------------------------------------------------------------------------
# Latent projection
# ---------------------------------------------------------------------------


def latent_projection(digraph: nx.DiGraph, latent_nodes: Iterable[str]) -> nx.DiGraph:
    """Marginalises `latent_nodes` out of a DAG.

    For each latent node, every parent is connected to every child, then the
    node is removed. A latent mediator L in ``A -> L -> B`` therefore leaves
    ``A -> B``: the dependence survives marginalisation, so a method run on the
    observed variables can and should find it.

    Latents are removed one at a time and in a fixed (sorted) order, so a chain
    of two latents projects transitively and the result is deterministic.

    This is the "observable" target. The induced subgraph -- which simply drops
    every edge touching a latent -- is the stricter alternative and is scored
    alongside it, because the two disagree exactly where marginalisation
    matters.

    Note this returns a DAG, not a PAG: a latent *confounder* (a common cause of
    two observed nodes) genuinely induces a bidirected edge that no DAG can
    express, and networkx has no PAG type. For the case at hand every latent is
    a mediator or a source, so nothing is lost; a confounding latent would need
    an ADMG and is rejected below rather than silently mis-projected.
    """
    latents = sorted(set(latent_nodes) & set(digraph.nodes))
    projected = digraph.copy()

    for latent in latents:
        parents = list(projected.predecessors(latent))
        children = list(projected.successors(latent))
        # A latent with two or more children and no parent is a pure confounder:
        # marginalising it produces a bidirected edge, not a directed one.
        if len(children) > 1 and not parents:
            raise ValueError(
                f"latent node {latent!r} is a common cause of {children} with no "
                "parents; marginalising it needs a bidirected edge, which a DAG "
                "cannot represent. Handle this case explicitly before projecting."
            )
        for parent in parents:
            for child in children:
                if parent != child:
                    projected.add_edge(parent, child)
        projected.remove_node(latent)

    return projected


def induced_subgraph(digraph: nx.DiGraph, keep_nodes: Iterable[str]) -> nx.DiGraph:
    """The subgraph on `keep_nodes`, dropping every edge that touches a dropped
    node. Strictly less than the latent projection, and scored as the secondary
    target."""
    keep = [n for n in digraph.nodes if n in set(keep_nodes)]
    return digraph.subgraph(keep).copy()


# ---------------------------------------------------------------------------
# DAG -> CPDAG
# ---------------------------------------------------------------------------


def dag_to_cpdag(digraph: nx.DiGraph, var_names: Sequence[str]) -> CausalGraph:
    """The CPDAG (Markov equivalence class) of a DAG.

    A perfect observational method cannot do better than the CPDAG: every DAG in
    the class implies the same conditional independencies, so any edge the class
    leaves undirected is undirectable from observational data by *any* method.
    Scoring a CPDAG-returning algorithm against the DAG would charge it for that
    theoretical limit.

    Implemented as the standard two-step construction -- orient the unshielded
    colliders, then apply Meek's rules R1-R3 to a fixed point -- because
    networkx has no CPDAG routine and the alternatives were ruled out as
    dependencies. R4 needs background knowledge and does not apply to a plain
    DAG.
    """
    if not nx.is_directed_acyclic_graph(digraph):
        raise ValueError("dag_to_cpdag requires an acyclic input graph")

    index = {name: i for i, name in enumerate(var_names)}
    skeleton = {
        (min(index[u], index[v]), max(index[u], index[v]))
        for u, v in digraph.edges()
        if u != v
    }
    neighbours: dict[int, set[int]] = {index[n]: set() for n in var_names}
    for i, j in skeleton:
        neighbours[i].add(j)
        neighbours[j].add(i)

    true_arcs = {(index[u], index[v]) for u, v in digraph.edges() if u != v}
    arcs: set[tuple[int, int]] = set()

    # Step 1: unshielded colliders a -> b <- c, with a and c non-adjacent.
    for b in neighbours:
        for a, c in combinations(sorted(neighbours[b]), 2):
            if (min(a, c), max(a, c)) in skeleton:
                continue
            if (a, b) in true_arcs and (c, b) in true_arcs:
                arcs.add((a, b))
                arcs.add((c, b))

    # Step 2: Meek's rules to a fixed point.
    def undirected(a: int, b: int) -> bool:
        return (a, b) not in arcs and (b, a) not in arcs

    changed = True
    while changed:
        changed = False
        for i, j in sorted(skeleton):
            for a, b in ((i, j), (j, i)):
                if not undirected(a, b):
                    continue
                # R1: c -> a, a - b, c not adjacent to b  =>  a -> b
                rule = any(
                    (c, a) in arcs and c != b and c not in neighbours[b]
                    for c in neighbours[a]
                )
                # R2: a -> c -> b, a - b  =>  a -> b
                if not rule:
                    rule = any((a, c) in arcs and (c, b) in arcs for c in neighbours[a])
                # R3: a - c, a - d, c -> b, d -> b, c and d non-adjacent => a -> b
                if not rule:
                    parents = [
                        c
                        for c in neighbours[a]
                        if c != b and (c, b) in arcs and undirected(a, c)
                    ]
                    rule = any(d not in neighbours[c] for c, d in combinations(parents, 2))
                if rule:
                    # Only ever commit an orientation the true DAG agrees with;
                    # Meek's rules are sound, so this is an assertion, not a
                    # correction, and it catches a rule implemented wrongly.
                    if (a, b) not in true_arcs:
                        raise AssertionError(
                            f"Meek rule oriented {var_names[a]} -> {var_names[b]}, "
                            "which contradicts the input DAG"
                        )
                    arcs.add((a, b))
                    changed = True
                    break

    cpdag = CausalGraph.empty(list(var_names), tau_max=0)
    for i, j in skeleton:
        if (i, j) in arcs:
            cpdag.links[i, j, 0], cpdag.links[j, i, 0] = "-->", "<--"
        elif (j, i) in arcs:
            cpdag.links[j, i, 0], cpdag.links[i, j, 0] = "-->", "<--"
        else:
            cpdag.links[i, j, 0] = cpdag.links[j, i, 0] = "o-o"
    cpdag.meta = {"derived_from": "dag_to_cpdag"}
    return cpdag


def count_v_structures(digraph: nx.DiGraph) -> int:
    """Unshielded colliders in a DAG. Reported because they are what determines
    how much of the CPDAG is orientable."""
    total = 0
    for node in digraph.nodes:
        parents = sorted(digraph.predecessors(node))
        for a, c in combinations(parents, 2):
            if not (digraph.has_edge(a, c) or digraph.has_edge(c, a)):
                total += 1
    return total
