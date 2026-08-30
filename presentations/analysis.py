"""Supplies the presentation with real results rather than transcribed numbers.

Runs the analysis on the Tennessee Eastman dataset and returns the recovered
graph, a readable layout for it, and a short slice of the raw series. Keeping
this in one module means the deck can never drift out of step with the results.
"""

import math
import random
import sys
from pathlib import Path

PC_DIR = Path(__file__).resolve().parent.parent / "PC_Algorithm"
sys.path.insert(0, str(PC_DIR))

import pandas as pd  # noqa: E402
from utils import load_dataset, load_ground_truth, pc_algorithm  # noqa: E402

DATASET = PC_DIR / "datasetTE.csv"
GROUND_TRUTH = PC_DIR / "TEGroundTruth.txt"
ALPHA = 0.01
MAX_COND = 2


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def tennessee_result():
    """Returns a dict with the recovered graph and its comparison to the
    reference graph, in a form the slides can consume directly."""
    names, data, _ = load_dataset(DATASET)
    result = pc_algorithm(data, alpha=ALPHA, max_cond_set_size=MAX_COND,
                          variable_names=names)
    reference, _, info = load_ground_truth(GROUND_TRUTH, names)
    n = len(names)

    skeleton = result.skeleton_edges()
    reference_skeleton = {
        (i, j) for i in range(n) for j in range(i + 1, n)
        if reference[i, j] or reference[j, i]
    }

    return {
        "names": names,
        "n_vars": n,
        "n_samples": data.shape[0],
        "skeleton": sorted(skeleton),
        "reference_skeleton": sorted(reference_skeleton),
        "agreeing": sorted(skeleton & reference_skeleton),
        "additional": sorted(skeleton - reference_skeleton),
        "missed": sorted(reference_skeleton - skeleton),
        "directed": set(result.directed_edges),
        "undirected": set(result.undirected_edges),
        "conflicting": set(result.conflicting_edges),
        "components": _components(result.adjacency, n),
        "reference_info": info,
    }


def _components(adjacency, n):
    """Connected components of the skeleton, largest first."""
    seen, groups = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            group.append(node)
            stack.extend(adjacency[node] - seen)
        groups.append(sorted(group))
    return sorted(groups, key=len, reverse=True)


def series_slice(variable_index=0, count=400):
    """A short slice of one variable, for the autocorrelation slide."""
    names, data, _ = load_dataset(DATASET)
    values = data[:count, variable_index]
    frame = pd.read_csv(DATASET)
    lags = [frame[name].autocorr(1) for name in names]
    return {
        "name": names[variable_index],
        "values": [float(v) for v in values],
        "max_lag1": round(max(lags), 3),
        "median_lag1": round(float(pd.Series(lags).median()), 3),
    }


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def spring_layout(nodes, edges, iterations=400, seed=11):
    """Fruchterman-Reingold layout, normalised into the unit square.

    A sparse graph like this one (average degree 2.3) lays out far more legibly
    under a force model than in a circle, because the natural chains and
    branches separate instead of turning into crossing chords.
    """
    if len(nodes) == 1:
        return {nodes[0]: (0.5, 0.5)}

    rnd = random.Random(seed)
    pos = {v: [rnd.uniform(0, 1), rnd.uniform(0, 1)] for v in nodes}
    neighbours = {v: set() for v in nodes}
    for a, b in edges:
        if a in neighbours and b in neighbours:
            neighbours[a].add(b)
            neighbours[b].add(a)

    k = math.sqrt(1.0 / len(nodes))
    temperature = 0.12

    for step in range(iterations):
        disp = {v: [0.0, 0.0] for v in nodes}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = math.hypot(dx, dy) or 1e-4
                force = (k * k) / dist          # repulsion
                ux, uy = dx / dist, dy / dist
                disp[a][0] += ux * force
                disp[a][1] += uy * force
                disp[b][0] -= ux * force
                disp[b][1] -= uy * force
        for a in nodes:
            for b in neighbours[a]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = math.hypot(dx, dy) or 1e-4
                force = (dist * dist) / k       # attraction
                disp[a][0] -= (dx / dist) * force
                disp[a][1] -= (dy / dist) * force
        for v in nodes:
            dx, dy = disp[v]
            dist = math.hypot(dx, dy) or 1e-4
            step_len = min(dist, temperature)
            pos[v][0] += (dx / dist) * step_len
            pos[v][1] += (dy / dist) * step_len
        temperature = max(0.002, temperature * (1 - step / iterations) ** 0.5)

    return _normalise(pos)


def _normalise(pos):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    return {
        v: ((p[0] - min(xs)) / span_x, (p[1] - min(ys)) / span_y)
        for v, p in pos.items()
    }


def place_components(components, edges, panels):
    """Lays out each component inside its own rectangular panel.

    `panels` is a list of (x, y, w, h) in inches, one per component. Returns
    {node_index: (x_inches, y_inches)}. Clustering this way is what keeps 31
    nodes readable: the groups separate visually instead of interleaving.
    """
    placed = {}
    for group, panel in zip(components, panels):
        px, py, pw, ph = panel
        inner = [(a, b) for a, b in edges if a in group and b in group]
        layout = spring_layout(list(group), inner)
        if len(group) == 1:
            layout = {group[0]: (0.5, 0.5)}
        for v, (fx, fy) in layout.items():
            placed[v] = (px + fx * pw, py + fy * ph)
    return placed
