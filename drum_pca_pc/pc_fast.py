"""Fast PC algorithm (Fisher-z, stable, depth <= 1) with the same rules as causal-learn's
``pc(data, alpha, "fisherz", stable=True, uc_rule=0, uc_priority=2, max_k=1)``.

Why this module exists: causal-learn finds unshielded triples, triangles and kites by comparing every pair of
list entries (``GraphClass.find_triangles`` / ``find_kites``). On a dense graph (~250 nodes, ~20k edges) that is
~1e11-1e12 Python steps, i.e. days to months. Here the same tests and rules are evaluated with numpy arrays.

Steps (identical to causal-learn):
  1. depth 0: test every pair; remove x-y if p > alpha; sepset(x, y) = {}.
  2. depth 1 (stable): with the adjacency frozen after depth 0, test x _||_ y | z for every z adjacent to x or to y;
     remove x-y if any test has p > alpha; sepset(x, y) = every such z (union over both sides, as causal-learn).
  3. colliders (uc_sepset, priority 2 = "prioritize existing"): for unshielded x - y - z (x < z), y not in
     sepset(x, z) -> x -> y <- z, unless y -> x or y -> z is already oriented. Triples visited in causal-learn's
     order (middle node y first, then x, then z).
  4. Meek rules 1-3, repeated until nothing changes; an edge is not oriented if that would close a directed cycle.

Graph encoding is causal-learn's: G[i, j] = -1, G[j, i] = 1  means i -> j;  both -1 means i - j.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

_EPS1 = 1.0 - np.finfo(float).eps


def _fisher_p(r: np.ndarray, n: int, k: int) -> np.ndarray:
    """causal-learn FisherZ p-value for partial correlation r with conditioning-set size k."""
    r = np.where(np.abs(r) >= 1, _EPS1 * np.sign(r), r)
    x = np.sqrt(n - k - 3) * np.abs(0.5 * np.log((1 + r) / (1 - r)))
    return 2 * (1 - norm.cdf(x))


def skeleton(data: np.ndarray, alpha: float, max_k: int = 1, log=print):
    """Stable PC skeleton up to depth max_k (0 or 1).

    Returns:
        A: bool (p, p) adjacency after the search.
        sep: bool (p, p, p); sep[x, y, z] = True if z is in sepset(x, y). Symmetric in x, y.
        removed_at: int8 (p, p); -1 = edge kept, 0 / 1 = removed at that depth.
        R: correlation matrix used (np.corrcoef, as causal-learn).
    """
    if max_k > 1:
        raise NotImplementedError("pc_fast supports depth <= 1 only")
    n, p = data.shape
    R = np.corrcoef(data.T)
    t0 = time.time()

    # depth 0
    A = _fisher_p(R, n, 0) <= alpha
    np.fill_diagonal(A, False)
    removed_at = np.full((p, p), -1, np.int8)
    removed_at[~A] = 0
    np.fill_diagonal(removed_at, -1)
    sep = np.zeros((p, p, p), bool)
    log(f"  depth 0: {p * (p - 1) // 2:,} tests, {int(A.sum() // 2):,} edges left, "
        f"max degree {int(A.sum(1).max())} ({time.time() - t0:.1f} s)")

    # depth 1, adjacency frozen at its depth-0 state (stable)
    if max_k >= 1 and A.sum(1).max() - 1 > 0:
        t1 = time.time()
        A0 = A.copy()
        sep_x = np.zeros((p, p, p), bool)           # sep_x[x, y, z]: z separates x, y and z is adjacent to x
        n_tests = 0
        for x in range(p):
            nb = A0[x]
            if not nb.any():
                continue
            rx = R[x]
            num = rx[:, None] - R * rx[None, :]      # [y, z]: r_xy - r_yz * r_xz
            den = np.sqrt(np.clip((1 - rx ** 2)[None, :] * (1 - R ** 2), 0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                pr = num / den
            mask = nb[:, None] & nb[None, :]         # y adjacent to x, z adjacent to x
            np.fill_diagonal(mask, False)            # z != y
            n_tests += int(mask.sum())
            pv = np.ones((p, p))
            pv[mask] = _fisher_p(pr[mask], n, 1)
            sep_x[x] = mask & (pv > alpha)
        sep = sep_x | sep_x.transpose(1, 0, 2)       # union of x's side and y's side
        rem = sep.any(2) & A0
        A = A0 & ~rem
        removed_at[rem] = 1
        log(f"  depth 1: {n_tests:,} tests, {int(rem.sum() // 2):,} edges removed, {int(A.sum() // 2):,} left "
            f"({time.time() - t1:.1f} s)")
    return A, sep, removed_at, R


def orient_colliders(A: np.ndarray, sep: np.ndarray, log=print) -> np.ndarray:
    """uc_sepset, priority 2. Returns the graph matrix G (causal-learn encoding)."""
    t0 = time.time()
    p = len(A)
    G = np.where(A, -1, 0).astype(np.int8)
    idx = np.arange(p)
    n_uc = n_applied = 0
    # causal-learn's find_adj lists (x, y) ordered by (y, x), so its triples come in (y, x, z) order
    for y in range(p):
        for x in np.flatnonzero(A[y]):
            zs = np.flatnonzero(A[y] & ~A[x] & (idx > x) & ~sep[x, :, y])
            for z in zs:
                n_uc += 1
                if (G[y, x] == -1 and G[x, y] == 1) or (G[y, z] == -1 and G[z, y] == 1):
                    continue                          # y -> x or y -> z already oriented: keep it
                G[x, y], G[y, x] = -1, 1
                G[z, y], G[y, z] = -1, 1
                n_applied += 1
    log(f"  colliders: {n_uc:,} unshielded colliders found, {n_applied:,} applied ({time.time() - t0:.1f} s)")
    return G


def _closure(D: np.ndarray) -> np.ndarray:
    """Reachability: anc[a, b] = True if a directed path a -> ... -> b exists."""
    reach = D.copy()
    while True:
        nxt = reach | ((reach.astype(np.float32) @ reach.astype(np.float32)) > 0)
        if (nxt == reach).all():
            return reach
        reach = nxt


def meek(G: np.ndarray, max_passes: int = 300, time_limit_s: float = 300.0, log=print):
    """Meek rules 1-3 until no change. Returns (G, info)."""
    t0 = time.time()
    G = G.copy()
    p = len(G)
    adj = G != 0
    nonadj = (~adj).astype(np.float32)
    np.fill_diagonal(nonadj, 0)
    nonadj_b = nonadj > 0
    anc = _closure((G == -1) & (G.T == 1))
    counts = {"R1": 0, "R2": 0, "R3": 0, "skipped_cycle": 0}

    def orient(a, b):                                 # a - b  ->  a -> b, unless b is an ancestor of a
        if not (G[a, b] == -1 and G[b, a] == -1):
            return False
        if anc[b, a]:
            counts["skipped_cycle"] += 1
            return False
        G[a, b], G[b, a] = -1, 1
        up = anc[:, a].copy()
        up[a] = True
        down = anc[b].copy()
        down[b] = True
        anc[np.ix_(up, down)] = True
        return True

    passes, stopped = 0, None
    while True:
        if passes >= max_passes:
            stopped = f"pass cap {max_passes} reached"
            break
        if time.time() - t0 > time_limit_s:
            stopped = f"time limit {time_limit_s:.0f} s reached"
            break
        passes += 1
        changed = False

        # R1: i -> j, j - k, i and k not adjacent  =>  j -> k
        D = ((G == -1) & (G.T == 1)).astype(np.float32)
        U = (G == -1) & (G.T == -1)
        cand = U & ((D.T @ nonadj) > 0)
        js, ks = np.nonzero(cand)
        if len(js):
            first_i = np.argmax((D[:, js] > 0) & nonadj_b[:, ks], axis=0)
            for o in np.lexsort((ks, js, first_i)):
                if orient(js[o], ks[o]):
                    counts["R1"] += 1
                    changed = True

        # R2: i -> j -> k, i - k  =>  i -> k
        D = ((G == -1) & (G.T == 1)).astype(np.float32)
        U = (G == -1) & (G.T == -1)
        cand = U & ((D @ D) > 0)
        for i, k in zip(*np.nonzero(cand)):
            if orient(i, k):
                counts["R2"] += 1
                changed = True

        # R3: i - j, i - k, j -> l, k -> l, i - l, j and k not adjacent  =>  i -> l
        D = ((G == -1) & (G.T == 1)).astype(np.float32)
        U = (G == -1) & (G.T == -1)
        Uf = U.astype(np.float32)
        for i in np.flatnonzero(U.any(1)):
            C = Uf[i][:, None] * D                    # C[j, l] = 1 if i - j and j -> l
            hit = (((C.T @ nonadj) * C.T).sum(1) > 0) & U[i]
            for l in np.flatnonzero(hit):
                if orient(i, l):
                    counts["R3"] += 1
                    changed = True

        if not changed:
            break
    info = {"passes": passes, "stopped": stopped, **counts, "seconds": time.time() - t0}
    log(f"  Meek rules: {passes} pass(es), oriented R1 {counts['R1']:,} / R2 {counts['R2']:,} / R3 {counts['R3']:,}, "
        f"skipped (would close a cycle) {counts['skipped_cycle']:,} ({info['seconds']:.1f} s)"
        + (f"  WARNING: stopped early, {stopped}" if stopped else ""))
    return G, info


def sepset_lists(sep: np.ndarray, removed_at: np.ndarray):
    """{(x, y): sorted sepset} for every removed pair x < y (for saving / comparison)."""
    out = {}
    xs, ys = np.nonzero(np.triu(removed_at >= 0, 1))
    for x, y in zip(xs, ys):
        out[(int(x), int(y))] = sorted(int(z) for z in np.flatnonzero(sep[x, y]))
    return out
