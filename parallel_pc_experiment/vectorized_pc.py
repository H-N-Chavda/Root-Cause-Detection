"""Vectorized, CPU-only PC-stable (Fisher-z) that reproduces standard_pc.py exactly.

Reference: standard_pc.py = causal-learn pc(stable=True, uc_rule=0, uc_priority=2), Fisher-z.
Stages and how each one matches it:

1. corr       C = np.corrcoef(X.T), float64 (the matrix causal-learn's FisherZ uses).
2. skeleton   PC-stable. At depth l the adjacency is frozen, every edge (i, j) is tested with every
              S subset of adj(i)\\{j} and of adj(j)\\{i} with |S| = l, and deletions are applied at
              the end of the level. Runs while max degree - 1 > l and l <= max_l (as causal-learn).
                l = 0 : one threshold on |C|.
                l = 1 : closed-form partial correlations, edges x blocks of k.
                l >= 2: numba prange over edges, partial correlation from a Gauss-Jordan inverse.
              Independent iff |r| < tau_l = tanh(z_{1-alpha/2} / sqrt(N - l - 3)), i.e. causal-learn's
              p > alpha. Any test with |r| within BOUNDARY_TOL of tau (or non-finite, or |r| ~ 1) is
              re-run with causal-learn's exact formula (same sub-matrix order, np.linalg.inv, log,
              norm.cdf, p > alpha), so floating-point noise cannot flip a decision.
3. sepsets    causal-learn's stable mode does NOT stop at the first separating set: it tests every S
              and keeps the union of all separating sets found (both sides) at the depth where the
              edge is removed. sepset[(i, j)] (i < j) is that union, as a sorted tuple.
4. colliders  uc_sepset priority 2: triples x - y - z (x < z non-adjacent, y not in sepset) visited
              in causal-learn's order (y, then x, then z); x -> y <- z is applied unless y -> x or
              y -> z is already oriented. Order-dependent, so replicated exactly (cheap loop).
5. Meek       R1-R3 as boolean matrix products. If the pattern from step 4 has a consistent DAG
              extension (Dor-Tarsi check), Meek's rules give the same graph in any order, so the
              matrix result equals causal-learn's sequential one. Otherwise (conflicting colliders,
              common with sampling errors) the order matters, and causal-learn's sequential sweep is
              replicated exactly: same UT / triangle / kite order, same ancestor check through its
              incrementally maintained dpath matrix (including its incomplete transitive closure).

Output A: bool n x n; A[i, j] & ~A[j, i] means i -> j, A[i, j] & A[j, i] means i - j.
Threads: see set_threads(). Call warmup() once before timing to exclude numba compilation.
"""
from __future__ import annotations

import os
import time
import warnings
from itertools import combinations
from math import log, sqrt

import numpy as np
from scipy.stats import norm

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # not installed or incompatible with this numpy: same results, pure-Python speed
    numba = None
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    warnings.warn("numba is not available: vectorized_pc runs its loops in pure Python (slow for large n).")

BOUNDARY_TOL = 1e-7      # |r| this close to tau is re-tested with causal-learn's exact formula
DEGENERATE = 1e-12       # |r| > 1 - DEGENERATE (near-singular) is also re-tested exactly
BLOCK_ELEMS = 1_000_000  # l = 1: (edge, k) pairs evaluated per block
LAST_INFO: dict = {}     # filled by pc(): stage times, Meek path, number of exact re-tests

_BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")


# ----------------------------------------------------------------------------------------------------
# threads
# ----------------------------------------------------------------------------------------------------
def set_threads(numba_threads: int | None = 10, blas_threads: int | None = 10) -> None:
    """Thread setup for a 10-core limit.

    The l >= 2 numba kernel is scalar code with no BLAS/LAPACK calls, and the Meek matmuls run after it,
    so the numba pool and the BLAS pool are never busy at the same time. With max_l <= 1 numba's thread
    pool is not used at all.

    BLAS thread pools are limited in the current process; environment variables are also set for
    subprocesses started later. For a fresh process, export the variables before Python starts:
        OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 \\
        NUMBA_NUM_THREADS=10 python compare.py ...
    numba_threads is applied immediately (capped at NUMBA_NUM_THREADS).
    """
    if blas_threads is not None:
        for var in _BLAS_VARS:
            os.environ[var] = str(int(blas_threads))
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(limits=max(1, int(blas_threads)))
        except ImportError:
            pass
    if numba_threads is not None and HAVE_NUMBA:
        numba.set_num_threads(max(1, min(int(numba_threads), numba.config.NUMBA_NUM_THREADS)))


# ----------------------------------------------------------------------------------------------------
# Fisher-z test
# ----------------------------------------------------------------------------------------------------
def _fisherz_pvalue(C, N, x, y, S) -> float:
    """causal-learn FisherZ.__call__, operation for operation (used for near-threshold tests)."""
    x, y = (int(x), int(y)) if x < y else (int(y), int(x))
    S = sorted(set(map(int, S)))
    var = [x, y] + S
    sub = C[np.ix_(var, var)]
    try:
        inv = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        raise ValueError("Data correlation matrix is singular. Cannot run fisherz test. Please check your data.")
    r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
    if abs(r) >= 1:
        r = (1. - np.finfo(float).eps) * np.sign(r)
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(N - len(S) - 3) * abs(Z)
    return 2 * (1 - norm.cdf(abs(X)))


def _tau(alpha: float, N: int, l: int) -> float:
    """|r| threshold at depth l: p > alpha  <=>  |r| < tau."""
    if N - l - 3 <= 0:
        raise ValueError(f"Fisher-z at depth {l} needs N > {l + 3} samples (N = {N}).")
    return float(np.tanh(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(N - l - 3)))


def _classify_np(r: np.ndarray, tau: float):
    """Fast decision for an array of partial correlations: (independent, needs exact re-test)."""
    with np.errstate(invalid="ignore"):
        a = np.abs(r)
        amb = ~np.isfinite(r) | (a > 1.0 - DEGENERATE) | (np.abs(a - tau) <= BOUNDARY_TOL)
        ind = ~amb & (a < tau)
    return ind, amb


@njit(cache=True)
def _classify(r, tau, tol):
    """0 dependent, 1 independent, 2 needs exact re-test."""
    if not np.isfinite(r):
        return 2
    a = abs(r)
    if a > 1.0 - DEGENERATE or abs(a - tau) <= tol:
        return 2
    if a < tau:
        return 1
    return 0


@njit(cache=True)
def _pcorr(C, var, m, w, inv):
    """Partial correlation of var[0], var[1] given var[2:m]: -P01 / sqrt(P00 P11), P = inv(C[var, var]).
    Gauss-Jordan with partial pivoting; NaN on a zero pivot (then the exact test decides / raises)."""
    for a in range(m):
        for b in range(m):
            w[a, b] = C[var[a], var[b]]
            inv[a, b] = 1.0 if a == b else 0.0
    for col in range(m):
        p = col
        best = abs(w[col, col])
        for rr in range(col + 1, m):
            v = abs(w[rr, col])
            if v > best:
                best = v
                p = rr
        if best == 0.0 or not np.isfinite(best):
            return np.nan
        if p != col:
            for c in range(m):
                tmp = w[col, c]
                w[col, c] = w[p, c]
                w[p, c] = tmp
                tmp = inv[col, c]
                inv[col, c] = inv[p, c]
                inv[p, c] = tmp
        piv = w[col, col]
        for c in range(m):
            w[col, c] /= piv
            inv[col, c] /= piv
        for rr in range(m):
            if rr != col:
                f = w[rr, col]
                if f != 0.0:
                    for c in range(m):
                        w[rr, c] -= f * w[col, c]
                        inv[rr, c] -= f * inv[col, c]
    d = abs(inv[0, 0] * inv[1, 1])
    if d == 0.0:                                   # numba raises on float division by zero
        return np.nan
    return -inv[0, 1] / np.sqrt(d)


@njit(parallel=True, cache=True)
def _level_kernel(C, adj, I, J, l, tau, tol, removed, amb, members):
    """Depth l >= 2, one edge per prange iteration. Every S of size l from adj(i)\\{j}, then adj(j)\\{i}
    (frozen adj), is tested; removed[e] = some S separates, members[e, s] = s is in a separating set,
    amb[e] = some test was near the threshold (the edge is then redone exactly in Python)."""
    n = C.shape[0]
    for e in prange(I.shape[0]):
        i = I[e]
        j = J[e]
        idx = np.empty(l, np.int64)
        var = np.empty(l + 2, np.int64)
        cand = np.empty(n, np.int64)
        w = np.empty((l + 2, l + 2))
        inv = np.empty((l + 2, l + 2))
        var[0] = i
        var[1] = j
        for side in range(2):
            a = i if side == 0 else j
            b = j if side == 0 else i
            m = 0
            for v in range(n):
                if adj[a, v] and v != b:
                    cand[m] = v
                    m += 1
            if m < l:
                continue
            for t in range(l):
                idx[t] = t
            while True:                            # combinations(cand[:m], l), lexicographic
                for t in range(l):
                    var[2 + t] = cand[idx[t]]
                cls = _classify(_pcorr(C, var, l + 2, w, inv), tau, tol)
                if cls == 2:
                    amb[e] = 1
                elif cls == 1:
                    removed[e] = 1
                    for t in range(l):
                        members[e, var[2 + t]] = 1
                t = l - 1
                while t >= 0 and idx[t] == m - l + t:
                    t -= 1
                if t < 0:
                    break
                idx[t] += 1
                for u in range(t + 1, l):
                    idx[u] = idx[u - 1] + 1


# ----------------------------------------------------------------------------------------------------
# skeleton
# ----------------------------------------------------------------------------------------------------
def _record(I, J, e, k, sepset):
    """sepset[(I[e], J[e])] = sorted union of the separating nodes k of edge e; returns removed edges."""
    order = np.lexsort((k, e))
    e, k = e[order], k[order]
    uniq, start = np.unique(e, return_index=True)
    stop = np.append(start[1:], len(e))
    kl = k.tolist()
    for u, s0, s1 in zip(uniq.tolist(), start.tolist(), stop.tolist()):
        sepset[(int(I[u]), int(J[u]))] = tuple(sorted(set(kl[s0:s1])))
    return I[uniq], J[uniq]


def _level0(C, N, alpha, adj, sepset, stats):
    I, J = np.nonzero(np.triu(adj, 1))
    ind, amb = _classify_np(C[I, J], _tau(alpha, N, 0))
    for e in np.flatnonzero(amb):
        stats["exact_tests"] += 1
        ind[e] = _fisherz_pvalue(C, N, I[e], J[e], ()) > alpha
    for a, b in zip(I[ind].tolist(), J[ind].tolist()):
        sepset[(a, b)] = ()
    return I[ind], J[ind]


def _level1(C, N, alpha, adj, sepset, stats):
    """rho_ij|k = (r_ij - r_ik r_jk) / sqrt((1 - r_ik^2)(1 - r_jk^2)) for all frozen edges (i, j) and
    every k adjacent to i or j (k != i, j); all separating k are kept (union)."""
    n = len(C)
    tau = _tau(alpha, N, 1)
    I, J = np.nonzero(np.triu(adj, 1))
    cij = C[I, J][:, None]
    kb = max(1, BLOCK_ELEMS // max(len(I), 1))
    hit_e, hit_k = [], []
    for k0 in range(0, n, kb):
        ks = np.arange(k0, min(n, k0 + kb))
        cik, cjk = C[np.ix_(I, ks)], C[np.ix_(J, ks)]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (cij - cik * cjk) / np.sqrt((1.0 - cik * cik) * (1.0 - cjk * cjk))
        tested = ((adj[np.ix_(I, ks)] | adj[np.ix_(J, ks)])
                  & (ks[None, :] != I[:, None]) & (ks[None, :] != J[:, None]))
        ind, amb = _classify_np(r, tau)
        ind &= tested
        for e, t in zip(*np.nonzero(amb & tested)):
            stats["exact_tests"] += 1
            ind[e, t] = _fisherz_pvalue(C, N, I[e], J[e], (ks[t],)) > alpha
        e, t = np.nonzero(ind)
        hit_e.append(e)
        hit_k.append(ks[t])
    return _record(I, J, np.concatenate(hit_e), np.concatenate(hit_k), sepset)


def _exact_edge(C, N, alpha, adj, i, j, l):
    """Edge (i, j) at depth l with causal-learn's exact test: (removed, union of separating sets)."""
    found, mem = False, set()
    for a, b in ((i, j), (j, i)):
        cand = [int(v) for v in np.flatnonzero(adj[a]) if v != b]
        for S in combinations(cand, l):
            if _fisherz_pvalue(C, N, i, j, S) > alpha:
                found = True
                mem.update(S)
    return found, mem


def _level_ge2(C, N, alpha, adj, l, sepset, stats):
    n = len(C)
    I, J = np.nonzero(np.triu(adj, 1))
    I, J = I.astype(np.int64), J.astype(np.int64)
    removed = np.zeros(len(I), np.uint8)
    amb = np.zeros(len(I), np.uint8)
    members = np.zeros((len(I), n), np.uint8)
    _level_kernel(C, adj, I, J, l, _tau(alpha, N, l), BOUNDARY_TOL, removed, amb, members)
    for e in np.flatnonzero(amb):
        stats["exact_edges"] += 1
        found, mem = _exact_edge(C, N, alpha, adj, int(I[e]), int(J[e]), l)
        removed[e] = found
        members[e] = 0
        if mem:
            members[e, sorted(mem)] = 1
    rem = np.flatnonzero(removed)
    for e in rem.tolist():
        sepset[(int(I[e]), int(J[e]))] = tuple(np.flatnonzero(members[e]).tolist())
    return I[rem], J[rem]


def _skeleton(X, alpha, max_l, verbose):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (N samples x n variables).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")
    N, n = X.shape
    times = {"corr": 0.0, "l=0": 0.0, "l=1": 0.0, "l>=2": 0.0}
    stats = {"exact_tests": 0, "exact_edges": 0}

    t = time.perf_counter()
    C = np.ascontiguousarray(np.atleast_2d(np.corrcoef(X.T)))
    times["corr"] = time.perf_counter() - t

    adj = ~np.eye(n, dtype=bool)
    sepset: dict = {}
    depth = -1
    while n > 0 and adj.sum(1).max() - 1 > depth:
        depth += 1
        if max_l is not None and depth > max_l:
            break
        t = time.perf_counter()
        if depth == 0:
            I, J = _level0(C, N, alpha, adj, sepset, stats)
        elif depth == 1:
            I, J = _level1(C, N, alpha, adj, sepset, stats)
        else:
            I, J = _level_ge2(C, N, alpha, adj, depth, sepset, stats)
        adj[I, J] = False                          # deletions applied after the whole level
        adj[J, I] = False
        times["l=0" if depth == 0 else "l=1" if depth == 1 else "l>=2"] += time.perf_counter() - t
        if verbose and depth <= 1:
            deg = adj.sum(1)
            print(f"[vectorized_pc] after l={depth}: mean degree {deg.mean():.2f}, max degree {deg.max()}, "
                  f"edges {int(adj.sum()) // 2}")
    return adj, sepset, times, stats


def skeleton(X, alpha: float = 0.05, max_l: int | None = None, verbose: bool = False):
    """PC-stable skeleton. Returns (adj, sepset): adj bool n x n symmetric; sepset {(i, j): tuple}, i < j,
    for every non-adjacent pair = union of all separating sets found at its removal depth."""
    adj, sepset, _, _ = _skeleton(X, alpha, max_l, verbose)
    return adj, sepset


# ----------------------------------------------------------------------------------------------------
# colliders (uc_sepset, priority 2)
# ----------------------------------------------------------------------------------------------------
def _collider_triples(adj, sepset):
    """(x, y, z) with x < z non-adjacent, both adjacent to y, y not in sepset(x, z); causal-learn order
    (y, then x, then z)."""
    n = len(adj)
    codes = np.array(sorted((x * n + z) * n + y for (x, z), S in sepset.items() for y in S), dtype=np.int64)
    out = []
    for y in range(n):
        nb = np.flatnonzero(adj[y])
        if len(nb) < 2:
            continue
        a, b = np.nonzero(np.triu(~adj[np.ix_(nb, nb)], 1))
        if len(a) == 0:
            continue
        x, z = nb[a], nb[b]
        if len(codes):
            c = (x * n + z) * n + y
            hit = codes[np.minimum(np.searchsorted(codes, c), len(codes) - 1)] == c
            x, z = x[~hit], z[~hit]
        out.append(np.stack([x, np.full(len(x), y), z], axis=1))
    if not out:
        return np.zeros((0, 3), np.int64)
    return np.ascontiguousarray(np.concatenate(out).astype(np.int64))


@njit(cache=True)
def _orient_colliders(A, trip, ops_a, ops_y, ops_was):
    """Apply x -> y <- z unless y -> x or y -> z exists. Logs every written edge a -> y and whether it was
    already directed (causal-learn then rebuilds dpath). Returns the number of logged writes."""
    n_ops = 0
    for t in range(trip.shape[0]):
        x = trip[t, 0]
        y = trip[t, 1]
        z = trip[t, 2]
        if (A[y, x] and not A[x, y]) or (A[y, z] and not A[z, y]):
            continue
        for s in range(2):
            a = x if s == 0 else z
            ops_a[n_ops] = a
            ops_y[n_ops] = y
            ops_was[n_ops] = A[a, y] and not A[y, a]
            n_ops += 1
            A[a, y] = True
            A[y, a] = False
    return n_ops


# ----------------------------------------------------------------------------------------------------
# causal-learn dpath bookkeeping (only needed for the sequential Meek path)
# ----------------------------------------------------------------------------------------------------
@njit(cache=True)
def _adjust_dpath(dpath, i, j):
    """GeneralGraph.adjust_dpath(i, j) for a new edge i -> j (dpath[b, a] = 1: a is an ancestor of b)."""
    n = dpath.shape[0]
    dpath[j, i] = 1
    for k in range(n):
        if dpath[i, k] == 1:
            dpath[j, k] = 1
        if dpath[k, j] == 1:
            dpath[k, i] = 1


@njit(cache=True)
def _reconstitute_dpath(A, dpath):
    """GeneralGraph.reconstitute_dpath(get_graph_edges()): identity, then adjust for each directed edge,
    popped from the end of the (i < j, row-major) edge list."""
    n = A.shape[0]
    for a in range(n):
        for b in range(n):
            dpath[a, b] = 0
        dpath[a, a] = 1
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            if A[i, j] and not A[j, i]:
                _adjust_dpath(dpath, i, j)
            elif A[j, i] and not A[i, j]:
                _adjust_dpath(dpath, j, i)


@njit(cache=True)
def _collider_dpath(adj, ops_a, ops_y, ops_was, n_ops):
    """dpath after uc_sepset. Re-writing an already directed a -> y removes it first, which rebuilds dpath
    from scratch, so only the last rebuild and the adjustments from there on matter."""
    n = adj.shape[0]
    dpath = np.zeros((n, n), np.int64)
    for a in range(n):
        dpath[a, a] = 1
    last = -1
    for t in range(n_ops):
        if ops_was[t]:
            last = t
    start = 0
    if last >= 0:
        snap = adj.copy()
        for t in range(last):
            snap[ops_a[t], ops_y[t]] = True
            snap[ops_y[t], ops_a[t]] = False
        snap[ops_a[last], ops_y[last]] = False    # the edge being removed when dpath is rebuilt
        snap[ops_y[last], ops_a[last]] = False
        _reconstitute_dpath(snap, dpath)
        start = last
    for t in range(start, n_ops):
        _adjust_dpath(dpath, ops_a[t], ops_y[t])
    return dpath


# ----------------------------------------------------------------------------------------------------
# Meek
# ----------------------------------------------------------------------------------------------------
@njit(cache=True)
def _is_extendable(A):
    """Dor-Tarsi: does PDAG A have a DAG extension with the same skeleton and v-structures?"""
    n = A.shape[0]
    alive = np.ones(n, np.bool_)
    for _ in range(n):
        found = -1
        for x in range(n):
            if not alive[x]:
                continue
            ok = True
            for y in range(n):                     # x has no outgoing directed edge
                if alive[y] and A[x, y] and not A[y, x]:
                    ok = False
                    break
            if ok:
                for y in range(n):                 # each undirected neighbour of x is adjacent to x's others
                    if alive[y] and A[x, y] and A[y, x]:
                        for z in range(n):
                            if z != y and alive[z] and (A[x, z] or A[z, x]) and not (A[y, z] or A[z, y]):
                                ok = False
                                break
                        if not ok:
                            break
            if ok:
                found = x
                break
        if found < 0:
            return False
        alive[found] = False
    return True


def _meek_matrix(A0):
    """R1-R3 fired simultaneously per pass until no change. Returns (A, conflict)."""
    A = A0.copy()
    adj = A | A.T
    Nn = ~adj
    np.fill_diagonal(Nn, False)
    Nf = Nn.astype(np.float32)
    while True:
        U = A & A.T
        if not U.any():
            return A, False
        D = A & ~A.T
        Df, Uf = D.astype(np.float32), U.astype(np.float32)
        new = U & ((Df.T @ Nf) > 0)                # R1: i -> j - k, i, k non-adjacent      => j -> k
        new |= U & ((Df @ Df) > 0)                 # R2: i -> j -> k, i - k                 => i -> k
        for l in np.flatnonzero(D.sum(0) >= 2):    # R3: i - j -> l <- k - i, i - l, j, k non-adj => i -> l
            if not U[:, l].any():
                continue
            P = np.flatnonzero(D[:, l])
            UP = Uf[:, P]
            hit = ((UP @ Nf[np.ix_(P, P)]) * UP).sum(1) > 0
            new[:, l] |= hit & U[:, l]
        if not new.any():
            return A, False
        if (new & new.T).any():
            return A, True
        A &= ~new.T                                # a -> b: drop the mark A[b, a]


def _reference_order(A, adj):
    """causal-learn's find_adj() list (tails then arrow heads, each as (col, row) in row-major order) as
    (first, second) arrays, and each node's neighbours sorted by the list position of (node, neighbour)."""
    n = len(A)
    r1, c1 = np.nonzero(A)                        # G[r, c] == -1
    r2, c2 = np.nonzero(adj & ~A)                 # G[r, c] == 1
    first = np.ascontiguousarray(np.concatenate([c1, c2]).astype(np.int64))
    second = np.ascontiguousarray(np.concatenate([r1, r2]).astype(np.int64))
    pos = np.full((n, n), -1, np.int64)
    pos[first, second] = np.arange(len(first))
    nb_ptr = np.zeros(n + 1, np.int64)
    nb_ptr[1:] = np.cumsum(adj.sum(1))
    nb_idx = np.empty(nb_ptr[-1], np.int64)
    for b in range(n):
        nb = np.flatnonzero(adj[b])
        nb_idx[nb_ptr[b]:nb_ptr[b + 1]] = nb[np.argsort(pos[b, nb], kind="stable")]
    return first, second, nb_ptr, nb_idx


@njit(cache=True)
def _meek_sequential(A, adj, first, second, nb_ptr, nb_idx, dpath):
    """causal-learn Meek.meek: sweep R1 over UT, R2 over triangles, R3 over kites (lists in causal-learn's
    order, generated on the fly), orienting one edge at a time; repeat until a full pass changes nothing."""
    M = first.shape[0]
    loop = True
    while loop:
        loop = False
        for p in range(M):                        # R1, UT = (i, j, k)
            i = first[p]
            j = second[p]
            for t in range(nb_ptr[j], nb_ptr[j + 1]):
                k = nb_idx[t]
                if k == i or adj[i, k]:
                    continue
                if A[i, j] and not A[j, i] and A[j, k] and A[k, j]:
                    if dpath[j, k] == 1:          # k is an ancestor of j
                        continue
                    A[k, j] = False
                    _adjust_dpath(dpath, j, k)
                    loop = True
        for p in range(M):                        # R2, Tri = (i, j, k)
            i = first[p]
            j = second[p]
            for t in range(nb_ptr[j], nb_ptr[j + 1]):
                k = nb_idx[t]
                if k == i or not adj[i, k]:
                    continue
                if A[i, j] and not A[j, i] and A[j, k] and not A[k, j] and A[i, k] and A[k, i]:
                    if dpath[i, k] == 1:          # k is an ancestor of i
                        continue
                    A[k, i] = False
                    _adjust_dpath(dpath, i, k)
                    loop = True
        for p in range(M):                        # R3, Kite = (i, j, k, l) from triangles (i, j, l), (i, k, l)
            i = first[p]
            j = second[p]
            for t in range(nb_ptr[j], nb_ptr[j + 1]):
                l = nb_idx[t]
                if l == i or not adj[i, l]:
                    continue
                for t2 in range(nb_ptr[i], nb_ptr[i + 1]):
                    k = nb_idx[t2]
                    if k <= j or not adj[k, l] or adj[j, k]:
                        continue
                    if (A[i, j] and A[j, i] and A[i, k] and A[k, i] and A[j, l] and not A[l, j]
                            and A[k, l] and not A[l, k] and A[i, l] and A[l, i]):
                        if dpath[i, l] == 1:      # l is an ancestor of i
                            continue
                        A[l, i] = False
                        _adjust_dpath(dpath, i, l)
                        loop = True


def _meek(A0, adj, ops, n_ops):
    if _is_extendable(A0):
        A, conflict = _meek_matrix(A0)
        if not conflict:                           # cannot happen for an extendable pattern; kept as a guard
            return A, "matrix"
    dpath = _collider_dpath(adj, ops[0], ops[1], ops[2], n_ops)
    first, second, nb_ptr, nb_idx = _reference_order(A0, adj)
    A = A0.copy()
    _meek_sequential(A, adj, first, second, nb_ptr, nb_idx, dpath)
    return A, "sequential"


# ----------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------
def pc(X, alpha: float = 0.05, max_l: int | None = None, verbose: bool = False):
    """PC-stable with Fisher-z. Returns (A, sepset); see the module docstring for the conventions."""
    adj, sepset, times, stats = _skeleton(X, alpha, max_l, verbose)

    t = time.perf_counter()
    trip = _collider_triples(adj, sepset)
    A = adj.copy()
    ops = (np.zeros(2 * len(trip), np.int64), np.zeros(2 * len(trip), np.int64),
           np.zeros(2 * len(trip), np.bool_))
    n_ops = _orient_colliders(A, trip, ops[0], ops[1], ops[2])
    times["v-structures"] = time.perf_counter() - t

    t = time.perf_counter()
    A, path = _meek(A, adj, ops, n_ops)
    times["meek"] = time.perf_counter() - t

    LAST_INFO.clear()
    LAST_INFO.update(times=times, meek_path=path, **stats)
    if verbose:
        print("[vectorized_pc] time: " + " | ".join(f"{k} {v:.3f}s" for k, v in times.items())
              + f"  (Meek path: {path}; exact re-tests: {stats['exact_tests']} tests, "
                f"{stats['exact_edges']} edges)")
    return A, sepset


def to_causallearn(A: np.ndarray) -> np.ndarray:
    """A -> causal-learn endpoint matrix (G[i, j] = -1, G[j, i] = 1 for i -> j; both -1 for i - j)."""
    G = np.zeros(A.shape, dtype=int)
    D = A & ~A.T
    G[A & A.T] = -1
    G[D] = -1
    G[D.T] = 1
    return G


def warmup() -> None:
    """Compile every numba function on tiny inputs (same dtypes as real calls) so timings exclude JIT."""
    if not HAVE_NUMBA:
        return
    n = 4
    C = np.eye(n)
    adj = ~np.eye(n, dtype=bool)
    one = np.zeros(1, np.uint8)
    _level_kernel(C, adj, np.array([0], np.int64), np.array([1], np.int64), 2, 0.1, BOUNDARY_TOL,
                  one, one.copy(), np.zeros((1, n), np.uint8))
    skel = adj.copy()
    skel[0, 2] = skel[2, 0] = False                # 0 - 1 - 2 unshielded
    A = skel.copy()
    trip = np.array([[0, 1, 2]], np.int64)
    ops = (np.zeros(2, np.int64), np.zeros(2, np.int64), np.zeros(2, np.bool_))
    k = _orient_colliders(A, trip, ops[0], ops[1], ops[2])
    _is_extendable(A)
    dpath = _collider_dpath(skel, ops[0], ops[1], ops[2], k)
    first, second, nb_ptr, nb_idx = _reference_order(A, skel)
    _meek_sequential(A, skel, first, second, nb_ptr, nb_idx, dpath)
