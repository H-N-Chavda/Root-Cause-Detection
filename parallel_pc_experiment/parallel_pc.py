"""Parallel PC (Le et al., 2015/2016, "A fast PC algorithm ... with multi-core PCs", Alg. 2).

causal-learn has no Parallel-PC, so the skeleton phase is implemented here, following Alg. 2:
  for each depth d:
    1. freeze adj(X) for every X (taken from the graph at the start of the level);
    2. split the unordered adjacent pairs {X, Y} over P worker processes;
       each worker tests X _||_ Y | S for S subset adj(X)\\{Y}, |S| = d, then
       S subset adj(Y)\\{X} (early stop or not depends on SEPSET_RULE, below);
    3. barrier: delete every separated edge and record its separating set.
Orientation is not parallelised in the paper. Here it is vectorised with numpy (orient(), below) and
reproduces causal-learn's uc_sepset (priority 2) + Meek exactly: same rule order, same triple / triangle /
kite list order, and the same ancestor bookkeeping (causal-learn's dpath, including its incomplete
transitive closure). So the two pipelines still differ only in the skeleton phase.

Same CI test object (causal-learn CIT 'fisherz'), alpha and independence rule (p > alpha).

SEPSET_RULE selects how separating sets are recorded (the skeleton is the same either way):
  "first" - Le et al. / pcalg: stop at the first separating set S and store S.
  "union" - causal-learn pc(stable=True): test every S on both sides (no early stop) and store,
            per side, the union of all separating sets. A collider x->y<-z is then oriented
            only if y is in none of them. Use this for a like-for-like comparison with
            causal-learn's stable PC (verified against SkeletonDiscovery.skeleton_discovery).

Output: outputs/parallel_pc.npy (causal-learn endpoint matrix, see standard_pc.py).
"""
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.Helper import append_value

ALPHA, CI_TEST, UC_PRIORITY, WORKERS, SEPSET_RULE = 0.05, "fisherz", 2, 4, "union"
OUT = Path(__file__).parent / "outputs"
_cit = None  # per-worker CI test


def _init(data: np.ndarray, ci_test: str = CI_TEST) -> None:
    global _cit
    _cit = CIT(data, ci_test)


def _test_edge(task):
    """Alg. 2 inner loop for one edge (X-side, then Y-side).

    Returns (x, y, sepsets): sepsets is None if the edge stays, else the list of
    separating sets to store ("first": [S]; "union": [union_X, union_Y]).
    """
    x, y, nbr_x, nbr_y, d, rule = task
    if rule == "first":
        for nbrs in (nbr_x, nbr_y):
            for S in combinations(nbrs, d):
                if _cit(x, y, S) > ALPHA:
                    return x, y, [S]
        return x, y, None
    unions, separated = [], False
    for nbrs in (nbr_x, nbr_y):
        u = set()
        for S in combinations(nbrs, d):
            if _cit(x, y, S) > ALPHA:
                separated = True
                u.update(S)
        unions.append(tuple(sorted(u)))
    return x, y, (unions if separated else None)


def skeleton(data: np.ndarray, workers: int = WORKERS, rule: str = SEPSET_RULE,
             pool: Pool | None = None, ci_test: str = CI_TEST) -> CausalGraph:
    """pool: optional already-started Pool(initializer=_init, initargs=(data, ci_test)) to reuse
    (used by timing.py to separate worker start-up from the algorithm itself)."""
    p = data.shape[1]
    cg = CausalGraph(p)
    cg.set_ind_test(CIT(data, ci_test))
    own = pool is None
    if own:
        pool = Pool(workers, initializer=_init, initargs=(data, ci_test))
    try:
        d = 0
        while cg.max_degree() - 1 >= d:
            adj = {v: sorted(int(u) for u in cg.neighbors(v)) for v in range(p)}  # frozen
            tasks = [(x, y, [u for u in adj[x] if u != y], [u for u in adj[y] if u != x], d, rule)
                     for x in range(p) for y in adj[x] if x < y]
            results = pool.map(_test_edge, tasks)                               # parallel
            for x, y, sepsets in results:                                       # barrier
                if sepsets is not None:
                    cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]))
                    for S in sepsets:
                        append_value(cg.sepset, x, y, S)
                        append_value(cg.sepset, y, x, S)
            d += 1
    finally:
        if own:
            pool.close()
            pool.join()
    return cg


# ---------------------------------------------------------------------------------------------------
# Vectorised orientation, identical to causal-learn's UCSepset.uc_sepset(cg, 2) followed by Meek.meek.
# Encoding (causal-learn): G[a, b] = -1, G[b, a] = 1  means a -> b;  both -1 means a - b.
# dpath (causal-learn GeneralGraph.dpath): dpath[a, b] = 1 means "b is an ancestor of a"; Meek skips an
# orientation a -> b when dpath[a, b] = 1. causal-learn updates dpath with adjust_dpath (an incomplete
# transitive closure) and rebuilds it (reconstitute_dpath) whenever a directed edge is removed; both are
# reproduced here so the cycle checks, and therefore the output, are the same.
# ---------------------------------------------------------------------------------------------------
_CHUNK = 1 << 16  # list items whose rule condition is evaluated at once in a Meek sweep


def _adjust_dpath(dpath: np.ndarray, i: int, j: int) -> None:
    """causal-learn GeneralGraph.adjust_dpath(i, j) for a new edge i -> j (equal to its k-loop because
    the diagonal of dpath is always 1)."""
    dpath[j, i] = 1
    row = dpath[i] == 1
    col = dpath[:, j] == 1
    dpath[j, row] = 1
    dpath[col, i] = 1


def _reconstitute_dpath(G: np.ndarray) -> np.ndarray:
    """causal-learn reconstitute_dpath(get_graph_edges()): identity, then adjust_dpath for every directed
    edge, popped from the end of get_graph_edges() (pairs i < j in row-major order)."""
    p = len(G)
    dpath = np.eye(p, dtype=int)
    iu, ju = np.triu_indices(p, 1)
    for i, j in zip(iu[::-1].tolist(), ju[::-1].tolist()):
        if G[j, i] == 1 and G[i, j] == -1:
            _adjust_dpath(dpath, i, j)
        elif G[i, j] == 1 and G[j, i] == -1:
            _adjust_dpath(dpath, j, i)
    return dpath


def _orient_colliders(cg: CausalGraph) -> tuple[np.ndarray, np.ndarray]:
    """uc_sepset, priority 2 ("prioritise existing colliders"). Returns (G, dpath).

    Triples are visited in causal-learn's order (find_unshielded_triples on the skeleton: middle node y,
    then x, then z, with x < z). x - y - z is a collider if y is in none of cg.sepset[x, z]; it is applied
    unless y -> x or y -> z is already oriented."""
    G = cg.G.graph.astype(np.int8)
    A = G != 0
    p = len(G)
    idx = np.arange(p)
    insep = np.zeros((p, p, p), bool)  # insep[x, z, y]: y is in some separating set of (x, z), x < z
    for x in range(p):
        for z in range(x + 1, p):
            if not A[x, z] and cg.sepset[x, z] is not None:
                for S in cg.sepset[x, z]:
                    insep[x, z, list(S)] = True
    G0 = G.copy()
    ops = []  # (a, y, a->y was already directed): every edge a -> y written, in causal-learn's order
    for y in range(p):
        for x in np.flatnonzero(A[y]):
            for z in np.flatnonzero(A[y] & ~A[x] & (idx > x) & ~insep[x, :, y]):
                if (G[y, x] == -1 and G[x, y] == 1) or (G[y, z] == -1 and G[z, y] == 1):
                    continue
                for a in (int(x), int(z)):
                    ops.append((a, y, bool(G[a, y] == -1 and G[y, a] == 1)))
                    G[a, y], G[y, a] = -1, 1
    # dpath: causal-learn rebuilds it each time it removes an already directed a -> y before re-adding it,
    # and adjusts it for every edge added. Only the last rebuild and the additions after it matter.
    last = max((k for k, op in enumerate(ops) if op[2]), default=None)
    if last is None:
        dpath, start = np.eye(p, dtype=int), 0
    else:
        snap = G0.copy()
        for a, y, _ in ops[:last]:
            snap[a, y], snap[y, a] = -1, 1
        a, y, _ = ops[last]
        snap[a, y] = snap[y, a] = 0  # the directed edge being removed at the last rebuild
        dpath, start = _reconstitute_dpath(snap), last
    for a, y, _ in ops[start:]:
        _adjust_dpath(dpath, a, y)
    return G, dpath


def _expand(seg_start: np.ndarray, seg_len: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For segments [seg_start[t], seg_start[t] + seg_len[t]): (owner t, position) of every element, in order."""
    owner = np.repeat(np.arange(len(seg_len)), seg_len)
    pos = np.arange(int(seg_len.sum())) - np.repeat(np.cumsum(seg_len) - seg_len, seg_len)
    return owner, np.repeat(seg_start, seg_len) + pos


def _meek_lists(G: np.ndarray):
    """causal-learn's UT, Tri and Kite lists, in the same order, built with array operations instead of
    pairwise list scans (permutations(find_adj(), 2), permutations(find_triangles(), 2))."""
    p = len(G)
    r, c = np.nonzero(G == -1)                   # find_tails: (col, row), row-major
    r2, c2 = np.nonzero(G == 1)                  # find_arrow_heads
    first = np.concatenate([c, c2])
    second = np.concatenate([r, r2])             # find_adj() = tails + heads, as (first, second)
    by_first = np.argsort(first, kind="stable")
    cnt = np.bincount(first, minlength=p)
    off = np.concatenate([[0], np.cumsum(cnt)])
    t, k = _expand(off[second], cnt[second])     # pair (adj[t], adj[by_first[k]]) with adj[t][1] == its [0]
    a, b, cc = first[t], second[t], second[by_first[k]]
    ok = a != cc
    UT = np.stack([a, b, cc], 1)[ok & (G[a, cc] == 0)]
    Tri = np.stack([a, b, cc], 1)[ok & (G[a, cc] != 0)]
    # kites: ordered pairs of triangles (t1, t2) with the same (i, l), t1.j < t2.j and t1.j, t2.j not adjacent
    key = Tri[:, 0] * p + Tri[:, 2]
    by_key = np.argsort(key, kind="stable")
    kcnt = np.bincount(key, minlength=p * p)
    koff = np.concatenate([[0], np.cumsum(kcnt)])
    kites = []
    for s in range(0, len(Tri), 50_000):         # blocks of t1 keep memory bounded
        t1 = np.arange(s, min(s + 50_000, len(Tri)))
        o, q = _expand(koff[key[t1]], kcnt[key[t1]])
        u1, u2 = t1[o], by_key[q]
        j1, j2 = Tri[u1, 1], Tri[u2, 1]
        keep = (j1 < j2) & (G[j1, j2] == 0)
        kites.append(np.stack([Tri[u1, 0], j1, j2, Tri[u1, 2]], 1)[keep])
    Kite = np.concatenate(kites) if kites else np.zeros((0, 4), int)
    return UT, Tri, Kite


def _directed(G, a, b):
    return (G[a, b] == -1) & (G[b, a] == 1)


def _undirected(G, a, b):
    return (G[a, b] == -1) & (G[b, a] == -1)


def _sweep(G, dpath, items, cond, src, dst) -> bool:
    """One causal-learn loop over a rule's list: items are visited in order and an item whose condition
    holds orients src -> dst unless dpath says dst is an ancestor of src. Conditions are evaluated for a
    block of items at once; after an orientation the block is re-evaluated from the next item, so the
    result equals the item-by-item loop."""
    changed, s, n = False, 0, len(items)
    while s < n:
        blk = items[s:s + _CHUNK]
        nxt = s + len(blk)
        for h in np.flatnonzero(cond(G, blk)):
            a, b = int(blk[h, src]), int(blk[h, dst])
            if dpath[a, b] == 1:                 # is_ancestor_of(b, a): causal-learn skips (continue)
                continue
            G[a, b], G[b, a] = -1, 1
            _adjust_dpath(dpath, a, b)
            changed, nxt = True, s + h + 1
            break
        s = nxt
    return changed


def _meek(G: np.ndarray, dpath: np.ndarray) -> np.ndarray:
    """causal-learn Meek.meek: rules R1, R2, R3 swept in that order until a full pass changes nothing."""
    UT, Tri, Kite = _meek_lists(G)
    r1 = lambda G, x: _directed(G, x[:, 0], x[:, 1]) & _undirected(G, x[:, 1], x[:, 2])
    r2 = lambda G, x: (_directed(G, x[:, 0], x[:, 1]) & _directed(G, x[:, 1], x[:, 2])
                       & _undirected(G, x[:, 0], x[:, 2]))
    r3 = lambda G, x: (_undirected(G, x[:, 0], x[:, 1]) & _undirected(G, x[:, 0], x[:, 2])
                       & _directed(G, x[:, 1], x[:, 3]) & _directed(G, x[:, 2], x[:, 3])
                       & _undirected(G, x[:, 0], x[:, 3]))
    loop = True
    while loop:
        c1 = _sweep(G, dpath, UT, r1, 1, 2)      # i -> j - k, i, k not adjacent      =>  j -> k
        c2 = _sweep(G, dpath, Tri, r2, 0, 2)     # i -> j -> k, i - k                 =>  i -> k
        c3 = _sweep(G, dpath, Kite, r3, 0, 3)    # kite i - j -> l <- k - i, i - l    =>  i -> l
        loop = c1 or c2 or c3
    return G


def orient(cg: CausalGraph) -> np.ndarray:
    """Colliders (uc_sepset, priority 2) + Meek on the skeleton cg; returns the endpoint matrix."""
    assert UC_PRIORITY == 2, "the vectorised orientation implements uc_sepset priority 2 only"
    G, dpath = _orient_colliders(cg)
    return _meek(G, dpath).astype(cg.G.graph.dtype)


def run(data: np.ndarray, workers: int = WORKERS, rule: str = SEPSET_RULE,
        pool: Pool | None = None, ci_test: str = CI_TEST) -> np.ndarray:
    cg = skeleton(data, workers, rule, pool, ci_test)
    return orient(cg)


def main() -> None:
    data = np.loadtxt(OUT / "data.csv", delimiter=",", skiprows=1)
    G = run(data)
    np.save(OUT / "parallel_pc.npy", G)
    print(f"parallel PC ({WORKERS} workers, sepset rule '{SEPSET_RULE}'): "
          f"{int((np.triu(G != 0)).sum())} edges")


if __name__ == "__main__":
    main()
