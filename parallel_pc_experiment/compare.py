"""Compare standard_pc.py (causal-learn PC-stable) with vectorized_pc.py.

    python compare.py                                   # small + medium suites
    python compare.py --stages large --large-n 300 --large-N 20000 [--large-max-l2]
    python compare.py --stages large --large-csv my_data.csv           # real data, header row
    python compare.py --legacy-parallel                 # original standard vs parallel_pc comparison

Per run: skeleton equal (Y/N) and number of differing adjacencies, sepsets equivalent in effect
(same collider decision for every unshielded triple) plus whether the sepset unions are identical,
final CPDAG equal (Y/N) with every differing pair listed as (i, j, standard vs vectorized), and the
wall-clock time of both with the speedup. The reference is causal-learn pc() with standard_pc.py's
settings (CI test, stable, uc_priority); alpha and max_l (causal-learn's max_k) vary per run.
Exit code 1 if any run differs.

Legacy (--legacy-parallel): CPDAGs from standard_pc.py and parallel_pc.py, with controls
  A. Parallel PC with 1 vs 4 workers                -> does parallelism itself change anything?
  B. Both algorithms under 10 column permutations    -> order dependence of skeleton and CPDAG.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from causallearn.search.ConstraintBased.PC import pc as causallearn_pc  # noqa: E402

import make_data  # noqa: E402
import parallel_pc  # noqa: E402
import standard_pc  # noqa: E402
import vectorized_pc as vpc  # noqa: E402

OUT = Path(__file__).parent / "outputs"

# ---- sizes (all overridable from the command line where noted) ----
SMALL_P, SMALL_N, SMALL_SEEDS = 20, 1000, (0, 1, 2, 3, 4)       # variables, samples, seeds
MEDIUM_P, MEDIUM_N, MEDIUM_SEEDS = 80, 5000, (0, 1)
ALPHAS = (0.01, 0.05, 0.1)                                       # small and medium suites
EXPECTED_DEGREE = 3.0        # synthetic DAGs: edge probability = EXPECTED_DEGREE / (p - 1)   (--degree)
LARGE_P, LARGE_N = 300, 20000                                    # --large-n, --large-N
LARGE_SEED = 0                                                   # --large-seed
MAX_DEGREE_FOR_L2 = 30       # large suite: skip max_l=2 if the max degree after l=1 exceeds this


def edge_type(G, i, j):
    """'->', '<-', '--' or None for pair (i, j) in a causal-learn endpoint matrix."""
    if G[i, j] == 0 and G[j, i] == 0:
        return None
    if G[i, j] == -1 and G[j, i] == 1:
        return "->"
    if G[i, j] == 1 and G[j, i] == -1:
        return "<-"
    return "--" if G[i, j] == G[j, i] == -1 else "<>"


def diff(G1, G2):
    p = len(G1)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    adj = [(i, j) for i, j in pairs if (edge_type(G1, i, j) is None) != (edge_type(G2, i, j) is None)]
    ori = [(i, j) for i, j in pairs if edge_type(G1, i, j) and edge_type(G2, i, j)
           and edge_type(G1, i, j) != edge_type(G2, i, j)]
    return adj, ori


def unpermute(G, perm):
    """Map a graph learned on data[:, perm] back to original column labels."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return G[np.ix_(inv, inv)]


def draw(ax, G, title, highlight, pos):
    p = len(G)
    g = nx.DiGraph()
    g.add_nodes_from(range(p))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#dbe7f3", edgecolors="#2f5d8a", node_size=420)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={i: f"X{i}" for i in range(p)}, font_size=8)
    for i in range(p):
        for j in range(i + 1, p):
            t = edge_type(G, i, j)
            if t is None:
                continue
            col = "#c0392b" if (i, j) in highlight else "#333333"
            u, v = (j, i) if t == "<-" else (i, j)
            nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[(u, v)], edge_color=col, width=1.6,
                                   arrows=True, arrowsize=14, node_size=420,
                                   arrowstyle="-|>" if t in ("->", "<-") else "-")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def compare_parallel() -> None:
    data = np.loadtxt(OUT / "data.csv", delimiter=",", skiprows=1)
    true = np.load(OUT / "true_dag.npy")
    Gs, Gp = np.load(OUT / "standard_pc.npy"), np.load(OUT / "parallel_pc.npy")
    adj, ori = diff(Gs, Gp)
    n_edges = lambda G: int(np.triu((G != 0) | (G.T != 0)).sum())
    true_skel = (true + true.T) > 0
    skel_err = lambda G: int(np.triu(((G != 0) | (G.T != 0)) != true_skel, 1).sum())

    print(f"=== Standard PC (stable={standard_pc.STABLE}) vs Parallel PC "
          f"(sepset rule '{parallel_pc.SEPSET_RULE}'), Fisher-z, alpha=0.05 ===")
    print(f"edges: standard={n_edges(Gs)}  parallel={n_edges(Gp)}  true DAG={int(true.sum())}")
    print(f"adjacencies identical : {not adj}   ({len(adj)} differing pairs: "
          f"{[(f'X{i}', f'X{j}') for i, j in adj]})")
    print(f"orientations identical (on shared edges): {not ori}   ({len(ori)} differing: "
          + ", ".join(f"X{i}{edge_type(Gs, i, j)}X{j} vs X{i}{edge_type(Gp, i, j)}X{j}"
                      for i, j in ori) + ")")
    print(f"total differing node pairs: {len(adj) + len(ori)}")
    print(f"skeleton errors vs true DAG: standard={skel_err(Gs)}  parallel={skel_err(Gp)}")

    # Control A: parallelism itself
    G1 = parallel_pc.run(data, workers=1)
    print(f"\n[A] parallel PC, 1 vs 4 workers identical: {np.array_equal(G1, Gp)}")

    # Control B: order dependence (10 column permutations)
    rng = np.random.default_rng(0)
    skel = lambda G: ((G != 0) | (G.T != 0)).tobytes()
    res = {"standard": [], "parallel": []}
    for _ in range(10):
        perm = rng.permutation(data.shape[1])
        res["standard"].append(unpermute(standard_pc.run(data[:, perm]), perm))
        res["parallel"].append(unpermute(parallel_pc.run(data[:, perm]), perm))
    for k, Gl in res.items():
        print(f"[B] {k:8s} over 10 column orders: distinct skeletons={len({skel(G) for G in Gl})}"
              f"  distinct CPDAGs={len({G.tobytes() for G in Gl})}")

    # Figure: same layout, differing pairs in red
    pos = nx.circular_layout(range(data.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    hl = set(adj) | set(ori)
    draw(axes[0], Gs, f"Standard PC (causal-learn, stable={standard_pc.STABLE}): {n_edges(Gs)} edges", hl, pos)
    draw(axes[1], Gp, f"Parallel PC (Alg. 2, 4 workers, sepset={parallel_pc.SEPSET_RULE}): {n_edges(Gp)} edges", hl, pos)
    fig.suptitle(f"Red = differs between the two CPDAGs ({len(hl)} node pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    print(f"\nfigure: {OUT / 'comparison.png'}")


# ---------------------------------------------------------------------------------------------------
# standard_pc vs vectorized_pc
# ---------------------------------------------------------------------------------------------------
def run_standard(X, alpha, max_l):
    """causal-learn pc() with standard_pc.py's settings. Returns (endpoint matrix, {(i, j): sepset union})."""
    cg = causallearn_pc(X, alpha=alpha, indep_test=standard_pc.CI_TEST, stable=standard_pc.STABLE, uc_rule=0,
                        uc_priority=standard_pc.UC_PRIORITY, show_progress=False, max_k=max_l)
    p = X.shape[1]
    sep = {}
    for i in range(p):
        for j in range(i + 1, p):
            if cg.sepset[i, j] is not None:
                sep[(i, j)] = {int(v) for S in cg.sepset[i, j] for v in S}
    return cg.G.graph.copy(), sep


def sepset_check(Gs, Gv, sep_s, sep_v):
    """(# unshielded triples x - y - z, in both skeletons, where 'y in sepset(x, z)' differs,
        # pairs non-adjacent in both whose sepset unions differ)."""
    adj_s, adj_v = (Gs != 0) | (Gs.T != 0), (Gv != 0) | (Gv.T != 0)
    both_adj = adj_s & adj_v
    effect = exact = 0
    p = len(Gs)
    for x in range(p):
        for z in range(x + 1, p):
            if adj_s[x, z] or adj_v[x, z]:
                continue
            Ss, Sv = sep_s.get((x, z), set()), set(sep_v.get((x, z), ()))
            exact += Ss != Sv
            effect += sum((y in Ss) != (y in Sv) for y in np.flatnonzero(both_adj[x] & both_adj[z]).tolist())
    return effect, exact


def compare_run(label, X, alpha, max_l, run_std=True):
    """One standard vs vectorized comparison; prints a report and returns True if everything is equal."""
    t0 = time.perf_counter()
    A, sep_v = vpc.pc(X, alpha=alpha, max_l=max_l)
    tv = time.perf_counter() - t0
    info = vpc.LAST_INFO
    Gv = vpc.to_causallearn(A)
    head = f"[{label} alpha={alpha} max_l={max_l}] N={X.shape[0]} n={X.shape[1]}"
    if not run_std:
        print(f"{head} | vectorized {tv:.2f}s (standard skipped) | edges {int(np.triu(Gv != 0).sum())}"
              f" | Meek path {info['meek_path']}")
        return True
    t0 = time.perf_counter()
    Gs, sep_s = run_standard(X, alpha, max_l)
    ts = time.perf_counter() - t0
    adj_d, ori_d = diff(Gs, Gv)
    effect, exact = sepset_check(Gs, Gv, sep_s, sep_v)
    cpdag_eq = not adj_d and not ori_d and np.array_equal(Gs, Gv)
    yn = lambda b: "Y" if b else "N"
    print(f"{head} | skeleton equal: {yn(not adj_d)} ({len(adj_d)} differing)"
          f" | sepsets equal in effect: {yn(effect == 0)} ({effect} triples; unions identical: {yn(exact == 0)},"
          f" {exact} pairs) | CPDAG equal: {yn(cpdag_eq)} ({len(adj_d) + len(ori_d)} differing)"
          f" | standard {ts:.2f}s vectorized {tv:.2f}s speedup {ts / max(tv, 1e-9):.1f}x"
          f" | Meek path {info['meek_path']}, exact re-tests {info['exact_tests']}+{info['exact_edges']} edges")
    for i, j in sorted(adj_d + ori_d):
        print(f"    ({i}, {j}, {edge_type(Gs, i, j)} vs {edge_type(Gv, i, j)})")
    return not adj_d and effect == 0 and cpdag_eq


def synthetic(p, n_samples, degree, seed):
    X, _ = make_data.simulate(p=p, n=n_samples, p_edge=min(1.0, degree / max(p - 1, 1)), seed=seed)
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="standard_pc (causal-learn PC-stable) vs vectorized_pc")
    ap.add_argument("--stages", default="small,medium", help="comma list of: small, medium, large")
    ap.add_argument("--degree", type=float, default=EXPECTED_DEGREE, help="expected degree of synthetic DAGs")
    ap.add_argument("--large-n", type=int, default=LARGE_P, help="large suite: number of variables")
    ap.add_argument("--large-N", type=int, default=LARGE_N, help="large suite: number of samples")
    ap.add_argument("--large-seed", type=int, default=LARGE_SEED)
    ap.add_argument("--large-csv", default=None, help="large suite: CSV (header row, N x n) instead of synthetic data")
    ap.add_argument("--large-alpha", type=float, default=standard_pc.ALPHA)
    ap.add_argument("--large-max-l2", action="store_true",
                    help=f"also compare max_l=2 (skipped if max degree after l=1 > {MAX_DEGREE_FOR_L2})")
    ap.add_argument("--skip-standard", action="store_true", help="large suite: time vectorized_pc only")
    ap.add_argument("--legacy-parallel", action="store_true", help="run the original standard vs parallel_pc comparison")
    args = ap.parse_args()
    if args.legacy_parallel:
        compare_parallel()
        return

    if not standard_pc.STABLE:
        print("WARNING: standard_pc.STABLE is False; vectorized_pc implements PC-stable, so results can differ.")
    print(f"numba available: {vpc.HAVE_NUMBA}; compiling kernels ...")
    vpc.set_threads()
    vpc.warmup()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    results = []

    if "small" in stages:
        print(f"\n=== small: n={SMALL_P}, N={SMALL_N}, seeds {SMALL_SEEDS}, alphas {ALPHAS}, max_l=None ===")
        for seed in SMALL_SEEDS:
            X = synthetic(SMALL_P, SMALL_N, args.degree, seed)
            for alpha in ALPHAS:
                results.append((f"small seed={seed} alpha={alpha}", compare_run(f"small seed={seed}", X, alpha, None)))

    if "medium" in stages:
        print(f"\n=== medium: n={MEDIUM_P}, N={MEDIUM_N}, seeds {MEDIUM_SEEDS}, alphas {ALPHAS}, max_l=None ===")
        for seed in MEDIUM_SEEDS:
            X = synthetic(MEDIUM_P, MEDIUM_N, args.degree, seed)
            for alpha in ALPHAS:
                results.append((f"medium seed={seed} alpha={alpha}", compare_run(f"medium seed={seed}", X, alpha, None)))

    if "large" in stages:
        if args.large_csv:
            X = np.loadtxt(args.large_csv, delimiter=",", skiprows=1)
            src = args.large_csv
        else:
            X = synthetic(args.large_n, args.large_N, args.degree, args.large_seed)
            src = f"synthetic seed={args.large_seed}"
        print(f"\n=== large: N={X.shape[0]}, n={X.shape[1]} ({src}), alpha={args.large_alpha} ===")
        results.append(("large max_l=1", compare_run("large", X, args.large_alpha, 1, not args.skip_standard)))
        if args.large_max_l2:
            adj1, _ = vpc.skeleton(X, args.large_alpha, max_l=1)
            deg = adj1.sum(1)
            print(f"degree after l=1: mean {deg.mean():.2f}, max {deg.max()}")
            if deg.max() > MAX_DEGREE_FOR_L2:
                print(f"WARNING: max degree {deg.max()} > {MAX_DEGREE_FOR_L2}; skipping max_l=2 "
                      f"(the reference would test O(degree^2) sets per edge).")
            else:
                results.append(("large max_l=2",
                                compare_run("large", X, args.large_alpha, 2, not args.skip_standard)))

    bad = [lab for lab, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} runs identical" + (f"; differing: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
