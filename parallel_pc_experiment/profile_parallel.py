"""Where does Parallel PC's time go, and when does parallelism pay off?

Sections (all times are wall-clock, perf_counter):
  1. CI-test cost: calls, cache hits and mean cost per Fisher-z test inside causal-learn PC-stable.
  2. Per-level breakdown of the parallel skeleton on an already-started pool:
       compute  = sum of time spent inside _test_edge (summed over tasks, i.e. CPU work),
       ideal    = compute / workers (perfect parallelism, zero overhead),
       map      = measured wall time of pool.map for that level,
       overhead = map - ideal (IPC, pickling, scheduling, idle cores at the barrier),
       ipc_only = wall time of pool.map on the same task list with a no-op function.
  3. Worker start-up (spawn + import of causal-learn in each worker).
  4. Heavy CI test (KCI): does parallelism win when each test is expensive? Run with OpenBLAS
     limited to 1 thread and with its default, because KCI's linear algebra is itself multithreaded.

Usage:  python profile_parallel.py            (sections 1-4)
        python profile_parallel.py --kci-only (section 4 only; used internally with env vars)
"""
import os
import statistics
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "outputs" / "data.csv"


# ------------------------------------------------------------------ helpers (picklable, top level)
def _timed_task(task):
    import parallel_pc
    t = time.perf_counter()
    r = parallel_pc._test_edge(task)
    return r, time.perf_counter() - t


def _noop(task):
    return task[0], task[1], None


def _ready(pool, w):
    pool.map(abs, range(4 * w))  # blocks until every worker has finished start-up


# ------------------------------------------------------------------ 1. CI-test cost
def ci_test_cost(data):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils import cit as citmod

    stats = {"calls": 0, "hits": 0, "t": 0.0}
    orig = citmod.FisherZ.__call__

    def wrapped(self, X, Y, condition_set=None):
        key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)[3]
        stats["calls"] += 1
        stats["hits"] += key in self.pvalue_cache
        t = time.perf_counter()
        p = orig(self, X, Y, condition_set)
        stats["t"] += time.perf_counter() - t
        return p

    citmod.FisherZ.__call__ = wrapped
    t = time.perf_counter()
    pc(data, 0.05, "fisherz", stable=True, uc_rule=0, uc_priority=2, show_progress=False)
    total = time.perf_counter() - t
    citmod.FisherZ.__call__ = orig
    computed = stats["calls"] - stats["hits"]
    print("1. causal-learn PC-stable, Fisher-z")
    print(f"   total {total:.3f} s | CI calls {stats['calls']} | cache hits {stats['hits']} | "
          f"computed {computed} | time inside CI calls {stats['t']:.3f} s | "
          f"per computed test ~{1e6 * stats['t'] / max(computed, 1):.0f} us\n")
    return total


# ------------------------------------------------------------------ 2. per-level breakdown
def level_breakdown(data, w):
    import parallel_pc
    from causallearn.graph.GraphClass import CausalGraph

    p = data.shape[1]
    cg = CausalGraph(p)
    rows = []
    with Pool(w, parallel_pc._init, (data,)) as pool:
        _ready(pool, w)
        t_all = time.perf_counter()
        d = 0
        while cg.max_degree() - 1 >= d:
            t0 = time.perf_counter()
            adj = {v: sorted(int(u) for u in cg.neighbors(v)) for v in range(p)}
            tasks = [(x, y, [u for u in adj[x] if u != y], [u for u in adj[y] if u != x], d,
                      parallel_pc.SEPSET_RULE) for x in range(p) for y in adj[x] if x < y]
            t1 = time.perf_counter()
            out = pool.map(_timed_task, tasks)
            t2 = time.perf_counter()
            pool.map(_noop, tasks)
            t3 = time.perf_counter()
            for (x, y, sepsets), _ in out:
                if sepsets is not None:
                    cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]))
            t4 = time.perf_counter()
            comp = [c for _, c in out]
            rows.append((d, len(tasks), sum(comp), max(comp), t2 - t1, t3 - t2, (t1 - t0) + (t4 - t3)))
            d += 1
        total = time.perf_counter() - t_all
    print(f"2. parallel skeleton, {w} worker(s), pool already started (times in ms)")
    print("   lvl  tasks   compute     ideal   max_task      map   overhead  ipc_only  serial")
    for d, n, comp, mx, mp, ipc, ser in rows:
        ideal = comp / w
        print(f"   {d:>3} {n:>6} {1e3*comp:9.1f} {1e3*ideal:9.1f} {1e3*mx:9.2f} {1e3*mp:9.1f} "
              f"{1e3*(mp-ideal):9.1f} {1e3*ipc:9.1f} {1e3*ser:7.1f}")
    comp = sum(r[2] for r in rows)
    mp = sum(r[4] for r in rows)
    print(f"   sum: compute {comp:.3f} s, ideal {comp / w:.3f} s, map {mp:.3f} s, "
          f"skeleton total {total:.3f} s (includes no-op IPC probe {sum(r[5] for r in rows):.3f} s)\n")
    return comp, mp


# ------------------------------------------------------------------ 3. start-up
def startup(data):
    import parallel_pc
    print("3. worker start-up (spawn + import causal-learn + build CIT), median of 3")
    for w in (1, 2, 4, 8):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            with Pool(w, parallel_pc._init, (data,)) as pool:
                _ready(pool, w)
                ts.append(time.perf_counter() - t)
        print(f"   {w} worker(s): {statistics.median(ts):.2f} s")
    print()


# ------------------------------------------------------------------ 4. heavy CI test
def kci_section():
    import parallel_pc
    from causallearn.search.ConstraintBased.PC import pc

    X = np.loadtxt(DATA, delimiter=",", skiprows=1)[:200, :8]  # small: KCI is O(n^3) per test
    label = os.environ.get("OPENBLAS_NUM_THREADS", "default")
    t = time.perf_counter()
    Gs = pc(X, 0.05, "kci", stable=True, uc_rule=0, uc_priority=2, show_progress=False).G.graph
    ts = time.perf_counter() - t
    print(f"   OpenBLAS threads={label:>7}: standard PC-stable {ts:6.2f} s", flush=True)
    for w in (2, 4):
        with Pool(w, parallel_pc._init, (X, "kci")) as pool:
            _ready(pool, w)
            t = time.perf_counter()
            Gp = parallel_pc.run(X, workers=w, pool=pool, ci_test="kci")
            ta = time.perf_counter() - t
        print(f"   OpenBLAS threads={label:>7}: parallel {w} workers {ta:6.2f} s "
              f"(algorithm only) -> {ts / ta:4.2f}x, same graph: {np.array_equal(Gs, Gp)}",
              flush=True)


def main():
    if "--kci-only" in sys.argv:
        kci_section()
        return
    data = np.loadtxt(DATA, delimiter=",", skiprows=1)
    print(f"data n={data.shape[0]}, p={data.shape[1]}; Apple M1 (4 performance + 4 efficiency cores)\n")
    ci_test_cost(data)
    for w in (1, 4, 8):
        level_breakdown(data, w)
    startup(data)
    print("4. KCI test (n=200, p=8): expensive CI tests")
    for threads in ("1", None):
        env = dict(os.environ)
        if threads:
            env.update(OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads)
        subprocess.run([sys.executable, __file__, "--kci-only"], env=env, check=True)


if __name__ == "__main__":
    main()
