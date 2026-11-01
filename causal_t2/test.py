"""Diagnostic: run the Stage 2 PC call on the first 100 rows of the preprocessed matrix.

Purpose: tell whether the stack-limit failure on one month of data comes from the number of tags (it should
reproduce here) or from the number of rows (it should not). Nothing in src/ is changed and no output file is written.

    python causal_t2/test.py > causal_t2/outputs/logs/test_100rows.log 2>&1

Needs outputs/data/Z.parquet from Stage 1 (python causal_t2/run_all.py runs it first if missing).
The recursion limit is only printed, never raised.
"""
import faulthandler
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import DATA, PC_ALPHA, read_registry  # noqa: E402  (reads .env)
from causallearn.graph.GraphClass import CausalGraph  # noqa: E402
from causallearn.graph.GraphNode import GraphNode  # noqa: E402
from causallearn.search.ConstraintBased.PC import pc  # noqa: E402
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge  # noqa: E402

faulthandler.enable()                 # a segfault (C stack overflow) still prints the Python stack
N_ROWS = 100
MAX_K = 1                             # same depth cap Stage 2 chose for the failing one-month run
DUP_R = 0.999999                      # |r| at or above this in the test rows = duplicate tag (singular Fisher-Z)

z_path = DATA / "Z.parquet"
if not z_path.exists():
    sys.exit(f"{z_path} not found: run Stage 1 first (python causal_t2/run_all.py stops after it fails at s2_pc).")
Z = pd.read_parquet(z_path)
print(f"Z.parquet: {Z.shape[0]} rows x {Z.shape[1]} tags")
X = Z.head(N_ROWS)
del Z
names = list(X.columns)
D = X.to_numpy(dtype=np.float64)

reg = read_registry()
flare_of = dict(zip(reg["tag"], reg["is_flare"]))
fl = np.array([bool(flare_of.get(t, False)) for t in names])

# ---- facts about the 5-row input that change how a failure should be read
sd = D.std(axis=0)
with np.errstate(invalid="ignore", divide="ignore"):
    R = np.corrcoef(D, rowvar=False)
iu = np.triu_indices(len(names), 1)
print(f"loaded rows: {D.shape[0]} rows x {D.shape[1]} tags, flare tags {int(fl.sum())}")
print(f"constant tags in these rows (correlation undefined): {int((sd == 0).sum())}")
print(f"tag pairs with |r| >= {DUP_R} (singular Fisher-Z submatrix): {int((np.abs(R[iu]) >= DUP_R).sum())}")

# ---- drop only what makes Fisher-Z impossible on so few rows: constant tags, then one tag of each duplicate pair
keep = sd > 0
r = np.abs(np.nan_to_num(R[iu]))
for h in np.flatnonzero(r >= DUP_R):
    i, j = iu[0][h], iu[1][h]
    if keep[i] and keep[j]:
        keep[i if (fl[j] and not fl[i]) else j] = False      # never drop a flare tag if avoidable
names = [t for t, k in zip(names, keep) if k]
D = D[:, keep]
fl = fl[keep]
print(f"test input after drops: {D.shape[0]} rows x {D.shape[1]} tags, flare tags {int(fl.sum())}")
print(f"Python {sys.version.split()[0]}, recursion limit {sys.getrecursionlimit()} (unchanged)")
print(f"PC: fisherz, alpha = {PC_ALPHA}, stable = True, uc_rule = 0, uc_priority = 2, max_k = {MAX_K}, "
      "flare tags in tier 1 (as in src/s2_pc.py)")

# ---- same background knowledge and the same find_adj replacement as src/s2_pc.py
node_names = [f"X{i + 1}" for i in range(len(names))]
bk = BackgroundKnowledge()
for i, nm in enumerate(node_names):
    bk.add_node_to_tier(GraphNode(nm), 1 if fl[i] else 0)
bk.forbid_within_tier(1)


def _find_adj(self):
    g = self.G.graph
    t, a = np.where(g == -1), np.where(g == 1)
    return list(zip(t[1], t[0])) + list(zip(a[1], a[0]))


CausalGraph.find_adj = _find_adj

tracemalloc.start()
t0 = time.perf_counter()
try:
    cg = pc(D, alpha=PC_ALPHA, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2, background_knowledge=bk,
            verbose=False, show_progress=False, max_k=MAX_K)
    result = "SUCCESS"
except BaseException as exc:          # RecursionError, MemoryError, ValueError, KeyboardInterrupt ...
    cg = None
    result = f"FAILURE: {type(exc).__name__}: {exc}"
    print("---- traceback")
    traceback.print_exc(file=sys.stdout)
secs = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("---- result")
print(result)
print(f"PC time: {secs:.2f} s")
print(f"peak Python memory during PC (tracemalloc): {peak / 2**20:.1f} MiB")
try:
    import resource
    print(f"peak process RSS (ru_maxrss, KiB on Linux): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
except ImportError:
    pass                              # Windows has no resource module
if cg is not None:
    G = np.asarray(cg.G.graph)
    print(f"edges in output graph: {int(((G != 0) | (G.T != 0))[np.triu_indices(len(names), 1)].sum())}")
