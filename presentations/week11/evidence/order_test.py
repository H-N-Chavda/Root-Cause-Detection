"""Column-order test on the drum PC input (drum_pca_pc/outputs/Dk.npy, 31,044 rows x 254 PCs).

Runs vectorized_pc (alpha 0.05, max_l 1) on the original column order, the reversed order and 4 random orders
(seed 0) and compares each result with the original: skeleton identical? how many edge directions flip?
    python presentations/week11/evidence/order_test.py > presentations/week11/evidence/order_test.log
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "parallel_pc_experiment"))
import vectorized_pc as vpc  # noqa: E402

X = np.load(ROOT / "drum_pca_pc" / "outputs" / "Dk.npy")
vpc.set_threads(8, 8)
vpc.warmup()


def run(perm):
    A, _ = vpc.pc(X[:, perm], alpha=0.05, max_l=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return A[np.ix_(inv, inv)], vpc.LAST_INFO["meek_path"]


p = X.shape[1]
base, _ = run(np.arange(p))
rng = np.random.default_rng(0)
perms = [("reversed", np.arange(p)[::-1])] + [(f"random {k}", rng.permutation(p)) for k in range(1, 5)]
D0 = base & ~base.T
for name, perm in perms:
    A, path = run(perm)
    D1 = A & ~A.T
    both = np.triu((A | A.T) & (base | base.T), 1)
    i, j = np.nonzero(both)
    flipped = int(((D0[i, j] & D1[j, i]) | (D0[j, i] & D1[i, j])).sum())
    print(f"{name}: skeleton_identical={np.array_equal(A | A.T, base | base.T)} edges={int(both.sum())} "
          f"flipped={flipped} pct={100 * flipped / both.sum():.1f} meek_path={path}")
