"""Generate a small linear-Gaussian dataset from a random DAG (20 nodes).

Model:  X_j = sum_{i in Pa(j)} B[i, j] * X_i + e_j,   e_j ~ N(0, 1)
DAG:    Erdos-Renyi over a random causal order, edge probability P_EDGE.
Weights: |B| ~ U(0.5, 1.5) with random sign (bounded away from 0 -> faithfulness is plausible).
Columns are written in a shuffled order, so column order != causal order.

Outputs (in ./outputs): data.csv (n x p), true_dag.npy (B != 0, i -> j iff A[i, j] = 1).
"""
from pathlib import Path

import numpy as np

SEED, P, N, P_EDGE = 42, 40, 300, 0.2
OUT = Path(__file__).parent / "outputs"


def main() -> None:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(P)                          # causal order
    B = np.zeros((P, P))
    for a in range(P):
        for b in range(a + 1, P):
            if rng.random() < P_EDGE:
                B[order[a], order[b]] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    X = np.zeros((N, P))
    for j in order:                                     # sample in causal order
        X[:, j] = X @ B[:, j] + rng.normal(size=N)
    X = (X - X.mean(0)) / X.std(0)                      # standardise (Fisher-z is scale-free anyway)
    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / "data.csv", X, delimiter=",", header=",".join(f"X{i}" for i in range(P)),
               comments="")
    np.save(OUT / "true_dag.npy", (B != 0).astype(int))
    print(f"data: n={N}, p={P}, true edges={(B != 0).sum()}, seed={SEED}")


if __name__ == "__main__":
    main()
