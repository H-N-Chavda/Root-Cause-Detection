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


def simulate(p: int = P, n: int = N, p_edge: float = P_EDGE, seed: int = SEED):
    """Return (X, true_dag) without writing files; main() uses the module constants."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(p)                          # causal order
    B = np.zeros((p, p))
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < p_edge:
                B[order[a], order[b]] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    X = np.zeros((n, p))
    for j in order:                                     # sample in causal order
        X[:, j] = X @ B[:, j] + rng.normal(size=n)
    X = (X - X.mean(0)) / X.std(0)                      # standardise (Fisher-z is scale-free anyway)
    return X, (B != 0).astype(int)


def main() -> None:
    X, true_dag = simulate()
    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / "data.csv", X, delimiter=",", header=",".join(f"X{i}" for i in range(P)),
               comments="")
    np.save(OUT / "true_dag.npy", true_dag)
    print(f"data: n={N}, p={P}, true edges={true_dag.sum()}, seed={SEED}")


if __name__ == "__main__":
    main()
