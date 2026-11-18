"""Standard PC via causal-learn: pc(..., stable=STABLE).

stable=False: original Spirtes-Glymour PC; an edge is deleted the moment a separating set is
              found (cg.G.remove_edge(...) + break inside the test loop), so it is order-dependent.
stable=True:  causal-learn's PC-stable; adjacency frozen per depth, every conditioning set is
              tested, and the sepset is the union of all separating sets (no early stop).
Verified in causallearn/utils/PCUtils/SkeletonDiscovery.py.

Shared settings (also used by parallel_pc.py): Fisher-z test, alpha = 0.05, collider rule
uc_sepset with priority 2, then Meek rules.

Output: outputs/standard_pc.npy = causal-learn endpoint matrix
        (G[i,j] = -1 and G[j,i] = 1  ->  i -> j;  both -1  ->  i - j).
"""
from pathlib import Path

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

ALPHA, CI_TEST, UC_PRIORITY, STABLE = 0.05, "fisherz", 2, True
OUT = Path(__file__).parent / "outputs"


def run(data: np.ndarray) -> np.ndarray:
    cg = pc(data, alpha=ALPHA, indep_test=CI_TEST, stable=STABLE, uc_rule=0,
            uc_priority=UC_PRIORITY, show_progress=False)
    return cg.G.graph.copy()


def main() -> None:
    data = np.loadtxt(OUT / "data.csv", delimiter=",", skiprows=1)
    G = run(data)
    np.save(OUT / "standard_pc.npy", G)
    print(f"standard PC: {int((np.triu(G != 0)).sum())} edges")


if __name__ == "__main__":
    main()
