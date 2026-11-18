# Standard PC vs Parallel PC (20-node synthetic check)

```bash
python make_data.py      # outputs/data.csv, outputs/true_dag.npy (seed 42, n=300, p=20)
python standard_pc.py    # causal-learn pc(stable=False)          -> outputs/standard_pc.npy
python parallel_pc.py    # Le et al. Alg. 2, 4 worker processes    -> outputs/parallel_pc.npy
python compare.py        # metrics, controls A/B, outputs/comparison.png
```

Both pipelines use causal-learn's Fisher-z `CIT`, alpha = 0.05, the same independence rule (p > alpha),
and the same orientation code (`uc_sepset` priority 2 + Meek). They differ only in the skeleton phase.
Requires causal-learn 0.1.4.8, networkx, matplotlib.
