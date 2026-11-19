"""Time vectorized_pc on the real plant input (Dk, about 31,044 x 254), outside the notebook.

Runs the same call as the notebook's Step 4 cell, vpc.pc(Dk, alpha, max_l), twice in one process:
run 1 = what the notebook timed, run 2 = same call again (anything numba still compiled in run 1 is
cached by then). If run 2 is much faster than run 1, the notebook's 25 s was compile time, not compute.

    python parallel_pc_experiment/time_real_data.py                 # depth 1, alpha 0.05
    python parallel_pc_experiment/time_real_data.py --max-l 2
    python parallel_pc_experiment/time_real_data.py --dk path/to/Dk.npy

Input: drum_pca_pc/outputs/checkpoints/pc_input.pkl (the notebook's checkpoint, key "Dk"),
or a .npy file given with --dk.
"""
import argparse
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parallel_pc_experiment"))
import vectorized_pc as vpc  # noqa: E402


def load_dk(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh)["data"]["Dk"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk", type=Path, default=REPO / "drum_pca_pc" / "outputs" / "checkpoints" / "pc_input.pkl")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-l", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="numba/BLAS threads (default: notebook's set_threads())")
    a = ap.parse_args()

    print(f"python {platform.python_version()} | {platform.machine()} | cores {os.cpu_count()} | "
          f"numpy {np.__version__} | numba {vpc.numba.__version__ if vpc.HAVE_NUMBA else 'NOT INSTALLED'}")
    Dk = load_dk(a.dk)
    print(f"Dk {Dk.shape} from {a.dk}")

    if a.threads is None:
        vpc.set_threads()
    else:
        vpc.set_threads(a.threads, a.threads)
    t = time.perf_counter()
    vpc.warmup()
    print(f"warmup (numba compile) {time.perf_counter() - t:.1f} s")

    for run in (1, 2):
        t = time.perf_counter()
        A, _ = vpc.pc(Dk, alpha=a.alpha, max_l=a.max_l, verbose=(run == 1))
        wall = time.perf_counter() - t
        times = {k: round(v, 3) for k, v in vpc.LAST_INFO["times"].items()}
        und = A | A.T
        print(f"run {run}: wall {wall:.2f} s | {times} | Meek path {vpc.LAST_INFO['meek_path']} | "
              f"edges {int(und.sum() // 2)}")


if __name__ == "__main__":
    main()
