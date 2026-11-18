"""Fair wall-clock comparison of standard_pc.run vs parallel_pc.run on outputs/data.csv.

There is no on-disk cache to clear: __pycache__ only holds bytecode (import speed), and
causal-learn's CIT p-value cache is in memory, created fresh by every run() (no cache_path).

Measured (median of REPEATS, after one untimed warm-up):
  standard      : standard_pc.run(data)
  par end-to-end: parallel_pc.run(data, workers=w), includes starting w worker processes;
                  on macOS (spawn) each worker re-imports causal-learn (~2 s).
  par algorithm : the same run on a pool whose workers are already started; the pool is new
                  for every repeat, so worker p-value caches are never warm.
"""
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import parallel_pc
import standard_pc

REPEATS, WORKER_COUNTS = 5, (1, 2, 4, 8)
OUT = Path(__file__).parent / "outputs"


def median_time(fn) -> float:
    fn()                                                    # warm-up, not timed
    times = []
    for _ in range(REPEATS):
        times.append(fn())
    return statistics.median(times)


def timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def algorithm_only(data: np.ndarray, w: int) -> float:
    with Pool(w, parallel_pc._init, (data,)) as pool:
        pool.map(abs, range(4 * w))                         # wait until every worker is up
        return timed(lambda: parallel_pc.run(data, workers=w, pool=pool))


def main() -> None:
    data = np.loadtxt(OUT / "data.csv", delimiter=",", skiprows=1)
    print(f"data n={data.shape[0]}, p={data.shape[1]}; median of {REPEATS} runs\n")
    ts = median_time(lambda: timed(lambda: standard_pc.run(data)))
    print(f"standard PC (stable={standard_pc.STABLE}): {ts:6.3f} s\n")
    print("workers   end-to-end   algorithm-only   speed-up vs standard (e2e / algo)")
    for w in WORKER_COUNTS:
        te = median_time(lambda: timed(lambda: parallel_pc.run(data, workers=w)))
        ta = median_time(lambda: algorithm_only(data, w))
        print(f"{w:>7}   {te:8.3f} s   {ta:12.3f} s   {ts / te:6.2f}x / {ts / ta:5.2f}x")


if __name__ == "__main__":
    main()
