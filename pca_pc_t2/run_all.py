"""One command for the whole pipeline (paths in pca_pc_t2/.env). Copied from causal_t2/run_all.py; only the stage
list, the package list and these lines changed.

    python pca_pc_t2/run_all.py            # skips stages already finished (cached in outputs/data/*.done)
    python pca_pc_t2/run_all.py --force    # reruns every stage
    python pca_pc_t2/run_all.py --from s5_t2

Each stage runs as  python src/<stage>.py > outputs/logs/<stage>.log 2>&1  from this folder. Once a stage runs, all
later stages run too. Every result is appended to outputs/LOG.md; on a failure the last 80 log lines are printed.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NEED = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
        "networkx": "networkx", "causallearn": "causal-learn", "dotenv": "python-dotenv", "sklearn": "scikit-learn"}
missing = [pip for mod, pip in NEED.items() if importlib.util.find_spec(mod) is None]
if missing:
    sys.exit("Missing packages. Install with:  pip install " + " ".join(missing))

sys.path.insert(0, str(PROJECT / "src"))
from common import DATA, LOGS, OUT, WINDOW_TEXT  # noqa: E402  (reads .env, creates the output folders)

STAGES = ["s1_load_clean", "s2_pca", "s3_pc", "s4_dag", "s5_t2", "s6_figures", "s7_summary"]
args = sys.argv[1:]
force = "--force" in args
start = args[args.index("--from") + 1] if "--from" in args else None
if start is not None and start not in STAGES:
    sys.exit(f"--from must be one of {STAGES}")
cached_window = DATA / "window.txt"
old_window = cached_window.read_text(encoding="utf-8").strip() if cached_window.exists() else "not recorded"
if (DATA / "s1_load_clean.done").exists() and old_window != WINDOW_TEXT and not force:
    sys.exit(f"Cached results in {OUT} are for the time window '{old_window}', but .env now says "
             f"'{WINDOW_TEXT}'. Rerun with --force, or set another OUT_DIR in .env.")


def log_row(st, result, seconds, note):
    with open(OUT / "LOG.md", "a", encoding="utf-8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {st} | {result} | {seconds:.0f} | {note} |\n")


env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
rerun = force
for st in STAGES:
    rerun = rerun or st == start or st == "s7_summary"
    marker = DATA / f"{st}.done"
    if marker.exists() and not rerun:
        print(f"=== {st}: cached, skipped ===")
        continue
    rerun = True
    marker.unlink(missing_ok=True)
    log_path = LOGS / f"{st}.log"
    print(f"=== {st}: running, log -> {log_path.relative_to(PROJECT).as_posix()} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        code = subprocess.call([sys.executable, f"src/{st}.py"], cwd=PROJECT, stdout=fh, stderr=subprocess.STDOUT,
                               env=env)
    secs = time.time() - t0
    rel = log_path.relative_to(OUT).as_posix()
    if code != 0 or not marker.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        last = next((ln for ln in reversed(lines) if ln.strip()), "").replace("|", "/")[:200]
        log_row(st, "FAIL", secs, f"{rel}: {last}")
        print("\n".join(lines[-80:]))
        sys.exit(f"{st} failed (exit code {code}). Full log: {rel}. Add the error and fix to outputs/LOG.md.")
    log_row(st, "PASS", secs, rel)
    print(f"    done in {secs:.0f} s")
print("All stages passed. Read outputs/SUMMARY.md and outputs/figures_index.csv.")
