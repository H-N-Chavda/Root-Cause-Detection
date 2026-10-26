# Commands: diagnose causal_t2 s2_pc crash (exit -11 / segfault)

Run from the repository root on the lab machine. Each command rewrites `causal_t2/outputs/logs/s2_pc.log`.
Send that log back after each run.

## 1. Get the crash traceback
macOS / Linux:
```bash
cd causal_t2 && PYTHONFAULTHANDLER=1 python src/s2_pc.py > outputs/logs/s2_pc.log 2>&1
```
Windows PowerShell:
```powershell
cd causal_t2; $env:PYTHONFAULTHANDLER="1"; python src/s2_pc.py *> outputs/logs/s2_pc.log
```

## 2. Same run, BLAS limited to one thread (tests the threading cause)
macOS / Linux:
```bash
cd causal_t2 && PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/s2_pc.py > outputs/logs/s2_pc.log 2>&1
```
Windows PowerShell:
```powershell
cd causal_t2; $env:PYTHONFAULTHANDLER="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python src/s2_pc.py *> outputs/logs/s2_pc.log
```
While it runs, watch RAM (Task Manager / `top`). RAM filling up just before the crash = memory cause.

## 3. Record the library versions
```bash
python -c "import sys, numpy, scipy, causallearn; print(sys.version, numpy.__version__, scipy.__version__, causallearn.__version__)"
```

## Reading the result
| Result | Likely cause |
|---|---|
| Crash in step 1, none in step 2 | BLAS threading |
| RAM full before the crash | memory (PC problem too large: 267 tags, ~13M CI tests) |
| Crash in both, RAM fine | library / Python version (venv is Python 3.14) |
