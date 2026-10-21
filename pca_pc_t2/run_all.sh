#!/usr/bin/env bash
# Runs the PCA -> PC -> T² pipeline s0 -> s5 from a clean state.
# Usage (from any folder):  bash pca_pc_t2/run_all.sh        Python:  PYTHON=/path/to/python bash run_all.sh
# Set DATA_ROOT in pca_pc_t2/.env first. Any failed stage gate stops the run (see outputs/run.log).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

"$PY" - <<'EOF'
import importlib.util, sys
need = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy", "sklearn": "scikit-learn", "scipy": "scipy",
        "matplotlib": "matplotlib", "networkx": "networkx", "causallearn": "causal-learn"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    sys.exit("Missing packages. Install with:  pip install " + " ".join(missing))
EOF

rm -rf outputs RESULTS.md          # generated files only; raw data is never touched
mkdir -p outputs
for s in s0_audit s1_build_matrix s2_pca s3_pc_algorithm s4_t2_analysis s5_figures; do
  echo "=== $s ==="
  "$PY" "src/$s.py"
done
echo "Done. Report: RESULTS.md, audit: outputs/00_data_audit.md, log: outputs/run.log"
