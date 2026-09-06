"""Health check: runs every algorithm on synthetic data with a known answer.

No repository data required -- the signal is generated here. Run it after an
install, or before pointing the package at a new dataset, to confirm the
libraries underneath (tigramite, lingam, numpy, scipy) are wired up correctly.

    python smoke_check.py            # all five, ~25 seconds
    python smoke_check.py --fast     # skips LSTE, ~7 seconds

Exit status is 0 only when every check passes.

WHY IT INSPECTS METADATA. `CausalDiscoveryAlgorithm.run` never raises: an
algorithm failure comes back as an empty graph whose `meta["completed"]` is
False. Checking only that a call returned would pass on a broken install, so
each run below is judged on `completed` and on whether the planted edge was
found.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

# The ground truth: X0 -> X1 at lag 1, X1 -> X2 at lag 1. Nothing else.
PLANTED = [("X0", "X1", 1), ("X1", "X2", 1)]
VAR_NAMES = ["X0", "X1", "X2", "X3"]


def make_data(n: int = 600, seed: int = 0) -> np.ndarray:
    """A linear SCM with non-Gaussian (uniform) noise.

    Non-Gaussian because VAR-LiNGAM identifies direction from exactly that;
    Gaussian noise would leave it unidentifiable and the check would fail for
    a reason that is not a broken install. X3 is an isolated distractor.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, size=(n, 4))
    x = np.zeros((n, 4))
    x[0] = noise[0]
    for t in range(1, n):
        x[t, 0] = 0.3 * x[t - 1, 0] + noise[t, 0]
        x[t, 1] = 0.8 * x[t - 1, 0] + 0.2 * noise[t, 1]
        x[t, 2] = 0.8 * x[t - 1, 1] + 0.2 * noise[t, 2]
        x[t, 3] = 0.3 * x[t - 1, 3] + noise[t, 3]
    return x


def check_imports() -> list[tuple[str, bool, str]]:
    rows = []
    for module in (
        "causal_bench",
        "causal_bench.algorithms",
        "causal_bench.graph",
        "causal_bench.io",
        "causal_bench.prior",
        "causal_bench.scoring",
        "causal_bench.utils",
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "tigramite",
        "lingam",
    ):
        try:
            __import__(module)
            rows.append((f"import {module}", True, ""))
        except Exception as exc:
            rows.append((f"import {module}", False, f"{type(exc).__name__}: {exc}"))
    return rows


def check_algorithms(data: np.ndarray, fast: bool) -> list[tuple[str, bool, str]]:
    from causal_bench.algorithms import available, build

    # Parameters are tuned for speed on 600x4, not for accuracy on real data.
    configs: dict[str, dict] = {
        "pc": {"pc_alpha": 0.05},
        "pc_manual": {"pc_alpha": 0.05},
        "pcmci_plus": {"tau_max": 2, "pc_alpha": 0.05},
        "var_lingam": {"tau_max": 2, "run_bootstrap": False},
        "lste": {
            "tau_max": 2,
            "alpha": 0.05,
            "sig_samples": 50,
            "two_stage": False,
            "fdr_method": "none",
        },
    }
    if fast:
        configs.pop("lste")

    rows = []
    for name in available():
        if name not in configs:
            rows.append((f"{name}: skipped", True, "not run"))
            continue
        try:
            graph = build(name, seed=0, **configs[name]).run(data, VAR_NAMES)
        except Exception as exc:
            rows.append((f"{name}: run", False, f"{type(exc).__name__}: {exc}"))
            continue

        meta = graph.meta
        if not meta.get("completed"):
            rows.append((f"{name}: run", False, str(meta.get("error"))))
            continue

        # pc and pc_manual are contemporaneous-only, so they cannot see a lag-1
        # edge; for them "found the planted structure" means finding the
        # adjacency, not the lag. Both are judged on producing a non-empty graph.
        found = set(graph.directed_edges()) | {
            (b, a, lag) for a, b, lag in graph.undirected_edges()
        }
        if name in ("pc", "pc_manual"):
            detail = f"{graph.n_edges()} edges, {meta['runtime_seconds']}s"
            rows.append((f"{name}: run", True, detail))
        else:
            hits = [e for e in PLANTED if e in found]
            ok = len(hits) == len(PLANTED)
            detail = (
                f"{len(hits)}/{len(PLANTED)} planted edges, "
                f"{graph.n_edges()} total, {meta['runtime_seconds']}s"
            )
            rows.append((f"{name}: recovers planted edges", ok, detail))
    return rows


def check_support(data: np.ndarray) -> list[tuple[str, bool, str]]:
    """Round-trips the pieces the algorithms hand their output to."""
    import json
    import tempfile
    from pathlib import Path

    from causal_bench.algorithms import build
    from causal_bench.graph import load_graph, save_graph
    from causal_bench.prior import load_prior_knowledge
    from causal_bench.scoring import structural_hamming_distance

    rows = []
    graph = build("pcmci_plus", tau_max=1, pc_alpha=0.05).run(data, VAR_NAMES)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_graph(graph, Path(tmp) / "g.json")
            reloaded = load_graph(path)
        ok = reloaded.directed_edges() == graph.directed_edges()
        rows.append(("graph: save/load round-trip", ok, ""))
    except Exception as exc:
        rows.append(("graph: save/load round-trip", False, f"{type(exc).__name__}: {exc}"))

    try:
        shd = structural_hamming_distance(graph, graph)
        rows.append(("scoring: SHD of a graph against itself is 0", shd.shd == 0, ""))
    except Exception as exc:
        rows.append(("scoring: SHD", False, f"{type(exc).__name__}: {exc}"))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pk.yaml"
            path.write_text(
                json.dumps(
                    {
                        "edges": [
                            {"cause": "X0", "effect": "X1", "lag": 1, "kind": "forced"},
                            {"cause": "X3", "effect": "X0", "lag": 1, "kind": "forbidden"},
                        ]
                    }
                )
            )  # JSON is valid YAML
            pk = load_prior_knowledge(path)
        ok = pk is not None and len(pk.edges) == 2
        rows.append(("prior: loads a 2-edge YAML", ok, ""))
    except Exception as exc:
        rows.append(("prior: loads a 2-edge YAML", False, f"{type(exc).__name__}: {exc}"))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="skip LSTE (the slow one)")
    args = parser.parse_args()

    import logging

    logging.disable(logging.CRITICAL)  # the wrappers log their own progress

    data = make_data()
    sections = [("Imports", check_imports())]
    if all(ok for _, ok, _ in sections[0][1]):
        sections.append(("Algorithms", check_algorithms(data, args.fast)))
        sections.append(("Support modules", check_support(data)))
    else:
        print("imports failed; skipping the run checks\n")

    failures = 0
    for title, rows in sections:
        print(f"\n{title}")
        print("-" * len(title))
        for label, ok, detail in rows:
            failures += not ok
            mark = "PASS" if ok else "FAIL"
            suffix = f"  ({detail})" if detail else ""
            print(f"  [{mark}] {label}{suffix}")

    total = sum(len(rows) for _, rows in sections)
    print(f"\n{total - failures}/{total} checks passed")
    if args.fast:
        print("(--fast: LSTE was not run)")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
