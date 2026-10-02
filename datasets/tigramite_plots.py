"""Every plot tigramite.plotting offers, drawn on the qtank data and saved graphs.

Purpose: find the limits of tigramite's native plotting versus the
matplotlib/seaborn figures in results/qtank/figures. Each call is wrapped so a
failure is recorded in LIMITATIONS.md instead of aborting the sweep.

    python tigramite_plots.py [--run run-copula-tau20-s3] [--stride 3]
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tigramite.plotting as tp
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import LinearMediation
from tigramite.pcmci import PCMCI

from causal_bench.eda.copula import gaussian_copula_fit, gaussian_copula_transform

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "generated" if False else Path(__file__).resolve().parent / "generated"

notes: list[str] = []


def attempt(label: str, fn):
    try:
        fn()
        plt.close("all")
        notes.append(f"| {label} | ok | |")
    except Exception as exc:  # noqa: BLE001 - this sweep records failures
        plt.close("all")
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        notes.append(f"| {label} | FAIL | {msg} |")
        print(f"FAIL {label}: {msg}")
        traceback.print_exc(limit=1)


def load_data(ds: str, stride: int, var_names: list[str]):
    raw = np.genfromtxt(DATA / f"qtank_{ds}.csv", delimiter=",", names=True)
    data = np.column_stack([raw[c] for c in var_names])[::stride]
    df = pd.DataFrame(data, columns=var_names)
    cop = gaussian_copula_transform(df, gaussian_copula_fit(df)).to_numpy()
    return data, cop


def parents_from_links(links: np.ndarray) -> dict:
    n, _, taus = links.shape
    out = {j: [] for j in range(n)}
    for i in range(n):
        for j in range(n):
            for tau in range(taus):
                if links[i, j, tau] == "-->" and tau > 0:
                    out[j].append((i, -tau))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run-copula-tau20-s3")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--datasets", default="P_minus,P_plus")
    ap.add_argument("--tau-max", type=int, default=20)
    args = ap.parse_args()

    run_dir = REPO / "results" / "qtank" / args.run
    out = REPO / "results" / "qtank" / f"tigramite-plots-{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads((DATA / "ground_truth_edges.json").read_text())
    names = truth["columns"]
    n = len(names)

    # Ground truth has no lag information: drawn as contemporaneous links.
    # Lag-0 links must be symmetric ("-->" / "<--"), so draw them at lag 1.
    gt = np.full((n, n, 2), "", dtype="<U3")
    for a, b in truth["edges"]:
        gt[names.index(a), names.index(b), 1] = "-->"
    attempt("ground_truth plot_graph (drawn at lag 1; no lags known)", lambda: tp.plot_graph(
        graph=gt, var_names=names, save_name=str(out / "ground_truth_graph.png"),
        show_colorbar=False))

    for ds in args.datasets.split(","):
        rawd, copd = load_data(ds, args.stride, names)
        for tag, arr in (("raw", rawd), ("copula", copd)):
            df = pp.DataFrame(arr, var_names=names, datatime={0: np.arange(len(arr)) * 5 * args.stride})
            p = out / f"{ds}_{tag}"
            attempt(f"{ds}/{tag} plot_timeseries", lambda: tp.plot_timeseries(
                dataframe=df, save_name=str(p) + "_timeseries.png", figsize=(10, 8),
                time_label="t [s]"))
            attempt(f"{ds}/{tag} plot_scatterplots", lambda: tp.plot_scatterplots(
                dataframe=df, name=str(p) + "_scatter.png",
                setup_args={"figsize": (10, 10)}))
            attempt(f"{ds}/{tag} plot_densityplots", lambda: tp.plot_densityplots(
                dataframe=df, name=str(p) + "_density.png",
                setup_args={"figsize": (10, 10)}))

            # Lag functions: unconditional lagged dependencies via ParCorr.
            def lagfuncs():
                pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
                dep = pcmci.get_lagged_dependencies(tau_max=args.tau_max, val_only=True)
                np.save(str(p) + "_lagfuncs_val.npy", dep["val_matrix"])
                tp.plot_lagfuncs(val_matrix=dep["val_matrix"], name=str(p) + "_lagfuncs.png",
                                 setup_args={"var_names": names, "x_base": 5, "y_base": 0.5})
            attempt(f"{ds}/{tag} plot_lagfuncs (ParCorr lagged deps)", lagfuncs)

        # Saved graphs from the run.
        for gpath in sorted((run_dir / "graphs").glob(f"{ds}__*.json")):
            algo = gpath.stem.split("__")[1]
            g = json.loads(gpath.read_text())
            links = np.array(g["links"])
            val = np.array(g["val_matrix"], dtype=float) if g["val_matrix"] else None
            p = out / f"{ds}_{algo}"
            vmax = float(np.nanmax(np.abs(val))) if val is not None else 1.0
            kw = dict(graph=links, val_matrix=val, var_names=names,
                      vmin_edges=-vmax, vmax_edges=vmax)
            attempt(f"{ds}/{algo} plot_graph", lambda: tp.plot_graph(
                save_name=str(p) + "_graph.png", show_autodependency_lags=True,
                vmin_nodes=-vmax, vmax_nodes=vmax, **kw))
            attempt(f"{ds}/{algo} plot_time_series_graph", lambda: tp.plot_time_series_graph(
                save_name=str(p) + "_tsg.png", figsize=(16, 6), **kw))
            attempt(f"{ds}/{algo} write_csv", lambda: tp.write_csv(
                graph=links, val_matrix=val, var_names=names, save_name=str(p) + "_links.csv"))

            # plot_tsg: path-highlighting time-series graph, needs a dict of links.
            def tsg():
                link_dict = {j: [((i, -tau), 1.0) for i in range(n) for tau in range(links.shape[2])
                                 if links[i, j, tau] == "-->"] for j in range(n)}
                X = [(names.index("v1"), -args.tau_max)]
                Y = [(names.index("h2"), 0)]
                fig, ax = tp.plot_tsg(links=link_dict, X=X, Y=Y)
                fig.savefig(str(p) + "_plot_tsg.png", dpi=120)
            attempt(f"{ds}/{algo} plot_tsg (v1 -> h2 path)", tsg)

            # Mediation needs a fitted linear model on the discovered *lagged*
            # parents (LinearMediation ignores contemporaneous links). The
            # (cause, effect, lag) drawn is the one whose path touches the most
            # nodes, so an empty picture means no multi-step path exists.
            def mediation():
                df = pp.DataFrame(copd if "copula" in args.run else rawd, var_names=names)
                med = LinearMediation(dataframe=df, data_transform=None)
                med.fit_model(all_parents=parents_from_links(links), tau_max=args.tau_max)
                best = (0, None)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        for tau in range(1, 7):
                            gd = med.get_mediation_graph_data(i=i, tau=tau, j=j)
                            k = int(np.count_nonzero(gd["path_node_array"]))
                            if k > best[0]:
                                best = (k, (i, tau, j, gd))
                if best[1] is None:
                    raise RuntimeError("no lagged path between any pair; nothing to draw")
                i, tau, j, gd = best[1]
                tag = f"{names[i]}_to_{names[j]}_tau{tau}"
                tp.plot_mediation_graph(
                    var_names=names, path_val_matrix=gd["path_val_matrix"],
                    path_node_array=gd["path_node_array"],
                    save_name=str(p) + f"_mediation_{tag}.png")
                tp.plot_mediation_time_series_graph(
                    var_names=names, path_node_array=gd["path_node_array"],
                    tsg_path_val_matrix=gd["tsg_path_val_matrix"],
                    save_name=str(p) + f"_mediation_tsg_{tag}.png", figsize=(16, 6))
            attempt(f"{ds}/{algo} plot_mediation_graph + _time_series_graph", mediation)

    (out / "LIMITATIONS.md").write_text(
        "| plot | status | error |\n|---|---|---|\n" + "\n".join(notes) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
