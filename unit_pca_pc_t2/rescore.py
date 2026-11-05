"""Re-score saved outputs (no PC run): python unit_pca_pc_t2/rescore.py (main.py also calls it at the end).

Moving-window T² baseline (TA's Main.py), x T2_LIMIT_SCALE limits, alarms, TP/FP/TN/FN metrics, root cause per alarm
and event, and outputs/RCA_report.xlsx. Reads graph/pc_scores.parquet and graph/dag_edges.csv from main.py.
"""
import config as C
import evaluation as E
import helpers as H


def main() -> None:
    """Rewrite t2/ and tables/ with the moving-window scores and write RCA_report.xlsx."""
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "rescore.log")
    E.rescore(P.out)
    E.write_excel(P.out)


if __name__ == "__main__":
    main()
