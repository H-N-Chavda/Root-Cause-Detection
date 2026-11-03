"""Write outputs/SUMMARY.md from saved outputs: python unit_pca_pc_t2/summary.py (run main.py first)."""
import config as C
import helpers as H


def main():
    """Read outputs/ and write SUMMARY.md (data and cleaning, PCA variance, graph and loops, T² and root causes)."""
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "summary.log")
    H.build_summary(P.out)


if __name__ == "__main__":
    main()
