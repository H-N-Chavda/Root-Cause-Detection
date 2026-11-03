"""Build the contributor deck from saved outputs: python unit_pca_pc_t2/deck.py (run main.py first)."""
import config as C
import helpers as H


def main() -> None:
    """Read outputs/ and write figures/contributors.pptx: native PowerPoint charts of T²_j / own limit per PC.

    Raises:
        SystemExit: If main.py's outputs, python-pptx or XlsxWriter are missing.
    """
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "deck.log")
    H.build_contributor_deck(H.load_plot_inputs(P.out), d["figures"] / "contributors.pptx")


if __name__ == "__main__":
    main()
