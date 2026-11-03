"""Draw the figures from saved pipeline outputs: python unit_pca_pc_t2/plots.py (run main.py first)."""
import config as C
import helpers as H


def main() -> None:
    """Read outputs/, label the T² peaks, write the root-cause table and draw Figures 1-5 plus 'individual PCs'.

    Outputs: tables/t2_peaks.csv, tables/root_cause_by_event.csv, figures/fig1_flare.png, fig2_t2_overall.png,
    fig3_event_zooms.png, fig4_pc_terms.png, fig5_causal_graph.png and figures/individual PCs/*.png.

    Raises:
        SystemExit: If main.py's outputs are missing.
    """
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "plots.log")
    D = H.load_plot_inputs(P.out)
    start, end = D["flare"].index.min(), D["flare"].index.max()
    sections = H.time_sections(start, end)
    spans = H.context_spans(D)
    peaks = H.label_t2_peaks(D, sections)
    peaks.to_csv(d["tables"] / "t2_peaks.csv", index=False)
    H.root_cause_table(D).to_csv(d["tables"] / "root_cause_by_event.csv", index=False)
    H.plot_flare_overview(D, spans, sections, d["figures"] / "fig1_flare.png")
    H.plot_t2_overview(D, spans, sections, peaks, d["figures"] / "fig2_t2_overall.png")
    H.plot_event_zooms(D, d["figures"] / "fig3_event_zooms.png")
    H.plot_pc_terms(D, peaks, spans, d["figures"] / "fig4_pc_terms.png")
    H.plot_causal_graph(D, d["figures"] / "fig5_causal_graph.png")
    H.plot_individual_pcs(D, peaks, spans, H.time_sections(start, end, C.PLOT_PC_SECTION_HOURS),
                          d["figures"] / "individual PCs")
    H.log("figures finished")


if __name__ == "__main__":
    main()
