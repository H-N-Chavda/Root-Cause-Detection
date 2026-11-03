"""Draw the figures from saved pipeline outputs: python unit_pca_pc_t2/plots.py (run main.py first)."""
import config as C
import helpers as H


def main():
    """Read outputs/, label the T² peaks, draw Figures 1-5 and one PNG per PC in 'individual PCs'."""
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "plots.log")
    D = H.load_plot_inputs(P.out)
    sections = H.time_sections(D["flare"].index.min(), D["flare"].index.max())
    spans = H.context_spans(D)
    peaks = H.label_t2_peaks(D, sections)
    peaks.to_csv(d["tables"] / "t2_peaks.csv", index=False)
    H.plot_flare_overview(D, spans, sections, d["figures"] / "fig1_flare.png")
    H.plot_t2_overview(D, spans, sections, peaks, d["figures"] / "fig2_t2_overall.png")
    H.plot_event_zooms(D, d["figures"] / "fig3_event_zooms.png")
    H.plot_pc_terms(D, peaks, spans, d["figures"] / "fig4_pc_terms.png")
    H.plot_causal_graph(D, d["figures"] / "fig5_causal_graph.png")
    H.plot_individual_pcs(D, peaks, spans, sections, d["figures"] / "individual PCs")
    H.log("figures finished")


if __name__ == "__main__":
    main()
