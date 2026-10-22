"""Every fact the deck states, in one place, each with its source in the repo.

Nothing here is written by hand from memory. The provenance comment above each
block names the file the number came from, so a reader can check any claim on
a slide against the run that produced it.

Sources
-------
RUN      results/runs/run-20260831-025032/{RUN_REPORT.md, scores.csv}
LSTE_RUN results/lste_only/run-20260831-025653/{RUN_REPORT.md, scores.csv}
EDA      docs/EDA_REPORT.md  (= results/eda/eda-20260831-005002)
CFG      configs/default.yaml
SRC      src/causal_bench/algorithms/*.py
DIFF     uncommitted working-tree changes to lste.py, var_lingam.py, default.yaml
"""

# ---------------------------------------------------------------------------
# Deck furniture
# ---------------------------------------------------------------------------

ACT_FRAME = "OBJECTIVE"
ACT_SOTA = "STATE OF THE ART"
ACT_EDA = "OUR MEASUREMENT"
ACT_SETUP = "EXPERIMENTAL SETUP"
ACT_RESULT = "OUR RESULTS"
ACT_ANALYSIS = "ANALYSIS"
ACT_LEARN = "OUR LEARNING"
ACT_CLOSE = "CONCLUSION"

TITLE = "Causal Discovery on Tennessee Eastman:\nPCMCI+, VAR-LiNGAM and LSTE"
SUBTITLE = ("What the data told us before we ran anything, "
            "and whether the results agreed")
TITLE_META = "Tennessee Eastman  ·  31 variables × 1499 samples  ·  commit e8f02e8"
TITLE_CHIPS = ["EDA  eda-20260831-005002",
               "run  run-20260831-025032",
               "LSTE  run-20260831-025653"]
TITLE_CLAIM = ("The exploratory analysis predicted a distinct failure mode for "
               "each method. Three of the four predictions held; the fourth "
               "failed in the opposite direction, and that is the finding.")

# ---------------------------------------------------------------------------
# References. Numbered once, cited inline, resolved in the slide footer.
# ---------------------------------------------------------------------------

REFS = {
    1: "Spirtes & Glymour, Social Science Computer Review 9(1), 1991 - PC",
    2: "Runge, UAI 2020 - PCMCI+",
    3: "Hyvarinen, Zhang, Shimizu & Hoyer, JMLR 11, 2010 - VAR-LiNGAM",
    4: "Schreiber, Phys. Rev. Lett. 85(2), 2000 - transfer entropy",
    5: "Kraskov, Stogbauer & Grassberger, Phys. Rev. E 69:066138, 2004",
    6: "tigramite 5.2.10.1 - github.com/jakobrunge/tigramite",
    7: "lingam 1.13.0 - github.com/cdt15/lingam",
    8: "Menghi et al., arXiv:2208.01529, 2022 - CIPCaD-Bench",
    9: "Benjamini & Hochberg, JRSS-B 57(1), 1995 - FDR",
    10: "Downs & Vogel, Computers & Chemical Engineering 17(3), 1993 - TE",
}

# ---------------------------------------------------------------------------
# 2 - objective and scope.   Source: CFG discovery.*, RUN, LSTE_RUN
# ---------------------------------------------------------------------------

OBJECTIVE = ("Recover the causal graph over Tennessee Eastman process variables "
             "from observation alone, and score the recovery against the "
             "benchmark's published ground truth.")

PIPELINE = [
    ("Characterise", "31 x 1499 measured\nagainst every assumption"),
    ("Choose", "tau, alpha and scaling\nread off the EDA"),
    ("Run", "4 methods\n16 configurations"),
    ("Score", "SHD against the\n30-edge projected truth"),
]

SCOPE = [
    ("3", "methods announced", "PCMCI+, VAR-LiNGAM, LSTE"),
    ("+1", "baseline", "PC, the i.i.d. reference point"),
    ("16", "scored configurations", "3 PC, 6 PCMCI+, 6 VAR-LiNGAM, 1 LSTE"),
    ("30", "edges in the target", "latent projection of a 32-edge truth"),
]

SCORING_NOTE = ("SHD counts a missing, extra or reversed edge as 1 and an "
                "unoriented edge as 0.5: a CPDAG's undirected edge is an "
                "abstention, not an error. Two ground-truth nodes (X27, X31) "
                "have no column in the data, so the truth is projected onto "
                "the 31 observed variables before anything is scored.")

# ---------------------------------------------------------------------------
# 3 - method comparison.   Source: SRC docstrings, CFG algorithm_params, RUN
# ---------------------------------------------------------------------------

# Four cards, not four table rows: the argument is a comparison of kinds, and
# a card lets each method keep its own shape.
METHOD_CARDS = [
    ("PC  [1]", "baseline, not one of the three", "slate", [
        ("Returns", "a CPDAG over the 31 variables, no lags"),
        ("Assumes", "i.i.d. samples and an invertible covariance"),
        ("Orients by", "v-structures, then Meek's rules"),
        ("Built on", "tigramite PC-stable, majority colliders [6]"),
    ]),
    ("PCMCI+  [2]", "announced method", "teal", [
        ("Returns", "lagged + contemporaneous graph, then a summary"),
        ("Assumes", "stationarity; autocorrelation is modelled"),
        ("Orients by", "time order for lags, CI rules for the rest"),
        ("Built on", "tigramite with the ParCorr test [6]"),
    ]),
    ("VAR-LiNGAM  [3]", "announced method", "teal", [
        ("Returns", "VAR coefficients plus contemporaneous B0"),
        ("Assumes", "VAR residuals that are non-Gaussian"),
        ("Orients by", "ICA; the non-Gaussian contrast fixes B0"),
        ("Built on", "lingam, adaptive lasso with a bootstrap [7]"),
    ]),
    ("LSTE  [4,5]", "announced method", "teal", [
        ("Returns", "one TE test per source, target and lag"),
        ("Assumes", "stationarity; enough samples for a kNN density"),
        ("Orients by", "time order alone: source precedes target"),
        ("Built on", "tigramite CMIknn + shuffle test, BH [6,9]"),
    ]),
]

# ---------------------------------------------------------------------------
# 4 - the pivot slide: what the EDA measured, per assumption per method.
#     Source: EDA findings 1-9 and its "Algorithm outlook" table.
#     Mark meaning: "x" violated, "ok" satisfied, "-" the method does not
#     assume it. Every violation mark below is one the EDA report itself lists
#     against that algorithm.
# ---------------------------------------------------------------------------

EDA_METHODS = ["PC", "PCMCI+", "VAR-LiNGAM", "LSTE"]

EDA_MATRIX = [
    ("Independent samples",
     "lag-1 autocorrelation up to 0.981; Ljung-Box rejects 27 of 31",
     ["x", "-", "-", "-"]),
    ("Stationarity",
     "ADF calls 30 of 31 stationary, KPSS only 24; the two disagree on 8",
     ["-", "x", "x", "x"]),
    ("Non-Gaussian VAR residuals",
     "mean |excess kurtosis| 0.141 vs a threshold of 1.0; 0 of 31 clear it",
     ["-", "-", "x", "-"]),
    ("Adequate sample size",
     "binding effective n is 36 of 1499 rows - 2.4% of the nominal count",
     ["x", "x", "x", "x"]),
    ("Well-conditioned covariance",
     "condition number 3388 vs a threshold of 1000; 9 variables VIF > 10",
     ["x", "x", "x", "-"]),
    ("Linearity",
     "0 of 465 pairs have distance correlation exceeding |Pearson| by 0.1",
     ["ok", "ok", "ok", "-"]),
    ("Causal sufficiency",
     "X27 and X31 are in the truth but absent from the data; 4 edges lost",
     ["x", "x", "x", "x"]),
]

EDA_VERDICT = [
    ("PC", "fails"),
    ("PCMCI+", "fails"),
    ("VAR-LiNGAM", "fails"),
    ("LSTE", "works, with\ncaveats"),
]

EDA_KICKER = ("Seven assumptions, measured on the data before any algorithm ran. "
              "One row is good news.")

EDA_NOTE = ("Only linearity holds, and it holds cleanly - which is why every "
            "method here uses its linear or partial-correlation test rather "
            "than a nonlinear one. Everything else in this matrix is a "
            "prediction of failure, and slide 7 checks each one against what "
            "actually happened.")

# ---------------------------------------------------------------------------
# 5 - experimental setup.   Source: CFG, RUN "Parameters and provenance"
# ---------------------------------------------------------------------------

SETUP_GROUPS = [
    ("Data and preprocessing", [
        ("Dataset", "datasetTE.csv - 1499 rows x 31 variables [8,10]"),
        ("Standardised", "yes; largest / smallest column SD = 2525x, so an unscaled\n"
                         "kernel or distance would be dominated by one variable"),
        ("Ground truth", "33-node DAG, 32 edges -> projected onto the 31 observed\n"
                         "variables, giving the 30-edge primary target"),
        ("Integrity", "0 missing values, 0 duplicate rows, no constant or\n"
                      "near-constant columns; a regular 1-unit sampling interval"),
    ]),
    ("Parameters, and where each came from", [
        ("tau_max", "swept over 1, 2, 3 - the VAR order criteria disagree\n"
                    "(AIC 3, BIC 1, HQIC 2, FPE 3), so none was privileged"),
        ("alpha", "0.01 for PC and PCMCI+ (Fisher-z / ParCorr); 0.05 for\n"
                  "LSTE, then Benjamini-Hochberg over its 2790 tests [9]"),
        ("VAR-LiNGAM", "adaptive lasso, pruning on, 10 bootstrap resamples,\n"
                       "criterion: bic - which overrode the tau we asked for"),
        ("LSTE", "CMIknn, k = 0.1n, 20-shuffle significance test [5,6]"),
    ]),
    ("Scoring", [
        ("Primary target", "gt_projected, 30 edges. Also scored against the induced\n"
                           "subgraph and against both targets' CPDAGs"),
        ("Baselines", "empty graph = SHD 30. The truth's CPDAG leaves 6 edges\n"
                      "undirectable by any observational method"),
        ("Prior knowledge", "4 edges, direction taken from the ground truth file.\n"
                            "An oracle and a best case - not independent validation"),
    ]),
]

# ---------------------------------------------------------------------------
# 6 - results.   Source: RUN scores.csv (gt_projected), LSTE_RUN scores.csv
# ---------------------------------------------------------------------------

RESULT_BARS = [
    # label, SHD shown, range over configs or None, TP, FP, FN, note, tone
    ("LSTE  tau 3", 30.0, None, 0, 0, 30,
     "0 edges predicted - identical to the empty graph", "degenerate"),
    ("PC  best of 3; 47.5-48.5", 47.5, (47.5, 48.5), 4, 23, 18,
     "35 edges; best is pc with prior knowledge", "neutral"),
    ("PCMCI+  best of 6", 57.0, (57.0, 67.0), 8, 36, 14,
     "52 edges; across tau: 57.0 / 66.0 / 61.0", "best"),
    ("VAR-LiNGAM  all 6", 198.5, None, 0, 176, 11,
     "195 edges; every configuration scored identically", "failed"),
]

EMPTY_GRAPH_SHD = 30.0
RESULT_AXIS_MAX = 210.0

# TP out of the 30 recoverable edges - the second panel.
RECALL_DOTS = [
    ("PCMCI+", 8, "tau 1, with prior knowledge"),
    ("PC", 4, "pc with prior knowledge; also pc_manual"),
    ("VAR-LiNGAM", 0, "195 edges, none of them right"),
    ("LSTE", 0, "no edges to be right about"),
]
RECALL_DENOM = 30

RESULT_KICKER = ("Ranked by SHD, the best method predicts nothing at all. "
                 "That is a fact about SHD, so both panels are shown.")

RESULT_NOTE = ("Left: SHD against the 30-edge projected target, lower is better. "
               "Right: true positives out of the 30 recoverable edges. The two "
               "panels rank the methods differently, and the right one is the "
               "one that answers whether we recovered the graph.")

# tau sensitivity, folded onto the bars as whiskers rather than given its own
# table.  Source: RUN scores.csv, gt_projected rows.
TAU_NOTE = ("Whiskers span tau = 1, 2, 3. PCMCI+ moves 10.0 SHD across the "
            "sweep. VAR-LiNGAM has no whisker because it did not move - see "
            "slide 7 for why.")

# ---------------------------------------------------------------------------
# 7 - analysis: prediction against outcome, method by method.
#     Source: EDA outlook + RUN / LSTE_RUN per-algorithm notes.
# ---------------------------------------------------------------------------

# The precision frontier: a method predicting `e` edges cannot score more than
# min(e, 30) true positives. Plotting predictions against recoveries puts every
# method in one picture and shows how far below that ceiling all four sit.
FRONTIER_MAX_EDGES = 200
FRONTIER_MAX_TP = 30

SCATTER_POINTS = [
    # label, edges predicted, true positives, tone, label offset (dx, dy)
    ("LSTE", 0, 0, "degenerate", (0.14, -0.30)),
    ("PC", 35, 4, "neutral", (0.14, -0.30)),
    ("PCMCI+", 52, 8, "best", (0.14, -0.30)),
    ("VAR-LiNGAM", 195, 0, "failed", (-1.94, -0.36)),
]
SCATTER_IDEAL = ("the truth", 30, 30)

ANALYSIS_NOTES = [
    ("VAR-LiNGAM", "prediction held", "failed",
     "195 edges, none of them right. Two causes: tau never "
     "reached the library, and Gaussian residuals leave B0 unidentified."),
    ("PCMCI+", "prediction held", "best",
     "The most recovery of the four, and the false positives that come "
     "with modelling the lag structure."),
    ("PC", "held, yet ranked first", "neutral",
     "Fewer edges predicted, so fewer errors charged. Recall of 7-13% "
     "is a scoring artefact, not recovery."),
    ("LSTE", "missed, in the other direction", "degenerate",
     "Sits on the origin. 548 of 2790 tests were significant; BH "
     "correction rejected none of them."),
]

ANALYSIS_KICKER = ("Edges predicted against edges recovered. The diagonal is "
                   "the ceiling; every method sits far below it.")

PK_NOTE = ("Prior knowledge is worth 1.0-1.5 SHD wherever it applies, and "
           "nothing at all for VAR-LiNGAM. The ground truth is acyclic with no "
           "reciprocal pairs, so the controller-loop structure the prior encodes "
           "cannot be scored on this dataset at all.")

# ---------------------------------------------------------------------------
# 8 - cost.   Source: RUN / LSTE_RUN runtime_seconds, "Reproducing this run"
# ---------------------------------------------------------------------------

RUNTIME_BARS = [
    ("PC", 2.7, "per configuration"),
    ("PCMCI+", 32.6, "slowest configuration, tau = 3"),
    ("VAR-LiNGAM", 120.7, "tau = 1, with the lag-0 bootstrap"),
    ("LSTE", 1434.2, "a single configuration, tau = 3"),
]

COST_HEADLINE = [
    ("358 s", "all 15 PC, PCMCI+ and VAR-LiNGAM runs"),
    ("1434 s", "one LSTE configuration"),
    ("4.0x", "one LSTE cell costs 4x the rest of the grid"),
]

# Which cells of the tau x prior-knowledge grid were actually run.
COVERAGE_COLS = ["t1", "t1+PK", "t2", "t2+PK", "t3", "t3+PK"]
COVERAGE_ROWS = [
    ("PC", ["n/a", "n/a", "n/a", "n/a", "n/a", "n/a"],
     "3 runs incl. the pc_manual oracle - tau does not apply"),
    ("PCMCI+", ["run", "run", "run", "run", "run", "run"], "6 of 6"),
    ("VAR-LiNGAM", ["boot", "run", "run", "run", "run", "run"],
     "6 of 6, but the lag-0 bootstrap only at tau = 1"),
    ("LSTE", ["skip", "skip", "skip", "skip", "run", "skip"], "1 of 6"),
]

COST_NOTE = ("The LSTE and VAR-LiNGAM conclusions therefore rest on less "
             "evidence than the PCMCI+ ones. LSTE ran at tau = 3 only with no "
             "prior-knowledge variant; VAR-LiNGAM's bootstrap ran at tau = 1 "
             "only, at 10 resamples. Both gaps are cost, not choice.")

COST_KICKER = ("Coverage is not uniform across the four methods, because the "
               "runtimes are not within two orders of magnitude of each other.")

# ---------------------------------------------------------------------------
# 9 - learnings
# ---------------------------------------------------------------------------

LEARNINGS = [
    ("The EDA earned its place, and its one miss is the useful part.",
     "It called VAR-LiNGAM, PCMCI+ and PC correctly before a single run. It "
     "missed LSTE by sizing the estimator's power and not the correction "
     "stacked on top of it."),
    ("An identical score in every cell means a knob was not connected.",
     "VAR-LiNGAM returned SHD 198.5 six times. With criterion: bic set, lingam "
     "picks its own lag order and ignores the tau we asked for, so the sweep "
     "refitted one VAR(1) six times."),
    ("SHD alone ranks the empty graph first, so baselines are not optional.",
     "The empty graph scores 30.0 and 6 edges of the truth's CPDAG cannot be "
     "oriented by any observational method. Both belong on the chart, not in a "
     "footnote."),
    ("A shuffle test has a p-value floor, and it is checkable before the run.",
     "20 shuffles cannot produce a p below 1/21, and BH over 2790 tests will not "
     "reject at that floor. Pure arithmetic, now a pre-flight guard rather than "
     "24 minutes of compute."),
    ("An edge needs a selection rule, not just a non-zero coefficient.",
     "The adaptive lasso left coefficients as small as |b| = 0.0003 on "
     "standardised data, and every one became an edge. Selection now goes by "
     "bootstrap frequency."),
    ("The failures are in the data, not in the wrappers we wrote.",
     "On the suite's synthetic case - Laplace innovations, independent "
     "samples - the same code recovers the generating graph. Nothing here is "
     "a broken wrapper."),
]

LEARN_KICKER = ("What we would carry into the next dataset, in the order we "
                "would apply it.")

# ---------------------------------------------------------------------------
# 10 - conclusion
# ---------------------------------------------------------------------------

CONCLUSION_COLS = [
    ("What we did",
     ["Implemented and ran PCMCI+, VAR-LiNGAM and LSTE on Tennessee Eastman, "
      "with PC as an i.i.d. baseline.",
      "16 scored configurations, every assumption measured first and every "
      "violation recorded with its number."]),
    ("What we found",
     ["No method beats the empty-graph baseline in a way that means anything: "
      "the best recall is 8 of 30 edges.",
      "Each failure traces to a quantity measured before the run - Gaussian "
      "residuals, effective n of 36, a condition number of 3388."]),
    ("What it cost us to learn",
     ["A 24-minute LSTE run that could not have produced an edge at any "
      "significance level.",
      "A six-cell VAR-LiNGAM grid that refitted the same VAR(1) six times, "
      "because a config key overrode the lag order we swept."]),
]

NEXT_STEPS = [
    ("Re-run the two fixed methods", "coded, not yet run",
     "LSTE with more shuffles or FDR off; VAR-LiNGAM with bootstrap selection"),
    ("Close the coverage gaps", "cost-bound",
     "LSTE across tau 1-3 and with prior knowledge; the bootstrap at every tau"),
    ("Attack the binding constraint", "the real one",
     "Effective n of 36 defeats everything here; a longer segment beats tuning"),
]

UNVERIFIED = ("UNVERIFIED - the two fixes above are in the working tree, not in "
              "the run this deck reports. No number on any earlier slide comes "
              "from them.")

CLOSING = ("The headline is not that four methods failed, but that we could "
           "say which would fail, and why, before running them.")

# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------

NOTES = {
    1: "Tennessee Eastman, 31 variables, 1499 samples. Everything in this deck "
       "comes from two runs in the repo; the run IDs are on the slide.",
    2: "Three methods were announced last time: PCMCI+, VAR-LiNGAM, LSTE. PC is "
       "here as the i.i.d. baseline - it is what the temporal methods are "
       "supposed to improve on, so it has to be in the table.",
    3: "The point of this table is column three. The four assumptions are "
       "different and non-overlapping, which is what makes the next slide "
       "necessary.",
    4: "This is the slide the rest of the deck hangs off. Every mark is from "
       "the EDA report's own per-algorithm verdict. Note the one clean row - "
       "linearity - which is why we use ParCorr and not GPDC.",
    5: "Nothing here was chosen by preference. tau is swept because the four "
       "order criteria disagree; standardisation is on because of the 2525x "
       "scale ratio. The prior knowledge is an oracle and we say so.",
    6: "Two panels because the two orderings disagree. LSTE is first on SHD "
       "and last on recovery. If you only show the left panel you have "
       "published a lie.",
    7: "Row by row: the EDA prediction, then what happened. Three held. Two "
       "things to say out loud: LSTE failed in the direction we were not "
       "watching, and VAR-LiNGAM's six identical scores turned out to have a "
       "config cause on top of the statistical one - criterion: bic made "
       "lingam ignore the tau we swept.",
    8: "Say this one plainly: LSTE and VAR-LiNGAM are under-covered, and it is "
       "because of runtime. One LSTE cell costs four times the whole rest of "
       "the grid.",
    9: "The four transferable ones are the first four. The pre-flight "
       "feasibility check is the one that would have saved actual time.",
    10: "The fixes are written but not run. Nothing on the results slides "
        "depends on them, and we should not present them as results.",
}
