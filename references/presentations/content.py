"""Every number and string used in the deck, in one place.

Values come from the analysis reported in docs/legacy/Results.txt. Keeping them
here means a re-run of the analysis needs edits in exactly one file, and no
figure can drift between slides.
"""

# ---------------------------------------------------------------------------
# Acts — the section name shown in each slide's header rail
# ---------------------------------------------------------------------------

ACT_FRAME = "Objective"
ACT_METHOD = "The PC algorithm"
ACT_DATA = "Dataset"
ACT_BUILD = "Building the graph"
ACT_RESULT = "Results"
ACT_CLOSE = "Outlook"

# ---------------------------------------------------------------------------
# Slide 1
# ---------------------------------------------------------------------------

TITLE = "Causal Discovery for Root-Cause Detection"
SUBTITLE = "The PC algorithm applied to the Tennessee Eastman process"
TITLE_CHIPS = ["31 variables", "1,499 samples", "4,066 independence tests"]

# ---------------------------------------------------------------------------
# Slide 2 / 20 — the pipeline
# ---------------------------------------------------------------------------

PIPELINE = [
    ("Process data", "sensor time series"),
    ("Causal graph", "which variable drives which"),
    ("Fault detected", "T² process monitoring"),
    ("Trace backwards", "walk the graph upstream"),
    ("Root cause", "the originating variable"),
]
PIPELINE_DELIVERS = 1          # index of the stage this work delivers
PIPELINE_NEXT = [2, 3, 4]      # stages still ahead

# ---------------------------------------------------------------------------
# Slides 3 / 4 — the three structures
# ---------------------------------------------------------------------------

STRUCTURES = [
    {
        "name": "Chain",
        "form": "X → Y → Z",
        "marginal": ("X ⫫ Z", "false"),
        "conditional": ("X ⫫ Z | Y", "true"),
    },
    {
        "name": "Fork",
        "form": "X ← Y → Z",
        "marginal": ("X ⫫ Z", "false"),
        "conditional": ("X ⫫ Z | Y", "true"),
    },
    {
        "name": "Collider",
        "form": "X → Z ← Y",
        "marginal": ("X ⫫ Y", "true"),
        "conditional": ("X ⫫ Y | Z", "false"),
    },
]

CORRELATION_MATRIX = [
    ["", "X", "Y", "Z"],
    ["X", "1.00", "0.71", "0.50"],
    ["Y", "0.71", "1.00", "0.71"],
    ["Z", "0.50", "0.71", "1.00"],
]

# ---------------------------------------------------------------------------
# Slide 5 — the four frames
# ---------------------------------------------------------------------------

FRAMES = [
    ("Complete graph", "every pair adjacent", "10 edges"),
    ("Skeleton", "remove when separated", "5 edges"),
    ("Colliders", "orient unshielded triples", "2 arrows"),
    ("Propagate", "follow the consequences", "4 arrows"),
]

# ---------------------------------------------------------------------------
# Slide 8 — orientation rules
# ---------------------------------------------------------------------------

ORIENTATION_RULES = [
    ("Collider rule",
     "a — b — c with a, c non-adjacent, and b not in the set that separated them",
     "b is a common effect, so both arrows point into it"),
    ("No new collider",
     "a → b — c with a, c non-adjacent",
     "pointing c → b would create a collider we already ruled out"),
    ("No cycle",
     "a → c → b with a — b",
     "pointing b → a would close a directed cycle"),
    ("Two parents",
     "a — b, a — c, a — d with c → b, d → b, and c, d non-adjacent",
     "every remaining option for a — b creates a collider or a cycle"),
]

# ---------------------------------------------------------------------------
# Slides 9 / 10 — the dataset
# ---------------------------------------------------------------------------

PROCESS_UNITS = ["Reactor", "Condenser", "Separator", "Stripper"]
PROCESS_FEEDS = "Feed streams"
PROCESS_PRODUCT = "Product"
PROCESS_RECYCLE = "Recycle (compressor)"

DATA_STATS = [
    ("1,499", "time-ordered samples"),
    ("31", "process variables"),
    ("28", "reference relationships"),
]

PROCESS_NOTE = (
    "The Tennessee Eastman simulation is the standard benchmark for process "
    "monitoring: a real plant topology with published reference relationships. "
    "The variable-to-sensor mapping is not part of the data we hold, so the "
    "recovered graph is assessed structurally rather than interpreted per sensor."
)

PREPARATION = [
    ("31 process variables retained",
     "the file's row-counter column is an index, not a process measurement"),
    ("Nothing imputed or discarded",
     "no missing values and no constant variables in the 1,499 samples"),
    ("No rescaling applied",
     "variable scales span a factor of 2,525, and partial correlation is "
     "scale-invariant — a property of the statistic, not an oversight"),
]

PREPARATION_REFERENCE = (
    "The reference graph covers 33 variables; two of them are absent from the "
    "data, so 28 of its 32 relationships are recoverable. Recall is scored "
    "against 28, not 32."
)

# ---------------------------------------------------------------------------
# Slide 11 — parameters
# ---------------------------------------------------------------------------

PARAMETERS = [
    {
        "name": "Significance level  α",
        "chosen": "0.01",
        "ticks": ["0.05", "0.01", "10⁻³", "10⁻⁵", "10⁻¹⁰"],
        "pin": 1,
        "note": "A stricter threshold demands stronger evidence before keeping "
                "an edge, giving a sparser graph.",
    },
    {
        "name": "Largest conditioning set",
        "chosen": "2",
        "ticks": ["0", "1", "2", "3", "29"],
        "pin": 2,
        "note": "Not a compromise here: the graph stops changing at 2, so "
                "searching deeper would cost time and find nothing.",
    },
]

# ---------------------------------------------------------------------------
# Slide 12 — the skeleton search
# ---------------------------------------------------------------------------

FUNNEL = [
    ("All candidate pairs", 465, None),
    ("Conditioning on nothing", 217, 682),
    ("Conditioning on one variable", 54, 2728),
    ("Conditioning on two variables", 35, 656),
]
FUNNEL_TOTAL_TESTS = 4066
FUNNEL_CAPTION = (
    "Three quarters of the surviving pairs are separated by a single "
    "conditioning variable — that is the algorithm's central bet paying off, "
    "and it is where most of the 4,066 tests are spent."
)

# ---------------------------------------------------------------------------
# Slide 13 — orientation outcome
# ---------------------------------------------------------------------------

ORIENTATION_SPLIT = [
    ("Directed", 20, "the data determines which way the influence runs"),
    ("Undirected", 3, "no collider touches the edge, so either direction fits"),
    ("Unresolved", 12, "colliders point both ways, so no direction is claimed"),
]
ORIENTATION_CAPTION = (
    "An undirected edge is a result, not a gap: it states that these data "
    "cannot distinguish the two directions."
)

# ---------------------------------------------------------------------------
# Slides 14 / 15 — the graph and its assessment
# ---------------------------------------------------------------------------

GRAPH_COUNTS = [
    ("35", "relationships found"),
    ("11", "confirmed by the reference graph"),
    ("24", "additional candidates"),
    ("20", "with a direction established"),
]

METRICS = [
    ("Precision", "0.31", "of the relationships we report, this share is in the reference graph"),
    ("Recall", "0.39", "of the 28 recoverable reference relationships, this share was found"),
    ("F1", "0.35", "the balance of the two"),
    ("SHD", "48", "structural edits between our graph and the reference"),
]

METRICS_READING = (
    "The reference graph records the plant's designed relationships. Edges we "
    "report that it does not contain are therefore candidates for undocumented "
    "coupling, not automatically mistakes — which is why the additional 24 are "
    "shown in their own colour rather than as errors."
)

# ---------------------------------------------------------------------------
# Slide 16 — sensitivity
# ---------------------------------------------------------------------------

SENSITIVITY = [
    {
        "title": "Significance level",
        "x_labels": ["0.05", "0.01", "10⁻³", "10⁻⁵", "10⁻¹⁰"],
        "values": [0.310, 0.349, 0.367, 0.364, 0.340],
        "verdict": "flat — the answer barely moves",
    },
    {
        "title": "Conditioning set size",
        "x_labels": ["0", "1", "2", "3", "all"],
        "values": [0.155, 0.341, 0.349, 0.349, 0.349],
        "verdict": "converged at 2 — deeper changes nothing",
    },
    {
        "title": "Using every nth sample",
        "x_labels": ["1", "5", "20", "50"],
        "values": [0.349, 0.377, 0.267, 0.350],
        "verdict": "sensitive — and slide 17 explains why",
    },
]
SENSITIVITY_AXIS = (0.10, 0.40)
SENSITIVITY_CAPTION = (
    "Thinning to every 5th sample raises precision from 0.31 to 0.40. Beyond "
    "that, too few samples remain for the tests to resolve anything."
)

# ---------------------------------------------------------------------------
# Slide 17 — the independence assumption
# ---------------------------------------------------------------------------

AUTOCORR_HEADLINE = "0.982"
AUTOCORR_BODY = (
    "Consecutive samples are near-copies of one another. The strongest "
    "lag-one autocorrelation across the 31 variables is 0.982; the median "
    "is 0.571."
)
AUTOCORR_CONSEQUENCE = (
    "So the effective sample size is a fraction of 1,499. The independence "
    "tests are more confident than the evidence warrants, the graph is denser "
    "than it should be — and that is precisely why using fewer, "
    "further-apart samples improved precision."
)

# ---------------------------------------------------------------------------
# Slide 18 — the other two cases
# ---------------------------------------------------------------------------

OTHER_CASES = {
    "header": ["Case", "Variables", "Samples", "Found", "Precision", "Recall", "Converged"],
    "rows": [
        ["Tennessee Eastman", "31", "1,499", "35", "0.31", "0.39", "yes, at 2"],
        ["Ultra-processed food", "17", "23,132", "55", "0.55", "0.36", "no"],
        ["…with machine dependencies", "17", "23,132", "55", "0.80", "0.41", "no"],
    ],
    "highlight_row": 0,
    "note": "The third case's reference graph contains feedback loops. PC "
            "assumes an acyclic structure, so those relationships cannot be "
            "recovered by construction and its recall is capped from the start.",
}

# ---------------------------------------------------------------------------
# Slide 19 — claims
# ---------------------------------------------------------------------------

SUPPORTED = [
    "A converged, stable skeleton for the Tennessee Eastman process.",
    "11 of 28 documented relationships recovered from observation alone.",
    "A direction established for 20 of the 35 relationships found.",
    "The same procedure transfers unchanged to two further datasets.",
]

NOT_SUPPORTED = [
    ("That the 24 additional relationships are real.",
     "an independence test that accounts for time ordering"),
    ("Any claim about specific sensors or process units.",
     "the variable-to-sensor mapping for the benchmark"),
    ("Relationships involving the two absent variables.",
     "a method that admits unmeasured confounders"),
    ("Feedback loops of any kind.",
     "a method that does not assume acyclicity"),
]

# ---------------------------------------------------------------------------
# Slide 20 — next
# ---------------------------------------------------------------------------

NEXT_STEPS = [
    ("Time-aware independence testing", "addresses the assumption above"),
    ("T² fault detection on the recovered graph", "turns structure into monitoring"),
    ("Backward traversal to a named root cause", "the objective we started from"),
]

# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------

NOTES = {
    1: "30-minute slot. The deck walks one dataset end to end: preparation, "
       "graph construction, the graph, its assessment, next steps.",
    2: "Frame the work: the causal graph is not the goal, locating a fault's "
       "origin is. We deliver the graph stage.",
    3: "Key point: all three structures imply the same correlation matrix. "
       "Correlation cannot distinguish them, so it cannot give direction.",
    4: "The collider is the asymmetry that makes causal discovery possible. "
       "Conditioning on a common effect creates dependence rather than "
       "removing it. Everything PC orients traces back to this.",
    5: "Two phases. Phase 1 removes edges; phase 2 orients what remains. "
       "Note the edge count stops falling after phase 1.",
    6: "Say: under joint normality, conditional independence is exactly a zero "
       "partial correlation. The recursion peels one variable off the "
       "conditioning set at a time.",
    7: "α = 0.01 means we keep an edge unless the evidence for independence is "
       "strong. Flag the independence assumption here — slide 17 returns to it.",
    8: "Four rules, all forced. Each avoids either inventing a collider we "
       "ruled out or creating a cycle. What stays undirected is genuinely "
       "undetermined by the data.",
    9: "If asked which variable is which: the sensor mapping is not in the "
       "dataset we hold. That is listed as a next step on slide 19.",
    10: "Emphasise the third point — no rescaling is needed because partial "
        "correlation is scale-invariant. It is a consequence of the maths, "
        "not an omission.",
    11: "The second parameter is the interesting one: 2 is not a budget cut, "
        "the answer genuinely stops changing there. Slide 16 shows it.",
    12: "465 candidate pairs down to 35. Most of the work, and most of the "
        "4,066 tests, happen at the one-variable stage.",
    13: "12 unresolved edges are an honest outcome: colliders imply arrows in "
        "both directions, so we claim neither.",
    14: "The deliverable. Teal edges are confirmed by the reference graph, "
        "blue are additional candidates. Four natural clusters.",
    15: "Read precision carefully: the reference graph is the plant's designed "
        "topology, so extra edges are candidates for undocumented coupling. "
        "If earlier figures for this dataset were reported differently, they "
        "reflected a reference-alignment problem since resolved; these are the "
        "corrected numbers.",
    16: "Left: α hardly matters. Middle: converged at 2 — a strength. Right: "
        "sample spacing matters a lot, which slide 17 explains.",
    17: "The headline limitation. 1,499 samples are not 1,499 independent "
        "observations. This is the single biggest thing to fix next.",
    18: "The method is not tuned to one dataset. Third row scores highest, but "
        "its reference graph has feedback loops PC cannot express.",
    19: "Give the right-hand column equal time. Each gap has a concrete "
        "remedy — this is the research agenda.",
    20: "Close on the objective: the graph exists, so the root-cause traversal "
        "is now buildable.",
}
