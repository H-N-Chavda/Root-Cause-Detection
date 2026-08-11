# Stage 2 — Presentation Plan (20 slides, 30-minute meeting)

**Deliverable being planned:** `presentations/build_presentation.py` → `presentations/PC_Algorithm.pptx`
**Status:** plan only. No `.pptx` and no build code written yet.

**Revision 3.** Two instructions applied:

1. **No implementation content.** Nothing about how the logic was written, no code, no file or function
   names, no test counts. The deck demonstrates: we understand the algorithm, we understand the
   mathematics, we applied it to a dataset, here are the results.
2. **No before/after, no audit narrative.** The earlier errors were a colleague's and were corrected;
   the deck presents the method as it now stands, without pointing at anyone.

**One running example throughout** — Tennessee Eastman — carried end to end:
`data preparation → graph creation → the graph itself → what it means → further steps`.

§7 records what was dropped from revision 2 and the two trade-offs I want you to be aware of.

---

## 1. The governing design rule

> **No slide is "title + bulleted list."** Each slide's layout is chosen from the shape of its own
> argument. §8 audits the outcome: **20 distinct devices across 20 slides**, one deliberate callback.

Slides 2, 3, 14 and 17 stay deliberately sparse. With no pacing slides left, those four are the
deck's only breathing room.

---

## 2. Why Tennessee Eastman is the running example

I tested the alternative before choosing, because "one example throughout" makes this the single most
consequential decision in the plan.

| | **Tennessee Eastman (chosen)** | Ultra-Processed Food |
|---|---|---|
| Variables / edges found | 31 / 35 | 17 / 55 |
| Average degree | **2.26** | 6.47 |
| Layout legibility | clean — 4 natural clusters, max degree 5 | hairball — 4× the edge crossings |
| Converged? | **yes, at conditioning size 2** — deeper search changes nothing | no — still shedding edges at the limit |
| Recognisable? | **the canonical process-control benchmark** | in-house dataset |
| Reference graph | sparse, properly acyclic | contains feedback loops PC cannot express |
| Agreement with reference | 11 of 28 found (modest) | 44 of 108 (precision 0.80) |

The first four rows decide it. You asked for a graph that is understandable at a glance, and TE is the
only one of the two that draws legibly. It is also the scientifically safer spine: **its answer is
converged and stable**, whereas UF's is still changing when the search stops — so any UF number needs
a caveat that TE's does not.

The cost is row 7: TE's agreement with its reference graph is modest. **UF's stronger precision (0.80)
still gets shown**, on slide 18, as corroboration from two further cases. See §7 for the honest read
on this trade-off.

---

## 3. Toolkit: pure `python-pptx`

Recommendation unchanged: `python-pptx` alone, no matplotlib. Output stays **vector and editable in
PowerPoint** so you can fix a typo or nudge a box yourself, and Unicode formulas let me colour-tint
individual terms to match the nodes in the adjacent diagram — which teaches better than a monochrome
LaTeX image. Unicode covers `ρ σ √ ∑ ⊥ ⫫ | ² ₀₁₂₃ ᵢ ⱼ → ← ↔ ⇒`.

**Verified working** before planning around it: native line / scatter / stacked-bar charts, connectors
with real arrowheads, freeform polygons, table cell fill and merge, per-slide speaker notes, Unicode
maths glyphs.

**File layout** — four small modules, plain functions, no classes:

```
presentations/
├── PRESENTATION_PLAN.md
├── theme.py                ← palette, fonts, grid, ~10 drawing helpers
├── content.py              ← every number and string in one place
├── slides.py               ← one function per slide
└── build_presentation.py   ← entry point, saves the .pptx
```

---

## 4. Visual system

**Canvas** 16:9, 13.333″ × 7.5″, margins 0.65″, 12-column grid every slide snaps to.

**Palette** — 5 colours, meaning only:

| Role | Hex | Used for |
|---|---|---|
| Paper | `#FAFAF7` | background |
| Ink | `#1A1D21` | body text |
| Ink-dark | `#14181D` | title slide, and the graph slide's ground |
| **Teal** | `#1A7F72` | edges confirmed by the reference graph; established claims |
| **Slate blue** | `#3D6B99` | edges we found that the reference does not contain |
| **Amber** | `#B8860B` | limitations, caveats, unresolved orientations |
| Rule grey | `#C8C6C0` | hairlines, reference edges we did not find |

Note the deliberate choice: edges not in the reference graph are **slate blue, not red**. They are
candidate findings under a reference that is itself incomplete, not errors — and colouring 24 of 35
edges red would misrepresent the result on the deck's centrepiece slide.

**Type** — `Aptos` with `Calibri` fallback (both ship with PowerPoint on Mac and Windows). `Consolas`
for the few numeric readouts. Title 30pt semibold · statement 40pt light · hero number 88pt ·
subhead 16pt · body 13.5pt · caption 10pt.

**Furniture** — a hairline below the top margin with the section name at 9pt left, slide number right.
Section name colour shifts per act, which is what supplies navigation without divider slides.

---

## 5. Slide-by-slide plan

Each entry: **message** → **device** → **why that device** → content → density.

### Frame

---

**1 · Title** · *light*
- **Message:** this is a causal-discovery study on a benchmark process plant.
- **Device:** full-bleed dark; title left-aligned against a 3pt teal vertical rule; three small stat
  chips lower-right. No centred stack.
- **Why:** asymmetric weight and the rule read as "designed" immediately; the chips pre-load scale
  before a word is spoken.
- **Content:** "Causal Discovery for Root-Cause Detection" / "The PC algorithm applied to the
  Tennessee Eastman process" / chips `31 variables` · `1,499 samples` · `4,066 independence tests`.

**2 · The goal** · *light*
- **Message:** the causal graph is a means; locating the root cause of a fault is the end.
- **Device:** a horizontal five-stage pipeline — process data → **causal graph** → fault detected →
  trace the graph backwards → root cause — with only the graph stage filled teal and tagged
  "what this work delivers."
- **Why:** one glance places the work in its context; a bulleted objectives slide would take longer to
  read and say less. Slide 20 calls this diagram back with the later stages filled.

### The algorithm and its mathematics

---

**3 · Why correlation is not enough** · *light*
- **Message:** three different causal structures produce identical correlations, so correlation alone
  cannot recover causal direction.
- **Device:** three small 3-node graphs across the upper band (chain `X→Y→Z`, fork `X←Y→Z`, collider
  `X→Z←Y`), and beneath them **one shared 3×3 correlation matrix** with a single brace fanning up to
  all three.
- **Why:** the "one matrix, three structures" convergence *is* the argument, and it is spatial — it can
  be seen in a way a sentence about identifiability cannot. First appearance of a motif that slide 4
  reuses.

**4 · Conditional independence: the one asymmetry that gives direction** · *moderate*
- **Message:** conditioning separates chains and forks but *connects* colliders, and that asymmetry is
  the only source of direction in observational data.
- **Device:** the same three diagrams, now in three columns with two independence statements stacked
  beneath each. The **collider column is promoted** — teal tint, larger, with a callout: "the only
  column that yields an arrow."
- **Why:** reusing the identical diagram means zero re-orientation cost; only the annotation is new.
  Promoting one of three columns encodes importance in the layout instead of asserting it.
- **Content:** chain & fork — `X ⫫ Z` false, `X ⫫ Z | Y` **true**. Collider — `X ⫫ Z` **true**,
  `X ⫫ Z | Y` false. Footer: "PC's entire orientation phase rests on the bottom-right cell."

**5 · PC in four frames** · *light*
- **Message:** two phases — thin a complete graph, then orient what the thinning revealed.
- **Device:** a storyboard strip: four panels showing *the same* 5-node graph at successive stages
  (complete K₅ → skeleton → colliders oriented → propagated), with a thin phase rail beneath.
- **Why:** small multiples of one evolving object show a process that transforms a thing. A flowchart
  of boxes would describe the algorithm; this shows it happening.
- **Content:** captions "every pair adjacent" / "remove when separated" / "orient unshielded
  colliders" / "propagate the consequences". Edge counts 10 → 5 → 5 → 5.

**6 · The quantity being tested: partial correlation** · *moderate*
- **Message:** for Gaussian data, conditional independence reduces to a partial correlation of zero.
- **Device:** the recursive reduction formula at 26pt, **terms colour-tinted to match the nodes of a
  small adjacent 3-node diagram**, with the recursion shown as one "peel a variable off the
  conditioning set" step.
- **Why:** colour-linking a symbol to the node it denotes removes the "which variable is which"
  friction that makes formula slides fail. The peel step shows why the definition is recursive rather
  than just asserting it.
- **Content:** `ρ(X,Y | Z∪{z}) = [ρ(X,Y|Z) − ρ(X,z|Z)·ρ(Y,z|Z)] / √[(1−ρ(X,z|Z)²)(1−ρ(Y,z|Z)²)]`.
  One line: `X ⫫ Y | Z` ⟺ `ρ(X,Y|Z) = 0` under joint normality.

**7 · The decision rule: Fisher's z** · *moderate*
- **Message:** each independence claim is a hypothesis test, and α sets how much evidence we demand.
- **Device:** formula stack on the left third; on the right, a plotted null distribution with the
  two-sided rejection region shaded and α marked by a leader line. An amber flag pinned to the sample-
  size term.
- **Why:** α is otherwise an abstract knob — seeing the tail area makes `α = 0.01` mean something. The
  amber flag plants the independence assumption ten slides before slide 17 cashes it in, so that slide
  reads as a payoff rather than a complaint.
- **Content:** `z = ½·√(n − |Z| − 3)·ln((1+ρ̂)/(1−ρ̂))`, `p = 2(1 − Φ(|z|))`; the edge is removed when
  `p ≥ α`. Amber: "assumes n independent observations — we return to this."

**8 · Orientation rules** · *moderate*
- **Message:** four rules take the skeleton to the most-oriented graph the data can justify.
- **Device:** **four rule cards** in a 2×2 grid, each a tiny before → after graph pair with the
  precondition written beneath. No prose.
- **Why:** each rule *is* a small graph transformation, so a picture of the transformation is a
  complete statement of it. Prose would be strictly longer and less precise.
- **Content:** collider rule (`a—b—c`, `a,c` non-adjacent, `b ∉ sep(a,c)` ⇒ `a→b←c`); then the three
  propagation rules — avoid creating a new collider; avoid creating a cycle; and the two-parent case.
  Footer: "what stays undirected genuinely cannot be determined from the data."

### The dataset

---

**9 · The Tennessee Eastman process** · *moderate*
- **Message:** a standard benchmark simulation of a real chemical plant, which is why it is the right
  test case.
- **Device:** a simplified process schematic across the top two thirds — feed streams into a reactor,
  then condenser → separator → stripper, with a recycle loop back to the reactor — and a stat rail
  beneath.
- **Why:** the schematic tells the audience what domain the variables live in and, crucially, *why a
  causal graph should exist at all*: the plant has real material and control flows. A table of counts
  alone would not motivate the exercise.
- **Content:** 1,499 time-ordered samples · 31 process variables · a published reference causal graph.
  Honest note in grey: "the variable-to-sensor mapping is not part of the dataset we hold, so the
  graph is assessed structurally rather than interpreted per sensor."

**10 · Preparing the data** · *light*
- **Message:** three preparation decisions, each following from the mathematics.
- **Device:** a two-panel "as provided → as analysed" comparison, with three short annotations bridging
  them.
- **Why:** preparation is a set of *decisions*, and a before/after of the data shape shows what each
  decision did. This is the one place a compact side-by-side is exactly right.
- **Content:** (1) 31 process variables retained; the file's row-counter column is not a process
  variable. (2) No missing values, no constant variables — nothing to impute or drop. (3) Variable
  scales span a factor of 2,525, and **no rescaling is applied because partial correlation is
  scale-invariant** — a property of the statistic, not an oversight. Plus: the reference graph covers
  33 variables; two are absent from the data, so 28 of its 32 edges are recoverable, and recall is
  scored against 28.

**11 · Parameters, and what each one controls** · *moderate*
- **Message:** two parameters, each chosen for a stated reason, not left at a default.
- **Device:** two horizontal dials — a track per parameter with the chosen value pinned and the
  consequence of moving it written along the track.
- **Why:** a table of two values is inert. Drawing them as ranges with a chosen point communicates that
  each was a decision with a trade-off, and slide 16 later shows both trade-offs measured.
- **Content:** significance level α = 0.01 (track 0.05 → 10⁻¹⁰; "stricter demands stronger evidence,
  giving a sparser graph"). Maximum conditioning-set size = 2 (track 0 → 29; "**and here it is not a
  compromise — the graph stops changing at 2**, so searching deeper would cost time and find nothing").

### Building the graph

---

**12 · Phase 1: from every pair to 35 edges** · *moderate*
- **Message:** the search tests progressively larger conditioning sets, and most edges fall early.
- **Device:** a descending funnel — four stacked bands, each labelled with edges remaining and the
  number of independence tests spent reaching it, narrowing left to right.
- **Why:** the message is attrition, and a funnel is the canonical form for attrition. The test counts
  in the same visual show where the computational cost actually goes, which a bar chart of edge counts
  alone would hide.
- **Content:** 465 candidate pairs → **217** after conditioning on nothing (682 tests) → **54** after
  conditioning on one variable (2,728 tests) → **35** after two (656 tests). Total 4,066 tests.
  Caption: "over half the pairs are separated by a single conditioning variable — that is the
  algorithm's central bet paying off."

**13 · Phase 2: how much direction the data supports** · *light*
- **Message:** the data determines direction for some edges, and honestly cannot for others.
- **Device:** a single proportional horizontal bar over all 35 edges, split into oriented / undirected /
  unresolved, with one small worked triple drawn beside it.
- **Why:** a proportional bar over a fixed total answers "how much of the graph is directed" in one
  mark. The adjacent worked triple grounds it in a concrete example rather than leaving it abstract.
- **Content:** 20 edges oriented · 3 left undirected · 12 where the collider rule points both ways and
  no direction is claimed. Caption: "an undirected edge is a result, not a gap — it says the data
  cannot distinguish the two directions."

**14 · The recovered causal graph** · *light — the centrepiece*
- **Message:** this is the deliverable.
- **Device:** **full-bleed node-link diagram on the dark ground**, 31 nodes laid out by their four
  natural clusters rather than in a circle, edges coloured teal where the reference graph agrees and
  slate blue where it does not. Arrowheads only on oriented edges. A compact legend and three counts
  in one corner. Almost no text.
- **Why:** the whole deck exists to produce this picture, so it gets the most space, the most contrast
  and the least competition. Clustering by component rather than circularly is what makes 31 nodes
  legible — the four groups separate visually and average degree is only 2.26. Reversing to the dark
  ground makes it the visual peak of the deck.
- **Content:** 35 edges — 11 agreeing with the reference, 24 additional. 20 directed. Cluster sizes
  21 / 7 / 2 / 1.

### What it means

---

**15 · Measured against the reference graph** · *moderate*
- **Message:** the recovered graph agrees with the reference on 11 of 28 findable relationships, with
  24 additional candidates.
- **Device:** *(revised at build time)* a 31 × 31 **agreement matrix** on the left — one square per
  variable pair, teal where both graphs agree, slate where only ours has the relationship, grey where
  only the reference does — with the metric block and its reading on the right.
- **Why:** the plan originally called for twin node-link graphs sharing node positions. Building it
  showed two problems: the reference graph connects nodes our graph leaves in separate clusters, so
  its edges cut across the cluster layout and made both panels harder to read than either alone; and
  two graphs at half scale put every node below legible size. The matrix has no layout risk, shows all
  three categories at once including the misses, and is a genuinely different device from slide 14 —
  so the two result slides no longer look like the same picture twice.
- **Content:** precision 0.31 · recall 0.39 · F1 0.35 · structural Hamming distance 48. Caption framing
  the honest reading: "the reference graph records the plant's designed relationships; additional edges
  are candidates for relationships the design does not document, not automatically errors."

**16 · What the result depends on** · *light*
- **Message:** the answer is stable in conditioning depth and insensitive to α, but sensitive to how
  much of the time series we use.
- **Device:** three small multiples on a shared axis — α sweep, conditioning-depth sweep, sample-
  thinning sweep.
- **Why:** the insight is a comparison of *slopes*, which only works on a shared axis. Three full-size
  charts would destroy the comparison; small multiples preserve it.
- **Content:** α from 0.05 to 10⁻¹⁰ moves F1 0.31 → 0.34 (flat). Conditioning depth: **identical at 2,
  3 and full — converged.** Thinning to every 5th sample *raises* precision 0.31 → 0.40. Callout: "the
  third panel is the interesting one, and slide 17 explains it."

**17 · The assumption we know we are stretching** · *light*
- **Message:** these are 1,499 time-ordered samples, not 1,499 independent ones, so the tests are
  optimistic.
- **Device:** a real line chart of one process variable over ~400 consecutive samples — wide and short —
  with `lag-1 autocorrelation = 0.982` set large in amber beside it.
- **Why:** the smooth serial trace makes the point in a second: neighbouring samples are near-copies.
  Stating "the independence assumption is violated" never lands this way. Cashes in slide 7's amber
  flag and explains slide 16's third panel.
- **Content:** max lag-1 0.982, median 0.571 across the 31 variables. Consequence: the effective sample
  size is a fraction of 1,499, so p-values are smaller than they should be and the graph is denser than
  the evidence strictly supports — which is exactly why thinning *improved* precision.

**18 · The same procedure, two further cases** · *moderate*
- **Message:** the method is not tuned to one dataset.
- **Device:** a compact three-row table, TE row tinted as the case just walked through, the two
  Ultra-Processed-Food cases beneath.
- **Why:** having spent 16 slides on one example, corroboration needs to be brief and comparable — a
  small table is the honest, low-cost form. It is also where UF's stronger numbers get their airing.
- **Content:** TE 31 vars, precision 0.31, converged. UF 17 vars, precision 0.55. UF with internal
  machine dependencies, 17 vars, **precision 0.80** — with a one-line note that this reference graph
  contains feedback loops, which PC assumes away, so its recall is capped by construction.

**19 · What we can and cannot claim** · *moderate*
- **Message:** an explicit inventory, with each gap paired to what would close it.
- **Device:** two facing columns of **deliberately equal visual weight** — teal left "supported,"
  amber right "not yet supported" — same width, type size and item count, each right-hand item carrying
  a short "what it would take."
- **Why:** limitations are normally a smaller, quieter, later list. Identical real estate is a design
  decision saying the uncertainty is part of the result. Pairing each gap with a remedy turns a caveat
  list into a research agenda — the right register for a progress review.
- **Content:** Supported — a converged, stable skeleton for TE; 11 of 28 documented relationships
  recovered from data alone; direction established for 20 of 35 edges; the method transfers to two
  further datasets. Not yet supported — that the 24 additional edges are real (→ time-aware
  independence testing, or block-resampled p-values); per-sensor causal claims (→ the variable-to-
  sensor mapping); relationships involving the two absent variables (→ a method admitting unmeasured
  confounders); feedback loops (→ a method that does not assume acyclicity).

**20 · Further steps** · *light*
- **Message:** the graph is now in place; the root-cause loop is what it unlocks.
- **Device:** the slide-2 pipeline returns, now with the graph stage filled teal and the next two stages
  outlined and dated — a visual "you are here."
- **Why:** closing the loop opened on slide 2 gives the deck structural completeness, and reusing the
  diagram costs the audience nothing to re-read.
- **Content:** next — a time-aware independence test to address the headline assumption; fault
  detection via T² monitoring; and the backward walk over the graph that turns a detected fault into a
  named root cause.

---

## 6. Density budget

| Density | Slides | Count | Target |
|---|---|---|---|
| **Light** (one visual, ≤ 25 words) | 1, 2, 3, 5, 10, 13, 14, 16, 17, 20 | **10** | 45–60 s |
| **Moderate** (two elements or a small table) | 4, 6, 7, 8, 9, 11, 12, 15, 18, 19 | **10** | ~2 min |
| **Dense** | — | **0** | — |

≈ 9 min light + 20 min moderate ≈ **29 minutes** before questions. If it runs long, slides 6 and 8 are
the safe ones to move through quickly — the mathematics is background, and slide 5 has already made the
structural point.

Speaker notes on all 20 carry the message, the numbers to say aloud, and the anticipated question.

---

## 7. What changed from revision 2, and two things to be aware of

### Removed for the no-implementation rule

| Removed | Was | Why |
|---|---|---|
| Every code excerpt | old 12, 13, 15 | Function names, assignments, a circled `break`, a struck-through assertion. All gone. |
| "What changed, and how we know it's right" | old 17 | A file-by-file change ledger — nothing survived the rule. |
| "16 tests" chip | old 1 | Replaced with `4,066 independence tests`, which is a property of the analysis, not the code. |
| Reproduce commands | old 20 | Implementation. |

### Removed for the no-before/after rule

| Removed | Was | Why |
|---|---|---|
| The Tennessee zero, the five-step trace, the raw-file evidence, the verdict badge | old 10–13 | The entire anomaly narrative. It documented a colleague's error. |
| "Eighteen findings, five that mattered" | old 14 | An audit-taxonomy slide. |
| The chain counter-example | old 15 | Its mathematics was good, but its purpose was to expose a defect. |
| "Fixing orientation made results look worse" | old 16 | Explicitly a before/after. Its *substance* survives on slide 13, reframed as "how much direction the data supports" — which is a result, not a correction. |
| Before → after dumbbell chart | old 18 | The chart type existed only to show a correction. |
| The index column as a discovery | old 8 | Now one neutral line on slide 10, as a preparation decision. |

Nine slots were freed and reinvested in the running example: the process schematic (9), preparation
(10), the two build slides (12, 13), **the graph itself (14)**, and the reference comparison (15). The
mathematics also regained the slide it had lost, so partial correlation and Fisher's z each get their
own (6, 7) plus a new orientation-rules slide (8).

### Two things I want to flag

1. **The spine dataset scores modestly.** TE agrees with its reference on 11 of 28 relationships
   (precision 0.31). I chose it anyway because it is the only legible graph, it is converged, and it is
   the recognisable benchmark — and because slides 15 and 19 let that number be framed accurately
   rather than defensively: the reference records the plant's *designed* relationships, so additional
   edges are candidates rather than automatic errors. UF's precision of 0.80 appears on slide 18. If
   you would rather lead with the stronger number, say so — the spine would become UF, and slide 14
   would have to become a 17×17 matrix view instead of a graph, because UF's node-link diagram is a
   hairball.
2. **No per-sensor interpretation is possible.** The variable-to-sensor mapping for X1…X33 is not in
   the data we hold, so the deck cannot say "reactor pressure drives separator level." Slide 9 states
   this in grey rather than leaving the audience to wonder, and slide 19 lists obtaining the mapping as
   a next step. If you can get that mapping, slide 14 becomes dramatically more valuable — it would let
   the graph be read as process knowledge rather than assessed only structurally.

### Suggested safety net, not on a slide

Your professor may already have seen the earlier Tennessee numbers. I'd put a short factual
explanation in the **speaker notes** for slide 15 — that earlier figures reflected a
reference-graph alignment problem since resolved, and the current numbers are the corrected ones. It
stays off the slides entirely, but you are not caught out if asked. Say if you'd rather it not exist
anywhere.

---

## 8. Format-variety audit

| Device | Slide |
|---|---|
| Dark full-bleed title | 1 |
| Horizontal pipeline strip | 2 *(recalled on 20)* |
| Three structures + one shared matrix | 3 |
| Three columns, one promoted | 4 |
| Storyboard small multiples | 5 |
| Colour-linked formula + peel step | 6 |
| Formula + shaded null distribution | 7 |
| Four before→after rule cards | 8 |
| Process schematic + stat rail | 9 |
| Two-panel as-provided → as-analysed | 10 |
| Parameter dials | 11 |
| Descending funnel with test counts | 12 |
| Proportional bar + worked triple | 13 |
| **Full-bleed node-link graph, dark ground** | 14 |
| Agreement matrix + metric block | 15 |
| Small multiples, shared axis | 16 |
| Real data line chart | 17 |
| Compact comparison table | 18 |
| Equal-weight facing columns | 19 |
| Pipeline callback | 20 |

**20 distinct devices across 20 slides.** No "title + bullets" slide. The single reuse (2 → 20) is a
deliberate callback.

---

## 9. Notes before building

1. **Slide 17 needs real data** — the build reads ~400 rows of one variable from the dataset, so it is
   not runnable standalone from `presentations/`.
2. **Slides 12, 14 and 15 are hand-drawn from shapes** and each needs a few rounds of coordinate
   tuning. Slide 14 especially: I will render it, look at it, and iterate on the cluster layout until
   31 nodes and 35 edges genuinely read at projection size. If it does not, the fallback is to drop the
   single isolated node and the 2-node pair into a footnote and draw only the 21- and 7-node clusters.
3. **Slide 9's schematic is illustrative.** It depicts the published benchmark process, not a mapping of
   our specific variables — labelled as such so it cannot be over-read.

---

## Awaiting your go-ahead

On approval I'll build the four modules, generate `PC_Algorithm.pptx`, then render and visually check
slides 12, 14 and 15 and iterate until the graph reads cleanly. Flag either item in §7 if you'd like
it decided differently — both are cheap to change now and expensive later.
