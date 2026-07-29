# SciSpace Report-Writing Agent — Evaluation Metrics Reference

This document records how the evaluation was conducted, defines every metric, states
exactly how each is computed, and reports the measured value for the four report tasks
(gut–brain axis, AI cancer detection, GLP-1 weight loss, diabetic retinopathy).

---

## Study Protocol — how this evaluation was conducted

**1. Task selection.** Five report-writing tasks were used. Three came from the
assignment's sample set; two — *GLP-1 receptor agonists for weight loss* and *deep
learning for diabetic retinopathy* — were generated with LLM assistance and selected
deliberately to stress distinct parts of the pipeline, in particular numeric
extraction and citation faithfulness on quantitatively dense clinical evidence, to
complement the more descriptive sample topics. One sample task (wearable devices) was
later excluded (step 4).

**2. Execution on SciSpace.** Each task was run manually through the SciSpace web app
in report-writing mode, which executes the full five-stage pipeline. Each session
produced a downloadable artifact bundle containing the per-source retrieval tables,
the consolidated evidence table, the extracted attribute table, and the final report
— the intermediate output of every stage, which is what makes stage-wise grading
possible.

**3. Artifact assembly.** Bundles were unzipped into one folder per task and wired
into the harness by their file layout (some sessions nested per-search sub-folders;
one multi-drug task split into three per-drug tables that the harness merges).

**4. Recovering citation traceability (manual step).** The downloaded report Markdown
was missing its numbered reference list — it renders in the SciSpace UI but was
truncated from the export. Citation-faithfulness grading requires resolving each
inline `[n]` to a specific paper, so the reference list was **copied from the UI by
hand into a `references.txt`** placed alongside each report. This step was
load-bearing: using an index-order fallback, cited authors matched the evidence table
in only 2 of 12 spot-checks; after adding the reference list, citations resolved to
the table 28/28 by DOI and matched authors 12/12. One task (wearables) had no
recoverable mapping even so — its report cited papers absent from its own table — and
was excluded rather than scored on a broken mapping.

**5. Evaluation harness.** A Python harness loads the artifacts, reconstructs the
`[n] → paper` map (parsing the reference list and matching each entry to the
consolidated table by DOI, then title), and runs the stage metrics. Evidence-integrity
checks (retrieval metadata via the CrossRef API; consolidation provenance) are
deterministic and use no model. Faithfulness stages use the LLM entailment judge
described below, validated against a hand-labelled gold set before use. To raise the
share of claims that could be adjudicated, open-access full text was fetched for the
cited papers (CrossRef/Unpaywall resolution plus PDF text extraction); coverage was
partial and is reported with every extraction metric.

**6. Discovering false-positive hallucinations.** An initial single-pass judge
returned alarmingly high hallucination counts (e.g., 25 flagged extraction
contradictions on the GLP-1 task). Manual inspection showed most were not genuine.
Three patterns recurred: the judge grading a claim against the **wrong passage** of a
long full-text source (the claim was in fact supported elsewhere); cells stating "not
reported" — the agent honestly describing sparse input — mislabelled as
contradictions; and **over-specifications** (a class-level result narrowed to one
drug) counted as conflicts.

**7. Adversarial verification (the fix).** To make the numbers trustworthy, every
flagged claim is re-adjudicated by a two-verifier panel that must search the whole
source for supporting evidence before ruling, with one verifier framed to seek
support and one to refute; a flag survives only if both confirm it. This collapsed the
first-pass flags substantially (roughly 70–95% were false positives) and produced the
post-verification figures reported here. The size of that correction is itself a
result: a single-pass LLM judge on dense sources is not, on its own, a reliable
hallucination detector.

---

## A. How claims are graded (shared machinery)

The agent is a five-stage retrieval-augmented pipeline: **retrieval → consolidation
→ criteria induction → attribute extraction → report synthesis.** Each stage is
graded against the output of the previous stage, so a failure is attributable to a
specific stage.

**Reference-grounded grading.** Every generated unit is graded against the evidence
available at that stage — the extracted table cell against the paper's text, and the
report sentence against the paper it cites (that paper's full text where we hold it,
else its abstract, unioned with the agent's extracted table row). Claims that cannot be
adjudicated from the evidence we hold (e.g., only an abstract is available and it is
silent on the claim) are labelled **non-adjudicable** and excluded from rates — they are
counted as neither faithful nor hallucinated.

**The judge.** Faithfulness metrics use an LLM as an entailment grader. For each
claim it returns one of three labels:

- **Supported** — the source states or entails the claim.
- **Contradicted** — the source asserts something incompatible (*intrinsic* hallucination).
- **Unsupported** — the source neither supports nor contradicts (*extrinsic* hallucination).

The judge must quote a verbatim supporting span (a "Supported" verdict with no
locatable span is demoted), and it is prompted adversarially (to attempt
refutation). It was checked against a hand-labelled set before use.

**Verification pass.** Every claim the first pass flags as a hallucination is
re-graded by a **two-verifier panel**: each verifier searches the entire source for
a supporting passage before it may rule against the claim; one is framed to seek
support, one to seek refutation. A hallucination label survives **only if both
verifiers confirm it.** All faithfulness metrics below are post-verification.

---

## B. Metrics by stage

### Stage 1 — Retrieval

**DOI resolution rate** — *does the cited identifier exist?*
Computed: each retrieved paper's DOI is looked up in CrossRef; resolved = the lookup
returns a real record. (A non-resolving DOI is reported as *unresolved*, not
*fabricated*, because some legitimate DOIs are not in CrossRef.)

**Metadata accuracy** — *is the recorded title/year correct?*
Computed: among DOI-resolved papers, the share whose agent-recorded title matches the
CrossRef title, and whose year matches exactly. Titles are compared after HTML-markup
cleanup, and a match is accepted when the titles are identical, when one contains the
other (a present/absent subtitle or a truncated-but-correct title — e.g. the agent's
`…: The SCALE Diabetes Randomized Clinical Trial` where CrossRef stores only the main
title), or when fuzzy similarity clears 0.85. Reported per source, then aggregated as
`title_matches ÷ resolved`. The few residual mismatches are **not fabrications** — they
are non-English original titles (the agent shows an English title, CrossRef the German/
French/Italian original) plus one mis-registered DOI whose CrossRef record is an
unrelated paper. *Completeness* (whether a record even carries a DOI) is tracked
separately from *accuracy*, because sources differ in how much metadata they return.

### Stage 2 — Consolidation

**Provenance integrity** — *is every consolidated paper real and unaltered?*
Computed (deterministic, no LLM): every row in the consolidated table must match a
retrieved source record — by normalized DOI, else by normalized title. A row with no
source match is a **fabricated** record. Also checks metadata drift (title/year
changing during the merge) and duplicate rows. Reported as
`rows_with_a_source ÷ total_rows`.

### Stage 3 — Criteria Induction

**Criteria coverage** — *were the query's requested comparison dimensions turned into
table columns?*
Computed: an LLM judge lists the distinct dimensions the query explicitly asks to
analyze and checks whether each is represented by a column. Reported as
`dimensions_covered ÷ dimensions_requested`.

**Spurious criteria** — *did the agent invent columns with no basis in the query?*
Computed: same judge marks each column as grounded in the query or not; spurious =
count of ungrounded columns.

### Stage 4 — Attribute Extraction

**Extraction hallucination rate** — *are the extracted table values grounded in the
paper?*
Computed: each non-empty table cell (paper × criterion) is entailment-graded against
that paper's source text. A cell is a hallucination if **Contradicted** (intrinsic)
or **Unsupported** (extrinsic). Reported as
`(contradicted + unsupported) ÷ adjudicable_cells`, post-verification.

**Under-extraction** — *did the agent report "no data" when the paper has the data?*
Computed: for cells stating the datum is absent, the verifier checks whether the
source in fact contains it. A confirmed case is an under-extraction (a recall error).
Reported as `confirmed ÷ "no-data" cells` (the cells that declared a datum absent).
It is tracked separately from hallucination because nothing is fabricated — the
failure is omission.

### Stage 5 — Report Synthesis

**Citation faithfulness (attribution precision)** — *does the paper a claim cites
actually support it?*
Computed: the report is segmented into sentences; each sentence carrying a citation
`[n]` is entailment-graded against the cited paper's **full text where we hold it (else
its abstract), unioned with the agent's extracted table row** — a claim is faithful if
it traces to either the real paper or what the agent extracted. Each cited sentence is
also classified by **claim kind**:

- **empirical** — asserts a specific finding, result, or quantitative outcome
  attributed to the cited study;
- **background** — general domain knowledge, definitions, or methodological framing
  that cites a source only illustratively (normal scientific practice, not an empirical
  attribution).

The **headline** metric is precision on **empirical** claims; **background** is reported
separately, for context, and is never counted as a citation failure. Reference-relativity
applies: an *Unsupported* verdict on a sentence whose cited papers we hold only as
abstracts is **non-adjudicable** (its support may lie in full text we do not have) and is
excluded rather than scored. Reported as `supported ÷ adjudicable`, post-verification.

**Report contradiction count** — *does any cited claim conflict with its source?*
Computed: count of cited sentences graded **Contradicted** (against the cited paper's
full text) after verification.

**Numerical consistency** — *do the numbers in the report appear in the evidence?*
Computed (deterministic, no LLM): every numeric token in a cited sentence (citation
markers stripped) is checked for presence in the cited paper's source text / table
row(s), tolerant of the `%` sign and thousands separators. Reported as
`numbers_found ÷ numbers_checked`. This is a strict lexical check — a number phrased
differently (rounded, reformatted) counts as not-found — so it is a conservative
signal.

**Unsupported synthesis** — *how many factual-looking sentences carry no citation?*
Computed: count of report sentences (above a length threshold, excluding headings and
lists) that contain no citation marker. These are untraceable to a specific source by
construction. Reported as a share of report sentences (with the count).

---

## C. Measured results (four tasks, post-verification)

Rates are shown as `percent (numerator/denominator)`. Rare, severe events
(fabricated records, spurious criteria, report contradictions) are reported as
`count / base` rather than a percentage, because a percentage at n ≤ 2 would imply
precision the sample does not support.

### Stage 1–3 — Evidence integrity

| Metric | Gut–brain | Cancer | GLP-1 | Retinopathy |
|---|---|---|---|---|
| Metadata accuracy | 99.5% (212/213) | 100% (208/208) | 99.1% (654/660) | 99.8% (629/630) |
| Provenance integrity | 100% (94/94) | 100% (98/98) | 100% (255/255) | 100% (278/278) |
| Fabricated records | 0 / 94 | 0 / 98 | 0 / 255 | 0 / 278 |
| Criteria coverage | 100% (3/3) | 100% (2/2) | 100% (3/3) | 80% (4/5) |
| Spurious criteria | 0 / 3 | 0 / 3 | 0 / 3 | 0 / 1 |

### Stage 4 — Extraction

| Metric | Gut–brain | Cancer | GLP-1 | Retinopathy |
|---|---|---|---|---|
| Extraction hallucination rate | 1.4% (1/72) | 0% (0/66) | 4.6% (9/195) | 0% (0/24) |
| — contradicted (intrinsic) | 0 | 0 | 4 | 0 |
| — unsupported (extrinsic) | 1 | 0 | 5 | 0 |
| Under-extraction | 0 / 3 | 0 / 5 | 6 / 17 | — (0 no-data cells) |

### Stage 5 — Report synthesis

Citation faithfulness is graded against the cited paper's full text (∪ table row) and
split by claim kind; the empirical row is the headline. *Non-adjudicable* counts the
cited sentences (abstract-only papers, claim not locatable) excluded from the rate under
reference-relativity. Report-contradiction denominators are the adjudicable cited
sentences.

| Metric | Gut–brain | Cancer | GLP-1 | Retinopathy |
|---|---|---|---|---|
| Citation faithfulness — **empirical claims** | 100% (98/98) | 100% (28/28) | 87.8% (36/41) | 96.4% (27/28) |
| Citation faithfulness — background *(context only)* | 100% (19/19) | 100% (18/18) | 80% (4/5) | 100% (7/7) |
| Non-adjudicable (abstract-only, excluded) | 21 | 25 | 27 | 15 |
| Report contradictions | 0 / 117 | 0 / 46 | 3 / 46 | 0 / 35 |
| Numerical consistency | 96.3% (26/27) | 93.6% (44/47) | 86.0% (257/299) | 89.5% (120/134) |
| Unsupported synthesis | 60.6% (212/350) | 73.6% (204/277) | 58.0% (101/174) | 54.9% (62/113) |

### Verification impact — a worked example (GLP-1, Stage 4 extraction)

All rates above are post-verification. To show why that qualifier matters, the table
below reports the same metric **before and after** the two-verifier panel, on the task
where the effect is largest (GLP-1 attribute extraction).

| | First-pass judge | After adversarial verification |
|---|---|---|
| Cells flagged as hallucinated | 48 | 9 confirmed |
| — contradicted (intrinsic) | 25 | 4 |
| — unsupported (extrinsic) | 23 | 5 |
| Extraction hallucination rate | 23.3% (48/206) | 4.6% (9/195) |

The single-pass judge flagged **48 of 206 adjudicable cells** as hallucinations. The
two-verifier panel — each verifier searching the whole source before ruling — confirmed
only **9**. The other 39 flags (**81%**) were overturned: 28 were in fact grounded
(the judge had checked the wrong passage of a long source), and 11 were reclassified as
a different, non-fabrication failure — 6 as under-extraction (a recall error, reported
separately) and 5 as not a substantive claim. This is why the adjudicable denominator
falls from 206 to 195.

Without this pass, the reported extraction hallucination rate would have been **~5×
higher (23% vs 4.6%)** and almost entirely wrong. A single-pass LLM judge on dense
full-text sources is not, by itself, a reliable hallucination detector; the verification
pass is what makes the headline numbers trustworthy.

---

## D. Next steps — evaluations beyond hallucination for production readiness

Faithfulness (this study) is necessary but not sufficient to ship Report Writing: a
report can be fully grounded in its sources and still cite the wrong papers, miss the
seminal work, faithfully summarise a predatory-journal preprint, or answer a different
question than the one asked. The next evaluations I would run — the shortlist a launch
decision should gate on, in priority order — are:

1. **Retrieval recall.** Did the agent find the papers that matter? It keeps ~94 of
   ~240 retrieved and reports on a top-30 subset; if it silently drops the landmark
   studies the report looks authoritative but is incomplete, and recall is invisible to
   a faithfulness eval. Measure Recall@k against expert-curated "must-cite" sets.

2. **Source credibility.** Are the sources trustworthy? The corpora observed here
   included predatory/low-tier venues and non-peer-reviewed items; a report that
   faithfully summarises junk is dangerous precisely because it looks rigorous. Score
   venue quality and check retraction status.

3. **Query-intent adherence.** Did it answer what was asked? The sample queries demand a
   *comparison* across specified dimensions; a grounded report that merely lists
   per-paper summaries has failed the task even at 100% faithfulness.

4. **Citation completeness & resolvable references.** Is every claim cited, and does
   every `[n]` resolve to a real, correct reference? The truncated reference list seen in
   this study is a hard ship-blocker for a research tool.

5. **Consensus & calibration.** Because the agent selects a top-30 subset, a selection
   biased toward positive findings can overstate consensus even with every sentence
   grounded — a subtle, high-severity failure for research.

6. **Expert acceptance (north star).** Would a domain expert accept the report with only
   minor edits, and how much time did it save versus writing it by hand? This, plus
   blind pairwise preference against a baseline, is the metric the launch decision rests
   on.

Full framing — question, why it matters, how to measure, priority, and which parts reuse
this harness — is in **`TASK2_additional_evals.md`**, which also covers the P1 hardening
axes (robustness/consistency, extraction completeness, latency and cost, editability).

---

## E. Glossary

- **Supported / Contradicted / Unsupported** — the three entailment labels the judge
  assigns a claim against its source.
- **Intrinsic hallucination** — a Contradicted claim (conflicts with the source).
- **Extrinsic hallucination** — an Unsupported claim (adds information not in the source).
- **Adjudicable** — a claim for which the available evidence is sufficient to render a
  verdict; non-adjudicable claims are excluded from all rates.
- **Post-verification** — the metric reflects the two-verifier panel re-check, not the
  first-pass judge.
