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
available at that stage — the extracted table cell against the paper's text, the
report sentence against the cited paper's table row. Claims that cannot be
adjudicated from the evidence we hold (e.g., only an abstract is available) are
labelled **non-adjudicable** and excluded from rates — they are counted as neither
faithful nor hallucinated.

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
CrossRef title (string-similarity ratio ≥ 0.85) and whose year matches exactly.
Reported per source, then aggregated as `title_matches ÷ resolved`.
*Completeness* (whether a record even carries a DOI) is tracked separately from
*accuracy*, because sources differ in how much metadata they return.

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

**Citation faithfulness (attribution precision)** — *does each cited claim's source
actually support it?*
Computed: the report is segmented into sentences; each sentence carrying a citation
`[n]` is entailment-graded against the cited paper(s)' table row(s). Reported as
`supported_cited_sentences ÷ graded_cited_sentences`, post-verification.

**Report contradiction count** — *does any cited claim conflict with its source?*
Computed: count of cited sentences graded **Contradicted** after verification.

**Numerical consistency** — *do the numbers in the report appear in the evidence?*
Computed (deterministic, no LLM): every numeric token in a cited sentence (citation
markers stripped) is checked for presence in the cited paper's table row(s), tolerant
of the `%` sign and thousands separators. Reported as
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
| Metadata accuracy | 99.5% (212/213) | 100% (208/208) | 97.3% (642/660) | 99.7% (628/630) |
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

| Metric | Gut–brain | Cancer | GLP-1 | Retinopathy |
|---|---|---|---|---|
| Citation faithfulness | 96.4% (133/138) | 67.1% (49/73) | 63.0% (46/73) | 64.7% (33/51) |
| Report contradictions | 0 / 138 | 0 / 73 | 2 / 73 | 0 / 51 |
| Numerical consistency | 96.3% (26/27) | 87.2% (41/47) | 80.3% (240/299) | 86.6% (116/134) |
| Unsupported synthesis | 60.6% (212/350) | 73.6% (204/277) | 58.0% (101/174) | 54.9% (62/113) |

---

## D. Glossary

- **Supported / Contradicted / Unsupported** — the three entailment labels the judge
  assigns a claim against its source.
- **Intrinsic hallucination** — a Contradicted claim (conflicts with the source).
- **Extrinsic hallucination** — an Unsupported claim (adds information not in the source).
- **Adjudicable** — a claim for which the available evidence is sufficient to render a
  verdict; non-adjudicable claims are excluded from all rates.
- **Post-verification** — the metric reflects the two-verifier panel re-check, not the
  first-pass judge.
