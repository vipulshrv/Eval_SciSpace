# SciSpace Report-Writing Agent — Hallucination Evaluation

An evaluation harness that measures the **faithfulness** (freedom from
hallucination) of SciSpace's report-writing agent — a retrieval-augmented system
that finds papers, extracts structured evidence, and synthesizes a cited report.

The full metric definitions and measured results are in
**`SciSpace_Evaluation_Metrics.pdf`** (source: `EVALUATION_METRICS.md`).
Additional production-readiness evaluations are proposed in
**`TASK2_additional_evals.md`**.

---

## Methodology at a glance

The agent is treated as a five-stage pipeline, and each stage is graded **against
the output of the previous stage**, so a failure is attributable to a specific
stage rather than only visible in the final report:

1. **Literature Retrieval** — validity/accuracy of retrieved paper metadata (checked against CrossRef).
2. **Consolidation** — every consolidated record traces to a retrieved source; no fabrication, drift, or duplication (deterministic).
3. **Criteria Induction** — the query's requested comparison dimensions become table columns.
4. **Attribute Extraction** — each extracted table cell is grounded in its paper's text.
5. **Report Synthesis** — each cited report claim is supported by the paper it cites.

Two principles underpin every faithfulness metric:

- **Reference-grounded grading.** A claim is judged only against the evidence the
  agent actually had; claims we cannot adjudicate from available evidence are
  excluded, never scored as hallucinations.
- **Hardened + verified judging.** Faithfulness uses an LLM-as-judge (three-way
  entailment: *supported / contradicted / unsupported*, must quote a supporting
  span, adversarially framed), validated against a hand-labelled gold set. Every
  flagged hallucination is then re-checked by a **two-verifier adversarial panel**
  that must search the whole source before ruling; a flag survives only if both
  verifiers confirm it. This second pass is essential — a single-pass judge
  over-reported hallucinations by roughly 3–15× on dense sources.

---

## Repository layout

```
evals/                     evaluation harness (one module per stage + shared machinery)
  data_loader.py           loads a query's artifacts; reconstructs the [n]→paper citation map
  step1_retrieval.py       Stage 1 — retrieval metadata (CrossRef, deterministic)
  step2_consolidation.py   Stage 2 — provenance integrity (deterministic)
  step3_criteria.py        Stage 3 — criteria coverage (judge)
  step4_extraction.py      Stage 4 — extraction faithfulness (judge)
  step5_report.py          Stage 5 — report/citation faithfulness (judge)
  judge.py                 hardened LLM-as-judge + token/cost tracking
  verify.py                two-verifier adversarial verification; folds into stages 4 & 5
  gold_set.py              judge validation vs hand-labelled gold set (Cohen's kappa)
  score.py                 turns stage summaries into the two-tier scorecard
  run_all.py               orchestrates all stages across all queries
  fetch_papers.py          downloads open-access full text to improve adjudicability
artifacts/                 SciSpace session inputs for the evaluated tasks (the eval's inputs)
results/                   evaluation outputs (per-query dashboards + combined_scores.json)
EVALUATION_METRICS.md/.pdf  metric definitions, computation, and measured results
TASK2_additional_evals.md  production-readiness evaluations beyond hallucination
```

The four evaluated tasks are registered in `evals/data_loader.py` (`QUERIES`):
`gut_brain`, `cancer`, `glp1_weight_loss`, `diabetic_retinopathy`.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic certifi PyPDF2

cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

- **`ANTHROPIC_API_KEY`** (in `.env`) is required — it powers the LLM judge.
- CrossRef and Unpaywall (used for metadata and full-text lookup) are public APIs
  and need no key.

---

## Running the evaluation

```bash
# 1. First pass — run all five stages across all four tasks.
#    Writes results/<query>/step{1..5}_*.json and results/<query>/dashboard.json
python -m evals.run_all

#    Options:
python -m evals.run_all --query cancer      # a single task
python -m evals.run_all --limit 10          # cheap smoke run (caps stages 4 & 5)

# 2. (Optional but recommended) Download open-access full text to raise the share
#    of extraction claims that can be adjudicated.
python -m evals.fetch_papers                 # all tasks  (or pass task names)

# 3. Verification + rescore — fold the adversarial panel into stages 4 & 5 and
#    regenerate the scorecard (results/combined_scores.json) from verified numbers.
python -m evals.verify

# Validate the judge against the gold set (Cohen's kappa):
python -m evals.gold_set

# Any single stage can also be run on its own, e.g.:
python -m evals.step2_consolidation
python -m evals.step4_extraction            # append a number to cap cells (0 = all)
```

**Order matters:** `run_all` (first pass) → `fetch_papers` (optional) → `verify`
(folds verification and rewrites the scorecard). Faithfulness numbers in the
committed results are post-verification.

---

## Outputs

- **`results/combined_scores.json`** — the two-tier scorecard across all tasks
  (integrity gates + hallucination profile).
- **`results/<query>/dashboard.json`** — per-task summaries, scorecard, and cost.
- **`results/<query>/step{1..5}_*.json`** — per-stage detail (per-cell / per-sentence
  verdicts, including verified states).
- **`results/gold_validation.json`** — judge-vs-human agreement.

Metrics are reported as `rate (numerator/denominator)`; rare severe events
(fabricated records, spurious criteria, report contradictions) are reported as
`count / base` rather than a percentage.

---

## Notes

- **Cost.** The judge and the two verification passes call the Anthropic API;
  a full four-task run with verification is on the order of tens of dollars
  (dominated by grading extraction cells against downloaded full text).
- **Evidence coverage.** Only open-access full text can be downloaded, so some
  claims are adjudicated against abstracts; unresolvable claims are excluded rather
  than guessed. Faithfulness rates are therefore conservative.
- **Excluded task.** A fifth task (wearable devices) was dropped because its
  exported report's citations could not be reconciled with its own evidence table.
- **`data/`** (downloaded paper full text) is gitignored — regenerate it with
  `python -m evals.fetch_papers`.
