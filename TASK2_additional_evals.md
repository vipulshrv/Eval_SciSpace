# Task 2 — Evals Beyond Hallucination for Production Readiness

**Framing.** Hallucination (faithfulness — are claims grounded in the cited sources?) is *necessary but not sufficient* for shipping Report Writing. A report can be 100% faithful to its sources and still be useless: it can cite the wrong papers for the topic, miss the seminal work, faithfully summarize a predatory-journal preprint, or answer a different question than the user asked. Task 1 measured faithfulness. Production readiness needs six more axes. Each below is stated as **the question**, **why it matters (grounded in what we saw in this dataset)**, **how to measure it**, and a **priority** (P0 = ship-blocker, P1 = important, P2 = nice-to-have).

Where useful I note reuse of the Task-1 harness (`evals/`) — the LLM judge, DOI resolver, and per-step data loader already cover several of these.

---

## 1. Retrieval Quality — the input bounds everything

The report can never be better than the papers retrieved. We only checked that retrieved papers *exist* and have correct metadata (Step 1); we never checked whether they're the *right* papers.

### 1a. Coverage / recall of the literature — **P0**
- **Question:** did the agent find the papers that matter for this topic?
- **Why:** the pipeline keeps only 94 of ~240 retrieved and reports on the "top 30." If it silently drops the seminal work (for gut-brain, e.g., the Cryan & Dinan foundational reviews; for a clinical query, the landmark RCTs), the report is authoritative-looking but incomplete. Recall is invisible to a faithfulness eval.
- **How:** build expert-curated "must-cite" reference sets for a benchmark of ~20 queries; measure **Recall@k** of the retrieved set and of the final 30. Cross-check against a systematic-review bibliography where one exists.
- **Reuse:** DOI matcher from `step2_consolidation`.

### 1b. Relevance / context precision — **P1**
- **Question:** are the retrieved papers actually on-topic?
- **Why:** the 240→94 filter could be keeping noise or dropping signal. Off-topic papers dilute the table and the synthesis.
- **How:** LLM-judge relevance of each paper's abstract to the query (RAGAS-style context precision); report precision@k. **Reuse:** `judge.py` with a relevance rubric.

### 1c. Source quality / authority — **P0 (biomedical)**
- **Question:** are the sources credible?
- **Why:** *directly observed.* The corpora contain predatory/low-tier venues ("Deleted Journal," IJFMR, IJITSS), 2026-dated preprints, and non-peer-reviewed items. A medical report that faithfully summarizes junk is dangerous precisely because it looks rigorous.
- **How:** score each source on venue quality (indexed in DOAJ/Scopus/PubMed vs not; journal impact tier; preprint flag) and **retraction status** (Crossref/Retraction Watch API). Report the credibility distribution of cited papers and flag reports that lean on weak sources.
- **Reuse:** DOI resolver; add a Retraction Watch lookup.

### 1d. Recency & bias — **P1**
- **Question:** is coverage current, and is it skewed?
- **Why:** research tools must surface recent work; and we saw English-dominant, geographically skewed corpora (one Portuguese paper, mostly Western journals). Publication/language bias distorts conclusions.
- **How:** publication-year distribution vs the field's; language/geography distribution; check that recent high-impact work appears.

---

## 2. Report Quality — beyond "is each sentence grounded"

### 2a. Query-intent adherence / answer relevance — **P0**
- **Question:** does the report answer *what was asked*?
- **Why:** the sample queries demand **comparison** ("comparing performance across imaging, genomics, and multimodal… using AUC, sensitivity, specificity"; "across human and animal studies"). A grounded report that merely lists per-paper summaries without comparing has failed the task even at 100% faithfulness.
- **How:** LLM-judge the report against the query's explicit asks (did it compare the requested dimensions? use the requested metrics?); expert rating. **Reuse:** extends `step3_criteria`.

### 2b. Extraction completeness / table density — **P1**
- **Question:** how many table cells are actually filled vs "not reported"?
- **Why:** *observed* — the wearable table was dominated by "Specific… not reported" cells. A sparse table produces a thin report. This is under-extraction at scale (Task 1 measured only per-cell false-N/A).
- **How:** % non-null cells per criterion; correlate with source availability (abstract-only papers yield sparse cells). **Reuse:** `step4_extraction` loader.

### 2c. Comparative / analytical depth — **P1**
- **Question:** does it synthesize (contrast, reconcile, weigh evidence) or just aggregate?
- **How:** rubric-based LLM + expert scoring for synthesis vs enumeration, presence of cross-study contrasts, handling of disagreement.

### 2d. Consensus faithfulness / calibration — **P0**
- **Question:** does the report's confidence match the evidence, and does it represent the *balance* of studies (including null results)?
- **Why:** Task 1 flagged 212 uncited-synthesis sentences and a selection step ("top 30 of 94"). If selection is biased toward positive findings, the report overstates consensus even with every sentence grounded — a subtle, high-severity failure for research.
- **How:** compare the report's directional claims against the full evidence set including dropped/contradicting studies; check hedging language tracks evidence strength.

### 2e. Citation completeness & format — **P0**
- **Question:** is every claim cited, and are references resolvable and well-formed?
- **Why:** *observed defect* — the gut-brain report shipped with **no reference list at all** (inline [n] with nothing to resolve them to). That's a hard production blocker for a research tool.
- **How:** % factual sentences with a citation; every [n] resolves to a real, correct reference; consistent citation style; clickable/verifiable links. **Reuse:** citation logic from `step5_report`.

---

## 3. Robustness & Consistency — **P1**

- **Determinism / variance:** run the same query N times — how much do the paper set, table, and *conclusions* drift? Research output that changes materially run-to-run erodes trust. Measure Jaccard overlap of retrieved sets and semantic stability of conclusions.
- **Query-phrasing sensitivity:** paraphrase the query; do criteria and papers stay stable?
- **Graceful degradation:** niche topics, sparse-evidence areas, ambiguous or non-English queries — does it fail loudly (say "insufficient evidence") or confabulate to fill the template?

## 4. Safety & Domain Risk — **P0 (these are medical topics)** 

- **Clinical over-claiming:** the queries are health-critical (cancer detection, chronic-disease management). Does the report state efficacy/clinical readiness beyond what evidence supports? A safety-specific judge for unsupported clinical recommendations.
- **Harm / misinformation guardrails:** flag reports that could be read as clinical advice without appropriate caveats.

## 5. Operational Readiness — **P1**

- **Latency & cost per report** at the SLA the product promises; cost scaling per query.
- **Export/format reliability** (the missing reference list is also an operational-format bug).
- **Human-in-the-loop:** can users edit the table / re-run a criterion / swap sources?

## 6. Outcome Metrics — the ones that actually decide "ship" — **P0**

- **Expert acceptance / usability:** would a domain expert accept the report with only minor edits? Measure % reports usable without major correction, and **time saved** vs writing manually. This is the north-star.
- **Pairwise preference vs baseline** (e.g., vs a strong single-LLM-with-search baseline and vs a human-written mini-review), blind-rated by experts.
- **Trust calibration:** are citations verifiable enough that a skeptical user can trust-but-verify quickly?

---

## Prioritized ship-gate (the P0 shortlist)

If I had to gate a production launch on a handful of evals beyond hallucination, in order:

1. **Retrieval recall** (1a) — you can't fix downstream what was never retrieved.
2. **Source credibility + retraction** (1c) — faithful-to-junk is the most dangerous failure in a medical tool.
3. **Query-intent adherence** (2a) — did it do the comparison the user asked for.
4. **Citation completeness & resolvable references** (2e) — the report currently ships without a bibliography.
5. **Consensus/calibration** (2d) — not overstating a biased selection as consensus.
6. **Expert acceptance + time-saved** (6) — the metric the business decision actually rests on.

Faithfulness (Task 1) plus these six is a defensible readiness bar. Everything else (robustness, latency, editability) is P1 hardening that can follow the initial gate.

**Effort note:** items 1a/1c/2a/2d/2e reuse the existing harness (LLM judge, DOI/retraction resolver, criteria and citation parsers); the expensive-but-decisive one is the expert-acceptance study (6), which requires human raters and a query benchmark, not just automation.
