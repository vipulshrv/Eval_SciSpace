"""
STEP 5 — REPORT GENERATION eval.

Question: for every report claim, does the paper it CITES actually support it,
and does it preserve the numbers?

Reference truth (clean report-vs-source / ALCE-style attribution): each cited
paper's FULLEST available source text — downloaded full text if we hold it, else
abstract/excerpt — UNIONED with the agent's own extracted table row. A claim is
faithful if it traces to EITHER the real paper OR what the agent extracted.
(Reference-relativity: an UNSUPPORTED verdict is only counted as a real
attribution failure when we had full text for all cited papers; otherwise the
support may live in text we do not hold, so the claim is non-adjudicable and
excluded — never scored as a hallucination. This mirrors Step 4.)

Unit of evaluation: the report sentence (an approximation of an atomic claim).
Each sentence is classified:
  - CITED    -> carries one or more [n] citations; graded 3-way against the union
                of the cited papers' source text + table rows.
  - UNCITED  -> a factual-looking sentence with no citation (Executive Summary /
                Discussion synthesis). Counted separately.

Every cited sentence is additionally classified by CLAIM KIND:
  - empirical  -> asserts a specific finding/result attributed to the cited study.
  - background -> general domain knowledge / framing that cites a source
                  illustratively. Normal scientific practice, not an empirical
                  attribution; reported separately (context only), never headlined
                  as a citation failure.

Metrics:
  - citation faithfulness (empirical) = supported / adjudicable empirical claims  [HEADLINE]
  - citation faithfulness (background) = supported / adjudicable background claims [context only]
  - report contradictions             = cited sentences the source contradicts
  - numeric fidelity                  = report numbers found in the cited source
  - uncited-synthesis count           = factual sentences with no citation
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data_loader import QueryDataset, gut_brain
from .judge import Judge, Judgment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# sentences too short or structural (headings, TOC) aren't claims
_MIN_CLAIM_LEN = 40
_CITE_RE = re.compile(r"\[(\d+)\]")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")

# richness ordering of source text available for a paper (worst -> best)
_LEVEL_ORDER = {
    "none": 0, "snippet_only": 1, "abstract_only": 2,
    "fulltext_excerpt": 3, "fulltext_downloaded": 4,
}


def _worst_level(levels: list[str]) -> str:
    """The weakest grounding across a sentence's cited papers (drives adjudicability)."""
    if not levels:
        return "none"
    return min(levels, key=lambda l: _LEVEL_ORDER.get(l, 0))


@dataclass
class SentenceResult:
    idx: int
    text: str
    citations: list[int]
    kind: str  # "cited" | "uncited"
    verdict: str | None = None
    confidence: str | None = None
    evidence_span: str | None = None
    claim_kind: str | None = None       # "empirical" | "background"  (cited only)
    grounding_level: str | None = None  # weakest source level across cited papers
    context_limited: bool = False       # UNSUPPORTED but not full-text-adjudicable
    numbers: list[str] = field(default_factory=list)
    numbers_grounded: list[str] = field(default_factory=list)
    numbers_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_paper_reference(ds: QueryDataset, n: int) -> tuple[str, str]:
    """Reference truth for citation [n] under report-vs-source grading (clean B):
    the cited paper's fullest source text (downloaded full text if held, else
    abstract/excerpt) UNIONED with the agent's extracted table row.
    Returns (reference_text, grounding_level)."""
    row = ds.citation_row(n)
    if not row:
        return "", "none"
    parts = [row.get("Paper Title", "")]
    for col in ds.criteria_columns:
        v = (row.get(col, "") or "").strip()
        if v:
            parts.append(f"{col}: {v}")
    rt = ds.reference_text(row)
    if rt.best:
        parts.append(f"Source text:\n{rt.best}")
    return "\n".join(p for p in parts if p.strip()), rt.grounding_level


# retained for backward-compat (older callers); superseded by build_paper_reference
def build_table_reference(ds: QueryDataset, n: int) -> str:
    row = ds.citation_row(n)
    if not row:
        return ""
    parts = [row.get("Paper Title", "")]
    for col in ds.criteria_columns:
        v = (row.get(col, "") or "").strip()
        if v:
            parts.append(f"{col}: {v}")
    ab = (row.get("Abstract", "") or "").strip()
    if ab:
        parts.append(f"Abstract: {ab}")
    return "\n".join(p for p in parts if p.strip())


def segment_sentences(report: str) -> list[str]:
    """Split report body into sentences, skipping headings / TOC / list scaffolding."""
    sentences: list[str] = []
    for line in report.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|") or line.startswith("-"):
            continue
        # numbered TOC / list lines like "1. Introduction"
        if re.match(r"^\d+\.\s", line) and len(line) < _MIN_CLAIM_LEN:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            s = s.strip()
            if len(s) >= _MIN_CLAIM_LEN:
                sentences.append(s)
    return sentences


def _numbers(text: str) -> list[str]:
    # strip citation markers so [9], [12] aren't read as data
    stripped = _CITE_RE.sub(" ", text)
    return [m.group(0) for m in _NUM_RE.finditer(stripped)]


def _num_in(num: str, ref: str) -> bool:
    # match the numeric core, tolerating the % sign and thousands separators
    core = num.rstrip("%")
    ref_norm = ref.replace(",", "")
    return core in ref_norm or (core + "%") in ref_norm


# ---- claim-kind classification (empirical vs background) ---------------------

_KIND_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["empirical", "background"]},
                },
                "required": ["idx", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

_KIND_SYS = (
    "You classify sentences from a scientific literature-review report. Each sentence "
    "carries one or more [n] citations. Label each by its FUNCTION, not by whether it "
    "contains a number:\n"
    "- \"empirical\": asserts a SPECIFIC finding, result, quantitative outcome, dataset, "
    "method, or characteristic attributed to the cited study/studies (e.g. a reported "
    "accuracy or effect size, a trial result, what a particular paper found or did).\n"
    "- \"background\": general domain knowledge, definitions, motivation, or "
    "methodological framing that cites a source only illustratively and is not a specific "
    "result of the cited work (e.g. 'CNNs learn hierarchical features', 'early detection "
    "improves outcomes', 'transfer learning is a standard paradigm')."
)


def classify_kinds(judge: Judge, cited: list[SentenceResult], chunk: int = 40) -> None:
    """Batch-label each cited sentence empirical/background (one judge call per chunk)."""
    kinds: dict[int, str] = {}
    for i in range(0, len(cited), chunk):
        batch = cited[i:i + chunk]
        listing = "\n".join(f"{r.idx}. {r.text}" for r in batch)
        out = judge.structured(_KIND_SYS, f"Classify each sentence by its idx:\n\n{listing}",
                               _KIND_SCHEMA, max_tokens=2000)
        for lab in (out.get("labels") or []):
            kinds[lab.get("idx")] = lab.get("kind")
    for r in cited:
        r.claim_kind = kinds.get(r.idx, "empirical")  # default to empirical (stricter)


@dataclass
class Step5Result:
    sentences: list[SentenceResult] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def _bucket(self, sents: list[SentenceResult]) -> dict:
        """Adjudicable = graded sentences minus context-limited UNSUPPORTED ones."""
        adj = [s for s in sents if not s.context_limited]
        sup = [s for s in adj if s.verdict == "SUPPORTED"]
        con = [s for s in adj if s.verdict == "CONTRADICTED"]
        return {
            "adjudicable": len(adj),
            "supported": len(sup),
            "contradicted": len(con),
            "context_limited_excluded": sum(1 for s in sents if s.context_limited),
            "precision": round(len(sup) / len(adj), 4) if adj else None,
        }

    def summary(self) -> dict:
        cited = [s for s in self.sentences if s.kind == "cited"]
        uncited = [s for s in self.sentences if s.kind == "uncited"]
        graded = [s for s in cited if s.verdict is not None]

        emp = self._bucket([s for s in graded if s.claim_kind == "empirical"])
        bg = self._bucket([s for s in graded if s.claim_kind == "background"])
        allb = self._bucket(graded)

        all_nums = sum(len(s.numbers) for s in cited)
        grounded_nums = sum(len(s.numbers_grounded) for s in cited)

        return {
            "step": "5_report_generation",
            "reference_truth": "report claim vs cited paper full-text (else abstract) ∪ extracted table row",
            "total_claim_sentences": len(self.sentences),
            "cited_sentences": len(cited),
            "uncited_synthesis_sentences": len(uncited),
            "graded_cited": len(graded),
            # HEADLINE citation faithfulness = empirical-claim precision
            "citation_precision": emp["precision"],
            "citation_precision_empirical": emp["precision"],
            "citation_precision_background": bg["precision"],
            "empirical": emp,
            "background": bg,
            "all_cited": allb,
            "report_contradictions": allb["contradicted"],
            "context_limited_excluded": allb["context_limited_excluded"],
            "verdict_breakdown": {
                "SUPPORTED": sum(1 for s in graded if s.verdict == "SUPPORTED"),
                "UNSUPPORTED": sum(1 for s in graded if s.verdict == "UNSUPPORTED"),
                "CONTRADICTED": sum(1 for s in graded if s.verdict == "CONTRADICTED"),
            },
            "numeric_fidelity": round(grounded_nums / all_nums, 4) if all_nums else None,
            "numbers_checked": all_nums,
            "numbers_missing": all_nums - grounded_nums,
        }


def run(ds: QueryDataset, limit: int | None = None, workers: int = 6,
        judge: Judge | None = None) -> Step5Result:
    judge = judge or Judge()
    sentences = segment_sentences(ds.report)

    records: list[SentenceResult] = []
    for i, text in enumerate(sentences):
        cites = [int(x) for x in _CITE_RE.findall(text)]
        kind = "cited" if cites else "uncited"
        records.append(SentenceResult(idx=i, text=text, citations=cites, kind=kind,
                                      numbers=_numbers(text)))

    cited = [r for r in records if r.kind == "cited"]
    if limit is not None:
        cited = cited[:limit]

    def grade_one(rec: SentenceResult) -> SentenceResult:
        refs, levels = [], []
        for n in rec.citations:
            r, lvl = build_paper_reference(ds, n)
            if r:
                refs.append(r)
                levels.append(lvl)
        ref = "\n\n".join(refs)
        rec.grounding_level = _worst_level(levels)
        j: Judgment = judge.grade(rec.text, ref, source_label="CITED SOURCE(S)")
        rec.verdict = j.verdict
        rec.confidence = j.confidence
        rec.evidence_span = j.evidence_span
        # reference-relativity: only count UNSUPPORTED as a real attribution failure
        # when full text was held for ALL cited papers; else it is non-adjudicable.
        if j.verdict == "UNSUPPORTED" and rec.grounding_level != "fulltext_downloaded":
            rec.context_limited = True
        for num in rec.numbers:
            (rec.numbers_grounded if _num_in(num, ref) else rec.numbers_missing).append(num)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(grade_one, cited))

    classify_kinds(judge, cited)
    return Step5Result(sentences=records, usage=judge.cost_report())


def _print(res: Step5Result) -> None:
    s = res.summary()
    print("=" * 68)
    print("STEP 5 — REPORT GENERATION  (report claim vs cited SOURCE, full-text)")
    print("=" * 68)
    print(f"  claim sentences            : {s['total_claim_sentences']}")
    print(f"  cited sentences            : {s['cited_sentences']}  (graded: {s['graded_cited']})")
    print(f"  uncited synthesis          : {s['uncited_synthesis_sentences']}")
    emp, bg = s["empirical"], s["background"]
    ep = f"{emp['precision']*100:.1f}%" if emp["precision"] is not None else "n/a"
    bp = f"{bg['precision']*100:.1f}%" if bg["precision"] is not None else "n/a"
    print(f"  citation faithfulness      :")
    print(f"    - EMPIRICAL claims       : {ep}  ({emp['supported']}/{emp['adjudicable']})   [HEADLINE]")
    print(f"    - background/illustrative: {bp}  ({bg['supported']}/{bg['adjudicable']})   [context only]")
    print(f"  report contradictions      : {s['report_contradictions']}")
    print(f"  context-limited (excluded) : {s['context_limited_excluded']}")
    nf = s["numeric_fidelity"]
    if nf is not None:
        print(f"  numeric fidelity           : {nf*100:.1f}%  "
              f"({s['numbers_checked']-s['numbers_missing']}/{s['numbers_checked']})")
    else:
        print("  numeric fidelity           : n/a")

    flagged = [r for r in res.sentences if r.verdict in ("UNSUPPORTED", "CONTRADICTED")
               and not r.context_limited]
    if flagged:
        print(f"\n  adjudicable flags ({len(flagged)}):")
        for r in flagged[:8]:
            print(f"    [{r.verdict}/{r.claim_kind}] cites {r.citations}: {r.text[:80]}...")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ds = gut_brain()
    print(f"grading up to {limit} cited sentences (pass a number to change; 0 = all)\n")
    res = run(ds, limit=None if limit == 0 else limit)
    _print(res)

    u = res.usage
    print(f"\n  tokens: in={u['input_tokens']} out={u['output_tokens']} "
          f"over {u['judge_calls']} calls  |  cost: ${u['cost_usd']['total']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "step5_report.json"
    out.write_text(json.dumps(
        {"summary": res.summary(), "usage": res.usage,
         "sentences": [s.to_dict() for s in res.sentences]},
        indent=2))
    print(f"  wrote {out}")
