"""
STEP 5 — REPORT GENERATION eval.

Question: does every report claim trace to the extraction table, cite the right
paper, and preserve its numbers?

Reference truth: the extraction (combined) table — specifically each cited
paper's criteria cells + abstract. (Reference-relativity: the report is graded
against what the agent PRODUCED in the table, not against the raw papers. That
localizes synthesis errors to this step.)

Unit of evaluation: the report sentence (an approximation of an atomic claim).
Each sentence is classified:
  - CITED    -> carries one or more [n] citations; graded for groundedness against
                the union of the cited papers' table rows (3-way judge verdict).
  - UNCITED  -> a factual-looking sentence with no citation (Executive Summary /
                Discussion synthesis). Counted separately as a distinct,
                higher-severity category — we can't trace it to any row.

Metrics:
  - citation precision      = supported cited-sentences / all cited-sentences
  - uncited-synthesis count = factual sentences with no citation
  - numeric fidelity        = report numbers found in the cited rows / all numbers

Numeric integrity is folded in here (per eval_criteria): every number in a cited
sentence is checked for presence in the referenced table rows.
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


@dataclass
class SentenceResult:
    idx: int
    text: str
    citations: list[int]
    kind: str  # "cited" | "uncited"
    verdict: str | None = None
    confidence: str | None = None
    evidence_span: str | None = None
    numbers: list[str] = field(default_factory=list)
    numbers_grounded: list[str] = field(default_factory=list)
    numbers_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_table_reference(ds: QueryDataset, n: int) -> str:
    """The reference truth for citation [n]: that paper's criteria cells + abstract."""
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


@dataclass
class Step5Result:
    sentences: list[SentenceResult] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def summary(self) -> dict:
        cited = [s for s in self.sentences if s.kind == "cited"]
        uncited = [s for s in self.sentences if s.kind == "uncited"]
        graded = [s for s in cited if s.verdict is not None]
        supported = [s for s in graded if s.verdict == "SUPPORTED"]
        contradicted = [s for s in graded if s.verdict == "CONTRADICTED"]
        unsupported = [s for s in graded if s.verdict == "UNSUPPORTED"]

        all_nums = sum(len(s.numbers) for s in cited)
        grounded_nums = sum(len(s.numbers_grounded) for s in cited)

        n_graded = len(graded) or 1
        return {
            "step": "5_report_generation",
            "total_claim_sentences": len(self.sentences),
            "cited_sentences": len(cited),
            "uncited_synthesis_sentences": len(uncited),
            "graded_cited": len(graded),
            "citation_precision": round(len(supported) / n_graded, 4),
            "verdict_breakdown": {
                "SUPPORTED": len(supported),
                "UNSUPPORTED": len(unsupported),
                "CONTRADICTED": len(contradicted),
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
        ref = "\n\n".join(
            build_table_reference(ds, n) for n in rec.citations if ds.citation_row(n)
        )
        j: Judgment = judge.grade(rec.text, ref, source_label="EXTRACTION TABLE ROW(S)")
        rec.verdict = j.verdict
        rec.confidence = j.confidence
        rec.evidence_span = j.evidence_span
        # numeric integrity against the same reference
        for num in rec.numbers:
            (rec.numbers_grounded if _num_in(num, ref) else rec.numbers_missing).append(num)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(grade_one, cited))

    return Step5Result(sentences=records, usage=judge.cost_report())


def _print(res: Step5Result) -> None:
    s = res.summary()
    print("=" * 68)
    print("STEP 5 — REPORT GENERATION")
    print("=" * 68)
    print(f"  claim sentences            : {s['total_claim_sentences']}")
    print(f"  cited sentences            : {s['cited_sentences']}  (graded: {s['graded_cited']})")
    print(f"  uncited synthesis          : {s['uncited_synthesis_sentences']}")
    print(f"  citation precision         : {s['citation_precision']*100:.1f}%")
    print(f"  verdicts                   : {s['verdict_breakdown']}")
    nf = s["numeric_fidelity"]
    print(f"  numeric fidelity           : {nf*100:.1f}%  "
          f"({s['numbers_checked']-s['numbers_missing']}/{s['numbers_checked']})"
          if nf is not None else "  numeric fidelity           : n/a")

    # show a few flagged claims
    flagged = [r for r in res.sentences if r.verdict in ("UNSUPPORTED", "CONTRADICTED")]
    if flagged:
        print(f"\n  flagged claims ({len(flagged)}):")
        for r in flagged[:8]:
            print(f"    [{r.verdict}] cites {r.citations}: {r.text[:90]}...")
    missing_nums = [(r.citations, r.numbers_missing, r.text) for r in res.sentences if r.numbers_missing]
    if missing_nums:
        print(f"\n  sentences with ungrounded numbers ({len(missing_nums)}):")
        for cites, nums, text in missing_nums[:8]:
            print(f"    cites {cites} missing {nums}: {text[:80]}...")


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
