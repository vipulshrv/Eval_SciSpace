"""
STEP 4 — DATA EXTRACTION eval  (core hallucination surface).

Question: is each extracted table cell grounded in that paper's source text?

Reference truth (reference-relativity): the paper's full-text "Relevant Excerpt"
where present, else its abstract, else a snippet. We grade against what the agent
plausibly had — and we LABEL each verdict with the grounding level so an
"unsupported" call made against an abstract-only source is treated as
context-limited, not a confirmed hallucination.

Grid: 30 extracted papers x 3 criteria columns = 90 cells.

Two cell modes:
  - CONTENT cell -> grade groundedness (3-way):
        SUPPORTED     = grounded
        CONTRADICTED  = INTRINSIC hallucination (conflicts with source) — the hard
                        signal; a contradiction of even the abstract can't be
                        excused by missing full text.
        UNSUPPORTED   = EXTRINSIC (adds info not in source). If the source was only
                        an abstract/snippet, this is flagged context_limited (the
                        full text the agent had may support it).
  - NO-DATA cell ("not reported"/"not addressed") -> false-N/A check: does the
        source actually contain data for this criterion? If yes -> under-extraction.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data_loader import QueryDataset, gut_brain
from .judge import Judge

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_NODATA_RE = re.compile(
    r"^\s*(not\s+(reported|addressed|available|specified|detailed)|n/?a)\b", re.I)


def _is_nodata(cell: str) -> bool:
    """A cell that only declares absence (no substantive extraction)."""
    c = (cell or "").strip()
    if not c:
        return True
    # short cell dominated by a 'not reported' phrase
    return bool(_NODATA_RE.match(c)) and len(c) < 60


@dataclass
class CellResult:
    paper_n: int
    paper_title: str
    criterion: str
    grounding_level: str          # fulltext_excerpt | abstract_only | snippet_only | none
    mode: str                     # "content" | "nodata"
    verdict: str | None = None    # SUPPORTED / UNSUPPORTED / CONTRADICTED (content)
    confidence: str | None = None
    evidence_span: str | None = None
    context_limited: bool = False # non-supported verdict against abstract/snippet only
    false_na: bool | None = None  # nodata cells: True = source had data (under-extraction)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Step4Result:
    cells: list[CellResult] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def summary(self) -> dict:
        content = [c for c in self.cells if c.mode == "content"]
        graded = [c for c in content if c.verdict is not None]
        supported = [c for c in graded if c.verdict == "SUPPORTED"]
        contradicted = [c for c in graded if c.verdict == "CONTRADICTED"]
        unsupported = [c for c in graded if c.verdict == "UNSUPPORTED"]
        # UNSUPPORTED against downloaded full text = likely extrinsic hallucination
        # (PDF-extraction caveat); against a fragment = unverifiable, needs full text.
        extrinsic_vs_fulltext = [c for c in unsupported if not c.context_limited]
        unverifiable = [c for c in unsupported if c.context_limited]

        nodata = [c for c in self.cells if c.mode == "nodata"]
        false_na = [c for c in nodata if c.false_na]

        by_level: dict[str, dict] = {}
        for c in graded:
            b = by_level.setdefault(c.grounding_level, {"n": 0, "supported": 0})
            b["n"] += 1
            b["supported"] += int(c.verdict == "SUPPORTED")

        n = len(graded) or 1
        return {
            "step": "4_data_extraction",
            "content_cells_graded": len(graded),
            # groundedness rate is a LOWER BOUND: we only hold a fragment (abstract or
            # excerpt), so cells grounded only in the un-held full text score UNSUPPORTED.
            "groundedness_rate_lower_bound": round(len(supported) / n, 4),
            "verdict_breakdown": {
                "SUPPORTED_grounded": len(supported),
                "CONTRADICTED_intrinsic_hallucination": len(contradicted),
                "UNSUPPORTED": len(unsupported),
            },
            "reliable_signals": {
                "grounded": len(supported),
                "intrinsic_hallucinations": len(contradicted),
                "extrinsic_hallucinations_vs_fulltext": len(extrinsic_vs_fulltext),
            },
            "unverifiable_needs_fulltext": len(unverifiable),
            "groundedness_by_source_level": {
                lvl: {"graded": b["n"], "supported": b["supported"],
                      "rate": round(b["supported"] / b["n"], 3)}
                for lvl, b in sorted(by_level.items())
            },
            "nodata_cells": len(nodata),
            "false_na_under_extraction": len(false_na),
        }


def run(ds: QueryDataset, limit: int | None = None, workers: int = 6,
        judge: Judge | None = None) -> Step4Result:
    judge = judge or Judge()

    tasks: list[CellResult] = []
    for i, row in enumerate(ds.extracted, start=1):
        ref = ds.reference_text(row)
        for col in ds.criteria_columns:
            cell = (row.get(col, "") or "").strip()
            mode = "nodata" if _is_nodata(cell) else "content"
            tasks.append(CellResult(
                paper_n=i,
                paper_title=(row.get("Paper Title", "") or "")[:70],
                criterion=col,
                grounding_level=ref.grounding_level,
                mode=mode,
            ))

    if limit is not None:
        tasks = tasks[:limit]

    rows_by_n = {i: row for i, row in enumerate(ds.extracted, start=1)}

    def grade_one(t: CellResult) -> CellResult:
        row = rows_by_n[t.paper_n]
        ref = ds.reference_text(row)
        source = ref.best
        cell = (row.get(t.criterion, "") or "").strip()

        if t.grounding_level == "none" or not source:
            t.note = "no source text available (cannot grade)"
            return t

        if t.mode == "nodata":
            has_info, reasoning = judge.source_has_topic(t.criterion, source)
            t.false_na = has_info
            t.note = reasoning[:160]
            return t

        j = judge.grade(cell, source, source_label=f"PAPER SOURCE ({t.grounding_level})")
        t.verdict = j.verdict
        t.confidence = j.confidence
        t.evidence_span = j.evidence_span
        # If we only hold a FRAGMENT (abstract / excerpt / snippet), an UNSUPPORTED
        # verdict can't be confirmed as a hallucination — the full paper the agent
        # extracted from may support it. Only when we have the DOWNLOADED full text
        # is UNSUPPORTED a confirmed extrinsic hallucination.
        if j.verdict == "UNSUPPORTED" and t.grounding_level != "fulltext_downloaded":
            t.context_limited = True
        return t

    with ThreadPoolExecutor(max_workers=workers) as ex:
        graded = list(ex.map(grade_one, tasks))

    return Step4Result(cells=graded, usage=judge.cost_report())


def _print(res: Step4Result) -> None:
    s = res.summary()
    print("=" * 68)
    print("STEP 4 — DATA EXTRACTION")
    print("=" * 68)
    print(f"  content cells graded       : {s['content_cells_graded']}")
    print(f"  groundedness rate (LOWER BOUND) : {s['groundedness_rate_lower_bound']*100:.1f}%")
    print(f"  verdicts                   : {s['verdict_breakdown']}")
    rs = s["reliable_signals"]
    print(f"  reliable signals           : grounded={rs['grounded']}, "
          f"intrinsic={rs['intrinsic_hallucinations']}, "
          f"extrinsic-vs-fulltext={rs['extrinsic_hallucinations_vs_fulltext']}")
    print(f"  unverifiable (fragment only): {s['unverifiable_needs_fulltext']}  "
          f"(UNSUPPORTED vs a fragment — needs downloaded paper)")
    print(f"  groundedness by source level (SUPPORTED rate):")
    for lvl, b in s["groundedness_by_source_level"].items():
        print(f"      {lvl:18s}: {b['supported']}/{b['graded']}  ({b['rate']*100:.0f}%)")
    print(f"  no-data cells              : {s['nodata_cells']}  "
          f"(false-N/A / under-extraction: {s['false_na_under_extraction']})")

    hard = [c for c in res.cells if c.verdict == "CONTRADICTED"]
    if hard:
        print(f"\n  INTRINSIC hallucinations — contradicts the source ({len(hard)}):")
        for c in hard[:8]:
            print(f"    [{c.paper_n}] {c.criterion}: {c.paper_title[:50]}")
    fna = [c for c in res.cells if c.false_na]
    if fna:
        print(f"\n  false-N/A (under-extraction) ({len(fna)}):")
        for c in fna[:8]:
            print(f"    [{c.paper_n}] {c.criterion}: {c.note[:70]}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ds = gut_brain()
    print(f"grading up to {limit} cells (0 = all 90)\n")
    res = run(ds, limit=None if limit == 0 else limit)
    _print(res)

    u = res.usage
    print(f"\n  tokens: in={u['input_tokens']} out={u['output_tokens']} "
          f"over {u['judge_calls']} calls  |  cost: ${u['cost_usd']['total']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "step4_extraction.json"
    out.write_text(json.dumps(
        {"summary": res.summary(), "usage": res.usage,
         "cells": [c.to_dict() for c in res.cells]}, indent=2))
    print(f"  wrote {out}")
