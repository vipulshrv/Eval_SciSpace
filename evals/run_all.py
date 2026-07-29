"""
Orchestrator: run every step eval across one or all queries, store per-query
results separately, and produce a combined score.

  results/<query>/step{1..5}_*.json   per-step detail
  results/<query>/dashboard.json      per-query summaries + scores + cost
  results/combined_scores.json        per-query + overall scores + total cost

Usage:
  python -m evals.run_all                       # all 3 queries, full
  python -m evals.run_all --query cancer        # one query
  python -m evals.run_all --limit 10            # cap steps 4 & 5 (cheap smoke run)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

from .data_loader import QUERIES, load_query
from . import (step1_retrieval, step2_consolidation, step3_criteria,
               step4_extraction, step5_report, score)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def run_query(name: str, limit: int | None) -> dict:
    ds = load_query(name)
    outdir = RESULTS_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*68}\n# {name}  ({ds.citation_mode} citations)\n{'#'*68}")

    summaries: dict = {}
    usage: dict = {}
    cost = 0.0

    r1 = step1_retrieval.run(ds); step1_retrieval._print(r1)
    summaries["step1_retrieval"] = r1["summary"]
    (outdir / "step1_retrieval.json").write_text(json.dumps(r1, indent=2))

    r2 = step2_consolidation.run(ds); print(); step2_consolidation._print(r2)
    summaries["step2_consolidation"] = r2.summary()
    (outdir / "step2_consolidation.json").write_text(json.dumps(r2.summary(), indent=2))

    r3 = step3_criteria.run(ds); print(); step3_criteria._print(r3)
    summaries["step3_criteria"] = r3["summary"]
    usage["step3_criteria"] = r3["usage"]; cost += r3["usage"]["cost_usd"]["total"]
    (outdir / "step3_criteria.json").write_text(json.dumps(r3, indent=2))

    r4 = step4_extraction.run(ds, limit=limit); print(); step4_extraction._print(r4)
    summaries["step4_extraction"] = r4.summary()
    usage["step4_extraction"] = r4.usage; cost += r4.usage["cost_usd"]["total"]
    (outdir / "step4_extraction.json").write_text(json.dumps(
        {"summary": r4.summary(), "usage": r4.usage, "cells": [c.to_dict() for c in r4.cells]}, indent=2))

    r5 = step5_report.run(ds, limit=limit); print(); step5_report._print(r5)
    summaries["step5_report"] = r5.summary()
    usage["step5_report"] = r5.usage; cost += r5.usage["cost_usd"]["total"]
    (outdir / "step5_report.json").write_text(json.dumps(
        {"summary": r5.summary(), "usage": r5.usage, "sentences": [s.to_dict() for s in r5.sentences]}, indent=2))

    card = score.profile(summaries, ds.citation_confidence)
    dashboard = {
        "query": name,
        "citation_mode": ds.citation_mode,
        "citation_confidence": ds.citation_confidence,
        "scorecard": card,
        "summaries": summaries,
        "usage_by_step": usage,
        "total_cost_usd": round(cost, 4),
    }
    (outdir / "dashboard.json").write_text(json.dumps(dashboard, indent=2))
    ex = card["hallucination_profile"]["extraction"]
    print(f"\n  >> {name}: extraction hallucination rate {ex['hallucination_rate']} "
          f"({ex['contradictions_intrinsic']} contradictions + {ex['extrinsic_vs_fulltext']} extrinsic "
          f"/ {ex['adjudicable_cells']} adjudicable)  |  cost ${round(cost,4)}")
    return dashboard


def main() -> None:
    q = _arg("--query")
    limit = _arg("--limit")
    limit = int(limit) if limit else None
    names = [q] if q else list(QUERIES.keys())

    RESULTS_DIR.mkdir(exist_ok=True)
    per_query = {name: run_query(name, limit) for name in names}
    board = build_board(per_query)
    (RESULTS_DIR / "combined_scores.json").write_text(json.dumps(board, indent=2))
    _print_board(board)


def build_board(per_query: dict) -> dict:
    board = {"per_query": {}, "total_cost_usd": 0.0}
    for name, d in per_query.items():
        board["per_query"][name] = {
            "citation_confidence": d.get("citation_confidence"),
            "scorecard": d["scorecard"],
            "cost_usd": d["total_cost_usd"],
        }
        board["total_cost_usd"] += d["total_cost_usd"]
    board["total_cost_usd"] = round(board["total_cost_usd"], 4)
    board["note"] = ("No single composite score: steps measure different constructs "
                     "over different denominators. Read integrity gates and the "
                     "hallucination profile separately.")
    return board


def _pct(v):
    return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "-"


def _print_board(board: dict) -> None:
    print(f"\n{'='*82}\nSCORECARD — per query (no composite; two tiers)\n{'='*82}")
    print("\nTIER A · Integrity gates")
    print(f"  {'query':22s}{'meta-acc':>10s}{'provenance':>12s}{'criteria-cov':>14s}{'spurious':>10s}")
    for name, r in board["per_query"].items():
        g = r["scorecard"]["integrity_gates"]
        print(f"  {name:22s}{_pct(g['retrieval_metadata_accuracy']):>10s}"
              f"{_pct(g['consolidation_provenance']):>12s}{_pct(g['criteria_coverage']):>14s}"
              f"{str(g['criteria_spurious_columns']):>10s}")

    print("\nTIER B · Hallucination profile   (VERIFIED — 2-panel best-span re-check, extraction + report)")
    print(f"  {'query':22s}{'extr-halluc':>12s}{'(con+ext/adj)':>15s}{'false-abs':>10s}"
          f"{'cite-prec':>11s}{'rep-contra':>11s}{'num-fidel':>10s}")
    for name, r in board["per_query"].items():
        e = r["scorecard"]["hallucination_profile"]["extraction"]
        rp = r["scorecard"]["hallucination_profile"]["report"]
        hr = e.get("hallucination_rate")
        hr_s = _pct(hr) + ("" if e.get("verified") else "?")
        detail = f"({e['contradictions_intrinsic']}+{e['extrinsic_vs_fulltext']}/{e['adjudicable_cells']})"
        fa = e.get("false_absence_under_extraction")
        cite = _pct(rp["citation_precision"]) if rp["citation_precision"] is not None else "excl"
        print(f"  {name:22s}{hr_s:>12s}{detail:>15s}{str(fa) if fa is not None else '-':>10s}"
              f"{cite:>11s}{str(rp['report_contradictions']):>11s}{_pct(rp['numeric_fidelity']):>10s}")

    print(f"\n  total cost ${board['total_cost_usd']}")
    print(f"  note: {board['note']}")
    print(f"  wrote {RESULTS_DIR / 'combined_scores.json'}")


if __name__ == "__main__":
    main()
