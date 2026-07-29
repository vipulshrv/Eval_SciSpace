"""
STEP 3 — CRITERIA IDENTIFICATION eval.

Question: do the agent's extracted criteria columns faithfully reflect the query?

Reference truth: the user query.
Two checks (one LLM-judge call):
  - Coverage: every dimension explicitly requested in the query is present as a
    column.
  - No invented criteria: every column is grounded in the query (a column with no
    basis in the query is a spurious fabrication surface downstream).
"""

from __future__ import annotations

import json
from pathlib import Path

from .data_loader import QueryDataset, gut_brain
from .judge import Judge

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_SCHEMA = {
    "type": "object",
    "properties": {
        "requested_dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "covered_by_column": {"type": "string"},  # "" if uncovered
                    "covered": {"type": "boolean"},
                },
                "required": ["dimension", "covered_by_column", "covered"],
                "additionalProperties": False,
            },
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "grounded_in_query": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": ["column", "grounded_in_query", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["requested_dimensions", "columns"],
    "additionalProperties": False,
}

_SYSTEM = """You audit whether an AI report-writer identified the right extraction \
criteria from a user's query. You are given the QUERY and the COLUMNS the agent \
created. Do two things:
1. Enumerate the distinct dimensions the query explicitly asks to analyze, and for \
each say whether some column covers it.
2. For each column, say whether it is grounded in the query (explicitly requested \
or a direct sub-aspect) or invented with no basis in the query.
Be strict but fair: a column that operationalizes an explicit request is grounded."""


def run(ds: QueryDataset, judge: Judge | None = None) -> dict:
    judge = judge or Judge()
    user = (f"QUERY:\n{ds.query}\n\n"
            f"COLUMNS the agent created:\n" + "\n".join(f"- {c}" for c in ds.criteria_columns))
    data = judge.structured(_SYSTEM, user, _SCHEMA)

    dims = data.get("requested_dimensions", [])
    cols = data.get("columns", [])
    covered = [d for d in dims if d.get("covered")]
    spurious = [c for c in cols if not c.get("grounded_in_query")]

    summary = {
        "step": "3_criteria_identification",
        "query_dimensions": len(dims),
        "dimensions_covered": len(covered),
        "coverage_rate": round(len(covered) / len(dims), 4) if dims else None,
        "uncovered_dimensions": [d["dimension"] for d in dims if not d.get("covered")],
        "columns_total": len(cols),
        "spurious_columns": [c["column"] for c in spurious],
        "spurious_count": len(spurious),
    }
    return {"summary": summary, "detail": data, "usage": judge.cost_report()}


def _print(res: dict) -> None:
    s = res["summary"]
    print("=" * 68)
    print("STEP 3 — CRITERIA IDENTIFICATION")
    print("=" * 68)
    cov = s["coverage_rate"]
    print(f"  query dimensions           : {s['query_dimensions']}")
    print(f"  coverage                   : {s['dimensions_covered']}/{s['query_dimensions']}"
          f"  ({cov*100:.0f}%)" if cov is not None else "  coverage: n/a")
    if s["uncovered_dimensions"]:
        print(f"  UNCOVERED dimensions       : {s['uncovered_dimensions']}")
    print(f"  columns                    : {s['columns_total']}")
    print(f"  spurious (invented) columns: {s['spurious_count']} {s['spurious_columns']}")
    for d in res["detail"].get("requested_dimensions", []):
        mark = "✓" if d.get("covered") else "✗"
        print(f"    {mark} {d['dimension']}  ->  {d.get('covered_by_column') or '(none)'}")


if __name__ == "__main__":
    res = run(gut_brain())
    _print(res)
    u = res["usage"]
    print(f"\n  cost: ${u['cost_usd']['total']} ({u['judge_calls']} call)")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "step3_criteria.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {out}")
