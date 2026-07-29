"""
Verification pass over flagged CONTRADICTED extraction cells.

The first-pass judge over-fires CONTRADICTED on dense full-text because it grades
against a single span. This pass fixes that with:

  1. Whole-source best-span search — the verifier must scan the ENTIRE source for a
     supporting span BEFORE it may rule contradiction.
  2. A taxonomy that separates a true contradiction from the three false-positive
     buckets we found:
        - NOT_A_FACTUAL_CLAIM : cell says "not reported in the provided metadata"
          (the agent describing its input, not asserting a fact).
        - over-specification  : source states it for a broader set (drug CLASS) and
          the claim narrows to a member -> UNSUPPORTED, not CONTRADICTED.
        - wrong-endpoint / subgroup-vs-aggregate : the span is about a different
          quantity than the claim -> not a contradiction.
  3. A 2-verifier panel (support-seeking + adversarial). CONTRADICTED stands ONLY
     if BOTH independently return CONTRADICTED; any SUPPORTED demotes it.

Run:  python -m evals.verify            # all queries' flagged contradictions
      python -m evals.verify glp1_weight_loss
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data_loader import QUERIES, load_query
from .judge import Judge

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SOURCE_CAP = 120_000  # chars of source passed to the verifier

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "NOT_A_FACTUAL_CLAIM")

_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "support_span": {"type": "string", "description": "Verbatim span supporting the claim's substantive content, or empty."},
        "contradiction_span": {"type": "string", "description": "Verbatim span asserting the OPPOSITE of the SAME quantity, or empty."},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reasoning", "support_span", "contradiction_span", "verdict"],
    "additionalProperties": False,
}

_RULES = """Taxonomy (apply strictly):
- SUPPORTED: the source states or entails the substantive claim (same quantity, \
entity, and direction). Quote support_span.
- CONTRADICTED: the source explicitly asserts the OPPOSITE of the SAME quantity the \
claim makes. Quote contradiction_span. This requires a direct conflict on the same \
measure — NOT any of the following, which are NOT contradictions:
    * the source reports the claim's figure for a broader set (e.g. the drug CLASS \
"GLP-1RAs") and the claim narrows it to one member (e.g. "semaglutide") -> UNSUPPORTED.
    * the span is about a DIFFERENT endpoint/outcome/subgroup than the claim (e.g. \
claim is the primary MACE composite, span is a heart-failure or microvascular \
secondary; or claim is a subgroup count, span is the aggregate) -> UNSUPPORTED.
- UNSUPPORTED: the source is silent on the claim, or only the above near-misses apply.
- NOT_A_FACTUAL_CLAIM: the cell is the agent stating that data was absent from its \
input (e.g. "not reported in the provided metadata", "no ... data is reported"). \
This is the agent describing its input, not asserting a fact — never a hallucination.

You MUST search the ENTIRE source for support before ruling. Judge only the source."""

_SUPPORTER = ("You verify an AI-extracted claim against a paper. Default to finding "
              "SUPPORT: search the whole source for any span that supports the claim's "
              "substantive content before considering any other verdict.\n\n" + _RULES)
_SKEPTIC = ("You adversarially verify an AI-extracted claim against a paper. Look hard "
            "for a span that DIRECTLY contradicts the same quantity — but do not count "
            "class-vs-member, different-endpoint, or subgroup-vs-aggregate near-misses "
            "as contradictions.\n\n" + _RULES)


def _verify_once(judge: Judge, system: str, claim: str, source: str) -> dict:
    user = (f"SOURCE (search all of it):\n\"\"\"\n{source[:SOURCE_CAP]}\n\"\"\"\n\n"
            f"CLAIM:\n\"\"\"\n{claim}\n\"\"\"\n\nClassify per the taxonomy.")
    return judge.structured(system, user, _SCHEMA, max_tokens=1200)


import re as _re
_ABSENCE = _re.compile(
    r"(^\s*no\b[^.]{0,70}\b(data|outcomes?|information|events?|results?|values?|numbers?|metrics?)\b"
    r"|\bnot\s+(reported|available|provided|specified|detailed|mentioned|given|addressed|quantified)"
    r"|\b(does|do|did)\s+not\s+(report|provide|specify|mention|detail|quantify|give)"
    r"|\bno\s+specific\b)",
    _re.I)


def _is_absence_claim(claim: str) -> bool:
    """The cell asserts that data is ABSENT rather than asserting a positive fact."""
    head = claim.strip()[:140]
    return bool(_ABSENCE.search(head))


def verify_finding(judge: Judge, claim: str, source: str) -> dict:
    sup = _verify_once(judge, _SUPPORTER, claim, source)
    skep = _verify_once(judge, _SKEPTIC, claim, source)
    v1, v2 = sup.get("verdict"), skep.get("verdict")
    # consensus
    if v1 == "CONTRADICTED" and v2 == "CONTRADICTED":
        # separate a genuine asserted-fact contradiction from a "false absence"
        # (agent wrongly said data is missing) — the latter is under-extraction.
        final = "FALSE_ABSENCE_under_extraction" if _is_absence_claim(claim) else "CONFIRMED_CONTRADICTION"
    elif "SUPPORTED" in (v1, v2):
        final = "DEMOTED_actually_supported"
    elif "NOT_A_FACTUAL_CLAIM" in (v1, v2):
        final = "DEMOTED_not_a_claim"
    elif v1 == v2 == "UNSUPPORTED":
        final = "DEMOTED_unsupported"
    else:
        final = "UNCERTAIN_panel_split"
    return {
        "final": final,
        "supporter_verdict": v1,
        "skeptic_verdict": v2,
        "support_span": sup.get("support_span") or skep.get("support_span") or "",
        "contradiction_span": skep.get("contradiction_span") or sup.get("contradiction_span") or "",
        "reasoning": skep.get("reasoning", "")[:300],
    }


def run(queries: list[str] | None = None, workers: int = 6) -> dict:
    queries = queries or list(QUERIES.keys())
    judge = Judge()
    findings = []
    for qname in queries:
        ds = load_query(qname)
        path = RESULTS_DIR / qname / "step4_extraction.json"
        if not path.exists():
            continue
        cells = json.loads(path.read_text())["cells"]
        for c in cells:
            if c.get("verdict") != "CONTRADICTED":
                continue
            row = ds.extracted[c["paper_n"] - 1]
            claim = (row.get(c["criterion"], "") or "").strip()
            source = ds.reference_text(row).best
            findings.append({"query": qname, "paper_n": c["paper_n"],
                             "paper_title": c["paper_title"], "criterion": c["criterion"],
                             "grounding_level": c["grounding_level"],
                             "claim": claim, "_source": source})

    def do(f):
        v = verify_finding(judge, f["claim"], f["_source"])
        f.pop("_source")
        f.update(v)
        return f

    with ThreadPoolExecutor(max_workers=workers) as ex:
        verified = list(ex.map(do, findings))

    counts: dict[str, int] = {}
    per_query: dict[str, dict] = {}
    for f in verified:
        counts[f["final"]] = counts.get(f["final"], 0) + 1
        pq = per_query.setdefault(f["query"], {"flagged": 0, "confirmed": 0})
        pq["flagged"] += 1
        pq["confirmed"] += int(f["final"] == "CONFIRMED_CONTRADICTION")

    return {"n_flagged": len(verified), "outcome_counts": counts,
            "per_query": per_query, "usage": judge.cost_report(),
            "findings": verified}


def _classify(orig_verdict: str, panel: dict, claim: str) -> str:
    """Map a flagged cell's panel outcome to a verified extraction state.

    orig_verdict: 'CONTRADICTED' or 'UNSUPPORTED' (the latter = extrinsic-vs-fulltext).
    Returns: grounded | contradiction | extrinsic | false_absence | not_a_claim | uncertain
    """
    v1, v2 = panel["supporter_verdict"], panel["skeptic_verdict"]
    if "SUPPORTED" in (v1, v2):
        return "grounded"                       # first pass picked the wrong span
    if "NOT_A_FACTUAL_CLAIM" in (v1, v2):
        return "not_a_claim"
    if v1 == "CONTRADICTED" and v2 == "CONTRADICTED":
        return "false_absence" if _is_absence_claim(claim) else "contradiction"
    if v1 == v2 == "UNSUPPORTED":
        # genuinely not in the paper. For an absence-claim cell that's under-extraction;
        # otherwise the claim adds info absent from the source = extrinsic hallucination.
        return "false_absence" if _is_absence_claim(claim) else "extrinsic"
    return "uncertain"


def fold_into_step4(query: str, judge: Judge, workers: int = 6, cache: dict | None = None) -> dict:
    """Verify a query's flagged cells and return verified counts + per-cell states.
    `cache` maps (paper_n, criterion) -> panel dict to reuse prior verifications."""
    ds = load_query(query)
    path = RESULTS_DIR / query / "step4_extraction.json"
    data = json.loads(path.read_text())
    cells = data["cells"]
    cache = cache or {}

    flagged = []
    for c in cells:
        is_contra = c.get("verdict") == "CONTRADICTED"
        is_extrinsic = c.get("verdict") == "UNSUPPORTED" and c.get("grounding_level") == "fulltext_downloaded"
        if is_contra or is_extrinsic:
            flagged.append(c)

    def do(c):
        key = (c["paper_n"], c["criterion"])
        row = ds.extracted[c["paper_n"] - 1]
        claim = (row.get(c["criterion"], "") or "").strip()
        panel = cache.get(key) or verify_finding(judge, claim, ds.reference_text(row).best)
        c["verified_state"] = _classify(c["verdict"], panel, claim)
        c["verified_panel"] = {k: panel[k] for k in ("supporter_verdict", "skeptic_verdict", "contradiction_span")}
        return c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, flagged))

    # verified accounting
    orig = data["summary"]["verdict_breakdown"]
    grounded0 = orig["SUPPORTED_grounded"]
    demoted_grounded = sum(1 for c in flagged if c["verified_state"] == "grounded")
    contradictions = sum(1 for c in flagged if c["verified_state"] == "contradiction")
    extrinsic = sum(1 for c in flagged if c["verified_state"] == "extrinsic")
    false_absence = sum(1 for c in flagged if c["verified_state"] == "false_absence")
    uncertain = sum(1 for c in flagged if c["verified_state"] in ("uncertain", "not_a_claim"))

    grounded = grounded0 + demoted_grounded
    adjudicable = grounded + contradictions + extrinsic
    verified = {
        "flagged_rechecked": len(flagged),
        "grounded": grounded,
        "confirmed_contradictions": contradictions,
        "confirmed_extrinsic": extrinsic,
        "false_absence_under_extraction": false_absence,
        "uncertain_or_not_claim": uncertain,
        "adjudicable": adjudicable,
        "hallucination_rate": round((contradictions + extrinsic) / adjudicable, 4) if adjudicable else None,
        "grounded_rate": round(grounded / adjudicable, 4) if adjudicable else None,
        "first_pass_flagged": len([c for c in cells if c.get("verdict") in ("CONTRADICTED",)]) ,
    }
    data["summary"]["verified"] = verified
    path.write_text(json.dumps(data, indent=2))
    return verified


def _classify_report(panel: dict, claim: str) -> str:
    v1, v2 = panel["supporter_verdict"], panel["skeptic_verdict"]
    if "SUPPORTED" in (v1, v2):
        return "supported"          # first pass picked the wrong cited row
    if "NOT_A_FACTUAL_CLAIM" in (v1, v2):
        return "not_a_claim"
    if v1 == "CONTRADICTED" and v2 == "CONTRADICTED":
        return "contradiction"
    if v1 == v2 == "UNSUPPORTED":
        return "unsupported"        # genuine citation failure (cited paper doesn't support it)
    return "uncertain"


def fold_into_step5(query: str, judge: Judge, workers: int = 6) -> dict:
    """Verify a query's non-SUPPORTED cited report sentences against their cited
    table rows; recompute citation precision + report contradictions as verified."""
    from .step5_report import build_table_reference
    ds = load_query(query)
    if ds.citation_confidence != "high":
        return {"skipped": "citation mapping unreliable"}
    path = RESULTS_DIR / query / "step5_report.json"
    data = json.loads(path.read_text())
    sents = data["sentences"]

    graded = [s for s in sents if s["kind"] == "cited" and s.get("verdict")]
    flagged = [s for s in graded if s["verdict"] in ("UNSUPPORTED", "CONTRADICTED")]

    def do(s):
        ref = "\n\n".join(build_table_reference(ds, n) for n in s["citations"] if ds.citation_row(n))
        panel = verify_finding(judge, s["text"], ref)
        s["verified_state"] = _classify_report(panel, s["text"])
        s["verified_panel"] = {k: panel[k] for k in ("supporter_verdict", "skeptic_verdict")}
        return s

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, flagged))

    orig_supported = sum(1 for s in graded if s["verdict"] == "SUPPORTED")
    regained = sum(1 for s in flagged if s["verified_state"] == "supported")
    contradictions = sum(1 for s in flagged if s["verified_state"] == "contradiction")
    unsupported = sum(1 for s in flagged if s["verified_state"] == "unsupported")
    supported = orig_supported + regained
    n = len(graded) or 1
    verified = {
        "graded_cited": len(graded),
        "flagged_rechecked": len(flagged),
        "supported": supported,
        "report_contradictions": contradictions,
        "unsupported": unsupported,
        "citation_precision": round(supported / n, 4),
        "first_pass_citation_precision": data["summary"].get("citation_precision"),
        "first_pass_contradictions": data["summary"].get("verdict_breakdown", {}).get("CONTRADICTED"),
    }
    data["summary"]["verified"] = verified
    path.write_text(json.dumps(data, indent=2))
    return verified


def _print(res: dict) -> None:
    print("=" * 70)
    print("VERIFICATION PASS — flagged CONTRADICTED extraction cells")
    print("=" * 70)
    print(f"  flagged contradictions re-checked: {res['n_flagged']}")
    print(f"  panel outcomes: {res['outcome_counts']}")
    print(f"\n  per query (confirmed / flagged):")
    for q, p in res["per_query"].items():
        print(f"    {q:22s}: {p['confirmed']} / {p['flagged']}")
    confirmed = [f for f in res["findings"] if f["final"] == "CONFIRMED_CONTRADICTION"]
    print(f"\n  CONFIRMED contradictions ({len(confirmed)}):")
    for f in confirmed:
        print(f"    [{f['query']}/{f['paper_n']}] {f['criterion']}: {f['paper_title'][:45]}")
        print(f"        claim: {f['claim'][:110]}")
        print(f"        source contradicts: {f['contradiction_span'][:110]}")
    u = res["usage"]
    print(f"\n  cost: ${u['cost_usd']['total']} ({u['judge_calls']} calls)")


def rescore(queries: list[str] | None = None) -> None:
    """Fold verification into stages 4 and 5 for each query, then regenerate the
    scorecard (results/combined_scores.json) from the verified numbers.
    Run this AFTER `python -m evals.run_all` has produced the first-pass results."""
    from . import score
    from .run_all import build_board, _print_board
    queries = queries or list(QUERIES.keys())
    judge = Judge()
    for q in queries:
        fold_into_step4(q, judge)
        fold_into_step5(q, judge)

    per_query = {}
    for name in QUERIES:
        dpath = RESULTS_DIR / name / "dashboard.json"
        if not dpath.exists():
            continue
        d = json.loads(dpath.read_text())
        d["summaries"]["step4_extraction"] = json.loads((RESULTS_DIR / name / "step4_extraction.json").read_text())["summary"]
        d["summaries"]["step5_report"] = json.loads((RESULTS_DIR / name / "step5_report.json").read_text())["summary"]
        ds = load_query(name)
        d["citation_confidence"] = ds.citation_confidence
        d["scorecard"] = score.profile(d["summaries"], ds.citation_confidence)
        dpath.write_text(json.dumps(d, indent=2))
        per_query[name] = d

    board = build_board(per_query)
    (RESULTS_DIR / "combined_scores.json").write_text(json.dumps(board, indent=2))
    _print_board(board)
    print(f"\n  verification cost: ${judge.cost_report()['cost_usd']['total']}")


if __name__ == "__main__":
    import sys
    rescore(sys.argv[1:] or None)
