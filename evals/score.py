"""
Scorecard builder.

We deliberately do NOT produce a single weighted average across steps — the steps
measure different constructs (metadata accuracy vs criteria coverage vs extraction
grounding vs citation precision), over different denominators, so averaging them is
a category error and it launders the real hallucination signal into a bland number.

Instead each query gets a two-tier scorecard:

  TIER A — Integrity gates (comparable, expected near-perfect):
    retrieval metadata accuracy, consolidation provenance, criteria coverage.

  TIER B — Hallucination profile (the actual deliverable), as raw counts + rates
    WITH their denominators and (for extraction) the full-text coverage that makes
    the rate interpretable:
      extraction: contradictions, extrinsic-vs-fulltext, grounded, hallucination rate
      report:     citation precision, report contradictions, numeric fidelity

The one cross-step summary we allow is an extraction-level hallucination RATE on a
single unit (cells) — never pooled with the report (different unit).
"""

from __future__ import annotations


def _rate(n, d):
    return round(n / d, 4) if d else None


def profile(summaries: dict, citation_confidence: str = "high") -> dict:
    s1 = summaries.get("step1_retrieval", {})
    s2 = summaries.get("step2_consolidation", {})
    s3 = summaries.get("step3_criteria", {})
    s4 = summaries.get("step4_extraction", {})
    s5 = summaries.get("step5_report", {})

    # Tier A — integrity gates
    tot_match = sum(r.get("title_match", 0) for r in s1.get("per_source", []))
    tot_resolved = sum(r.get("resolved", 0) for r in s1.get("per_source", []))
    integrity = {
        "retrieval_metadata_accuracy": _rate(tot_match, tot_resolved),
        "consolidation_provenance": s2.get("provenance_rate"),
        "consolidation_fabricated": s2.get("fabricated_count"),
        "criteria_coverage": s3.get("coverage_rate"),
        "criteria_spurious_columns": s3.get("spurious_count"),
        "criteria_uncovered": s3.get("uncovered_dimensions", []),
    }

    # Tier B — hallucination profile
    vb = s4.get("verdict_breakdown", {})
    grounded = vb.get("SUPPORTED_grounded", 0)
    intrinsic = vb.get("CONTRADICTED_intrinsic_hallucination", 0)
    extrinsic = s4.get("reliable_signals", {}).get("extrinsic_hallucinations_vs_fulltext", 0)
    adjudicable = grounded + intrinsic + extrinsic
    by_level = s4.get("groundedness_by_source_level", {})
    graded_total = sum(v.get("graded", 0) for v in by_level.values())
    ft_graded = by_level.get("fulltext_downloaded", {}).get("graded", 0)

    v = s4.get("verified")
    if v:  # verified numbers (2-panel best-span re-check) supersede the first pass
        extraction = {
            "verified": True,
            "adjudicable_cells": v["adjudicable"],
            "grounded": v["grounded"],
            "contradictions_intrinsic": v["confirmed_contradictions"],
            "extrinsic_vs_fulltext": v["confirmed_extrinsic"],
            "hallucination_rate": v["hallucination_rate"],
            "grounded_rate": v["grounded_rate"],
            "false_absence_under_extraction": v["false_absence_under_extraction"],
            "unverifiable_fragment_only": s4.get("unverifiable_needs_fulltext"),
            "fulltext_coverage": _rate(ft_graded, graded_total),
        }
    else:
        extraction = {
            "verified": False,
            "adjudicable_cells": adjudicable,
            "grounded": grounded,
            "contradictions_intrinsic": intrinsic,
            "extrinsic_vs_fulltext": extrinsic,
            "hallucination_rate_first_pass_upper_bound": _rate(intrinsic + extrinsic, adjudicable),
            "hallucination_rate": None,  # not verified
            "grounded_rate": _rate(grounded, adjudicable),
            "unverifiable_fragment_only": s4.get("unverifiable_needs_fulltext"),
            "false_na_under_extraction": s4.get("false_na_under_extraction"),
            "fulltext_coverage": _rate(ft_graded, graded_total),
        }

    rv = s5.get("verified")
    report = {
        "verified": bool(rv),
        "citation_confidence": citation_confidence,
        "cited_sentences": s5.get("cited_sentences"),
        "citation_precision": (rv["citation_precision"] if rv else s5.get("citation_precision"))
                              if citation_confidence == "high" else None,
        "report_contradictions": (rv["report_contradictions"] if rv
                                  else s5.get("verdict_breakdown", {}).get("CONTRADICTED")),
        "numeric_fidelity": s5.get("numeric_fidelity"),
        "uncited_synthesis": s5.get("uncited_synthesis_sentences"),
    }
    if citation_confidence != "high":
        report["note"] = "citation mapping unreliable — S5 metrics not scored"

    return {"integrity_gates": integrity, "hallucination_profile":
            {"extraction": extraction, "report": report}}
