"""
STEP 2 — PAPER CONSOLIDATION eval.

Question: did papers survive the raw -> consolidated merge without fabrication,
metadata drift, or double-counting?

Reference truth: the source retrieval CSVs. This step is DETERMINISTIC — no LLM.
Three checks:

  1. Provenance / no fabrication — every consolidated row must trace back to >=1
     source row (DOI, else fuzzy title). A row with no ancestor is a fabricated paper.

  2. Merge consistency — when a paper matched a source, does the consolidated row's
     metadata (DOI, year) agree with the source, or did the merge introduce drift?

  3. Dedup correctness — are the consolidated rows unique, or are near-duplicates
     (same DOI, or near-identical title) double-counted? Inflated counts propagate
     into the report's "N studies" claim.

Also logged (scoped to Task 2, not scored): the raw -> consolidated drop rate,
i.e. retrieval recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .data_loader import (
    QueryDataset,
    SourceIndex,
    gut_brain,
    norm_doi,
    norm_title,
)


@dataclass
class Step2Result:
    n_combined: int = 0
    n_raw_source_rows: int = 0
    n_unique_source_papers: int = 0

    # provenance
    provenance_by_doi: int = 0
    provenance_by_title: int = 0
    provenance_fuzzy: int = 0
    fabricated: list[str] = field(default_factory=list)  # untraceable combined rows

    # merge consistency
    merge_doi_drift: list[dict] = field(default_factory=list)
    merge_year_drift: list[dict] = field(default_factory=list)

    # dedup
    duplicate_groups: list[list[str]] = field(default_factory=list)

    def summary(self) -> dict:
        n = self.n_combined or 1
        traced = self.provenance_by_doi + self.provenance_by_title + self.provenance_fuzzy
        return {
            "step": "2_consolidation",
            "n_combined": self.n_combined,
            "provenance_rate": round(traced / n, 4),
            "provenance_breakdown": {
                "by_doi": self.provenance_by_doi,
                "by_title_exact": self.provenance_by_title,
                "by_title_fuzzy": self.provenance_fuzzy,
            },
            "fabricated_count": len(self.fabricated),
            "fabricated_titles": self.fabricated,
            "merge_doi_drift_count": len(self.merge_doi_drift),
            "merge_year_drift_count": len(self.merge_year_drift),
            "duplicate_group_count": len(self.duplicate_groups),
            "duplicate_extra_rows": sum(len(g) - 1 for g in self.duplicate_groups),
            "retrieval_recall_note": {
                "raw_source_rows": self.n_raw_source_rows,
                "unique_source_papers": self.n_unique_source_papers,
                "kept_in_combined": self.n_combined,
                "comment": "raw->combined drop is retrieval recall (Task 2), logged not scored",
            },
        }


def run(ds: QueryDataset) -> Step2Result:
    res = Step2Result(n_combined=len(ds.combined))
    idx = SourceIndex(ds.sources)

    # ── raw vs unique source papers (recall context) ────────────────────────
    all_rows = [r for rows in ds.sources.values() for r in rows]
    res.n_raw_source_rows = len(all_rows)
    unique_keys = set()
    for r in all_rows:
        unique_keys.add(norm_doi(r.get("DOI", "")) or norm_title(r.get("Paper Title", "")))
    res.n_unique_source_papers = len({k for k in unique_keys if k})

    # ── 1. provenance + 2. merge consistency ────────────────────────────────
    for row in ds.combined:
        matches = idx.match(row)
        title = (row.get("Paper Title", "") or "")[:70]

        if not matches:
            res.fabricated.append(title)
            continue

        how = matches[0].how
        if how == "doi":
            res.provenance_by_doi += 1
        elif how == "title":
            res.provenance_by_title += 1
        else:  # "title~"
            res.provenance_fuzzy += 1

        # merge consistency: compare combined metadata to its source rows
        c_doi = norm_doi(row.get("DOI", ""))
        src_dois = {norm_doi(m.row.get("DOI", "")) for m in matches if norm_doi(m.row.get("DOI", ""))}
        if c_doi and src_dois and c_doi not in src_dois:
            res.merge_doi_drift.append(
                {"title": title, "combined_doi": c_doi, "source_dois": sorted(src_dois)}
            )

        c_year = (row.get("Publication Year", "") or "").strip()
        src_years = {(m.row.get("Publication Year", "") or "").strip() for m in matches}
        src_years.discard("")
        if c_year and src_years and c_year not in src_years:
            res.merge_year_drift.append(
                {"title": title, "combined_year": c_year, "source_years": sorted(src_years)}
            )

    # ── 3. dedup within the combined table ──────────────────────────────────
    res.duplicate_groups = _find_duplicates(ds.combined)
    return res


def _find_duplicates(rows: list[dict]) -> list[list[str]]:
    """Group combined rows that are the same paper (same DOI, else same norm title)."""
    buckets: dict[str, list[str]] = {}
    for r in rows:
        key = norm_doi(r.get("DOI", "")) or norm_title(r.get("Paper Title", ""))
        if not key:
            continue
        buckets.setdefault(key, []).append((r.get("Paper Title", "") or "")[:60])
    return [titles for titles in buckets.values() if len(titles) > 1]


def _print(res: Step2Result) -> None:
    s = res.summary()
    print("=" * 68)
    print("STEP 2 — PAPER CONSOLIDATION")
    print("=" * 68)
    print(f"  consolidated rows          : {s['n_combined']}")
    print(f"  provenance rate            : {s['provenance_rate']*100:.1f}%  "
          f"(doi={s['provenance_breakdown']['by_doi']}, "
          f"title={s['provenance_breakdown']['by_title_exact']}, "
          f"fuzzy={s['provenance_breakdown']['by_title_fuzzy']})")
    print(f"  fabricated (untraceable)   : {s['fabricated_count']}")
    for t in s["fabricated_titles"]:
        print(f"      - {t}")
    print(f"  merge DOI drift            : {s['merge_doi_drift_count']}")
    for d in res.merge_doi_drift[:5]:
        print(f"      - {d['title']}  combined={d['combined_doi']} vs {d['source_dois']}")
    print(f"  merge year drift           : {s['merge_year_drift_count']}")
    for d in res.merge_year_drift[:5]:
        print(f"      - {d['title']}  combined={d['combined_year']} vs {d['source_years']}")
    print(f"  duplicate groups           : {s['duplicate_group_count']}  "
          f"(extra rows: {s['duplicate_extra_rows']})")
    for g in res.duplicate_groups[:5]:
        print(f"      - {g}")
    r = s["retrieval_recall_note"]
    print(f"  [Task2] recall context     : {r['raw_source_rows']} raw rows -> "
          f"{r['unique_source_papers']} unique -> {r['kept_in_combined']} kept")


if __name__ == "__main__":
    res = run(gut_brain())
    _print(res)
    print("\nJSON summary:")
    print(json.dumps(res.summary(), indent=2))
