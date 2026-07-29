"""
Loads the SciSpace agent artifacts for one query session and exposes the objects
the eval steps need:

  - the 4 source retrieval CSVs (the raw fan-out)
  - the consolidated table (combined_*.csv)
  - the extracted subset (rows that actually got criteria filled in)
  - the generated report text
  - a citation map:  [n]  ->  extracted table row  (report references are unnumbered
    in the file, but [n] corresponds to the n-th extracted row; verified against the
    report prose)
  - per-paper source text (abstract / full-text excerpt / snippet / tl;dr) pooled
    from every source the paper appeared in — this is the "reference truth" that
    Step 4 grades extraction against.

Stdlib only, so it runs without installing anything.
"""

from __future__ import annotations

import csv
import re
import difflib
from dataclasses import dataclass, field
from pathlib import Path

# ── artifact locations ────────────────────────────────────────────────────────

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "agent-artifacts-zip_b4fed69e-8a24-4fdf-b320-89da92df96da_1785232738"
)

# downloaded full text lives here, one file per paper keyed by normalized DOI
FULLTEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "fulltext"


@dataclass
class QueryDataset:
    """Everything for one query session, plus the config that describes its shape."""

    name: str
    query: str
    artifact_dir: Path
    combined_file: str | list[str]        # one table, or several to merge (e.g. per-drug)
    report_file: str
    # source retrieval files, keyed by source; each value is one path or several to merge
    source_files: dict[str, str | list[str]]
    # the columns in the combined table that are agent-extracted criteria
    criteria_columns: list[str]
    # optional separate references file ("[n]\n<citation>" or "[n] <citation>")
    references_file: str | None = None

    # populated by load()
    combined: list[dict] = field(default_factory=list)
    sources: dict[str, list[dict]] = field(default_factory=dict)
    extracted: list[dict] = field(default_factory=list)
    report: str = ""
    refmap: dict = field(default_factory=dict)   # [n] -> combined row, from a parsed ref list
    citation_mode: str = "index"                 # "reference_list" | "index"
    citation_confidence: str = "high"            # "high" | "low" — gates Step 5 trust

    # ── loading ────────────────────────────────────────────────────────────────

    def _read_many(self, spec: str | list[str]) -> list[dict]:
        files = [spec] if isinstance(spec, str) else spec
        rows: list[dict] = []
        for f in files:
            rows.extend(_read_csv(self.artifact_dir / f))
        return rows

    def load(self) -> "QueryDataset":
        self.combined = self._read_many(self.combined_file)
        self.sources = {src: self._read_many(spec) for src, spec in self.source_files.items()}
        self.report = (self.artifact_dir / self.report_file).read_text(encoding="utf-8")
        # extracted = combined rows where the agent filled in the criteria columns
        self.extracted = [
            r for r in self.combined if _any_filled(r, self.criteria_columns)
        ]
        self._build_refmap()
        return self

    # ── citation resolution: [n] -> paper row ────────────────────────────────────

    def _reference_entries(self) -> list[tuple[str, str]]:
        """Return [(n, citation_text)] from a separate references file if provided,
        else from a numbered list inside the report. Handles both '[n] text' and
        '[n]\\n text' (number on its own line) layouts."""
        text = ""
        if self.references_file:
            p = self.artifact_dir / self.references_file
            if p.exists():
                text = p.read_text(encoding="utf-8")
        if not text:
            text = self.report
        # split on [n] markers, keeping whatever follows until the next marker
        parts = re.split(r"(?m)^\s*\[(\d+)\]\s*", text)
        entries = []
        for i in range(1, len(parts) - 1, 2):
            num, body = parts[i], parts[i + 1].strip()
            if body:
                entries.append((num, body))
        return entries

    def _build_refmap(self) -> None:
        """Map [n] -> combined row by DOI/title using a reference list (separate file
        or in-report). Fall back to [n] = n-th extracted row (index mode)."""
        by_doi = {norm_doi(r.get("DOI", "")): r for r in self.combined if norm_doi(r.get("DOI", ""))}
        titles = {norm_title(r.get("Paper Title", "")): r for r in self.combined}
        resolved: dict[int, dict] = {}
        entries = self._reference_entries()
        for num, text in entries:
            n = int(num)
            m = re.search(r"10\.\d{4,9}/[^\s\"'>)]+", text)
            row = by_doi.get(norm_doi(m.group(0))) if m else None
            if row is None:
                qt = re.search(r'[“"]([^”"]+)[”"]', text)  # quoted title (incl. smart quotes)
                if qt:
                    key = norm_title(qt.group(1))
                    row = titles.get(key)
                    if row is None:
                        near = difflib.get_close_matches(key, titles.keys(), n=1, cutoff=0.9)
                        if near:
                            row = titles[near[0]]
            if row is not None:
                resolved[n] = row
        # accept reference-list mode only if it resolved most entries
        if entries and len(resolved) >= 0.6 * len(entries):
            self.refmap = resolved
            self.citation_mode = "reference_list"
            self.citation_confidence = "high"
        else:
            self.citation_mode = "index"
            # index mode is only trustworthy when the report cites exactly the
            # extracted set (citation n == n-th extracted row). If the counts
            # diverge, the report doesn't map cleanly onto this table — Step 5
            # citation grading would be measuring the wrong paper.
            n_cites = len(set(int(x) for x in re.findall(r"\[(\d+)\]", self.report)))
            self.citation_confidence = "high" if n_cites == len(self.extracted) else "low"

    def citation_row(self, n: int) -> dict | None:
        if self.citation_mode == "reference_list":
            return self.refmap.get(n)
        if 1 <= n <= len(self.extracted):
            return self.extracted[n - 1]
        return None

    # ── source-text pooling: the reference truth for a paper ─────────────────────

    def source_index(self) -> "SourceIndex":
        return SourceIndex(self.sources)

    def reference_text(self, row: dict) -> "ReferenceText":
        """Pool every piece of source text available for this paper, across sources.

        Returned object records WHICH texts exist so Step 4 can label a verdict
        'context-limited' when only an abstract was available.
        """
        idx = self.source_index()
        matches = idx.match(row)

        abstract = _first_nonempty([row.get("Abstract", "")] + [m.row.get("Abstract", "") for m in matches])
        excerpt = _first_nonempty([m.row.get("Relevant Excerpt", "") for m in matches])
        snippet = _first_nonempty([m.row.get("Snippet", "") for m in matches])
        tldr = _first_nonempty([m.row.get("TL;DR", "") for m in matches])

        # downloaded full text (fetch_papers.py) takes priority when present
        fulltext = ""
        d = norm_doi(row.get("DOI", ""))
        if d:
            f = FULLTEXT_DIR / f"{d.replace('/', '_')}.txt"
            if f.exists():
                fulltext = f.read_text(encoding="utf-8", errors="ignore")

        return ReferenceText(
            abstract=abstract,
            fulltext_excerpt=excerpt,
            snippet=snippet,
            tldr=tldr,
            fulltext=fulltext,
            matched_sources=sorted({m.source for m in matches}),
        )


# ── source matching ─────────────────────────────────────────────────────────

@dataclass
class SourceMatch:
    source: str
    row: dict
    how: str  # "doi" | "title"


class SourceIndex:
    """Indexes all source rows by normalized DOI and title for provenance matching."""

    TITLE_SIM_THRESHOLD = 0.90

    def __init__(self, sources: dict[str, list[dict]]):
        self.by_doi: dict[str, list[SourceMatch]] = {}
        self.by_title_norm: dict[str, list[SourceMatch]] = {}
        self._all: list[SourceMatch] = []
        for src, rows in sources.items():
            for r in rows:
                m = SourceMatch(source=src, row=r, how="")
                self._all.append(m)
                d = norm_doi(r.get("DOI", ""))
                if d:
                    self.by_doi.setdefault(d, []).append(m)
                t = norm_title(r.get("Paper Title", ""))
                if t:
                    self.by_title_norm.setdefault(t, []).append(m)

    def match(self, row: dict) -> list[SourceMatch]:
        """Return all source rows that are the same paper as `row`.

        DOI-exact first; else exact normalized title; else fuzzy title above
        threshold (guards against small metadata drift during consolidation).
        """
        d = norm_doi(row.get("DOI", ""))
        if d and d in self.by_doi:
            return [SourceMatch(m.source, m.row, "doi") for m in self.by_doi[d]]

        t = norm_title(row.get("Paper Title", ""))
        if t and t in self.by_title_norm:
            return [SourceMatch(m.source, m.row, "title") for m in self.by_title_norm[t]]

        if t:
            best = difflib.get_close_matches(t, self.by_title_norm.keys(), n=1, cutoff=self.TITLE_SIM_THRESHOLD)
            if best:
                return [SourceMatch(m.source, m.row, "title~") for m in self.by_title_norm[best[0]]]
        return []


@dataclass
class ReferenceText:
    abstract: str = ""
    fulltext_excerpt: str = ""
    snippet: str = ""
    tldr: str = ""
    fulltext: str = ""
    matched_sources: list[str] = field(default_factory=list)

    @property
    def best(self) -> str:
        """Richest available source text; downloaded full text wins."""
        return (self.fulltext or self.fulltext_excerpt or self.abstract
                or self.tldr or self.snippet)

    @property
    def grounding_level(self) -> str:
        if self.fulltext:
            return "fulltext_downloaded"
        if self.fulltext_excerpt:
            return "fulltext_excerpt"
        if self.abstract:
            return "abstract_only"
        if self.tldr or self.snippet:
            return "snippet_only"
        return "none"


# ── normalization / helpers ────────────────────────────────────────────────

def norm_doi(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return s.strip().rstrip(".,;)")  # drop trailing punctuation from ref-list DOIs


def norm_title(s: str) -> str:
    # collapse to alphanumerics; drop leading ellipsis fragments Google Scholar adds
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _any_filled(row: dict, cols: list[str]) -> bool:
    return any((row.get(c, "") or "").strip() for c in cols)


def _first_nonempty(values: list[str]) -> str:
    for v in values:
        if v and v.strip():
            return v.strip()
    return ""


# ── query registry (all three sessions) ──────────────────────────────────────

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"


def gut_brain() -> QueryDataset:
    return QueryDataset(
        name="microbiota_gut_brain_axis",
        query=(
            "Create a report on microbiota-gut-brain axis interactions, analyzing "
            "neurochemical signaling pathways and behavioral outcomes across human "
            "and animal studies."
        ),
        artifact_dir=ARTIFACTS_ROOT / "agent-artifacts-zip_b4fed69e-8a24-4fdf-b320-89da92df96da_1785232738",
        combined_file="combined_microbiota_gut_brain_axis_results.csv",
        report_file="microbiota_gut_brain_axis_report.md",
        source_files={
            "pubmed": "pubmed_microbiota_gut_brain.csv",
            "google_scholar": "google_scholar_microbiota_gut_brain.csv",
            "scispace_abstract": "scispace_microbiota_gut_brain_axis.csv",
            "scispace_fulltext": "scispace_fulltext_microbiota_gut_brain.csv",
        },
        criteria_columns=[
            "Neurochemical Pathways", "Study Design and Model", "Key Behavioral Outcomes",
        ],
    ).load()


def cancer_detection() -> QueryDataset:
    return QueryDataset(
        name="ai_cancer_detection",
        query=(
            "Create a report on AI-based early cancer detection methods, comparing "
            "performance across imaging, genomics, and multimodal approaches using "
            "metrics like AUC, sensitivity, and specificity."
        ),
        artifact_dir=ARTIFACTS_ROOT / "agent-artifacts-zip_e8d99547-b021-4e9c-8d20-8bf897ddbca4_1785248895",
        combined_file="combined_ai_cancer_detection_results.csv",
        report_file="AI_Cancer_Detection_Comparative_Report.md",
        source_files={
            "pubmed": "pubmed_ai_cancer_detection.csv",
            "google_scholar": "google_scholar_ai_cancer_detection.csv",
            "scispace_abstract": "scispace_ai_cancer_detection.csv",
            "scispace_fulltext": "scispace_fulltext_ai_cancer_detection.csv",
        },
        criteria_columns=[
            "Approach Type and Data Modalities", "Performance Metrics", "Cancer Type and AI Methodology",
        ],
    ).load()


def glp1_weight_loss() -> QueryDataset:
    # three per-drug sub-searches merged into one table; report cites a separate
    # references.txt (the reference list was cut from the report download).
    d = "agent-artifacts-zip_35a424fa-d596-40f7-87f5-6f7b24fdcd8e_1785261275"
    drugs = {
        "sema": "search_cvtncd", "lira": "search_lh5b6k", "tirz": "search_w3em9y",
    }
    combined = [f"{drugs['sema']}/combined_semaglutide_results.csv",
                f"{drugs['lira']}/combined_liraglutide_results.csv",
                f"{drugs['tirz']}/combined_tirzepatide_results.csv"]
    return QueryDataset(
        name="glp1_weight_loss",
        query=(
            "Create a report on GLP-1 receptor agonists for weight loss, comparing "
            "percent body-weight reduction, cardiovascular outcomes, and adverse-event "
            "rates across semaglutide, tirzepatide, and liraglutide."
        ),
        artifact_dir=ARTIFACTS_ROOT / d,
        combined_file=combined,
        report_file="GLP1_RA_Weight_Loss_Comparative_Report.md",
        references_file="references.txt",
        source_files={
            "pubmed": [f"{drugs['sema']}/pubmed_semaglutide_trials.csv",
                       f"{drugs['lira']}/pubmed_liraglutide_trials.csv",
                       f"{drugs['tirz']}/pubmed_tirzepatide_surmount.csv"],
            "google_scholar": [f"{drugs['sema']}/google_scholar_semaglutide.csv",
                               f"{drugs['lira']}/google_scholar_liraglutide.csv",
                               f"{drugs['tirz']}/google_scholar_tirzepatide.csv"],
            "scispace_abstract": [f"{drugs['sema']}/scispace_semaglutide_weight_loss.csv",
                                  f"{drugs['lira']}/scispace_liraglutide_clinical_trials.csv",
                                  f"{drugs['tirz']}/scispace_tirzepatide_surmount.csv"],
            "scispace_fulltext": [f"{drugs['sema']}/scispace_fulltext_semaglutide.csv",
                                  f"{drugs['lira']}/scispace_fulltext_liraglutide.csv",
                                  f"{drugs['tirz']}/scispace_fulltext_tirzepatide.csv"],
        },
        criteria_columns=["Weight Loss Outcomes", "Cardiovascular Outcomes", "Adverse Events"],
    ).load()


def diabetic_retinopathy() -> QueryDataset:
    d = "agent-artifacts-zip_6ec3cdd0-c62f-4612-b1a3-016d34a9e0ba_1785260843"
    sub = "search_0aftyq"
    return QueryDataset(
        name="diabetic_retinopathy",
        query=(
            "Create a report on deep learning for diabetic retinopathy screening from "
            "retinal fundus images, comparing sensitivity, specificity, and AUC across "
            "CNN architectures and datasets."
        ),
        artifact_dir=ARTIFACTS_ROOT / d,
        combined_file=f"{sub}/combined_search_results.csv",
        report_file="report_deep_learning_diabetic_retinopathy.md",
        references_file="references.txt",
        source_files={
            "pubmed": [f"{sub}/pubmed_dr_architectures.csv", f"{sub}/pubmed_dr_metrics.csv",
                       f"{sub}/pubmed_dr_screening.csv"],
            "google_scholar": [f"{sub}/gscholar_dr_architectures.csv", f"{sub}/gscholar_dr_metrics.csv",
                               f"{sub}/gscholar_dr_screening.csv"],
            "scispace_abstract": [f"{sub}/scispace_basic_dr_datasets.csv", f"{sub}/scispace_basic_dr_performance.csv",
                                  f"{sub}/scispace_basic_dr_screening.csv"],
            "scispace_fulltext": [f"{sub}/scispace_fulltext_dr_architectures.csv", f"{sub}/scispace_fulltext_dr_metrics.csv",
                                  f"{sub}/scispace_fulltext_dr_screening.csv"],
        },
        criteria_columns=["CNN Architecture and Performance Metrics"],
    ).load()


# NOTE: wearables was removed — its report's citations don't map onto its own
# consolidated table (no reference list; report↔table provenance mismatch).

QUERIES = {
    "gut_brain": gut_brain,
    "cancer": cancer_detection,
    "glp1_weight_loss": glp1_weight_loss,
    "diabetic_retinopathy": diabetic_retinopathy,
}


def load_query(name: str) -> QueryDataset:
    return QUERIES[name]()


if __name__ == "__main__":
    ds = gut_brain()
    print(f"query: {ds.name}")
    print(f"  combined rows : {len(ds.combined)}")
    print(f"  extracted rows: {len(ds.extracted)}  (top-N cited in report)")
    for src, rows in ds.sources.items():
        print(f"  source {src:20s}: {len(rows)} rows")
    print(f"  report chars  : {len(ds.report)}")

    print("\n  citation spot-check ([n] -> first author | title):")
    for n in (1, 2, 9, 12, 13):
        r = ds.citation_row(n)
        auth = (r.get("Author Names", "").split("\n")[0])[:24]
        print(f"    [{n:2d}] {auth:26s} | {r.get('Paper Title','')[:50]}")

    print("\n  reference-text grounding level for the 30 extracted papers:")
    levels: dict[str, int] = {}
    for r in ds.extracted:
        lvl = ds.reference_text(r).grounding_level
        levels[lvl] = levels.get(lvl, 0) + 1
    for lvl, c in sorted(levels.items()):
        print(f"    {lvl:20s}: {c}")
