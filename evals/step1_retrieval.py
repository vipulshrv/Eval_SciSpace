"""
STEP 1 — LITERATURE RETRIEVAL eval  (per source, deterministic — no LLM).

Question: are the retrieved papers real, with correct metadata?

Reference truth: external — the DOI record in CrossRef.

Graded PER SOURCE, because sources are heterogeneous:
  - PubMed / SciSpace: authoritative-ish, expect high DOI coverage + accuracy.
  - Google Scholar: notoriously sparse (missing DOI/year). Missing metadata is a
    retrieval-QUALITY issue, not a hallucination — so we report metadata
    COMPLETENESS separately from ACCURACY.

Existence handling (honest): a DOI that CrossRef resolves is definitely real. A
CrossRef 404 is reported as "unresolved" NOT "fabricated" — some legitimate DOIs
(regional journals, DataCite registrants) aren't in CrossRef, so a 404 alone can't
prove fabrication.

Metadata accuracy (for resolved DOIs): agent title vs CrossRef title (fuzzy) and
agent year vs CrossRef year.
"""

from __future__ import annotations

import re
import html
import json
import ssl
import time
import urllib.request
import urllib.parse
import difflib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .data_loader import gut_brain, norm_doi, norm_title

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAILTO = "vipulshrivastava@quvia.ai"
TITLE_MATCH_THRESHOLD = 0.85


def _clean_title(s: str) -> str:
    """Undo CrossRef's HTML markup before comparison (e.g. '&amp;lt;i&amp;gt;via'
    -> '<i>via' -> 'via'; '&#174;' -> '®'), so formatting isn't read as a mismatch."""
    s = s or ""
    for _ in range(2):
        s = html.unescape(s)
    return re.sub(r"<[^>]+>", " ", s)


def _title_matches(agent: str, xref: str) -> tuple[bool, float]:
    """Agent title vs CrossRef title. Counts as a match when identical, when one is
    contained in the other (subtitle present/absent, or a truncated-but-correct
    prefix — e.g. agent adds ': The SCALE ... Trial' that CrossRef omits), or when
    fuzzy similarity clears the threshold. Returns (matched, ratio)."""
    na = norm_title(_clean_title(agent))
    nx = norm_title(_clean_title(xref))
    if not na or not nx:
        return False, 0.0
    if na == nx:
        return True, 1.0
    # subtitle / truncation: one normalized title contains the other (guard against
    # a trivially short title matching inside an unrelated long one)
    if len(na) >= 15 and len(nx) >= 15 and (na in nx or nx in na):
        return True, 1.0
    ratio = difflib.SequenceMatcher(None, na, nx).ratio()
    return ratio >= TITLE_MATCH_THRESHOLD, ratio

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _CTX = ssl.create_default_context()
    _CTX.check_hostname = False
    _CTX.verify_mode = ssl.CERT_NONE


@dataclass
class DOIRecord:
    doi: str
    status: str            # "resolved" | "not_found" | "error"
    crossref_title: str = ""
    crossref_year: str = ""


def _crossref(doi: str) -> DOIRecord:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
    req = urllib.request.Request(url, headers={"User-Agent": f"SciSpaceEval/1.0 (mailto:{MAILTO})"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=15, context=_CTX) as r:
                m = json.load(r).get("message", {})
                title = (m.get("title") or [""])[0]
                dp = (m.get("issued", {}).get("date-parts") or [[None]])[0]
                year = str(dp[0]) if dp and dp[0] else ""
                return DOIRecord(doi, "resolved", title, year)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return DOIRecord(doi, "not_found")
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return DOIRecord(doi, "error")


@dataclass
class SourceReport:
    source: str
    rows: int
    with_doi: int
    resolved: int
    not_found: int
    errors: int
    title_match: int          # numerator; denominator = resolved
    year_match: int           # numerator; denominator = year_checked
    year_checked: int         # resolved rows where the agent supplied a year
    not_found_titles: list[str] = field(default_factory=list)
    mismatch_examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["doi_completeness"] = round(self.with_doi / self.rows, 3) if self.rows else None
        d["resolve_rate"] = round(self.resolved / self.with_doi, 3) if self.with_doi else None
        d["title_accuracy"] = round(self.title_match / self.resolved, 3) if self.resolved else None
        d["year_accuracy"] = round(self.year_match / self.year_checked, 3) if self.year_checked else None
        return d


def run(ds=None, workers: int = 8) -> dict:
    ds = ds or gut_brain()

    # gather all unique DOIs across sources, resolve once (cache)
    all_dois = set()
    for rows in ds.sources.values():
        for r in rows:
            d = norm_doi(r.get("DOI", ""))
            if d:
                all_dois.add(d)

    cache: dict[str, DOIRecord] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(_crossref, sorted(all_dois)):
            cache[rec.doi] = rec

    reports = []
    for source, rows in ds.sources.items():
        rep = SourceReport(source=source, rows=len(rows), with_doi=0, resolved=0,
                           not_found=0, errors=0, title_match=0, year_match=0, year_checked=0)
        for r in rows:
            d = norm_doi(r.get("DOI", ""))
            if not d:
                continue
            rep.with_doi += 1
            rec = cache.get(d)
            if not rec:
                continue
            if rec.status == "resolved":
                rep.resolved += 1
                # title accuracy (denominator = resolved); containment + HTML-cleaned
                matched, ratio = _title_matches(r.get("Paper Title", ""), rec.crossref_title)
                if matched:
                    rep.title_match += 1
                elif len(rep.mismatch_examples) < 30:
                    rep.mismatch_examples.append({
                        "agent_title": (r.get("Paper Title", "") or "")[:160],
                        "crossref_title": (rec.crossref_title or "")[:160],
                        "ratio": round(ratio, 2),
                    })
                # year accuracy (denominator = rows where agent supplied a year)
                ay = (r.get("Publication Year", "") or "").strip()
                if ay and rec.crossref_year:
                    rep.year_checked += 1
                    if ay == rec.crossref_year:
                        rep.year_match += 1
            elif rec.status == "not_found":
                rep.not_found += 1
                if len(rep.not_found_titles) < 5:
                    rep.not_found_titles.append((r.get("Paper Title", "") or "")[:60])
            else:
                rep.errors += 1
        reports.append(rep)

    return {
        "summary": {
            "step": "1_literature_retrieval",
            "unique_dois_checked": len(all_dois),
            "per_source": [r.to_dict() for r in reports],
        }
    }


def _print(res: dict) -> None:
    s = res["summary"]
    print("=" * 68)
    print("STEP 1 — LITERATURE RETRIEVAL  (CrossRef, per source)")
    print("=" * 68)
    print(f"  unique DOIs checked: {s['unique_dois_checked']}\n")
    hdr = f"  {'source':20s} {'rows':>4s} {'DOI%':>5s} {'resolv':>6s} {'title':>6s} {'year':>5s} {'404':>4s}"
    print(hdr)
    for r in s["per_source"]:
        comp = f"{(r['doi_completeness'] or 0)*100:.0f}"
        res_r = f"{(r['resolve_rate'] or 0)*100:.0f}" if r['resolve_rate'] is not None else "-"
        ta = f"{(r['title_accuracy'] or 0)*100:.0f}" if r['title_accuracy'] is not None else "-"
        ya = f"{(r['year_accuracy'] or 0)*100:.0f}" if r['year_accuracy'] is not None else "-"
        print(f"  {r['source']:20s} {r['rows']:>4d} {comp:>4s}% {res_r:>5s}% {ta:>5s}% {ya:>4s}% {r['not_found']:>4d}")
    print("\n  (DOI% = have a DOI; resolv = resolved in CrossRef; title/year = accuracy on resolved;")
    print("   404 = DOI not in CrossRef — 'unresolved', not proof of fabrication)")


if __name__ == "__main__":
    res = run()
    _print(res)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "step1_retrieval.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {out}")
