"""
Download full text for the extracted papers so Step 4's "unverifiable" cells can
be adjudicated against the real paper instead of a fragment.

Best-effort, in priority order per paper:
  1. the row's PDF Link (SciSpace / PMC / publisher)
  2. Unpaywall's best open-access PDF (via DOI)

Downloads the PDF, extracts text with PyPDF2, and writes it to
data/fulltext/<normalized-doi>.txt — which data_loader.reference_text() then picks
up automatically, upgrading those papers' grounding_level to "fulltext_downloaded".

Paywalled / non-OA papers will fail; that's expected and reported, not fatal.
"""

from __future__ import annotations

import io
import json
import ssl
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .data_loader import QUERIES, load_query, norm_doi, FULLTEXT_DIR

MAILTO = "vipulshrivastava@quvia.ai"
_UA = f"SciSpaceEval/1.0 (mailto:{MAILTO})"

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _CTX = ssl.create_default_context()
    _CTX.check_hostname = False
    _CTX.verify_mode = ssl.CERT_NONE

try:
    import PyPDF2  # type: ignore
except Exception:  # pragma: no cover
    PyPDF2 = None


@dataclass
class FetchResult:
    doi: str
    title: str
    status: str      # "ok" | "no_url" | "download_failed" | "not_pdf" | "extract_failed" | "no_doi"
    source_url: str = ""
    chars: int = 0


def _get(url: str, timeout: int = 25) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"})
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
            return r.read()
    except Exception:
        return None


def _unpaywall_pdf(doi: str) -> str:
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={MAILTO}"
    raw = _get(url, timeout=15)
    if not raw:
        return ""
    try:
        loc = json.loads(raw).get("best_oa_location") or {}
        return loc.get("url_for_pdf") or ""
    except Exception:
        return ""


def _pdf_to_text(data: bytes) -> str:
    if not data[:5].startswith(b"%PDF") or PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    except Exception:
        return ""


def fetch_one(row: dict) -> FetchResult:
    doi = norm_doi(row.get("DOI", ""))
    title = (row.get("Paper Title", "") or "")[:60]
    if not doi:
        return FetchResult("", title, "no_doi")

    out = FULLTEXT_DIR / f"{doi.replace('/', '_')}.txt"
    if out.exists() and out.stat().st_size > 500:
        return FetchResult(doi, title, "ok", "cached", len(out.read_text(errors="ignore")))

    urls = [u for u in [(row.get("PDF Link", "") or "").strip(), _unpaywall_pdf(doi)] if u]
    if not urls:
        return FetchResult(doi, title, "no_url")

    for url in urls:
        data = _get(url)
        if not data:
            continue
        text = _pdf_to_text(data)
        if len(text) < 500:  # too short = failed extraction / not real full text
            continue
        FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return FetchResult(doi, title, "ok", url, len(text))

    return FetchResult(doi, title, "download_failed", urls[0])


def run(query: str = "gut_brain", workers: int = 6) -> dict:
    ds = load_query(query)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_one, ds.extracted))

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    return {
        "query": query,
        "attempted": len(results),
        "succeeded": by_status.get("ok", 0),
        "by_status": by_status,
        "details": [r.__dict__ for r in results],
    }


if __name__ == "__main__":
    import sys
    if PyPDF2 is None:
        print("PyPDF2 not installed — run: pip install PyPDF2")
    targets = sys.argv[1:] or list(QUERIES.keys())
    for q in targets:
        res = run(q)
        print(f"[{q}] full text fetched: {res['succeeded']}/{res['attempted']}  "
              f"breakdown={res['by_status']}")
    print(f"saved to: {FULLTEXT_DIR}")
