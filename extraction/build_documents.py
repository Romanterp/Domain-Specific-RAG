"""
Build documents.json from catalog.json + extracted text metadata.

Produces clean RAG-ready metadata using the DocumentMetadata schema.
Fields populated:
  - Bibliographic: from catalog + PDF metadata
  - Web metadata: from catalog (scraped)
  - Processing status: from file existence checks
  - Enrichment fields: left empty for heuristic/LLM stages
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.schema import DocumentMetadata

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "texts"
CATALOG_PATH = DATA_DIR / "catalog.json"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
DOC_EXTENSIONS = (".pdf", ".docx", ".doc", ".xls", ".xlsx", ".pptx")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def find_file(slug: str) -> Path | None:
    """Find the downloaded document file for a slug."""
    for ext in DOC_EXTENSIONS:
        p = PDF_DIR / f"{slug}{ext}"
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def detect_language(text: str) -> str:
    """Simple language detection from text content."""
    # Common words for quick detection
    es_words = {"de", "la", "el", "en", "los", "las", "del", "por", "para", "con", "una"}
    fr_words = {"de", "la", "le", "les", "des", "du", "en", "et", "pour", "dans", "une"}
    ar_indicators = any("\u0600" <= c <= "\u06FF" for c in text[:500])
    zh_indicators = any("\u4e00" <= c <= "\u9fff" for c in text[:500])
    ru_indicators = any("\u0400" <= c <= "\u04FF" for c in text[:500])

    if ar_indicators:
        return "ar"
    if zh_indicators:
        return "zh"
    if ru_indicators:
        return "ru"

    words = set(text[:2000].lower().split())
    es_score = len(words & es_words)
    fr_score = len(words & fr_words)

    # Check for Spanish-specific words (not shared with French)
    if es_score > 4 and any(w in words for w in ("los", "las", "para", "pero", "como")):
        return "es"
    if fr_score > 4 and any(w in words for w in ("les", "des", "dans", "cette", "sont")):
        return "fr"

    return "en"


def parse_date(entry: dict) -> str:
    """Extract clean ISO date from catalog entry."""
    iso = entry.get("publication_date_iso", "")
    if iso:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    raw = entry.get("publication_date", "")
    if raw:
        # Try common formats
        for fmt in ("%d %B %Y", "%B %Y", "%Y"):
            try:
                return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def build_document(slug: str, entry: dict, parent_slug: str = "", download_url: str = "") -> DocumentMetadata:
    """Build a DocumentMetadata for a single file."""
    doc_file = find_file(slug)
    text_file = TEXT_DIR / f"{slug}.txt"

    # Per-file page/word count from text
    page_count = 0
    word_count = 0
    pages_file = TEXT_DIR / f"{slug}.pages.json"
    if pages_file.exists():
        try:
            with open(pages_file, encoding="utf-8") as f:
                pages = json.load(f)
            page_count = len(pages)
            word_count = sum(len(p["text"].split()) for p in pages)
        except Exception:
            pass
    if not word_count and text_file.exists():
        try:
            word_count = len(text_file.read_text(encoding="utf-8", errors="ignore").split())
        except Exception:
            pass

    # Detect language from this file's text
    language = "en"
    if text_file.exists():
        try:
            text = text_file.read_text(encoding="utf-8", errors="ignore")[:3000]
            language = detect_language(text)
        except Exception:
            pass

    return DocumentMetadata(
        slug=slug,
        parent_slug=parent_slug,

        # Bibliographic
        title=entry.get("title", ""),
        publication_date=parse_date(entry),
        language=language,
        page_count=page_count,
        word_count=word_count,
        document_type="other",
        source_url=entry.get("source_url", ""),
        filename=doc_file.name if doc_file else "",
        pdf_download_url=download_url or entry.get("pdf_download_url", ""),

        # Web metadata (shared from parent catalog entry)
        web_description=entry.get("web_description", ""),
        web_document_type=entry.get("web_document_type", ""),
        web_organizations=entry.get("web_organizations", []),

        # Processing status
        pdf_downloaded=doc_file is not None,
        text_extracted=text_file.exists() and text_file.stat().st_size > 0,
        heuristic_extracted=False,
        llm_extracted=False,
    )


def build_all():
    """Build documents.json — one entry per file."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    log.info(f"Processing {len(catalog)} catalog entries...")

    documents = []
    stats = {"total": 0, "with_pdf": 0, "with_text": 0}

    for entry in catalog:
        base_slug = entry.get("slug", "")
        urls = entry.get("pdf_download_urls", [])
        if not urls:
            single = entry.get("pdf_download_url", "")
            urls = [single] if single else []

        if not urls:
            # No PDF URLs at all — still create an entry for the catalog record
            doc = build_document(base_slug, entry)
            documents.append(doc.model_dump())
            stats["total"] += 1
            if doc.pdf_downloaded:
                stats["with_pdf"] += 1
            if doc.text_extracted:
                stats["with_text"] += 1
            continue

        for idx, url in enumerate(urls):
            if idx == 0:
                slug = base_slug
                parent = ""
            else:
                slug = f"{base_slug}_{idx + 1}"
                parent = base_slug

            doc = build_document(slug, entry, parent_slug=parent, download_url=url)
            documents.append(doc.model_dump())

            stats["total"] += 1
            if doc.pdf_downloaded:
                stats["with_pdf"] += 1
            if doc.text_extracted:
                stats["with_text"] += 1

    # Sort by date descending, then slug
    documents.sort(key=lambda d: (d.get("publication_date", ""), d.get("slug", "")), reverse=True)

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    log.info(f"Wrote {DOCUMENTS_PATH}")
    log.info(f"  Total entries: {stats['total']}")
    log.info(f"  With downloaded file: {stats['with_pdf']}")
    log.info(f"  With extracted text: {stats['with_text']}")


if __name__ == "__main__":
    build_all()
