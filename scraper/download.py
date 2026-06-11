"""
Download PDFs from discovered catalog entries.

Strategy:
1. Launch real Edge browser via CDP (avoids Cloudflare detection)
2. Navigate to detail pages to find /media/{id}/download links
3. Download PDFs using ?startDownload= parameter (bypasses interstitial)
4. Handles multiple PDFs per page (slug.pdf, slug_2.pdf, etc.)

Respects robots.txt Crawl-Delay: 10.
Supports resume — skips already-downloaded files.
Use --reset to re-check entries previously marked as having no PDF.
"""

import asyncio
import base64
import json
import logging
import random
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page

BASE_URL = "https://www.undrr.org"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
PDF_DIR = DATA_DIR / "pdfs"
PROFILE_DIR = DATA_DIR / "edge_profile"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_catalog() -> list[dict]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog: list[dict]):
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def launch_edge(port=9222):
    """Launch Microsoft Edge with remote debugging."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_path = next((p for p in edge_paths if Path(p).exists()), None)
    if not edge_path:
        raise RuntimeError("Microsoft Edge not found")

    log.info("Launching Edge browser...")
    proc = subprocess.Popen([
        edge_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
    ])
    time.sleep(3)
    return proc


async def wait_for_cf(page: Page, timeout: int = 90000) -> bool:
    """Wait for Cloudflare to resolve."""
    elapsed = 0
    while elapsed < timeout:
        try:
            title = await page.title()
            if "just a moment" not in title.lower():
                await asyncio.sleep(1)
                return True
        except Exception:
            pass
        await asyncio.sleep(3)
        elapsed += 3000
    return False


def extract_all_pdf_links(html: str) -> list[str]:
    """Extract all PDF download links from a detail page's HTML.

    Returns a deduplicated list of URLs, preserving order.
    """
    soup = BeautifulSoup(html, "lxml")
    urls = []
    seen = set()

    # Strategy 1: /media/{id} links (with or without /download suffix)
    for link in soup.find_all("a", href=re.compile(r"/media/\d+(/download)?$")):
        href = link["href"]
        if not href.rstrip("/").endswith("/download"):
            href = href.rstrip("/") + "/download"
        url = urljoin(BASE_URL, href)
        if url not in seen:
            urls.append(url)
            seen.add(url)

    # Strategy 2: direct .pdf links
    for link in soup.find_all("a", href=re.compile(r"\.pdf(\?|$)", re.I)):
        url = urljoin(BASE_URL, link["href"])
        if url not in seen:
            urls.append(url)
            seen.add(url)

    # Strategy 3: download buttons
    for link in soup.select("a.download, a[download], a[data-download]"):
        href = link.get("href")
        if href:
            url = urljoin(BASE_URL, href)
            if url not in seen:
                urls.append(url)
                seen.add(url)

    # Strategy 4: download-box links (older pages with external hosts)
    for link in soup.select("a.download-box__link"):
        href = link.get("href")
        if href:
            url = urljoin(BASE_URL, href)
            if url not in seen:
                urls.append(url)
                seen.add(url)

    return urls


def make_real_download_url(url: str) -> str:
    """Prepare a URL for actual download.

    Adds ?startDownload= to bypass interstitial pages on UNDRR and PreventionWeb.
    """
    # UNDRR media URLs need /download appended
    if re.search(r"/media/\d+", url):
        if "/download" not in url:
            url = url.rstrip("/") + "/download"

    # Both UNDRR and preventionweb use ?startDownload to bypass interstitial
    # UNDRR requires the date format (YYYYMMDD), not just "true"
    if ("preventionweb.net/" in url or "undrr.org/" in url) and "?" not in url:
        today = date.today().strftime("%Y%m%d")
        url = f"{url}?startDownload={today}"

    return url


DOCUMENT_MAGIC = {
    b"%PDF": ".pdf",
    b"PK\x03\x04": ".docx",  # ZIP archive (docx, xlsx, pptx, odt)
}


def detect_doc_ext(body: bytes, url: str = "") -> str | None:
    """Detect document type from magic bytes or URL extension.

    Returns file extension (.pdf, .docx, etc.) or None if not a document.
    """
    for magic, ext in DOCUMENT_MAGIC.items():
        if body[:len(magic)] == magic:
            return ext
    # Fall back to URL extension
    url_lower = url.lower().split("?")[0]
    for ext in (".pdf", ".docx", ".doc", ".xls", ".xlsx", ".pptx"):
        if url_lower.endswith(ext):
            return ext
    return None


def is_cross_origin(url: str) -> bool:
    """Check if URL is on a different domain than UNDRR."""
    return not url.startswith(BASE_URL)


async def download_cross_origin(context, doc_url: str, doc_path: Path) -> bool:
    """Download a document from a different domain using plain HTTP.

    Uses httpx for direct downloads — no browser needed for cross-origin files.
    Tries https:// if http:// fails.
    """
    urls_to_try = [doc_url]
    if doc_url.startswith("http://"):
        urls_to_try.insert(0, doc_url.replace("http://", "https://", 1))

    for url in urls_to_try:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                resp = await client.get(url)

            if resp.status_code >= 400:
                log.warning(f"    HTTP {resp.status_code} from {url}")
                continue

            body = resp.content
            if len(body) < 500:
                log.warning(f"    Too small ({len(body)} bytes)")
                continue

            if body.lstrip()[:15].lower().startswith(b"<!doctype html") or body.lstrip()[:5].lower() == b"<html":
                log.warning(f"    Got HTML instead of document")
                continue

            ext = detect_doc_ext(body, url)
            if not ext:
                log.warning(f"    Not a document (starts with: {body[:20]})")
                continue

            save_path = doc_path.with_suffix(ext)
            with open(save_path, "wb") as f:
                f.write(body)
            log.info(f"    Saved: {save_path.name} ({len(body):,} bytes)")
            return True

        except Exception as e:
            log.warning(f"    Cross-origin download error ({url}): {e}")

    return False


async def download_via_fetch(page: Page, doc_url: str, doc_path: Path) -> bool:
    """Download a document using JS fetch() within the authenticated page context."""
    try:
        result = await page.evaluate("""async (url) => {
            try {
                const resp = await fetch(url, {
                    credentials: 'include',
                    redirect: 'follow'
                });
                if (!resp.ok) return {error: resp.status, size: 0};
                const contentType = resp.headers.get('content-type') || '';
                const buf = await resp.arrayBuffer();
                let binary = '';
                const bytes = new Uint8Array(buf);
                const chunkSize = 32768;
                for (let i = 0; i < bytes.length; i += chunkSize) {
                    const chunk = bytes.subarray(i, i + chunkSize);
                    binary += String.fromCharCode.apply(null, chunk);
                }
                return {
                    error: null,
                    contentType: contentType,
                    size: buf.byteLength,
                    data: btoa(binary)
                };
            } catch(e) {
                return {error: e.message, size: 0};
            }
        }""", doc_url)

        if result.get("error"):
            log.warning(f"    Fetch error: {result['error']}")
            return False

        size = result.get("size", 0)
        if size < 500:
            log.warning(f"    Too small ({size} bytes), skipping")
            return False

        content_type = result.get("contentType", "")
        if "html" in content_type:
            log.warning(f"    Got HTML instead of document (content-type: {content_type})")
            return False

        data = base64.b64decode(result["data"])
        if data.lstrip()[:15].lower().startswith(b"<!doctype html") or data.lstrip()[:5].lower() == b"<html":
            log.warning(f"    Got HTML body, not a document")
            return False
        ext = detect_doc_ext(data, doc_url)
        if not ext:
            log.warning(f"    Not a document (starts with: {data[:20]})")
            return False

        save_path = doc_path.with_suffix(ext)
        with open(save_path, "wb") as f:
            f.write(data)
        log.info(f"    Saved: {save_path.name} ({len(data):,} bytes)")
        return True

    except Exception as e:
        log.error(f"    Download error: {e}")
        return False


DOC_EXTENSIONS = (".pdf", ".docx", ".doc", ".xls", ".xlsx", ".pptx")


def has_any_doc(slug: str) -> bool:
    """Check if any document file exists for this slug."""
    for ext in DOC_EXTENSIONS:
        main = PDF_DIR / f"{slug}{ext}"
        if main.exists() and main.stat().st_size > 1000:
            return True
    # Check for numbered variants
    for p in PDF_DIR.glob(f"{slug}_*.*"):
        if p.suffix.lower() in DOC_EXTENSIONS and p.stat().st_size > 1000:
            return True
    return False


async def run_downloads(max_downloads: int | None = None, reset: bool = False, retry_empty: bool = False):
    """Main download pipeline using real Edge browser."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()

    # Reset empty-string pdf_download_url entries so they get re-checked
    if reset or retry_empty:
        reset_count = 0
        for entry in catalog:
            if entry.get("pdf_download_url") == "":
                del entry["pdf_download_url"]
                if "pdf_download_urls" in entry:
                    del entry["pdf_download_urls"]
                reset_count += 1
        if reset_count:
            save_catalog(catalog)
            log.info(f"Reset {reset_count} entries for re-checking.")

    # Build work list
    work = []
    for entry in catalog:
        slug = entry.get("slug", "unknown")

        # Already checked and no PDF available?
        if entry.get("pdf_download_url") == "":
            continue

        if retry_empty:
            # Only process entries that have NO pdf_download_url at all (just reset above)
            if entry.get("pdf_download_url"):
                continue
            if has_any_doc(slug):
                continue
        else:
            # Normal mode: check if ALL PDFs are downloaded
            pdf_urls = entry.get("pdf_download_urls", [])
            if pdf_urls:
                all_done = True
                for idx in range(len(pdf_urls)):
                    stem = slug if idx == 0 else f"{slug}_{idx+1}"
                    found = any(
                        (PDF_DIR / f"{stem}{ext}").exists()
                        and (PDF_DIR / f"{stem}{ext}").stat().st_size > 1000
                        for ext in DOC_EXTENSIONS
                    )
                    if not found:
                        all_done = False
                        break
                if all_done:
                    continue
            elif has_any_doc(slug):
                continue

        work.append(entry)

    if max_downloads:
        work = work[:max_downloads]

    log.info(f"Documents to process: {len(work)}")
    if not work:
        log.info("Nothing to do.")
        return

    # Launch real Edge browser
    proc = launch_edge()

    try:
        async with async_playwright() as p:
            log.info("Connecting to Edge via CDP...")
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()

            # Step 1: Navigate to first detail page, solve Cloudflare if needed
            first_url = work[0]["source_url"]
            log.info(f"Navigating to first detail page...")
            log.info(f"  URL: {first_url}")
            log.info(f"  >>> Solve Cloudflare checkbox if it appears <<<")

            await page.goto(first_url, wait_until="domcontentloaded", timeout=60000)
            resolved = await wait_for_cf(page, timeout=120000)

            if not resolved:
                log.error("Could not get past Cloudflare. Exiting.")
                return

            log.info("Cloudflare OK! Starting PDF link discovery + download.\n")

            downloaded = 0
            failed = 0
            no_pdf = 0
            catalog_by_url = {e["source_url"]: e for e in catalog}

            for i, entry in enumerate(work):
              try:
                slug = entry.get("slug", "unknown")
                source_url = entry["source_url"]
                log.info(f"[{i+1}/{len(work)}] {slug}")

                # Ensure main page is still alive
                try:
                    await page.title()
                except Exception:
                    log.warning("  Main page lost, getting new one...")
                    page = context.pages[0] if context.pages else await context.new_page()

                # Check if we already have cached PDF URLs for this entry
                pdf_urls = entry.get("pdf_download_urls")  # list of URLs

                if not pdf_urls:
                    # Legacy single-URL field
                    single = entry.get("pdf_download_url")
                    if single:
                        pdf_urls = [single]

                if not pdf_urls:
                    # Visit detail page to discover PDF links
                    if i > 0:
                        await page.goto(source_url, wait_until="domcontentloaded", timeout=60000)
                        resolved = await wait_for_cf(page, timeout=60000)
                        if not resolved:
                            log.warning(f"  Cloudflare stuck, skipping")
                            failed += 1
                            continue

                    html = await page.content()
                    pdf_urls = extract_all_pdf_links(html)

                    if pdf_urls:
                        log.info(f"  Found {len(pdf_urls)} PDF link(s)")
                        for u in pdf_urls:
                            log.info(f"    - {u}")
                        entry["pdf_download_urls"] = pdf_urls
                        entry["pdf_download_url"] = pdf_urls[0]  # backwards compat
                        catalog_by_url[source_url] = entry
                    else:
                        log.info(f"  No PDF link found on detail page")
                        entry["pdf_download_url"] = ""
                        entry["pdf_download_urls"] = []
                        catalog_by_url[source_url] = entry
                        no_pdf += 1

                # Download all documents
                if pdf_urls:
                    for idx, pdf_url in enumerate(pdf_urls):
                        stem = slug if idx == 0 else f"{slug}_{idx+1}"
                        doc_path = PDF_DIR / f"{stem}.pdf"  # default ext, adjusted on save

                        # Skip if already downloaded (any extension)
                        existing = any(
                            (PDF_DIR / f"{stem}{ext}").exists()
                            and (PDF_DIR / f"{stem}{ext}").stat().st_size > 1000
                            for ext in DOC_EXTENSIONS
                        )
                        if existing:
                            log.info(f"    Already have: {stem}.*")
                            continue

                        real_url = make_real_download_url(pdf_url)
                        log.info(f"    Downloading: {real_url}")
                        if is_cross_origin(real_url):
                            success = await download_cross_origin(context, real_url, doc_path)
                        else:
                            success = await download_via_fetch(page, real_url, doc_path)
                        if success:
                            downloaded += 1
                        else:
                            failed += 1

              except Exception as e:
                log.error(f"  Error processing {entry.get('slug', '?')}: {e}")
                failed += 1

              finally:
                # Save progress every 20 entries
                if (i + 1) % 20 == 0:
                    save_catalog(list(catalog_by_url.values()))
                    log.info(f"  Checkpoint: {downloaded} saved, {failed} failed, {no_pdf} no PDF")

                # Respect robots.txt Crawl-Delay: 10
                if i < len(work) - 1:
                    delay = 10 + random.random() * 5
                    await asyncio.sleep(delay)

        # Final save
        save_catalog(list(catalog_by_url.values()))
        log.info(f"\nDone: {downloaded} downloaded, {failed} failed, {no_pdf} no PDF found.")

    finally:
        proc.terminate()


if __name__ == "__main__":
    args = sys.argv[1:]
    reset = "--reset" in args
    retry_empty = "--retry-empty" in args
    args = [a for a in args if not a.startswith("--")]
    max_dl = int(args[0]) if args else None
    asyncio.run(run_downloads(max_downloads=max_dl, reset=reset, retry_empty=retry_empty))
