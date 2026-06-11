"""Download missing document files without touching existing ones.

Scans catalog for entries where we have URLs but are missing files.
Uses httpx for cross-origin, Playwright fetch for same-origin (UNDRR).
"""

import asyncio
import base64
import json
import logging
import random
import re
import sys
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
CATALOG_PATH = DATA_DIR / "catalog.json"
DOC_EXTENSIONS = (".pdf", ".docx", ".doc", ".xls", ".xlsx", ".pptx")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Import shared helpers from download.py
from download import (
    BASE_URL,
    detect_doc_ext,
    is_cross_origin,
    make_real_download_url,
    launch_edge,
    wait_for_cf,
)


def file_exists(stem: str) -> bool:
    """Check if any document file exists for this stem."""
    return any(
        (PDF_DIR / f"{stem}{ext}").exists() and (PDF_DIR / f"{stem}{ext}").stat().st_size > 1000
        for ext in DOC_EXTENSIONS
    )


SKIP_DOMAINS = {"www.linkedin.com", "www.emerald.com", "cc.preventionweb.net"}

# URL patterns that are web pages, not downloadable documents
SKIP_URL_PATTERNS = [
    r"/v\.php\?id=",         # preventionweb view pages
    r"/arise/case-studies",  # web pages
    r"/publications/view/",  # web pages
    r"hfa-mtr/?$",           # web pages
    r"/home/?$",             # domain/site homepages
    r"/home/index\.html",
    r"\.html?(\?|$)",       # any HTML page
]


def is_downloadable_url(url: str) -> bool:
    """Check if URL looks like it points to a downloadable document."""
    from urllib.parse import urlparse
    parsed = urlparse(url)

    if parsed.netloc in SKIP_DOMAINS:
        return False

    # Skip bare domain roots (no path or just /)
    if not parsed.path or parsed.path.rstrip("/") == "":
        return False

    for pattern in SKIP_URL_PATTERNS:
        if re.search(pattern, url):
            return False

    return True


def find_gaps() -> list[dict]:
    """Find all (slug, stem, url) tuples where we have a URL but no file."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    gaps = []
    skipped = 0
    for entry in catalog:
        slug = entry.get("slug", "")
        urls = entry.get("pdf_download_urls", [])
        if not urls:
            single = entry.get("pdf_download_url", "")
            if single:
                urls = [single]
        if not urls:
            continue

        for idx, url in enumerate(urls):
            stem = slug if idx == 0 else f"{slug}_{idx+1}"
            if file_exists(stem):
                continue
            if not is_downloadable_url(url):
                skipped += 1
                continue
            gaps.append({"slug": slug, "stem": stem, "url": url})

    if skipped:
        log.info(f"Skipped {skipped} non-document URLs (web pages, dead domains, etc.)")
    return gaps


async def download_one(client: httpx.AsyncClient, url: str, stem: str, retries: int = 2) -> bool:
    """Download a single file via httpx with retries for connection drops."""
    real_url = make_real_download_url(url)
    for attempt in range(1, retries + 1):
        try:
            resp = await client.get(real_url)
            if resp.status_code >= 400:
                log.warning(f"    HTTP {resp.status_code}")
                return False

            body = resp.content
            if len(body) < 500:
                log.warning(f"    Too small ({len(body)} bytes)")
                return False

            if body.lstrip()[:15].lower().startswith(b"<!doctype html") or body.lstrip()[:5].lower() == b"<html":
                log.warning(f"    Got HTML instead of document")
                return False

            ext = detect_doc_ext(body, real_url)
            if not ext:
                log.warning(f"    Unknown file type (starts with: {body[:20]})")
                return False

            save_path = PDF_DIR / f"{stem}{ext}"
            with open(save_path, "wb") as f:
                f.write(body)
            log.info(f"    Saved: {save_path.name} ({len(body):,} bytes)")
            return True

        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
            if attempt < retries:
                log.info(f"    Connection dropped, retrying ({attempt}/{retries})...")
                await asyncio.sleep(3)
            else:
                log.warning(f"    Failed after {retries} attempts: {e}")
                return False
        except Exception as e:
            log.warning(f"    Error: {e}")
            return False
    return False


async def download_via_browser_fetch(page, url: str, stem: str) -> bool:
    """Download using JS fetch() in the authenticated browser page."""
    real_url = make_real_download_url(url)
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
        }""", real_url)

        if result.get("error"):
            log.warning(f"    Fetch error: {result['error']}")
            return False

        size = result.get("size", 0)
        if size < 500:
            log.warning(f"    Too small ({size} bytes)")
            return False

        data = base64.b64decode(result["data"])
        if data.lstrip()[:15].lower().startswith(b"<!doctype html") or data.lstrip()[:5].lower() == b"<html":
            # fetch() got interstitial HTML — fall back to navigation + download event
            return False

        ext = detect_doc_ext(data, real_url)
        if not ext:
            log.warning(f"    Unknown file type")
            return False

        save_path = PDF_DIR / f"{stem}{ext}"
        with open(save_path, "wb") as f:
            f.write(data)
        log.info(f"    Saved: {save_path.name} ({len(data):,} bytes)")
        return True

    except Exception as e:
        log.warning(f"    Browser fetch error: {e}")
        return False


async def download_via_browser_navigate(context, url: str, stem: str) -> bool:
    """Download by navigating in a new tab and catching the download event."""
    real_url = make_real_download_url(url)
    new_page = None
    try:
        new_page = await context.new_page()

        download_future = asyncio.get_event_loop().create_future()

        def on_download(dl):
            if not download_future.done():
                download_future.set_result(dl)

        new_page.on("download", on_download)

        try:
            await new_page.goto(real_url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            if "Download is starting" not in str(e):
                # Real navigation error — but still check for download event
                pass

        # Wait for the JS redirect / meta refresh to trigger a download
        try:
            download = await asyncio.wait_for(asyncio.shield(download_future), timeout=20)
        except asyncio.TimeoutError:
            log.warning(f"    No download triggered after navigation")
            await new_page.close()
            return False

        suggested = download.suggested_filename or ""
        ext = Path(suggested).suffix.lower() if suggested else None
        if not ext:
            ext = Path(url.split("?")[0]).suffix.lower() or ".pdf"

        save_path = PDF_DIR / f"{stem}{ext}"
        await download.save_as(str(save_path))
        await new_page.close()

        size = save_path.stat().st_size
        if size < 500:
            log.warning(f"    Too small ({size} bytes)")
            save_path.unlink(missing_ok=True)
            return False

        log.info(f"    Saved: {save_path.name} ({size:,} bytes)")
        return True

    except Exception as e:
        log.warning(f"    Navigate download error: {e}")
        if new_page:
            try:
                await new_page.close()
            except Exception:
                pass
        return False


async def run(max_downloads: int | None = None, browser_only: bool = False):
    gaps = find_gaps()
    log.info(f"Found {len(gaps)} missing files to download.")

    if not gaps:
        return

    if max_downloads:
        gaps = gaps[:max_downloads]

    downloaded = 0
    failed = 0

    if not browser_only:
        # Split into UNDRR (needs browser) and external (httpx first)
        undrr_gaps = [g for g in gaps if not is_cross_origin(g["url"])]
        external_gaps = [g for g in gaps if is_cross_origin(g["url"])]
        log.info(f"  UNDRR (browser): {len(undrr_gaps)}, External (httpx): {len(external_gaps)}")

        # --- External downloads via httpx ---
        httpx_failed = []
        if external_gaps:
            log.info("Downloading external files via httpx...")
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(connect=15, read=120, write=15, pool=15),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            ) as client:
                for i, gap in enumerate(external_gaps):
                    stem = gap["stem"]
                    if file_exists(stem):
                        continue

                    log.info(f"[{i+1}/{len(external_gaps)}] {stem}")
                    log.info(f"    URL: {make_real_download_url(gap['url'])}")

                    if await download_one(client, gap["url"], stem):
                        downloaded += 1
                    else:
                        httpx_failed.append(gap)
                        failed += 1

                    if i < len(external_gaps) - 1:
                        await asyncio.sleep(1 + random.random())

                    if (i + 1) % 50 == 0:
                        log.info(f"  Progress: {downloaded} saved, {failed} failed")

        # Combine UNDRR gaps + httpx failures for browser pass
        browser_gaps = undrr_gaps + httpx_failed
    else:
        browser_gaps = gaps

    # --- Browser downloads for everything that needs auth ---
    if browser_gaps:
        log.info(f"Downloading {len(browser_gaps)} files via browser...")
        from playwright.async_api import async_playwright

        proc = launch_edge()
        browser_downloaded = 0
        browser_failed = 0
        try:
            async with async_playwright() as p:
                browser = await p.chromium.connect_over_cdp("http://localhost:9222")
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else await context.new_page()

                # Navigate to UNDRR to pass Cloudflare
                first_url = "https://www.undrr.org/publication/documents-and-publications"
                log.info(f"Navigating to UNDRR for Cloudflare auth...")
                await page.goto(first_url, wait_until="domcontentloaded", timeout=60000)
                log.info("  >>> Solve Cloudflare checkbox if it appears <<<")
                resolved = await wait_for_cf(page, timeout=60000)
                if not resolved:
                    log.error("Cloudflare stuck, aborting browser downloads")
                    return

                log.info("Cloudflare OK! Starting downloads.")

                for i, gap in enumerate(browser_gaps):
                    stem = gap["stem"]
                    if file_exists(stem):
                        continue

                    log.info(f"[{i+1}/{len(browser_gaps)}] {stem}")
                    log.info(f"    URL: {make_real_download_url(gap['url'])}")

                    # Try fetch first, then navigate+download for interstitials
                    success = await download_via_browser_fetch(page, gap["url"], stem)
                    if not success and not file_exists(stem):
                        log.info(f"    Trying navigation download...")
                        success = await download_via_browser_navigate(context, gap["url"], stem)

                    if success:
                        downloaded += 1
                        browser_downloaded += 1
                    else:
                        browser_failed += 1

                    if i < len(browser_gaps) - 1:
                        await asyncio.sleep(1 + random.random())

                    if (i + 1) % 50 == 0:
                        log.info(f"  Browser progress: {browser_downloaded} saved, {browser_failed} failed")
        finally:
            proc.terminate()

    log.info(f"\nDone: {downloaded} downloaded, {failed} failed out of {len(gaps)} gaps.")


if __name__ == "__main__":
    browser_only = "--browser" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_dl = int(args[0]) if args else None
    asyncio.run(run(max_downloads=max_dl, browser_only=browser_only))
