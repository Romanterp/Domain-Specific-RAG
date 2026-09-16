"""
Discover all UNDRR document URLs and metadata from listing pages.

Uses Playwright to handle Cloudflare JS challenges.
Extracts rich metadata directly from listing page cards (title, date,
description, type, authors, countries, themes, hazards) — no need to
visit individual document detail pages.

Outputs a JSON catalog to data/catalog.json.
"""

import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, TimeoutError as PwTimeout

BASE_URL = "https://www.undrr.org"
LISTING_URL = f"{BASE_URL}/publications"
LISTING_PARAMS = "?page={page}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


async def wait_for_cloudflare(page: Page, timeout: int = 60000):
    """Wait for Cloudflare challenge to resolve (user may need to click checkbox)."""
    # Wait until the page title is no longer the Cloudflare challenge
    for _ in range(int(timeout / 2000)):
        title = await page.title()
        if "just a moment" not in title.lower():
            return
        await asyncio.sleep(2)
    log.warning("Cloudflare challenge may not have resolved — continuing anyway.")


def parse_listing_cards(html: str) -> list[dict]:
    """Parse document cards from listing page.

    Actual DOM structure (from UNDRR's Drupal site):
      div.views-row
        div.teaser-3-9.mg-card
          div.mg-card__content
            div.field--name-field-undrr-publication-types  -> subtype (Reports, Other, etc.)
            div.field--name-published-at > time            -> date
            div.field--name-node-title > header > a        -> title + URL
            p                                              -> description
            div.field--name-field-organization              -> organizations (each in .field--item)
    """
    soup = BeautifulSoup(html, "lxml")
    entries = []

    cards = soup.select("div.views-row")
    if not cards:
        return []

    for card in cards:
        entry = _extract_from_card(card)
        if entry:
            entries.append(entry)

    return entries


def _extract_from_card(card) -> dict | None:
    """Extract metadata from a single .views-row card."""
    # Title + URL from the title field link
    title_link = card.select_one(".field--name-node-title a, .mg-card__title a")
    if not title_link:
        return None

    href = title_link.get("href", "")
    if "/publication/" not in href:
        return None

    entry = {
        "source_url": urljoin(BASE_URL, href),
        "slug": href.rstrip("/").split("/")[-1],
        "title": title_link.get_text(strip=True),
    }

    # Publication subtype (Reports, Policy brief, Tool kit, etc.)
    type_el = card.select_one(".field--name-field-undrr-publication-types")
    if type_el:
        entry["web_document_type"] = type_el.get_text(strip=True)

    # Date
    time_el = card.select_one(".field--name-published-at time")
    if time_el:
        entry["publication_date"] = time_el.get_text(strip=True)
        iso_date = time_el.get("datetime", "")
        if iso_date:
            entry["publication_date_iso"] = iso_date

    # Description — plain <p> inside .mg-card__content
    content_div = card.select_one(".mg-card__content")
    if content_div:
        desc_p = content_div.find("p")
        if desc_p:
            entry["web_description"] = desc_p.get_text(strip=True)

    # Organizations — each in a separate .field--item inside .field--name-field-organization
    org_container = card.select_one(".field--name-field-organization")
    if org_container:
        org_items = org_container.select(".field--item")
        if org_items:
            entry["web_organizations"] = [item.get_text(strip=True) for item in org_items]
        else:
            text = org_container.get_text(strip=True)
            if text:
                entry["web_organizations"] = [text]

    # Taxonomy term IDs from data attributes (useful for linking)
    type_el_with_id = card.select_one("[class*='term-taxonomy--undrr_publication_type']")
    if type_el_with_id:
        term_id = type_el_with_id.get("data-term-id", "")
        if term_id:
            entry["web_type_term_id"] = term_id

    org_items_with_ids = card.select("[class*='term-taxonomy--organization']")
    if org_items_with_ids:
        entry["web_org_term_ids"] = [
            el.get("data-term-id", "") or el.get("data-org-id", "")
            for el in org_items_with_ids
            if el.get("data-term-id") or el.get("data-org-id")
        ]

    return entry


async def scrape_listing_page(page: Page, page_num: int) -> list[dict]:
    """Scrape a single listing page and return document entries with metadata."""
    url = LISTING_URL + LISTING_PARAMS.format(page=page_num)
    log.info(f"Scraping listing page {page_num}: {url}")

    for attempt in range(3):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            break
        except Exception as e:
            log.warning(f"Page {page_num}: attempt {attempt+1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(15 * (attempt + 1))
            else:
                log.error(f"Page {page_num}: giving up after 3 attempts.")
                return []

    await wait_for_cloudflare(page)

    # Give the page a moment to fully render
    await asyncio.sleep(3)

    content = await page.content()

    # Save first page HTML for debugging DOM structure
    if page_num == 0:
        debug_path = DATA_DIR / "debug_page_0.html"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(content)
        log.info(f"Saved debug HTML to {debug_path}")

    entries = parse_listing_cards(content)

    if not entries:
        # Check if we're past the last page
        soup = BeautifulSoup(content, "lxml")
        if not soup.select("div.views-row"):
            log.info(f"Page {page_num}: No document cards found — end of listing.")
            return []
        # Cards exist but parsing failed — save for debug
        log.warning(f"Page {page_num}: Found views-rows but parsing failed. Saving HTML.")
        debug_path = DATA_DIR / f"debug_page_{page_num}.html"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(content)
        return []

    log.info(f"Page {page_num}: Found {len(entries)} documents.")
    for e in entries:
        log.debug(f"  - {e.get('title', e.get('slug', '?'))[:60]}")

    return entries


async def run_discovery(
    max_pages: int | None = None,
    resume: bool = True,
):
    """Main discovery pipeline — listing pages only, no detail page visits."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing catalog for resume support
    catalog: dict[str, dict] = {}
    start_page = 0
    if resume and CATALOG_PATH.exists():
        with open(CATALOG_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        catalog = {e["source_url"]: e for e in existing}
        log.info(f"Resuming: loaded {len(catalog)} existing entries.")

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
                "Gecko/20100101 Firefox/120.0"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        # First visit: let user solve Cloudflare challenge
        log.info("Opening publications page — solve Cloudflare if prompted...")
        await page.goto(LISTING_URL + LISTING_PARAMS.format(page=0), wait_until="domcontentloaded")
        await wait_for_cloudflare(page, timeout=120000)  # Give user 2 min to solve
        await asyncio.sleep(3)

        # Now crawl listing pages
        log.info("=== Crawling listing pages for document metadata ===")
        page_num = start_page
        consecutive_empty = 0
        new_entries = 0

        while True:
            if max_pages is not None and page_num >= max_pages:
                break

            entries = await scrape_listing_page(page, page_num)

            if not entries:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    log.info("3 consecutive empty pages — stopping.")
                    break
                page_num += 1
                continue

            consecutive_empty = 0
            for entry in entries:
                url = entry["source_url"]
                if url not in catalog:
                    catalog[url] = entry
                    new_entries += 1
                else:
                    # Merge new metadata into existing entry
                    for k, v in entry.items():
                        if v and not catalog[url].get(k):
                            catalog[url][k] = v

            # Save progress every 10 pages
            if page_num % 10 == 0:
                _save_catalog(catalog)
                log.info(f"  Checkpoint: {len(catalog)} total, {new_entries} new.")

            page_num += 1
            # Respect robots.txt Crawl-Delay: 10
            delay = 10 + random.random() * 5
            await asyncio.sleep(delay)

        _save_catalog(catalog)
        await browser.close()

    log.info(f"Done: {len(catalog)} total documents, {new_entries} new. Saved to {CATALOG_PATH}")


def _save_catalog(catalog: dict):
    """Save catalog to JSON."""
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(list(catalog.values()), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(run_discovery(max_pages=max_pages))
