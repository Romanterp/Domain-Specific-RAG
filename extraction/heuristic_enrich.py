"""
Heuristic metadata enrichment for documents.json.

Extracts from text content using keyword/regex matching:
  - Countries (pycountry + common aliases)
  - Regions (derived from countries + direct mentions)
  - Frameworks referenced (keyword matching)
  - Organizations (from web metadata + text patterns)
  - Temporal coverage (year ranges)
  - Document type (mapped from web_document_type)

Also extracts:
  - Retrieval keywords (KeyBERT distinctive bigrams/trigrams; corpus-fitted vocabulary)

Does NOT attempt (reserved for LLM):
  - Hazard type classification (mentions ≠ "about")
  - Theme classification (needs semantic understanding)
  - Sendai priorities (needs interpretation)
  - Query anticipation
"""

import json
import logging
import re
import sys
from pathlib import Path

import pycountry
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.schema import FRAMEWORKS, REGIONS, DOCUMENT_TYPES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_DIR = DATA_DIR / "texts"
DOCUMENTS_PATH = DATA_DIR / "documents.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Country detection ────────────────────────────────────────────────────────

# Build lookup: lowercase name -> official name
COUNTRY_NAMES = {}
for country in pycountry.countries:
    COUNTRY_NAMES[country.name.lower()] = country.name
    if hasattr(country, "common_name"):
        COUNTRY_NAMES[country.common_name.lower()] = country.common_name

# Common aliases not in pycountry
COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "republic of korea": "Korea, Republic of",
    "south korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "dprk": "Korea, Democratic People's Republic of",
    "russia": "Russian Federation",
    "iran": "Iran, Islamic Republic of",
    "syria": "Syrian Arab Republic",
    "tanzania": "Tanzania, United Republic of",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "bolivia": "Bolivia, Plurinational State of",
    "vietnam": "Viet Nam",
    "laos": "Lao People's Democratic Republic",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "congo": "Congo",
    "drc": "Congo, The Democratic Republic of the",
    "democratic republic of congo": "Congo, The Democratic Republic of the",
    "myanmar": "Myanmar",
    "burma": "Myanmar",
    "east timor": "Timor-Leste",
    "cape verde": "Cabo Verde",
    "swaziland": "Eswatini",
    "czech republic": "Czechia",
    "palestine": "Palestine, State of",
    "micronesia": "Micronesia, Federated States of",
    "sao tome": "Sao Tome and Principe",
    "trinidad": "Trinidad and Tobago",
    "antigua": "Antigua and Barbuda",
    "st. lucia": "Saint Lucia",
    "st lucia": "Saint Lucia",
    "st. vincent": "Saint Vincent and the Grenadines",
}
for alias, official in COUNTRY_ALIASES.items():
    COUNTRY_NAMES[alias] = official

# Words that are also country names but commonly appear in other contexts
COUNTRY_FALSE_POSITIVES = {
    "chad", "chile", "china", "cuba", "dominica", "georgia", "guinea",
    "india", "jordan", "kenya", "mali", "monaco", "niger", "oman",
    "panama", "peru", "samoa", "togo", "turkey",
}

# Pre-compile patterns: match whole words, case-insensitive
# Sort by length descending so "United States of America" matches before "United States"
_sorted_names = sorted(COUNTRY_NAMES.keys(), key=len, reverse=True)
# Only compile patterns for names with 4+ chars to avoid false positives
COUNTRY_PATTERNS = []
for name in _sorted_names:
    if len(name) >= 4:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        COUNTRY_PATTERNS.append((pattern, COUNTRY_NAMES[name]))


def extract_countries(text: str) -> list[str]:
    """Extract country names from text."""
    found = set()
    # Use first 20k chars — countries are usually mentioned early
    sample = text[:20000]

    for pattern, official_name in COUNTRY_PATTERNS:
        if pattern.search(sample):
            found.add(official_name)

    return sorted(found)


# ── Country -> Region mapping ────────────────────────────────────────────────

# Simplified mapping using UN M49 groupings
# Region hierarchy is encoded in _COUNTRY_REGION mappings below

# Country to region(s) — built from pycountry + manual mapping
# This is a simplified version; a full mapping would use UN M49
_COUNTRY_REGION = {}

# African countries
for c in ["Nigeria", "Ghana", "Senegal", "Mali", "Niger", "Burkina Faso", "Guinea",
          "Sierra Leone", "Liberia", "Gambia", "Guinea-Bissau", "Cabo Verde",
          "Côte d'Ivoire", "Togo", "Benin", "Mauritania"]:
    _COUNTRY_REGION[c] = ["Africa", "West Africa"]

for c in ["Kenya", "Tanzania, United Republic of", "Uganda", "Rwanda", "Burundi",
          "Ethiopia", "Somalia", "Eritrea", "Djibouti", "South Sudan", "Sudan",
          "Comoros", "Madagascar", "Mauritius", "Seychelles", "Mozambique", "Malawi"]:
    _COUNTRY_REGION[c] = ["Africa", "East Africa"]

for c in ["South Africa", "Namibia", "Botswana", "Zimbabwe", "Zambia",
          "Eswatini", "Lesotho", "Angola"]:
    _COUNTRY_REGION[c] = ["Africa", "Southern Africa"]

for c in ["Cameroon", "Central African Republic", "Chad", "Congo",
          "Congo, The Democratic Republic of the", "Equatorial Guinea", "Gabon",
          "Sao Tome and Principe"]:
    _COUNTRY_REGION[c] = ["Africa", "Central Africa"]

for c in ["Egypt", "Libya", "Tunisia", "Algeria", "Morocco"]:
    _COUNTRY_REGION[c] = ["Africa", "North Africa", "Arab States"]

# Asian countries
for c in ["China", "Japan", "Korea, Republic of", "Korea, Democratic People's Republic of",
          "Mongolia"]:
    _COUNTRY_REGION[c] = ["Asia", "East Asia"]

for c in ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Bhutan",
          "Maldives", "Afghanistan"]:
    _COUNTRY_REGION[c] = ["Asia", "South Asia"]

for c in ["Thailand", "Viet Nam", "Myanmar", "Cambodia", "Lao People's Democratic Republic",
          "Philippines", "Indonesia", "Malaysia", "Singapore", "Brunei Darussalam",
          "Timor-Leste"]:
    _COUNTRY_REGION[c] = ["Asia", "Southeast Asia"]

for c in ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"]:
    _COUNTRY_REGION[c] = ["Asia", "Central Asia"]

for c in ["Iran, Islamic Republic of", "Iraq", "Syrian Arab Republic", "Lebanon",
          "Jordan", "Palestine, State of", "Yemen", "Saudi Arabia",
          "United Arab Emirates", "Qatar", "Bahrain", "Kuwait", "Oman"]:
    _COUNTRY_REGION[c] = ["Asia", "West Asia", "Arab States", "Middle East"]

# Europe
for c in ["France", "Germany", "Netherlands", "Belgium", "Luxembourg", "Austria",
          "Switzerland", "Liechtenstein"]:
    _COUNTRY_REGION[c] = ["Europe", "Western Europe"]

for c in ["United Kingdom", "Ireland", "Iceland", "Norway", "Sweden", "Finland",
          "Denmark", "Estonia", "Latvia", "Lithuania"]:
    _COUNTRY_REGION[c] = ["Europe", "Northern Europe"]

for c in ["Spain", "Portugal", "Italy", "Greece", "Malta", "Cyprus",
          "Slovenia", "Croatia", "Bosnia and Herzegovina", "Serbia",
          "Montenegro", "North Macedonia", "Albania", "Turkey"]:
    _COUNTRY_REGION[c] = ["Europe", "Southern Europe"]

for c in ["Russian Federation", "Ukraine", "Belarus", "Moldova, Republic of",
          "Poland", "Czechia", "Slovakia", "Hungary", "Romania", "Bulgaria",
          "Armenia", "Azerbaijan", "Georgia"]:
    _COUNTRY_REGION[c] = ["Europe", "Eastern Europe"]

# Americas
for c in ["United States", "Canada"]:
    _COUNTRY_REGION[c] = ["Americas", "North America"]

for c in ["Mexico", "Guatemala", "Belize", "Honduras", "El Salvador",
          "Nicaragua", "Costa Rica", "Panama"]:
    _COUNTRY_REGION[c] = ["Americas", "Central America", "Latin America and the Caribbean"]

for c in ["Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador",
          "Venezuela, Bolivarian Republic of", "Bolivia, Plurinational State of",
          "Paraguay", "Uruguay", "Guyana", "Suriname"]:
    _COUNTRY_REGION[c] = ["Americas", "South America", "Latin America and the Caribbean"]

for c in ["Cuba", "Haiti", "Dominican Republic", "Jamaica", "Trinidad and Tobago",
          "Barbados", "Bahamas", "Saint Lucia", "Grenada", "Saint Vincent and the Grenadines",
          "Antigua and Barbuda", "Dominica", "Saint Kitts and Nevis"]:
    _COUNTRY_REGION[c] = ["Americas", "Caribbean", "Latin America and the Caribbean",
                          "Small Island Developing States"]

# Pacific / Oceania
for c in ["Australia", "New Zealand"]:
    _COUNTRY_REGION[c] = ["Oceania"]

for c in ["Fiji", "Papua New Guinea", "Solomon Islands", "Vanuatu", "Samoa",
          "Tonga", "Kiribati", "Micronesia, Federated States of", "Palau",
          "Marshall Islands", "Nauru", "Tuvalu", "Cook Islands"]:
    _COUNTRY_REGION[c] = ["Oceania", "Pacific", "Small Island Developing States"]


def countries_to_regions(countries: list[str]) -> list[str]:
    """Derive regions from a list of countries."""
    regions = set()
    for country in countries:
        if country in _COUNTRY_REGION:
            regions.update(_COUNTRY_REGION[country])
    return sorted(regions)


# ── Framework detection ──────────────────────────────────────────────────────

FRAMEWORK_KEYWORDS = {
    "sendai_framework": [
        r"sendai\s+framework", r"sfdrr",
    ],
    "hyogo_framework": [
        r"hyogo\s+framework", r"hfa\b",
    ],
    "paris_agreement": [
        r"paris\s+agreement", r"paris\s+accord",
    ],
    "sdgs": [
        r"sustainable\s+development\s+goals?", r"\bsdgs?\b",
    ],
    "new_urban_agenda": [
        r"new\s+urban\s+agenda",
    ],
    "addis_ababa_action_agenda": [
        r"addis\s+ababa\s+action\s+agenda", r"\baaaa\b",
    ],
    "samoa_pathway": [
        r"samoa\s+pathway", r"s\.?a\.?m\.?o\.?a\.?\s+pathway",
        r"small\s+island\s+developing\s+states\s+accelerated\s+modalities",
    ],
    "istanbul_programme_of_action": [
        r"istanbul\s+programme\s+of\s+action", r"\bipoa\b",
    ],
}

_FRAMEWORK_PATTERNS = {
    fid: [re.compile(kw, re.IGNORECASE) for kw in kws]
    for fid, kws in FRAMEWORK_KEYWORDS.items()
}


def extract_frameworks(text: str) -> list[str]:
    """Detect referenced international frameworks."""
    sample = text[:30000]
    found = []
    for fid, patterns in _FRAMEWORK_PATTERNS.items():
        for p in patterns:
            if p.search(sample):
                found.append(fid)
                break
    return found


# ── Temporal coverage ────────────────────────────────────────────────────────

_YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_YEAR_RANGE_PATTERN = re.compile(r"\b(19[89]\d|20[0-4]\d)\s*[-–—]\s*(19[89]\d|20[0-4]\d)\b")


def extract_temporal_coverage(text: str) -> str:
    """Extract a temporal coverage range from years mentioned in text."""
    sample = text[:15000]

    # Check for explicit ranges first
    ranges = _YEAR_RANGE_PATTERN.findall(sample)
    if ranges:
        all_years = []
        for start, end in ranges:
            all_years.extend([int(start), int(end)])
        return f"{min(all_years)}-{max(all_years)}"

    # Fall back to individual years
    years = [int(y) for y in _YEAR_PATTERN.findall(sample)]
    if len(years) >= 2:
        return f"{min(years)}-{max(years)}"
    elif len(years) == 1:
        return str(years[0])

    return ""


# ── Document type mapping ────────────────────────────────────────────────────

_DOCTYPE_MAP = {
    "reports": "technical_report",
    "report": "technical_report",
    "technical report": "technical_report",
    "policy brief": "policy_brief",
    "policy briefs": "policy_brief",
    "case study": "case_study",
    "case studies": "case_study",
    "frameworks": "framework_document",
    "framework": "framework_document",
    "assessment": "assessment",
    "assessments": "assessment",
    "guidance note": "guidance_note",
    "guidance notes": "guidance_note",
    "guidelines": "guidance_note",
    "strategy": "strategy",
    "strategies": "strategy",
    "action plan": "action_plan",
    "action plans": "action_plan",
    "progress report": "progress_report",
    "working paper": "working_paper",
    "working papers": "working_paper",
    "infographic": "infographic",
    "infographics": "infographic",
    "tool kit": "toolkit",
    "toolkit": "toolkit",
    "toolkits": "toolkit",
    "newsletter": "newsletter",
    "newsletters": "newsletter",
    "resolution": "resolution",
    "resolutions": "resolution",
    "declaration": "declaration",
    "declarations": "declaration",
    "presentation": "presentation",
    "presentations": "presentation",
    "other": "other",
    "educational materials": "toolkit",
    "statements and messages": "declaration",
    "maps": "other",
    "words into action": "guidance_note",
    "plans": "action_plan",
    "academic and research papers": "working_paper",
}


def map_document_type(web_type: str) -> str:
    """Map web document type to controlled vocabulary."""
    return _DOCTYPE_MAP.get(web_type.lower().strip(), "other")


# ── Region detection from text ───────────────────────────────────────────────

_REGION_PATTERNS = {
    region: re.compile(r"\b" + re.escape(region) + r"\b", re.IGNORECASE)
    for region in REGIONS
}


def extract_regions_from_text(text: str) -> list[str]:
    """Detect region names mentioned directly in text."""
    sample = text[:20000]
    found = []
    for region, pattern in _REGION_PATTERNS.items():
        if pattern.search(sample):
            found.append(region)
    return found


# ── TF-IDF distinctive keyword extraction ───────────────────────────────────

MIN_WORDS_FOR_KEYWORDS = 500  # skip thin documents (flyers, cover pages)


def extract_retrieval_keywords(documents: list[dict], top_n: int = 10) -> dict[str, list[str]]:
    """Extract distinctive keywords per document using KeyBERT + corpus-aware filtering.

    Two-step approach:
    1. Fit a CountVectorizer on the full corpus to build a vocabulary of
       distinctive terms (max_df=0.15 filters generic DRR vocabulary,
       min_df=2 filters OCR noise).
    2. For each document, KeyBERT ranks vocabulary terms by semantic
       similarity to the document embedding (all-MiniLM-L6-v2).

    Only processes English documents with 500+ words.
    Returns dict: slug -> list of top distinctive terms.
    """
    slugs = []
    texts = []

    for doc in documents:
        if doc.get("language", "en") != "en":
            continue
        if doc.get("word_count", 0) < MIN_WORDS_FOR_KEYWORDS:
            continue
        slug = doc.get("slug", "")
        text_file = TEXT_DIR / f"{slug}.txt"
        if not text_file.exists() or text_file.stat().st_size == 0:
            continue
        text = text_file.read_text(encoding="utf-8", errors="ignore")[:30000]
        slugs.append(slug)
        texts.append(text)

    if len(texts) < 10:
        log.warning("Too few documents for keyword extraction, skipping")
        return {}

    log.info(f"Extracting keywords for {len(texts)} English documents (500+ words)...")

    # Step 1: Build corpus-level vocabulary of distinctive terms
    # max_features=50k keeps the top terms by corpus frequency — KeyBERT
    # has to embed every candidate, so 400k+ terms is way too slow
    corpus_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_df=0.15,
        min_df=2,
        max_features=50000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    corpus_vectorizer.fit(texts)
    vocab = corpus_vectorizer.vocabulary_
    log.info(f"  Corpus vocabulary: {len(vocab)} distinctive terms")

    # Step 2: Fixed-vocab vectorizer for per-doc KeyBERT extraction
    doc_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        vocabulary=vocab,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )

    import time
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Using device: {device}")

    log.info("  Loading KeyBERT model...")
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    if device == "cuda":
        kw_model.model.embedding_model.to(torch.device(device))
    log.info("  Model loaded.")

    # Resume from checkpoint if available
    checkpoint_path = DATA_DIR / "keywords_checkpoint.json"
    keywords_by_slug = {}
    if checkpoint_path.exists():
        try:
            keywords_by_slug = json.load(open(checkpoint_path, encoding="utf-8"))
            log.info(f"  Resuming from checkpoint: {len(keywords_by_slug)} already done")
        except Exception:
            pass

    t_start = time.time()
    for i, (slug, text) in enumerate(zip(slugs, texts)):
        if slug in keywords_by_slug:
            continue

        t0 = time.time()
        try:
            kw = kw_model.extract_keywords(
                text,
                vectorizer=doc_vectorizer,
                top_n=top_n,
                use_mmr=True,
                diversity=0.5,
            )
            terms = [word for word, score in kw]
            if terms:
                keywords_by_slug[slug] = terms
        except Exception:
            pass

        # Log first 3 docs individually so user can see it's working
        if i < 3:
            log.info(f"  [{i + 1}/{len(texts)}] {slug} ({time.time() - t0:.1f}s)")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            log.info(f"  Progress: {i + 1}/{len(texts)} ({len(keywords_by_slug)} with keywords) "
                     f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
        if (i + 1) % 200 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(keywords_by_slug, f)

    # Final save + cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    log.info(f"  Extracted keywords for {len(keywords_by_slug)} documents")
    return keywords_by_slug


# ── Main enrichment ──────────────────────────────────────────────────────────

def enrich_document(doc: dict) -> dict:
    """Apply heuristic enrichment to a single document entry."""
    slug = doc.get("slug", "")
    text_file = TEXT_DIR / f"{slug}.txt"

    if not text_file.exists() or text_file.stat().st_size == 0:
        return doc

    text = text_file.read_text(encoding="utf-8", errors="ignore")

    # Countries
    countries = extract_countries(text)
    doc["countries"] = countries

    # Regions — from countries + direct text mentions
    regions = set(countries_to_regions(countries))
    regions.update(extract_regions_from_text(text))
    doc["regions"] = sorted(regions)

    # Frameworks
    doc["frameworks_referenced"] = extract_frameworks(text)

    # Temporal coverage
    doc["temporal_coverage"] = extract_temporal_coverage(text)

    # Document type (from web metadata)
    if doc.get("web_document_type"):
        doc["document_type"] = map_document_type(doc["web_document_type"])

    # Organizations (carry over from web metadata)
    if doc.get("web_organizations") and not doc.get("organizations"):
        doc["organizations"] = doc["web_organizations"]

    doc["heuristic_extracted"] = True
    return doc


def enrich_all():
    """Run heuristic enrichment on all documents."""
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        documents = json.load(f)

    log.info(f"Enriching {len(documents)} documents...")

    enriched = 0
    skipped = 0

    for doc in documents:
        if not doc.get("text_extracted"):
            skipped += 1
            continue

        enrich_document(doc)
        enriched += 1

        if enriched % 500 == 0:
            log.info(f"  Progress: {enriched} enriched...")

    # TF-IDF keyword extraction (corpus-level)
    # Clear old keywords first so non-English docs don't keep stale results
    for doc in documents:
        doc["retrieval_keywords"] = []

    keywords_by_slug = extract_retrieval_keywords(documents, top_n=10)
    keywords_applied = 0
    for doc in documents:
        slug = doc.get("slug", "")
        if slug in keywords_by_slug:
            doc["retrieval_keywords"] = keywords_by_slug[slug]
            keywords_applied += 1

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # Stats
    with_countries = sum(1 for d in documents if d.get("countries"))
    with_frameworks = sum(1 for d in documents if d.get("frameworks_referenced"))
    with_temporal = sum(1 for d in documents if d.get("temporal_coverage"))

    log.info(f"Done: {enriched} enriched, {skipped} skipped (no text)")
    log.info(f"  With countries: {with_countries}")
    log.info(f"  With frameworks: {with_frameworks}")
    log.info(f"  With temporal coverage: {with_temporal}")
    log.info(f"  With retrieval keywords: {keywords_applied}")


if __name__ == "__main__":
    if "--keywords-only" in sys.argv:
        with open(DOCUMENTS_PATH, encoding="utf-8") as f:
            documents = json.load(f)
        log.info(f"Keywords-only mode: {len(documents)} documents")

        for doc in documents:
            doc["retrieval_keywords"] = []

        keywords_by_slug = extract_retrieval_keywords(documents, top_n=10)
        keywords_applied = 0
        for doc in documents:
            slug = doc.get("slug", "")
            if slug in keywords_by_slug:
                doc["retrieval_keywords"] = keywords_by_slug[slug]
                keywords_applied += 1

        with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        log.info(f"Done: {keywords_applied} documents with retrieval keywords")
    else:
        enrich_all()
