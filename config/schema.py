"""Metadata schema for UNDRR documents using Pydantic."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Controlled vocabularies ──────────────────────────────────────────────────

DOCUMENT_TYPES = [
    "technical_report",
    "policy_brief",
    "case_study",
    "framework_document",
    "assessment",
    "guidance_note",
    "strategy",
    "action_plan",
    "progress_report",
    "working_paper",
    "infographic",
    "toolkit",
    "newsletter",
    "resolution",
    "declaration",
    "presentation",
    "other",
]

HAZARD_TYPES = [
    "earthquake",
    "flood",
    "cyclone",
    "drought",
    "wildfire",
    "tsunami",
    "landslide",
    "volcanic_eruption",
    "storm",
    "heatwave",
    "cold_wave",
    "pandemic",
    "epidemic",
    "avalanche",
    "tornado",
    "coastal_erosion",
    "desertification",
    "industrial_accident",
    "nuclear",
    "biological",
    "chemical",
    "multi_hazard",
]

SENDAI_PRIORITIES = [
    "priority_1_understanding_risk",
    "priority_2_governance",
    "priority_3_investing_in_drr",
    "priority_4_preparedness",
]

FRAMEWORKS = [
    "sendai_framework",
    "hyogo_framework",
    "paris_agreement",
    "sdgs",
    "new_urban_agenda",
    "addis_ababa_action_agenda",
    "samoa_pathway",
    "istanbul_programme_of_action",
]

THEMES = [
    "early_warning_systems",
    "resilience",
    "climate_adaptation",
    "urban_risk",
    "gender",
    "indigenous_knowledge",
    "disability_inclusion",
    "governance",
    "financing_drr",
    "risk_assessment",
    "recovery",
    "build_back_better",
    "capacity_building",
    "technology_transfer",
    "data_and_statistics",
    "community_based_drr",
    "health",
    "education",
    "agriculture",
    "water",
    "infrastructure",
    "ecosystems",
    "displacement",
    "conflict",
    "poverty",
    "private_sector",
    "multi_hazard_early_warning",
    "loss_and_damage",
    "anticipatory_action",
    "nature_based_solutions",
]

REGIONS = [
    "Africa",
    "Sub-Saharan Africa",
    "North Africa",
    "East Africa",
    "West Africa",
    "Southern Africa",
    "Central Africa",
    "Asia",
    "Central Asia",
    "East Asia",
    "South Asia",
    "Southeast Asia",
    "West Asia",
    "Pacific",
    "Small Island Developing States",
    "Europe",
    "Eastern Europe",
    "Western Europe",
    "Northern Europe",
    "Southern Europe",
    "Americas",
    "Latin America and the Caribbean",
    "Caribbean",
    "Central America",
    "South America",
    "North America",
    "Middle East",
    "Arab States",
    "Oceania",
    "Least Developed Countries",
    "Landlocked Developing Countries",
]


# ── Metadata model ───────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Full metadata schema for a UNDRR document."""

    # Identity
    slug: str = ""
    parent_slug: str = ""  # empty for primary doc, set for sub-files (_2, _3, etc.)

    # Bibliographic
    title: str = ""
    publication_date: str = ""
    language: str = "en"
    page_count: int = 0
    word_count: int = 0
    document_type: str = "other"
    source_url: str = ""
    filename: str = ""
    pdf_download_url: str = ""

    # Web metadata (scraped from listing/detail pages)
    web_description: str = ""
    web_themes: list[str] = Field(default_factory=list)
    web_hazards: list[str] = Field(default_factory=list)
    web_countries: list[str] = Field(default_factory=list)
    web_regions: list[str] = Field(default_factory=list)
    web_document_type: str = ""
    web_organizations: list[str] = Field(default_factory=list)

    # Geographic & temporal scope (enriched)
    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    temporal_coverage: str = ""

    # Domain-specific taxonomy
    hazard_types: list[str] = Field(default_factory=list)
    sendai_priorities: list[str] = Field(default_factory=list)
    sdg_links: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    frameworks_referenced: list[str] = Field(default_factory=list)

    # Retrieval-optimized
    key_entities: list[str] = Field(default_factory=list)
    retrieval_keywords: list[str] = Field(default_factory=list)
    query_anticipation: str = ""

    # Processing status
    pdf_downloaded: bool = False
    text_extracted: bool = False
    heuristic_extracted: bool = False
    llm_extracted: bool = False
