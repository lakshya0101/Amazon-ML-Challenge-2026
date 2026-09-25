"""Configuration for blocking and candidate generation."""

from dataclasses import dataclass, field
from typing import Set, Optional


DEFAULT_NAME_STOPWORDS: Set[str] = {
    "group",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "llp",
    "pvt",
    "co",
    "company",
    "services",
    "service",
    "solutions",
    "solution",
    "international",
    "enterprises",
    "enterprise",
    "associates",
    "consulting",
    "management",
    "holdings",
    "holding",
    "the",
    "and",
    "of",
    "in",
    "for",
}

DEFAULT_ADDRESS_STOPWORDS: Set[str] = {
    "st",
    "street",
    "rd",
    "road",
    "ave",
    "avenue",
    "dr",
    "drive",
    "blvd",
    "boulevard",
    "ln",
    "lane",
    "hwy",
    "highway",
    "apt",
    "apartment",
    "ste",
    "suite",
    "fl",
    "floor",
    "unit",
    "box",
    "po",
    "bldg",
    "building",
}


@dataclass
class BlockingConfig:
    """Configuration options for multi-strategy candidate generation."""
    # Phase 1 Deterministic Strategies
    exact_name: bool = True
    exact_name_core: bool = True
    country_name_token: bool = True
    postal_code: bool = True
    house_number_address_token: bool = True

    # Token constraints
    min_name_token_length: int = 3
    min_address_token_length: int = 3
    name_stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_NAME_STOPWORDS))
    address_stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_ADDRESS_STOPWORDS))

    # Safety frequency threshold (skip tokens indexed in > max_bucket_size records; None or 0 to disable)
    max_bucket_size: Optional[int] = 5000
