"""
Address-based Blocker for Amazon ML Challenge 2026.
Generates candidate pairs by exploiting business address signals (country, zip/postal code, street tokens).
Uses canonical EntityRecord representation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

from src.data.record import EntityRecord, get_field
from src.normalization.name_normalizer import normalize_address, first_token

# Regex for postal/zip codes (5 digit US, 6 digit India, 5 digit France)
_POSTAL_CODE_RE = re.compile(r"\b(\d{5,6})\b")


def extract_postal_code(address: str | None) -> str:
    """Extract 5 or 6 digit postal/zip code from address if present."""
    if not address:
        return ""
    m = _POSTAL_CODE_RE.search(address)
    return m.group(1) if m else ""


def extract_address_key(address: str | None, business_name: str | None = None) -> str:
    """
    Generate an address blocking key.
    Combines clean street/number tokens with the first letter of business name
    to prevent dense address blocks from creating huge Cartesian explosions.
    """
    if not address:
        return ""
    norm_addr = normalize_address(address)
    tokens = norm_addr.split()
    if not tokens:
        return ""

    # Look for leading house/street number or postal code
    lead = tokens[0] if len(tokens[0]) >= 2 else ""
    first_char = (first_token(business_name) or " ")[0].lower()

    if lead:
        return f"{lead}_{first_char}"
    return ""


def blocker_country_postal_initial(
    s1_data: List[Any],
    s2_data: List[Any],
    s3_data: List[Any],
    max_candidates_per_block: int = 500,
) -> Tuple[Dict[str, Set[str]], str]:
    """
    Address Blocker:
    Matches records sharing (country, postal_code, name_first_char).
    Only applies to records where postal code can be extracted.
    """
    name = "B_address_postal_initial"
    idx_s2: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    idx_s3: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    # Index S2
    for row in s2_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        business_address = get_field(row, 'business_address', 2)
        country = get_field(row, 'country', 3)
        pcode = extract_postal_code(business_address)
        c = str(country).strip()
        if pcode and c:
            char = (first_token(business_name) or " ")[0].lower()
            idx_s2[(c, pcode, char)].append(entity_id)

    # Index S3
    for row in s3_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        business_address = get_field(row, 'business_address', 2)
        country = get_field(row, 'country', 3)
        pcode = extract_postal_code(business_address)
        c = str(country).strip()
        if pcode and c:
            char = (first_token(business_name) or " ")[0].lower()
            idx_s3[(c, pcode, char)].append(entity_id)

    candidates: Dict[str, Set[str]] = defaultdict(set)
    for row in s1_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        business_address = get_field(row, 'business_address', 2)
        country = get_field(row, 'country', 3)
        pcode = extract_postal_code(business_address)
        c = str(country).strip()
        if not pcode or not c:
            continue
        char = (first_token(business_name) or " ")[0].lower()
        key = (c, pcode, char)

        for cid in idx_s2.get(key, [])[:max_candidates_per_block]:
            candidates[entity_id].add(cid)
        for cid in idx_s3.get(key, [])[:max_candidates_per_block]:
            candidates[entity_id].add(cid)

    return dict(candidates), name
