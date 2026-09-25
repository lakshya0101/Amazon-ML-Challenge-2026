"""
Address-based Blocker for Amazon ML Challenge 2026.
Generates candidate pairs by exploiting business address signals (country, zip/postal code, street tokens).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

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
    for item in s2_data:
        eid, bname, addr, country = item[0], item[1], item[2], (item[3] if len(item) > 3 else "")
        pcode = extract_postal_code(addr)
        if pcode and country:
            char = (first_token(bname) or " ")[0].lower()
            idx_s2[(country.strip(), pcode, char)].append(eid)

    # Index S3
    for item in s3_data:
        eid, bname, addr, country = item[0], item[1], item[2], (item[3] if len(item) > 3 else "")
        pcode = extract_postal_code(addr)
        if pcode and country:
            char = (first_token(bname) or " ")[0].lower()
            idx_s3[(country.strip(), pcode, char)].append(eid)

    candidates: Dict[str, Set[str]] = defaultdict(set)
    for item in s1_data:
        eid, bname, addr, country = item[0], item[1], item[2], (item[3] if len(item) > 3 else "")
        pcode = extract_postal_code(addr)
        if not pcode or not country:
            continue
        char = (first_token(bname) or " ")[0].lower()
        key = (country.strip(), pcode, char)

        for cid in idx_s2.get(key, [])[:max_candidates_per_block]:
            candidates[eid].add(cid)
        for cid in idx_s3.get(key, [])[:max_candidates_per_block]:
            candidates[eid].add(cid)

    return dict(candidates), name
