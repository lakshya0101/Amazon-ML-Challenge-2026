"""Candidate retrieval strategies with provenance tracking."""

from typing import Dict, List, Set, Any, Optional
from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex, ensure_normalized_record


RULE_EXACT_NAME = "exact_name"
RULE_NAME_CORE = "name_core"
RULE_COUNTRY_TOKEN = "country_name_token"
RULE_POSTAL_CODE = "postal_code"
RULE_HOUSE_ADDRESS_TOKEN = "house_number_address_token"

ALL_RULES = [
    RULE_EXACT_NAME,
    RULE_NAME_CORE,
    RULE_COUNTRY_TOKEN,
    RULE_POSTAL_CODE,
    RULE_HOUSE_ADDRESS_TOKEN,
]


def retrieve_exact_name(record: Dict[str, Any], index: BlockingIndex, max_bucket_size: Optional[int] = None) -> List[str]:
    """Retrieves candidates sharing exact normalized name."""
    norm_name = record.get("normalized_name", "")
    if not norm_name:
        return []

    bucket = index.exact_name_index.get(norm_name, [])
    if max_bucket_size and len(bucket) > max_bucket_size:
        return []
    return bucket


def retrieve_exact_name_core(record: Dict[str, Any], index: BlockingIndex, max_bucket_size: Optional[int] = None) -> List[str]:
    """Retrieves candidates sharing exact name core."""
    name_core = record.get("name_core", "")
    if not name_core:
        return []

    bucket = index.name_core_index.get(name_core, [])
    if max_bucket_size and len(bucket) > max_bucket_size:
        return []
    return bucket


def retrieve_country_name_token(
    record: Dict[str, Any],
    index: BlockingIndex,
    config: BlockingConfig,
) -> List[str]:
    """Retrieves candidates sharing (country, informative_name_token)."""
    country = record.get("normalized_country", "")
    tokens = record.get("name_tokens", [])
    if not country or not tokens:
        return []

    candidates: List[str] = []
    seen_tokens: Set[str] = set()

    for token in tokens:
        if (
            len(token) >= config.min_name_token_length
            and token not in config.name_stopwords
            and token not in seen_tokens
        ):
            seen_tokens.add(token)
            bucket = index.country_token_index.get((country, token), [])
            if config.max_bucket_size and len(bucket) > config.max_bucket_size:
                continue
            candidates.extend(bucket)

    return candidates


def retrieve_postal_code(record: Dict[str, Any], index: BlockingIndex, max_bucket_size: Optional[int] = None) -> List[str]:
    """Retrieves candidates sharing exact postal code."""
    postal_code = record.get("postal_code", "")
    if not postal_code:
        return []

    bucket = index.postal_index.get(postal_code, [])
    if max_bucket_size and len(bucket) > max_bucket_size:
        return []
    return bucket


def retrieve_house_number_address_token(
    record: Dict[str, Any],
    index: BlockingIndex,
    config: BlockingConfig,
) -> List[str]:
    """Retrieves candidates sharing (house_number, informative_address_token)."""
    house_number = record.get("house_number", "")
    norm_address = record.get("normalized_address", "")
    if not house_number or not norm_address:
        return []

    addr_tokens = norm_address.split()
    candidates: List[str] = []
    seen_tokens: Set[str] = set()

    for token in addr_tokens:
        if (
            len(token) >= config.min_address_token_length
            and token != house_number
            and token not in config.address_stopwords
            and token not in seen_tokens
        ):
            seen_tokens.add(token)
            bucket = index.house_address_token_index.get((house_number, token), [])
            if config.max_bucket_size and len(bucket) > config.max_bucket_size:
                continue
            candidates.extend(bucket)

    return candidates


def retrieve_candidates_with_provenance(
    s1_record: Dict[str, Any],
    index: BlockingIndex,
    config: BlockingConfig,
) -> Dict[str, Set[str]]:
    """Retrieves candidates across all active strategies, tracking firing rules per candidate."""
    s1_norm = ensure_normalized_record(s1_record)
    candidate_provenance: Dict[str, Set[str]] = {}

    # 1. Exact Name
    if config.exact_name:
        for match_id in retrieve_exact_name(s1_norm, index, config.max_bucket_size):
            if match_id not in candidate_provenance:
                candidate_provenance[match_id] = set()
            candidate_provenance[match_id].add(RULE_EXACT_NAME)

    # 2. Exact Name Core
    if config.exact_name_core:
        for match_id in retrieve_exact_name_core(s1_norm, index, config.max_bucket_size):
            if match_id not in candidate_provenance:
                candidate_provenance[match_id] = set()
            candidate_provenance[match_id].add(RULE_NAME_CORE)

    # 3. Country + Informative Name Token
    if config.country_name_token:
        for match_id in retrieve_country_name_token(s1_norm, index, config):
            if match_id not in candidate_provenance:
                candidate_provenance[match_id] = set()
            candidate_provenance[match_id].add(RULE_COUNTRY_TOKEN)

    # 4. Postal Code
    if config.postal_code:
        for match_id in retrieve_postal_code(s1_norm, index, config.max_bucket_size):
            if match_id not in candidate_provenance:
                candidate_provenance[match_id] = set()
            candidate_provenance[match_id].add(RULE_POSTAL_CODE)

    # 5. House Number + Informative Address Token
    if config.house_number_address_token:
        for match_id in retrieve_house_number_address_token(s1_norm, index, config):
            if match_id not in candidate_provenance:
                candidate_provenance[match_id] = set()
            candidate_provenance[match_id].add(RULE_HOUSE_ADDRESS_TOKEN)

    return candidate_provenance
