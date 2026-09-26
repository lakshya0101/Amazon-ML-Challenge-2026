
from typing import Dict

from rapidfuzz import fuzz


FEATURE_COLUMNS = [
    # NAME
    "name_exact",
    "name_core_exact",
    "name_fuzzy",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_jaccard",
    "name_core_fuzzy",

    # ADDRESS
    "address_exact",
    "address_fuzzy",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
    "postal_match",
    "house_number_match",
    "address_missing_a",
    "address_missing_b",
    "both_addresses_missing",

    # COUNTRY
    "country_missing_a",
    "country_missing_b",
    "country_normalized_match",
    "country_similarity",
]


def safe_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def token_sort_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def partial_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b) / 100.0


def jaccard_tokens(tokens_a, tokens_b) -> float:
    set_a = set(tokens_a or [])
    set_b = set(tokens_b or [])

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def _exact_match(a: str, b: str) -> float:
    """
    Empty-vs-empty is not treated as a positive match.
    """
    if not a or not b:
        return 0.0

    return float(a == b)


def _missing(value: str) -> float:
    return float(not value)


def generate_pair_features(
    normalized_record_a: Dict,
    normalized_record_b: Dict,
) -> Dict[str, float]:
    """
    Generate the final 22 shared features.

    IMPORTANT:
    Inputs must already be normalized using normalize_record().

    No normalization is performed here.
    """

    # =====================================================
    # NAME
    # =====================================================

    name_a = normalized_record_a.get(
        "normalized_name",
        "",
    )

    name_b = normalized_record_b.get(
        "normalized_name",
        "",
    )

    core_a = normalized_record_a.get(
        "name_core",
        "",
    )

    core_b = normalized_record_b.get(
        "name_core",
        "",
    )

    tokens_a = normalized_record_a.get(
        "name_tokens",
        [],
    )

    tokens_b = normalized_record_b.get(
        "name_tokens",
        [],
    )

    name_exact = _exact_match(
        name_a,
        name_b,
    )

    name_core_exact = _exact_match(
        core_a,
        core_b,
    )

    name_fuzzy = safe_ratio(
        name_a,
        name_b,
    )

    name_token_sort = token_sort_similarity(
        name_a,
        name_b,
    )

    name_token_set = token_set_similarity(
        name_a,
        name_b,
    )

    name_partial = partial_similarity(
        name_a,
        name_b,
    )

    name_jaccard = jaccard_tokens(
        tokens_a,
        tokens_b,
    )

    name_core_fuzzy = safe_ratio(
        core_a,
        core_b,
    )

    # =====================================================
    # ADDRESS
    # =====================================================

    address_a = normalized_record_a.get(
        "normalized_address",
        "",
    )

    address_b = normalized_record_b.get(
        "normalized_address",
        "",
    )

    address_exact = _exact_match(
        address_a,
        address_b,
    )

    address_fuzzy = safe_ratio(
        address_a,
        address_b,
    )

    address_token_sort = token_sort_similarity(
        address_a,
        address_b,
    )

    address_token_set = token_set_similarity(
        address_a,
        address_b,
    )

    address_tokens_a = (
        address_a.split()
        if address_a
        else []
    )

    address_tokens_b = (
        address_b.split()
        if address_b
        else []
    )

    address_jaccard = jaccard_tokens(
        address_tokens_a,
        address_tokens_b,
    )

    postal_a = normalized_record_a.get(
        "postal_code",
        "",
    )

    postal_b = normalized_record_b.get(
        "postal_code",
        "",
    )

    postal_match = _exact_match(
        postal_a,
        postal_b,
    )

    house_a = normalized_record_a.get(
        "house_number",
        "",
    )

    house_b = normalized_record_b.get(
        "house_number",
        "",
    )

    house_number_match = _exact_match(
        house_a,
        house_b,
    )

    address_missing_a = _missing(
        address_a
    )

    address_missing_b = _missing(
        address_b
    )

    both_addresses_missing = float(
        not address_a and not address_b
    )

    # =====================================================
    # COUNTRY
    # =====================================================

    country_a = normalized_record_a.get(
        "normalized_country",
        "",
    )

    country_b = normalized_record_b.get(
        "normalized_country",
        "",
    )

    country_missing_a = _missing(
        country_a
    )

    country_missing_b = _missing(
        country_b
    )

    country_normalized_match = _exact_match(
        country_a,
        country_b,
    )

    country_similarity = safe_ratio(
        country_a,
        country_b,
    )

    # =====================================================
    # FINAL 22 FEATURES
    # =====================================================

    features = {
        # NAME
        "name_exact": name_exact,
        "name_core_exact": name_core_exact,
        "name_fuzzy": name_fuzzy,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_partial": name_partial,
        "name_jaccard": name_jaccard,
        "name_core_fuzzy": name_core_fuzzy,

        # ADDRESS
        "address_exact": address_exact,
        "address_fuzzy": address_fuzzy,
        "address_token_sort": address_token_sort,
        "address_token_set": address_token_set,
        "address_jaccard": address_jaccard,
        "postal_match": postal_match,
        "house_number_match": house_number_match,
        "address_missing_a": address_missing_a,
        "address_missing_b": address_missing_b,
        "both_addresses_missing": both_addresses_missing,

        # COUNTRY
        "country_missing_a": country_missing_a,
        "country_missing_b": country_missing_b,
        "country_normalized_match": country_normalized_match,
        "country_similarity": country_similarity,
    }

    # Ensure the implementation cannot silently drift
    # away from the agreed feature interface.
    assert list(features.keys()) == FEATURE_COLUMNS

    return features
