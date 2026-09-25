
from rapidfuzz import fuzz

from src.preprocessing.normalization import (
    normalize_name,
    normalize_name_core,
    normalize_address,
    normalize_country,
    name_tokens,
    extract_postal_code,
    extract_house_number,
)


def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(
        str(a),
        str(b)
    ) / 100.0


def token_sort_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(
        str(a),
        str(b)
    ) / 100.0


def token_set_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(
        str(a),
        str(b)
    ) / 100.0


def partial_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(
        str(a),
        str(b)
    ) / 100.0


def jaccard_tokens(a, b):

    if isinstance(a, str):
        a = a.split()

    if isinstance(b, str):
        b = b.split()

    a = set(a or [])
    b = set(b or [])

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def generate_pair_features(record_a, record_b):

    name_a = record_a.get(
        "business_name",
        record_a.get("raw_name", "")
    )

    name_b = record_b.get(
        "business_name",
        record_b.get("raw_name", "")
    )

    address_a = record_a.get(
        "business_address",
        record_a.get("raw_address", "")
    )

    address_b = record_b.get(
        "business_address",
        record_b.get("raw_address", "")
    )

    country_a = record_a.get(
        "country",
        record_a.get("raw_country", "")
    )

    country_b = record_b.get(
        "country",
        record_b.get("raw_country", "")
    )

    norm_name_a = normalize_name(name_a)
    norm_name_b = normalize_name(name_b)

    core_name_a = normalize_name_core(name_a)
    core_name_b = normalize_name_core(name_b)

    norm_address_a = normalize_address(address_a)
    norm_address_b = normalize_address(address_b)

    country_norm_a = normalize_country(country_a)
    country_norm_b = normalize_country(country_b)

    tokens_a = name_tokens(name_a)
    tokens_b = name_tokens(name_b)

    postal_a = extract_postal_code(address_a)
    postal_b = extract_postal_code(address_b)

    house_a = extract_house_number(address_a)
    house_b = extract_house_number(address_b)

    return {

        "name_exact": float(
            bool(norm_name_a)
            and norm_name_a == norm_name_b
        ),

        "name_core_exact": float(
            bool(core_name_a)
            and core_name_a == core_name_b
        ),

        "name_fuzzy": safe_ratio(
            norm_name_a,
            norm_name_b
        ),

        "name_token_sort": token_sort_similarity(
            norm_name_a,
            norm_name_b
        ),

        "name_token_set": token_set_similarity(
            norm_name_a,
            norm_name_b
        ),

        "name_partial": partial_similarity(
            norm_name_a,
            norm_name_b
        ),

        "name_jaccard": jaccard_tokens(
            tokens_a,
            tokens_b
        ),

        "name_core_fuzzy": safe_ratio(
            core_name_a,
            core_name_b
        ),

        "address_exact": float(
            bool(norm_address_a)
            and norm_address_a == norm_address_b
        ),

        "address_fuzzy": safe_ratio(
            norm_address_a,
            norm_address_b
        ),

        "address_token_sort": token_sort_similarity(
            norm_address_a,
            norm_address_b
        ),

        "address_token_set": token_set_similarity(
            norm_address_a,
            norm_address_b
        ),

        "address_jaccard": jaccard_tokens(
            norm_address_a,
            norm_address_b
        ),

        "postal_match": float(
            bool(postal_a)
            and bool(postal_b)
            and postal_a == postal_b
        ),

        "house_number_match": float(
            bool(house_a)
            and bool(house_b)
            and house_a == house_b
        ),

        "address_missing_a": float(
            not bool(norm_address_a)
        ),

        "address_missing_b": float(
            not bool(norm_address_b)
        ),

        "both_addresses_missing": float(
            not bool(norm_address_a)
            and not bool(norm_address_b)
        ),

        "country_normalized_match": float(
            bool(country_norm_a)
            and bool(country_norm_b)
            and country_norm_a == country_norm_b
        ),
    }
