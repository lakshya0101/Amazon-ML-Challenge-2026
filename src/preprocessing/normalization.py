
import re
import unicodedata
from typing import Dict, List

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# Longer legal suffixes are checked first.
# This is important for 3-token suffixes such as
# "limited liability company".
LEGAL_SUFFIXES = {
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "public limited": "public ltd",
    "public ltd": "public ltd",
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "llp": "llp",
    "l.l.p": "llp",
    "llc": "llc",
    "l.l.c": "llc",
    "inc": "inc",
    "corp": "corp",
    "co": "co",
    "pvt": "pvt",
}

# Longest suffix first.
LEGAL_SUFFIX_PATTERNS = sorted(
    LEGAL_SUFFIXES.items(),
    key=lambda item: len(item[0].split()),
    reverse=True,
)


def clean_unicode(text: str) -> str:
    if text is None:
        return ""

    return unicodedata.normalize("NFKC", str(text))


def normalize_base(text: str, transliterate: bool = True) -> str:
    """
    Basic text normalization:
    - Unicode normalization
    - transliteration where available
    - lowercase
    - punctuation normalization
    - whitespace normalization
    """
    if text is None:
        return ""

    text = clean_unicode(text)

    if transliterate and unidecode is not None:
        text = unidecode(text)

    text = text.lower()

    # Normalize ampersand to "and".
    text = text.replace("&", " and ")

    # Keep alphanumeric characters.
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _match_legal_suffix(tokens: List[str]):
    """
    Return (number_of_tokens, replacement) for the longest
    matching legal suffix.
    """
    if not tokens:
        return 0, ""

    for suffix, replacement in LEGAL_SUFFIX_PATTERNS:
        suffix_tokens = suffix.split()

        if len(tokens) >= len(suffix_tokens):
            if tokens[-len(suffix_tokens):] == suffix_tokens:
                return len(suffix_tokens), replacement

    return 0, ""


def normalize_name(name: str) -> str:
    """
    Normalize a business name while preserving a canonical
    legal suffix.
    """
    normalized = normalize_base(name)

    if not normalized:
        return ""

    tokens = normalized.split()

    suffix_length, replacement = _match_legal_suffix(tokens)

    if suffix_length:
        tokens = (
            tokens[:-suffix_length]
            + replacement.split()
        )

    return " ".join(tokens)


def normalize_name_core(name: str) -> str:
    """
    Normalize a business name and remove its trailing
    legal suffix.

    Examples:
        ABC Private Limited
            -> abc

        ABC Limited Liability Company
            -> abc
    """
    normalized = normalize_base(name)

    if not normalized:
        return ""

    tokens = normalized.split()

    suffix_length, _replacement = _match_legal_suffix(tokens)

    if suffix_length:
        tokens = tokens[:-suffix_length]

    return " ".join(tokens)


def name_tokens(name: str) -> List[str]:
    normalized = normalize_name(name)

    if not normalized:
        return []

    return normalized.split()


def normalize_address(address: str) -> str:
    """
    Normalize address without imposing a country-specific
    address schema.
    """
    if address is None:
        return ""

    address = clean_unicode(address)

    if unidecode is not None:
        address = unidecode(address)

    address = address.lower()

    address = address.replace("&", " and ")

    address = re.sub(r"[^a-z0-9]+", " ", address)

    address = re.sub(r"\s+", " ", address).strip()

    return address


def normalize_country(country: str) -> str:
    """
    Open-set country normalization.

    No country list is hardcoded.
    """
    if country is None:
        return ""

    country = clean_unicode(country)

    if unidecode is not None:
        country = unidecode(country)

    country = country.lower()

    country = re.sub(r"[^a-z0-9]+", " ", country)

    country = re.sub(r"\s+", " ", country).strip()

    return country


def extract_postal_code(address: str) -> str:
    """
    Extract common numeric or alphanumeric postal-code patterns.
    """
    if not address:
        return ""

    address = str(address)

    # Numeric postal codes:
    # India PIN, US ZIP, etc.
    matches = re.findall(
        r"\b\d{5,6}(?:-\d{4})?\b",
        address,
    )

    if matches:
        return matches[-1].lower()

    # Generic alphanumeric postal code.
    matches = re.findall(
        r"\b[A-Z0-9]{3,4}\s?[A-Z0-9]{2,4}\b",
        address.upper(),
    )

    if matches:
        return matches[-1].replace(" ", "").lower()

    return ""


def extract_house_number(address: str) -> str:
    """
    Extract a leading house/building number.

    Examples:
        123 Main Street -> 123
        123A Main Street -> 123a
        12-14 Main Street -> 12-14
    """
    if not address:
        return ""

    address = str(address).strip()

    match = re.match(
        r"^\s*(\d+[A-Za-z]?(?:[-/]\d+[A-Za-z]?)?)",
        address,
    )

    if match:
        return match.group(1).lower()

    return ""


def normalize_record(
    business_name: str,
    business_address: str,
    country: str,
) -> Dict:
    """
    Normalize one raw record exactly once.

    The returned dictionary is the shared interface consumed
    by generate_pair_features().
    """
    normalized_address = normalize_address(
        business_address
    )

    return {
        # NAME
        "raw_name": (
            "" if business_name is None
            else str(business_name)
        ),
        "normalized_name": normalize_name(
            business_name
        ),
        "name_core": normalize_name_core(
            business_name
        ),
        "name_tokens": name_tokens(
            business_name
        ),

        # ADDRESS
        "raw_address": (
            "" if business_address is None
            else str(business_address)
        ),
        "normalized_address": normalized_address,
        "postal_code": extract_postal_code(
            business_address
        ),
        "house_number": extract_house_number(
            business_address
        ),

        # COUNTRY
        "raw_country": (
            "" if country is None
            else str(country)
        ),
        "normalized_country": normalize_country(
            country
        ),
    }
