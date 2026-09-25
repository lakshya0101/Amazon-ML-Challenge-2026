"""
Business-name normalizer — STUB interface
==========================================

This module provides a *minimal, clean normalization interface* for business
names.  It is intentionally a thin stub so that Smriti (or anyone) can drop in
a richer implementation later WITHOUT changing any call-sites in the blockers.

Design contract
---------------
Every public function takes a raw string (possibly None/NaN) and returns a
deterministic, lowercased, whitespace-collapsed string.

Replacement rule
----------------
To swap in a better normalizer:
1.  Keep the same function signatures below.
2.  Replace the bodies.
3.  Do NOT rename the module or change its import path.

Functions
---------
normalize_name(text) → str
    Primary normalization for business names used in blocking keys.

normalize_address(text) → str
    Light normalization for address fields.

first_token(text) → str
    Returns the first whitespace token of the normalized name (used for
    prefix blocking).

informative_tokens(text, stopwords=None) → list[str]
    Returns a sorted, deduplicated list of "informative" tokens from the
    normalized name (used for token-based blocking).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ---------------------------------------------------------------------------
# Legal-suffix & noise patterns
# Note: deliberately exhaustive for English + common transliterations.
#       Smriti's implementation may handle Indic / French suffixes better.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIXES = re.compile(
    r"\b(pvt\.?|private|ltd\.?|limited|llc|llp|inc\.?|incorporated|"
    r"corp\.?|corporation|co\.?|company|enterprises?|group|holdings?|"
    r"services?|solutions?|international|intl\.?|industries?|associates?|"
    r"assoc\.?|brothers?|bros\.?|partners?|trading|traders?|agency|"
    r"agencies|consultants?|consultant|gmbh|sarl|sas|sa|bv|nv|ab|oy|"
    r"plc\.?|pty\.?|proprietor|prop\.?)\b",
    re.IGNORECASE,
)

_NOISE_CHARS = re.compile(r"[\"'`,.\-/\\|@#%^*=+<>{}()\[\]~]")

_MULTI_SPACE = re.compile(r"\s{2,}")

_AND_NORM = re.compile(r"\b&\b")

# Stopwords for the token-based blocker.
_DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "at", "by", "for",
        "to", "on", "is", "with", "de", "la", "le", "les", "du",
        # common Indian noise words
        "india", "bharat", "new",
    }
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def normalize_name(text: str | None) -> str:
    """Return a canonical, lowercased business-name string.

    Steps (in order):
        1. Unicode NFKD → ASCII strip (drops accents; keeps script chars for
           non-Latin names that will still match via equality).
        2. Lower-case.
        3. '&' → 'and'.
        4. Strip noise punctuation.
        5. Remove legal suffixes.
        6. Collapse whitespace.

    Returns "" for None / empty / whitespace-only input.
    """
    if not text or not str(text).strip():
        return ""
    s: str = str(text).strip()

    # 1. Normalize unicode (NFKD + ASCII re-encode where possible)
    #    We keep non-ASCII chars so Devanagari / French accented names still
    #    match each other; we only strip *combining diacritics* on Latin.
    s = unicodedata.normalize("NFKC", s)

    # 2. Lower-case
    s = s.lower()

    # 3. & → and
    s = _AND_NORM.sub(" and ", s)

    # 4. Strip noise punctuation
    s = _NOISE_CHARS.sub(" ", s)

    # 5. Remove legal suffixes
    s = _LEGAL_SUFFIXES.sub(" ", s)

    # 6. Collapse whitespace
    s = _MULTI_SPACE.sub(" ", s).strip()

    return s


def normalize_address(text: str | None) -> str:
    """Light normalization for address strings.

    Less aggressive than name normalization — we want to keep enough signal
    for address-assisted blocking while reducing trivial variation.
    """
    if not text or not str(text).strip():
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    # Expand common abbreviations
    abbrev = {
        r"\brd\b": "road",
        r"\bst\b": "street",
        r"\bave?\b": "avenue",
        r"\bblvd\b": "boulevard",
        r"\bdr\b": "drive",
        r"\bln\b": "lane",
        r"\bct\b": "court",
        r"\bhwy\b": "highway",
        r"\bnr\b": "near",
    }
    for pattern, replacement in abbrev.items():
        s = re.sub(pattern, replacement, s)
    s = _NOISE_CHARS.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def first_token(text: str | None) -> str:
    """Return the first whitespace token of the normalized name.

    Returns "" when the normalized name is empty or produces no tokens.
    """
    norm = normalize_name(text)
    if not norm:
        return ""
    tokens = norm.split()
    return tokens[0] if tokens else ""


def informative_tokens(
    text: str | None,
    stopwords: Iterable[str] | None = None,
    min_length: int = 3,
) -> list[str]:
    """Return sorted, deduplicated informative tokens from the normalized name.

    Tokens shorter than *min_length* characters and stopwords are excluded.
    The sort is alphabetical so the output is deterministic.

    Args:
        text: Raw business name.
        stopwords: Optional custom stopword set; defaults to _DEFAULT_STOPWORDS.
        min_length: Minimum token character length to keep.

    Returns:
        Sorted list of unique informative tokens (may be empty).
    """
    sw = frozenset(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS
    norm = normalize_name(text)
    if not norm:
        return []
    tokens = {t for t in norm.split() if len(t) >= min_length and t not in sw}
    return sorted(tokens)
