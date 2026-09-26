
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.preprocessing.normalization import (
    normalize_name,
    normalize_name_core,
    normalize_address,
    normalize_country,
    extract_postal_code,
    extract_house_number,
    normalize_record,
)

from src.matching.features import (
    FEATURE_COLUMNS,
    generate_pair_features,
)


def test_basic_name_normalization():
    assert normalize_name("ABC Pvt Ltd") == "abc pvt ltd"


def test_three_token_legal_suffix():
    result = normalize_name(
        "ABC Limited Liability Company"
    )

    assert result == "abc llc"


def test_legal_suffix_removed_from_name_core():
    result = normalize_name_core(
        "ABC Limited Liability Company"
    )

    assert result == "abc"


def test_transliteration():
    result = normalize_name(
        "राम मार्केटिंग प्राइवेट लिमिटेड"
    )

    # Unidecode may produce phonetic variants such as
    # "raam" and "maarketting".
    # The important requirement is that the non-Latin
    # input becomes usable Latin/ASCII text.
    assert result
    assert result.isascii()

    # Verify that the Hindi input was actually transliterated
    # rather than removed.
    assert len(result) > 10
    assert any(ch.isalpha() for ch in result)


def test_address_normalization():
    result = normalize_address(
        "123 Main Street, Noida"
    )

    assert result == "123 main street noida"


def test_postal_extraction():
    result = extract_postal_code(
        "123 Main Street, Noida 201301"
    )

    assert result == "201301"


def test_house_number_extraction():
    result = extract_house_number(
        "123A Main Street"
    )

    assert result == "123a"


def test_country_normalization():
    assert normalize_country("India") == "india"


def test_missing_address():
    record = normalize_record(
        "ABC Pvt Ltd",
        "",
        "India",
    )

    assert record["normalized_address"] == ""
    assert record["postal_code"] == ""
    assert record["house_number"] == ""


def test_pair_feature_generation():
    record_a = normalize_record(
        "ABC Private Limited",
        "123 Main Street, Noida 201301",
        "India",
    )

    record_b = normalize_record(
        "ABC Pvt Ltd",
        "123 Main St, Noida 201301",
        "India",
    )

    features = generate_pair_features(
        record_a,
        record_b,
    )

    assert features["name_core_exact"] == 1.0
    assert features["postal_match"] == 1.0
    assert features["house_number_match"] == 1.0
    assert features["country_normalized_match"] == 1.0


def test_normalized_record_interface():
    record = normalize_record(
        "ABC Private Limited",
        "123 Main Street, Noida 201301",
        "India",
    )

    required_fields = {
        "raw_name",
        "normalized_name",
        "name_core",
        "name_tokens",
        "raw_address",
        "normalized_address",
        "postal_code",
        "house_number",
        "raw_country",
        "normalized_country",
    }

    assert required_fields.issubset(
        record.keys()
    )

    features = generate_pair_features(
        record,
        record,
    )

    assert features["name_exact"] == 1.0
    assert features["address_exact"] == 1.0
    assert features["country_normalized_match"] == 1.0


def test_feature_count_is_22():
    assert len(FEATURE_COLUMNS) == 22

    record_a = normalize_record(
        "ABC Private Limited",
        "123 Main Street, Noida 201301",
        "India",
    )

    record_b = normalize_record(
        "XYZ Pvt Ltd",
        "456 Other Road, Delhi 110001",
        "France",
    )

    features = generate_pair_features(
        record_a,
        record_b,
    )

    assert len(features) == 22
    assert list(features.keys()) == FEATURE_COLUMNS


def test_country_features_are_open_set():
    record_a = normalize_record(
        "ABC Ltd",
        "123 Main Street",
        "France",
    )

    record_b = normalize_record(
        "ABC Ltd",
        "123 Main Street",
        "France",
    )

    features = generate_pair_features(
        record_a,
        record_b,
    )

    assert features["country_missing_a"] == 0.0
    assert features["country_missing_b"] == 0.0
    assert features["country_normalized_match"] == 1.0
    assert features["country_similarity"] == 1.0


def test_missing_country_features():
    record_a = normalize_record(
        "ABC Ltd",
        "123 Main Street",
        "",
    )

    record_b = normalize_record(
        "ABC Ltd",
        "123 Main Street",
        "India",
    )

    features = generate_pair_features(
        record_a,
        record_b,
    )

    assert features["country_missing_a"] == 1.0
    assert features["country_missing_b"] == 0.0
    assert features["country_normalized_match"] == 0.0
