"""Unit and synthetic integration tests for blocking and candidate generation."""

import pytest
import os
import tempfile
import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex
from src.blocking.candidate_generator import CandidateGenerator, generate_candidates
from src.blocking.evaluation import evaluate_candidates, evaluate_strategy_ablation
from src.blocking.retrieval import (
    RULE_EXACT_NAME,
    RULE_NAME_CORE,
    RULE_COUNTRY_TOKEN,
    RULE_POSTAL_CODE,
    RULE_HOUSE_ADDRESS_TOKEN,
    RULE_COUNTRY_ADDRESS_TOKENS,
)



@pytest.fixture
def synthetic_dataset():
    """Provides a synthetic dataset covering all required blocking strategies and edge cases."""
    s1 = [
        # 1. Exact name match candidate
        {"id": "s1_exact", "business_name": "Acme Logistics Inc", "business_address": "100 Pine St", "country": "USA"},
        # 2. Name core match candidate (differing legal suffix)
        {"id": "s1_core", "business_name": "Apex Innovations LLC", "business_address": "200 Oak Rd", "country": "USA"},
        # 3. Country + token match candidate (different word order / extra tokens)
        {"id": "s1_token", "business_name": "Global Blue Star Tech", "business_address": "300 Elm St", "country": "India"},
        # 4. Postal code match candidate (different business name, matching postal code)
        {"id": "s1_postal", "business_name": "Boutique Atelier", "business_address": "400 Rue Paris 75001", "country": "France"},
        # 5. House number + address token match candidate
        {"id": "s1_addr", "business_name": "Different Name Entirely", "business_address": "555 Market Boulevard Suite 400", "country": "USA"},
        # 6. Missing address candidate
        {"id": "s1_no_addr", "business_name": "CloudNine Networks Inc", "business_address": "", "country": "USA"},
        # 7. Generic / high frequency token candidate
        {"id": "s1_generic", "business_name": "Primary Care Health Group", "business_address": "10 Hospital Way", "country": "USA"},
        # 8. Zero match candidate
        {"id": "s1_zero", "business_name": "Completely Unique Unicorn Corp", "business_address": "999 Nowhere Land", "country": "Mars"},
        # 9. Multiple match candidate
        {"id": "s1_multi", "business_name": "Metro Pharmacy", "business_address": "12 Main St 90210", "country": "USA"},
    ]

    s2 = [
        # Target for s1_exact
        {"id": "s2_exact", "business_name": "Acme Logistics Inc", "business_address": "100 Pine St", "country": "USA"},
        # Target for s1_core (matches on 'apex innovations')
        {"id": "s2_core", "business_name": "Apex Innovations Corp", "business_address": "200 Oak Road", "country": "USA"},
        # Target for s1_token (matches on ('india', 'star'))
        {"id": "s2_token", "business_name": "Star Enterprises Pvt Ltd", "business_address": "Sector 5", "country": "India"},
        # Target for s1_postal (matches on postal '75001')
        {"id": "s2_postal", "business_name": "Cafe Parisien", "business_address": "12 Ave Champs 75001", "country": "France"},
        # Target 1 for s1_multi
        {"id": "s2_multi_1", "business_name": "Metro Pharmacy", "business_address": "12 Main St 90210", "country": "USA"},
        # Target with generic words (should be indexed cleanly)
        {"id": "s2_generic", "business_name": "Care Health Group", "business_address": "50 Medical Dr", "country": "USA"},
    ]

    s3 = [
        # Target for s1_addr (matches on house number '555' and token 'market')
        {"id": "s3_addr", "business_name": "Unrelated Name LLC", "business_address": "555 Market St Floor 2", "country": "USA"},
        # Target for s1_no_addr (matches on exact name)
        {"id": "s3_no_addr", "business_name": "CloudNine Networks Inc", "business_address": "888 Cloud Way", "country": "USA"},
        # Target 2 for s1_multi
        {"id": "s3_multi_2", "business_name": "Metro Pharmacy Ltd", "business_address": "12 Main St 90210", "country": "USA"},
    ]

    ground_truth = [
        ("s1_exact", "S2", "s2_exact"),
        ("s1_core", "S2", "s2_core"),
        ("s1_token", "S2", "s2_token"),
        ("s1_postal", "S2", "s2_postal"),
        ("s1_addr", "S3", "s3_addr"),
        ("s1_no_addr", "S3", "s3_no_addr"),
        ("s1_multi", "S2", "s2_multi_1"),
        ("s1_multi", "S3", "s3_multi_2"),
    ]

    return {
        "s1": pd.DataFrame(s1),
        "s2": pd.DataFrame(s2),
        "s3": pd.DataFrame(s3),
        "ground_truth": ground_truth,
    }


def test_exact_name_blocking(synthetic_dataset):
    config = BlockingConfig(
        exact_name=True,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s2_pairs = {(c.s1_id, c.match_id) for c in res["s2_candidates"]}
    assert ("s1_exact", "s2_exact") in s2_pairs
    assert ("s1_multi", "s2_multi_1") in s2_pairs


def test_name_core_blocking(synthetic_dataset):
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=True,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s2_pairs = {(c.s1_id, c.match_id) for c in res["s2_candidates"]}
    # "Apex Innovations LLC" and "Apex Innovations Corp" share name core "apex innovations"
    assert ("s1_core", "s2_core") in s2_pairs


def test_country_name_token_blocking(synthetic_dataset):
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=True,
        postal_code=False,
        house_number_address_token=False,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s2_pairs = {(c.s1_id, c.match_id) for c in res["s2_candidates"]}
    # Matches on ("india", "star")
    assert ("s1_token", "s2_token") in s2_pairs


def test_postal_code_blocking(synthetic_dataset):
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=True,
        house_number_address_token=False,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s2_pairs = {(c.s1_id, c.match_id) for c in res["s2_candidates"]}
    # Matches on postal code "75001"
    assert ("s1_postal", "s2_postal") in s2_pairs


def test_house_number_address_token_blocking(synthetic_dataset):
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=True,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s3_pairs = {(c.s1_id, c.match_id) for c in res["s3_candidates"]}
    # Matches on house number "555" and token "market"
    assert ("s1_addr", "s3_addr") in s3_pairs


def test_missing_address_handling(synthetic_dataset):
    """Missing address must not eliminate a record from name-based candidate generation."""
    config = BlockingConfig(
        exact_name=True,
        exact_name_core=True,
        country_name_token=True,
        postal_code=True,
        house_number_address_token=True,
    )
    generator = CandidateGenerator(config=config)
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    s3_pairs = {(c.s1_id, c.match_id) for c in res["s3_candidates"]}
    assert ("s1_no_addr", "s3_no_addr") in s3_pairs


def test_zero_match_and_multi_match_s1(synthetic_dataset):
    generator = CandidateGenerator()
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    # Zero match
    zero_pairs = [c for c in res["all_candidates"] if c.s1_id == "s1_zero"]
    assert len(zero_pairs) == 0

    # Multi match
    multi_pairs = [c for c in res["all_candidates"] if c.s1_id == "s1_multi"]
    assert len(multi_pairs) >= 2
    match_ids = {c.match_id for c in multi_pairs}
    assert "s2_multi_1" in match_ids
    assert "s3_multi_2" in match_ids


def test_s2_s3_separation_and_provenance(synthetic_dataset):
    generator = CandidateGenerator()
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    for c in res["s2_candidates"]:
        assert c.match_source == "S2"
        assert len(c.blocking_rules) > 0

    for c in res["s3_candidates"]:
        assert c.match_source == "S3"
        assert len(c.blocking_rules) > 0

    # Ensure provenance captures multiple rules firing
    multi_c = [c for c in res["s2_candidates"] if c.s1_id == "s1_multi" and c.match_id == "s2_multi_1"][0]
    assert RULE_EXACT_NAME in multi_c.blocking_rules
    assert RULE_NAME_CORE in multi_c.blocking_rules


def test_evaluation_and_ablation(synthetic_dataset):
    res = generate_candidates(
        synthetic_dataset["s1"],
        synthetic_dataset["s2"],
        synthetic_dataset["s3"],
    )

    metrics = evaluate_candidates(
        res,
        synthetic_dataset["ground_truth"],
    )

    assert "overall" in metrics
    assert "s2" in metrics
    assert "s3" in metrics
    assert metrics["overall"]["candidate_recall"] == 1.0
    assert metrics["s2"]["candidate_recall"] == 1.0
    assert metrics["s3"]["candidate_recall"] == 1.0
    assert metrics["overall"]["reduction_ratio"] > 0.5
    assert len(metrics["overall"]["missed_true_pairs"]) == 0

    # Test ablation runner
    ablation = evaluate_strategy_ablation(
        synthetic_dataset["s1"],
        synthetic_dataset["s2"],
        synthetic_dataset["s3"],
        synthetic_dataset["ground_truth"],
    )
    assert "individual_strategies" in ablation
    assert "cumulative_stages" in ablation
    assert ablation["cumulative_stages"]["Full Union (+ country_address_tokens)"]["overall"]["candidate_recall"] == 1.0


def test_tsv_export(synthetic_dataset):
    generator = CandidateGenerator()
    res = generator.generate(synthetic_dataset["s1"], synthetic_dataset["s2"], synthetic_dataset["s3"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "candidate_pairs.tsv")
        CandidateGenerator.export_candidate_pairs_tsv(res["all_candidates"], tsv_path)

        assert os.path.exists(tsv_path)
        df = pd.read_csv(tsv_path, sep="\t")
        assert list(df.columns) == ["s1_id", "match_source", "match_id"]
        assert len(df) == len(res["all_candidates"])


def test_generic_high_frequency_tokens_and_capping():
    """Verify that common stopwords like 'group' and explosive token buckets are safely handled."""
    s1 = [{"id": "s1_generic", "business_name": "Health Care Group LLC", "business_address": "123 Main St", "country": "USA"}]
    s2 = [
        {"id": f"s2_{i}", "business_name": f"Other Health Entity {i} Group", "business_address": f"{i} Main St", "country": "USA"}
        for i in range(20)
    ]
    s3 = []

    # With high bucket threshold, informative tokens match
    config = BlockingConfig(max_bucket_size=50)
    gen = CandidateGenerator(config=config)
    res = gen.generate(s1, s2, s3)
    # The stopwords ("group", "llc") were excluded from country-token index
    for c in res["s2_candidates"]:
        assert c.match_source == "S2"

    # With tight max_bucket_size=5, the large bucket is pruned
    config_tight = BlockingConfig(max_bucket_size=5)
    gen_tight = CandidateGenerator(config=config_tight)
    res_tight = gen_tight.generate(s1, s2, s3)
    # Exact name and name core won't match, token bucket > 5 is skipped
    assert len(res_tight["s2_candidates"]) == 0


def test_invalid_and_empty_records():
    """Verify that records with None or empty fields do not raise errors or create false positive matches."""
    s1 = [
        {"id": "s1_empty", "business_name": "", "business_address": None, "country": None},
        {"id": "s1_spaces", "business_name": "   ", "business_address": "   ", "country": "   "},
    ]
    s2 = [
        {"id": "s2_empty", "business_name": None, "business_address": "", "country": ""},
        {"id": "s2_valid", "business_name": "Valid Name", "business_address": "100 Pine St", "country": "USA"},
    ]
    s3 = []

    res = generate_candidates(s1, s2, s3)
    # Empty records should produce 0 candidates and not crash
    assert len(res["all_candidates"]) == 0


def test_missed_true_pairs_diagnosis():
    """Verify that evaluate_candidates correctly flags and details missed ground truth pairs."""
    s1 = [{"id": "s1_1", "business_name": "Alpha Corp", "business_address": "10 Pine", "country": "USA"}]
    s2 = [{"id": "s2_1", "business_name": "Beta Corp", "business_address": "20 Oak", "country": "USA"}]
    s3 = []
    # Ground truth asserts a match that shares no blocking keys
    gt = [("s1_1", "S2", "s2_1")]

    res = generate_candidates(s1, s2, s3)
    metrics = evaluate_candidates(res, gt)

    assert metrics["overall"]["candidate_recall"] == 0.0
    assert len(metrics["overall"]["missed_true_pairs"]) == 1
    assert ("s1_1", "S2", "s2_1") in metrics["overall"]["missed_true_pairs"]


def test_country_address_tokens_shared_threshold_and_country():
    """Verify >=2 shared address tokens retrieves, exactly 1 does not, different country does not."""
    # Only enable country_address_tokens rule
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
        country_address_tokens=True,
        country_address_token_min_shared=2,
    )

    s1 = [
        # Informative address tokens: "green", "valley", "center" (street is stopword)
        {
            "id": "s1_addr_match",
            "business_name": "Entity One",
            "business_address": "500 Green Valley Center Street",
            "country": "USA",
        }
    ]

    s2 = [
        # Shares 2 tokens: "green", "valley" -> should MATCH
        {
            "id": "s2_match_2_tokens",
            "business_name": "Totally Different Name A",
            "business_address": "123 Green Valley Road",
            "country": "USA",
        },
        # Shares only 1 token: "green" -> should NOT match (shared count = 1 < 2)
        {
            "id": "s2_1_token_only",
            "business_name": "Totally Different Name B",
            "business_address": "999 Green Hill Road",
            "country": "USA",
        },
        # Shares 2 tokens: "green", "valley", but in DIFFERENT country -> should NOT match
        {
            "id": "s2_diff_country",
            "business_name": "Totally Different Name C",
            "business_address": "123 Green Valley Road",
            "country": "UK",
        },
    ]
    s3 = []

    gen = CandidateGenerator(config=config)
    res = gen.generate(s1, s2, s3)

    retrieved_s2_ids = {c.match_id for c in res["s2_candidates"]}
    assert "s2_match_2_tokens" in retrieved_s2_ids
    assert "s2_1_token_only" not in retrieved_s2_ids
    assert "s2_diff_country" not in retrieved_s2_ids

    # Check provenance
    match_candidate = [c for c in res["s2_candidates"] if c.match_id == "s2_match_2_tokens"][0]
    assert RULE_COUNTRY_ADDRESS_TOKENS in match_candidate.blocking_rules


def test_country_address_tokens_duplicate_tokens_and_oversized_bucket():
    """Verify duplicate address tokens are counted once, and oversized buckets are ignored."""
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
        country_address_tokens=True,
        country_address_token_min_shared=2,
        max_bucket_size=5,
    )

    # S1 has repeated token "river" -> unique tokens: "river", "park"
    s1 = [
        {
            "id": "s1_dup",
            "business_name": "Delta Inc",
            "business_address": "River River River Road",  # Only 1 unique informative token: "river"
            "country": "USA",
        }
    ]

    # Target also has "river" repeated
    s2 = [
        {
            "id": "s2_dup_target",
            "business_name": "Echo LLC",
            "business_address": "River River River Blvd",
            "country": "USA",
        }
    ]
    s3 = []

    gen = CandidateGenerator(config=config)
    res = gen.generate(s1, s2, s3)

    # Since unique shared tokens is 1 ("river"), it should NOT meet threshold of 2
    assert len(res["s2_candidates"]) == 0

    # Test oversized bucket pruning
    s1_multi = [
        {
            "id": "s1_multi",
            "business_name": "Foxtrot",
            "business_address": "100 Popular Unique Place",  # "popular", "unique", "place"
            "country": "USA",
        }
    ]
    # Create 10 targets with "popular" -> bucket size for ("us", "popular") = 10 > max_bucket_size (5)
    s2_oversized = [
        {
            "id": f"s2_pop_{i}",
            "business_name": f"Pop {i}",
            "business_address": "Popular Random Highway",
            "country": "USA",
        }
        for i in range(10)
    ]
    gen_oversized = CandidateGenerator(config=config)
    res_oversized = gen_oversized.generate(s1_multi, s2_oversized, [])
    # "popular" bucket exceeds max_bucket_size of 5 so ignored, no match reaches threshold of 2
    assert len(res_oversized["s2_candidates"]) == 0


def test_country_address_tokens_missing_address_safety():
    """Verify missing address in S1 or target does not crash and produces no false matches."""
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
        country_address_tokens=True,
        country_address_token_min_shared=2,
    )

    s1 = [
        {"id": "s1_no_addr", "business_name": "No Address Co", "business_address": "", "country": "USA"},
        {"id": "s1_none_addr", "business_name": "None Address Co", "business_address": None, "country": "USA"},
        {"id": "s1_valid", "business_name": "Valid", "business_address": "100 Industrial Parkway Sector", "country": "USA"},
    ]
    s2 = [
        {"id": "s2_no_addr", "business_name": "No Address Target", "business_address": "", "country": "USA"},
        {"id": "s2_none_country", "business_name": "No Country Target", "business_address": "100 Industrial Parkway Sector", "country": ""},
    ]

    gen = CandidateGenerator(config=config)
    res = gen.generate(s1, s2, [])
    assert len(res["all_candidates"]) == 0


def test_s2_s3_independence_for_country_address_tokens():
    """Verify S2 and S3 candidate retrieval remain independent for country address tokens."""
    config = BlockingConfig(
        exact_name=False,
        exact_name_core=False,
        country_name_token=False,
        postal_code=False,
        house_number_address_token=False,
        country_address_tokens=True,
        country_address_token_min_shared=2,
    )

    s1 = [
        {"id": "s1_common", "business_name": "Entity S1", "business_address": "777 Highland Summit Trail", "country": "USA"}
    ]
    s2 = [
        {"id": "s2_match", "business_name": "Target S2", "business_address": "10 Highland Summit Road", "country": "USA"}
    ]
    s3 = [
        {"id": "s3_match", "business_name": "Target S3", "business_address": "20 Highland Summit Avenue", "country": "USA"}
    ]

    gen = CandidateGenerator(config=config)
    res = gen.generate(s1, s2, s3)

    assert len(res["s2_candidates"]) == 1
    assert res["s2_candidates"][0].match_id == "s2_match"
    assert res["s2_candidates"][0].match_source == "S2"

    assert len(res["s3_candidates"]) == 1
    assert res["s3_candidates"][0].match_id == "s3_match"
    assert res["s3_candidates"][0].match_source == "S3"


