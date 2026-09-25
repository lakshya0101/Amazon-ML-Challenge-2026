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
    assert ablation["cumulative_stages"]["Full Phase 1 Union"]["overall"]["candidate_recall"] == 1.0


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

