"""
Unit tests for ML Matching Model Component.

Covers:
1. positive training pair
2. negative training pair
3. hard negative selection
4. entity-level train/validation split
5. missing address feature
6. country mismatch feature
7. zero-match decision
8. one-match decision
9. multiple-match decision
10. threshold behavior
11. margin behavior
+ model save/load and evaluation metrics.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.matching.model import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_COLUMNS_CONTRACT,
    EntityMatcher,
)
from src.matching.training import (
    construct_labeled_pairs,
    create_batches,
    create_entity_split,
    optimize_memory_usage,
    sample_negatives,
    split_pairs_by_entity,
)
from src.matching.decision import (
    MatchDecider,
    apply_decision_logic,
    compute_f_beta,
    evaluate_entity_resolution,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_candidates():
    """Synthetic candidate pairs from blocking."""
    return pd.DataFrame(
        {
            "s1_entity_id": ["S1_1", "S1_1", "S1_1", "S1_2", "S1_2", "S1_3", "S1_4", "S1_4", "S1_4", "S1_4"],
            "candidate_entity_id": ["S2_101", "S2_102", "S3_103", "S2_201", "S3_202", "S2_301", "S2_401", "S2_402", "S3_403", "S3_404"],
            "candidate_source": ["S2", "S2", "S3", "S2", "S3", "S2", "S2", "S2", "S3", "S3"],
            "name_fuzzy_similarity": [0.95, 0.70, 0.40, 0.88, 0.30, 0.20, 0.99, 0.85, 0.82, 0.15],
            "name_exact": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "address_fuzzy_similarity": [0.90, 0.65, 0.20, 0.85, 0.10, 0.15, 0.95, 0.80, 0.75, 0.10],
            "country_match": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "address_missing": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def synthetic_ground_truth():
    """Synthetic ground truth pairs."""
    return pd.DataFrame(
        {
            "s1_entity_id": ["S1_1", "S1_2", "S1_4", "S1_4"],
            "candidate_entity_id": ["S2_101", "S2_201", "S2_401", "S3_403"],
        }
    )


# ==============================================================================
# TESTS
# ==============================================================================

def test_1_positive_training_pair(synthetic_candidates, synthetic_ground_truth):
    """Test 1: Positive training pair identification."""
    labeled = construct_labeled_pairs(synthetic_candidates, synthetic_ground_truth)
    
    # S1_1 -> S2_101 is in ground truth => label must be 1
    pos_match = labeled[
        (labeled["s1_entity_id"] == "S1_1") & (labeled["candidate_entity_id"] == "S2_101")
    ]
    assert len(pos_match) == 1
    assert pos_match["label"].iloc[0] == 1


def test_2_negative_training_pair(synthetic_candidates, synthetic_ground_truth):
    """Test 2: Negative training pair identification."""
    labeled = construct_labeled_pairs(synthetic_candidates, synthetic_ground_truth)
    
    # S1_1 -> S2_102 is NOT in ground truth => label must be 0
    neg_candidate = labeled[
        (labeled["s1_entity_id"] == "S1_1") & (labeled["candidate_entity_id"] == "S2_102")
    ]
    assert len(neg_candidate) == 1
    assert neg_candidate["label"].iloc[0] == 0

    # Verify total positives and negatives count
    assert labeled["label"].sum() == 4
    assert (labeled["label"] == 0).sum() == 6


def test_3_hard_negative_selection():
    """Test 3: Hard negative selection via similarity weighting and top-k."""
    # Create dataset with 2 positives and 10 negatives with varying similarity
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A"] * 6 + ["S1_B"] * 6,
            "candidate_entity_id": [f"C_{i}" for i in range(12)],
            "name_fuzzy_similarity": [0.99, 0.95, 0.90, 0.85, 0.20, 0.10, 0.98, 0.92, 0.88, 0.30, 0.15, 0.05],
            "label": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        }
    )

    # Sample negatives with ratio 2:1 (2 positives -> 4 negatives)
    sampled = sample_negatives(
        df,
        negative_to_positive_ratio=2.0,
        hard_negative_strategy="hard_first",
        similarity_col="name_fuzzy_similarity",
        random_state=42,
    )

    # Positives must be preserved
    assert len(sampled[sampled["label"] == 1]) == 2
    # Negatives sampled should be exactly 4
    assert len(sampled[sampled["label"] == 0]) == 4
    # With hard_first, sampled negatives must have the highest similarities among negatives
    neg_sims = sampled[sampled["label"] == 0]["name_fuzzy_similarity"].values
    assert min(neg_sims) >= 0.85

    # Test similarity_weighted strategy
    sampled_weighted = sample_negatives(
        df,
        negative_to_positive_ratio=2.0,
        hard_negative_strategy="similarity_weighted",
        similarity_col="name_fuzzy_similarity",
        random_state=42,
    )
    assert len(sampled_weighted) == 6


def test_4_entity_level_train_validation_split():
    """Test 4: Entity-level train/validation split ensuring 0 S1 overlap."""
    s1_list = [f"S1_{i}" for i in range(100)]
    df = pd.DataFrame(
        {
            "s1_entity_id": np.repeat(s1_list, 5),
            "candidate_entity_id": [f"C_{i}" for i in range(500)],
        }
    )

    train_ids, val_ids = create_entity_split(df, test_size=0.25, random_state=42)

    # 1. Total S1 entities preserved
    assert len(train_ids) + len(val_ids) == 100
    assert len(val_ids) == 25
    assert len(train_ids) == 75

    # 2. Strict ZERO overlap between train and validation S1 entities
    overlap = set(train_ids).intersection(set(val_ids))
    assert len(overlap) == 0

    # 3. Deterministic behavior with same seed
    train_ids_2, val_ids_2 = create_entity_split(df, test_size=0.25, random_state=42)
    np.testing.assert_array_equal(train_ids, train_ids_2)
    np.testing.assert_array_equal(val_ids, val_ids_2)

    # 4. Partition DataFrame
    train_df, val_df = split_pairs_by_entity(df, train_ids, val_ids)
    assert len(train_df) == 75 * 5
    assert len(val_df) == 25 * 5


def test_5_missing_address_feature():
    """Test 5: Model handles missing address features (NaNs) gracefully."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame(
        {
            "name_fuzzy_similarity": np.random.uniform(0.0, 1.0, n),
            "name_exact": np.random.choice([0.0, 1.0], n),
            "address_fuzzy_similarity": np.where(np.random.rand(n) > 0.3, np.random.rand(n), np.nan),
            "address_missing": np.random.choice([0.0, 1.0], n),
            "country_match": np.random.choice([0.0, 1.0], n),
        }
    )
    y = np.random.choice([0, 1], n)

    matcher = EntityMatcher(params={"n_estimators": 10, "min_child_samples": 5}, random_state=42)
    # Model should fit without raising error on NaNs (LightGBM native handling)
    matcher.fit(X, y)
    assert matcher.is_fitted

    # Predictions with NaNs should produce valid probabilities in [0.0, 1.0]
    X_test = pd.DataFrame(
        {
            "name_fuzzy_similarity": [0.9, 0.1],
            "name_exact": [1.0, 0.0],
            "address_fuzzy_similarity": [np.nan, 0.2],
            "address_missing": [1.0, 0.0],
            "country_match": [1.0, 0.0],
        }
    )
    probas = matcher.predict_proba(X_test)
    assert len(probas) == 2
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)


def test_6_country_mismatch_feature():
    """Test 6: Country mismatch feature behavior."""
    X = pd.DataFrame(
        {
            "name_fuzzy_similarity": [0.95, 0.95, 0.95, 0.95, 0.1, 0.1],
            "country_match": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = np.array([1, 1, 0, 0, 0, 0])

    matcher = EntityMatcher(params={"n_estimators": 30, "min_child_samples": 1}, random_state=42)
    matcher.fit(X, y)

    # A pair with high name similarity + country_match=1 should have higher probability than country_match=0
    test_pair_match = pd.DataFrame({"name_fuzzy_similarity": [0.95], "country_match": [1.0]})
    test_pair_mismatch = pd.DataFrame({"name_fuzzy_similarity": [0.95], "country_match": [0.0]})

    p_match = matcher.predict_proba(test_pair_match)[0]
    p_mismatch = matcher.predict_proba(test_pair_mismatch)[0]

    assert p_match > p_mismatch


def test_7_zero_match_decision():
    """Test 7: Zero-match decision when all candidates are below threshold."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A", "S1_A", "S1_A"],
            "candidate_entity_id": ["C1", "C2", "C3"],
            "candidate_source": ["S2", "S2", "S3"],
            "match_probability": [0.45, 0.30, 0.12],
        }
    )

    decider = MatchDecider(threshold=0.50)
    result = decider.decide(df)

    assert "is_match" in result.columns
    # All must be 0, never force the highest scoring 0.45 to match
    assert result["is_match"].sum() == 0

    filtered = decider.filter_matches(df)
    assert len(filtered) == 0


def test_8_one_match_decision():
    """Test 8: Exactly one match decision when only one candidate exceeds threshold."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A", "S1_A", "S1_A"],
            "candidate_entity_id": ["C1", "C2", "C3"],
            "candidate_source": ["S2", "S2", "S3"],
            "match_probability": [0.92, 0.42, 0.15],
        }
    )

    decider = MatchDecider(threshold=0.50)
    result = decider.decide(df)

    assert result["is_match"].sum() == 1
    matched_cand = result[result["is_match"] == 1]["candidate_entity_id"].iloc[0]
    assert matched_cand == "C1"


def test_9_multiple_match_decision():
    """Test 9: Multiple-match decision when multiple candidates exceed threshold."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A", "S1_A", "S1_A"],
            "candidate_entity_id": ["C1", "C2", "C3"],
            "candidate_source": ["S2", "S3", "S2"],
            "match_probability": [0.95, 0.89, 0.10],
        }
    )

    decider = MatchDecider(threshold=0.50)
    result = decider.decide(df)

    # Both C1 and C2 exceed threshold and should be matches
    assert result["is_match"].sum() == 2
    matched_cands = set(result[result["is_match"] == 1]["candidate_entity_id"])
    assert matched_cands == {"C1", "C2"}


def test_10_threshold_behavior():
    """Test 10: Strict threshold filtering behavior."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A", "S1_A", "S1_A", "S1_A"],
            "candidate_entity_id": ["C1", "C2", "C3", "C4"],
            "candidate_source": ["S2", "S2", "S3", "S3"],
            "match_probability": [0.90, 0.75, 0.60, 0.45],
        }
    )

    # At threshold 0.50 -> C1, C2, C3 match (3 matches)
    r_50 = apply_decision_logic(df, threshold=0.50)
    assert r_50["is_match"].sum() == 3

    # At threshold 0.70 -> C1, C2 match (2 matches)
    r_70 = apply_decision_logic(df, threshold=0.70)
    assert r_70["is_match"].sum() == 2

    # At threshold 0.85 -> C1 only matches (1 match)
    r_85 = apply_decision_logic(df, threshold=0.85)
    assert r_85["is_match"].sum() == 1

    # At threshold 0.95 -> 0 matches
    r_95 = apply_decision_logic(df, threshold=0.95)
    assert r_95["is_match"].sum() == 0


def test_11_margin_behavior():
    """Test 11: Top-1 vs Top-2 margin constraint behavior."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_A", "S1_A", "S1_B", "S1_B"],
            "candidate_entity_id": ["C1", "C2", "C3", "C4"],
            "candidate_source": ["S2", "S3", "S2", "S3"],
            "match_probability": [0.95, 0.90, 0.95, 0.60],
        }
    )

    # For S1_A: p1 = 0.95, p2 = 0.90 (gap = 0.05)
    # For S1_B: p1 = 0.95, p2 = 0.60 (gap = 0.35)
    
    # Margin = 0.10 -> S1_A gap (0.05) < 0.10, so candidate C2 is disqualified
    # S1_B gap (0.35) >= 0.10, so margin check passes
    decider = MatchDecider(threshold=0.50, margin=0.10)
    result = decider.decide(df)

    s1_a_matches = result[result["s1_entity_id"] == "S1_A"]
    assert s1_a_matches[s1_a_matches["candidate_entity_id"] == "C1"]["is_match"].iloc[0] == 1
    assert s1_a_matches[s1_a_matches["candidate_entity_id"] == "C2"]["is_match"].iloc[0] == 0


def test_model_save_and_load():
    """Test model serialization round-trip."""
    X = pd.DataFrame(
        {
            "name_fuzzy_similarity": [0.9, 0.1, 0.8, 0.2],
            "country_match": [1.0, 0.0, 1.0, 0.0],
        }
    )
    y = np.array([1, 0, 1, 0])

    matcher = EntityMatcher(params={"n_estimators": 5, "min_child_samples": 1}, random_state=42)
    matcher.fit(X, y)
    orig_probas = matcher.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "matcher.joblib")
        matcher.save(save_path)

        loaded_matcher = EntityMatcher.load(save_path)
        assert loaded_matcher.is_fitted
        loaded_probas = loaded_matcher.predict_proba(X)

        np.testing.assert_allclose(orig_probas, loaded_probas, rtol=1e-5)
        
        # Test feature importance
        imp_df = loaded_matcher.get_feature_importance()
        assert len(imp_df) == 2
        assert "feature" in imp_df.columns
        assert "importance" in imp_df.columns


def test_evaluation_metrics():
    """Test macro/micro F0.5 and match statistics computation."""
    # S1_1 has 1 true match and 1 predicted match (TP=1) -> P=1, R=1, F0.5=1
    # S1_2 has 1 true match and 0 predicted matches (FN=1) -> P=0, R=0, F0.5=0
    # S1_3 has 0 true matches and 0 predicted matches -> P=1, R=1, F0.5=1
    predictions = pd.DataFrame(
        {
            "s1_entity_id": ["S1_1", "S1_2"],
            "candidate_entity_id": ["C1", "C2"],
            "candidate_source": ["S2", "S2"],
            "is_match": [1, 0],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "s1_entity_id": ["S1_1", "S1_2"],
            "candidate_entity_id": ["C1", "C2"],
        }
    )

    stats = evaluate_entity_resolution(
        predictions,
        ground_truth,
        all_s1_ids=["S1_1", "S1_2", "S1_3"],
    )

    assert "macro_f05" in stats
    assert "macro_precision" in stats
    assert "macro_recall" in stats
    assert "match_count_stats" in stats
    assert stats["match_count_stats"]["total_s1_entities"] == 3
    assert stats["match_count_stats"]["zero_match_count"] == 2
    assert stats["match_count_stats"]["single_match_count"] == 1

    # Verify F0.5 calculation helper
    f05 = compute_f_beta(precision=0.8, recall=0.4, beta=0.5)
    # F0.5 = 1.25 * (0.8 * 0.4) / (0.25 * 0.8 + 0.4) = 1.25 * 0.32 / (0.20 + 0.40) = 0.40 / 0.60 = 0.6666...
    assert pytest.approx(f05, rel=1e-3) == 0.6667


def test_memory_optimization_and_batching():
    """Test memory optimization and batch generator utilities."""
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1_1", "S1_2", "S1_3"],
            "float_feat": [0.1, 0.2, 0.3],
            "int_feat": [1, 2, 3],
        }
    )
    df_opt = optimize_memory_usage(df)
    assert df_opt["float_feat"].dtype == np.float32

    batches = list(create_batches(df_opt, batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1
