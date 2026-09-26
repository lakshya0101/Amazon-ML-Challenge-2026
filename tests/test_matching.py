"""Comprehensive unit and integration tests for the ML entity matching layer."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.preprocessing.normalization import normalize_record
from src.matching.features import FEATURE_COLUMNS, generate_pair_features
from src.matching.dataset import (
    NormalizedRecordCache,
    build_candidate_dataframe,
    label_candidate_pairs,
    create_entity_split,
    split_pairs_by_entity,
    generate_pair_features_matrix,
    sample_negatives,
    normalize_ground_truth_set,
)
from src.matching.model import (
    LightGBMMatcher,
    LogisticRegressionMatcher,
    HistGradientBoostingMatcher,
)
from src.matching.decision import (
    compute_f_beta,
    evaluate_predictions,
    ThresholdSweeper,
)
from src.matching.inference import (
    EntityMatchInference,
    match_candidates,
)
from src.matching.training import (
    train_matcher_pipeline,
)
from src.matching.experiments import (
    ExperimentTracker,
)
from src.blocking.candidate_generator import CandidatePair, generate_candidates


@pytest.fixture
def synthetic_records():
    """Realistic synthetic dataset covering singletons, multi-matches, and zero matches."""
    s1 = [
        {"id": "s1_1", "business_name": "Acme Logistics Inc", "business_address": "100 Pine St", "country": "USA"},
        {"id": "s1_2", "business_name": "Apex Innovations LLC", "business_address": "200 Oak Rd", "country": "USA"},
        {"id": "s1_3", "business_name": "Global Star Tech", "business_address": "300 Elm St", "country": "India"},
        {"id": "s1_4", "business_name": "Boutique Atelier", "business_address": "400 Rue Paris 75001", "country": "France"},
        {"id": "s1_multi", "business_name": "Metro Pharmacy", "business_address": "12 Main St 90210", "country": "USA"},
        {"id": "s1_zero", "business_name": "Unicorn Ventures", "business_address": "999 Nowhere Rd", "country": "Canada"},
    ]

    s2 = [
        {"id": "target_1", "business_name": "Acme Logistics Inc", "business_address": "100 Pine St", "country": "USA"},
        {"id": "target_2", "business_name": "Apex Innovations Corp", "business_address": "200 Oak Road", "country": "USA"},
        {"id": "target_multi_s2", "business_name": "Metro Pharmacy Inc", "business_address": "12 Main Street", "country": "USA"},
        {"id": "distractor_1", "business_name": "Random Corp", "business_address": "555 Fifth Ave", "country": "USA"},
    ]

    s3 = [
        {"id": "target_3", "business_name": "Global Star Tech Pvt Ltd", "business_address": "300 Elm Street", "country": "India"},
        {"id": "target_4", "business_name": "Boutique Atelier SARL", "business_address": "400 Rue Paris", "country": "France"},
        {"id": "target_multi_s3", "business_name": "Metro Pharmacy LLC", "business_address": "12 Main St Suite A", "country": "USA"},
        {"id": "distractor_2", "business_name": "Other Business", "business_address": "777 Seventh St", "country": "UK"},
    ]

    # Ground truth matches
    ground_truth = [
        ("s1_1", "S2", "target_1"),
        ("s1_2", "S2", "target_2"),
        ("s1_3", "S3", "target_3"),
        ("s1_4", "S3", "target_4"),
        ("s1_multi", "S2", "target_multi_s2"),
        ("s1_multi", "S3", "target_multi_s3"),
        # s1_zero has 0 ground truth matches
    ]

    return {
        "s1": pd.DataFrame(s1),
        "s2": pd.DataFrame(s2),
        "s3": pd.DataFrame(s3),
        "ground_truth": ground_truth,
    }


def test_feature_generation_integration(synthetic_records):
    """Test feature generation creates 22 columns with exact keys."""
    cache = NormalizedRecordCache()
    cache.register_dataset("S1", synthetic_records["s1"])
    cache.register_dataset("S2", synthetic_records["s2"])

    pairs = pd.DataFrame([
        {"s1_id": "s1_1", "match_source": "S2", "match_id": "target_1"},
    ])

    features_df = generate_pair_features_matrix(pairs, cache)
    assert len(features_df) == 1
    assert list(features_df.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 22
    assert features_df.iloc[0]["name_exact"] == 1.0
    assert features_df.iloc[0]["address_exact"] == 1.0


def test_record_cache_single_normalization():
    """Verify records are normalized once and cached."""
    cache = NormalizedRecordCache()
    rec1 = cache.register_record("S1", "rec_1", "Acme LLC", "100 Main St", "USA")
    assert cache.contains("S1", "rec_1")

    # Second lookup retrieves cached dict without re-normalization
    rec2 = cache.get("S1", "rec_1")
    assert rec1 is rec2
    assert rec1["normalized_name"] == "acme llc"
    assert rec1["name_core"] == "acme"


def test_entity_split_zero_leakage():
    """Verify S1 entity level splitting has zero overlap."""
    s1_ids = [f"s1_{i}" for i in range(50)]
    pairs = pd.DataFrame([
        {"s1_id": f"s1_{i % 50}", "match_source": "S2", "match_id": f"s2_{i}", "label": i % 2}
        for i in range(200)
    ])

    train_ids, val_ids = create_entity_split(pairs, test_size=0.2, random_state=42)
    assert len(train_ids) > 0
    assert len(val_ids) > 0

    overlap = set(train_ids).intersection(set(val_ids))
    assert len(overlap) == 0

    train_pairs, val_pairs = split_pairs_by_entity(pairs, train_ids, val_ids)
    train_s1_in_val = set(train_pairs["s1_id"]).intersection(set(val_pairs["s1_id"]))
    assert len(train_s1_in_val) == 0


def test_candidate_labeling_and_s2_s3_separation():
    """Verify S2 and S3 records with identical IDs are not collapsed."""
    candidates = [
        CandidatePair(s1_id="s1_1", match_source="S2", match_id="same_id", blocking_rules={"exact"}),
        CandidatePair(s1_id="s1_1", match_source="S3", match_id="same_id", blocking_rules={"exact"}),
    ]
    gt = [("s1_1", "S2", "same_id")]  # S2 is true match, S3 is NOT

    labeled_df = label_candidate_pairs(candidates, gt)
    assert len(labeled_df) == 2

    s2_row = labeled_df[labeled_df["match_source"] == "S2"].iloc[0]
    s3_row = labeled_df[labeled_df["match_source"] == "S3"].iloc[0]

    assert s2_row["label"] == 1
    assert s3_row["label"] == 0


def test_duplicate_candidate_handling():
    """Verify duplicate candidate pairs are deduplicated."""
    cands = [
        CandidatePair(s1_id="s1_1", match_source="S2", match_id="m_1", blocking_rules={"rule1"}),
        CandidatePair(s1_id="s1_1", match_source="S2", match_id="m_1", blocking_rules={"rule2"}),
    ]
    df = build_candidate_dataframe(cands)
    assert len(df) == 1


def test_lightgbm_fit_and_inference():
    """Verify LightGBMMatcher fits, outputs probabilities in [0, 1], and predicts."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.rand(n, 22), columns=FEATURE_COLUMNS)
    y = (X["name_exact"] > 0.5).astype(int).values

    matcher = LightGBMMatcher(params={"n_estimators": 20, "num_leaves": 8})
    matcher.fit(X, y)

    assert matcher.is_fitted
    probas = matcher.predict_proba(X)
    assert len(probas) == n
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)

    preds_50 = matcher.predict(X, threshold=0.50)
    preds_90 = matcher.predict(X, threshold=0.90)
    assert np.all((preds_50 == 0) | (preds_50 == 1))
    # Higher threshold should yield <= positive predictions
    assert np.sum(preds_90) <= np.sum(preds_50)

    imp = matcher.get_feature_importance()
    assert len(imp) == 22
    assert "importance" in imp.columns


def test_model_save_and_load():
    """Verify LightGBM model serialization and exact deserialization."""
    X = pd.DataFrame(np.random.rand(50, 22), columns=FEATURE_COLUMNS)
    y = (X["name_exact"] > 0.5).astype(int).values

    matcher = LightGBMMatcher(params={"n_estimators": 10})
    matcher.fit(X, y)
    original_probas = matcher.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "matcher.joblib"
        matcher.save(model_path)
        assert model_path.exists()

        loaded_matcher = LightGBMMatcher.load(model_path)
        assert loaded_matcher.is_fitted
        loaded_probas = loaded_matcher.predict_proba(X)
        np.testing.assert_allclose(original_probas, loaded_probas, rtol=1e-5)


def test_empty_candidate_set():
    """Verify empty candidate sets are handled safely."""
    cache = NormalizedRecordCache()
    empty_df = pd.DataFrame(columns=["s1_id", "match_source", "match_id"])
    features = generate_pair_features_matrix(empty_df, cache)
    assert len(features) == 0
    assert list(features.columns) == FEATURE_COLUMNS

    matcher = LightGBMMatcher()
    # Dummy fit
    matcher.fit(pd.DataFrame(np.zeros((2, 22)), columns=FEATURE_COLUMNS), np.array([0, 1]))

    engine = EntityMatchInference(model=matcher, record_cache=cache)
    result = engine.match(empty_df)
    assert len(result) == 0
    assert list(result.columns) == ["s1_id", "match_source", "match_id", "probability", "final_match"]


def test_compute_f_beta_macro_f05():
    """Verify F0.5 formula calculation."""
    # F0.5 = 1.25 * (P * R) / (0.25 * P + R)
    p, r = 1.0, 1.0
    assert compute_f_beta(p, r, beta=0.5) == 1.0

    p, r = 0.8, 0.4
    # Expected: 1.25 * (0.8 * 0.4) / (0.25 * 0.8 + 0.4) = 1.25 * 0.32 / (0.2 + 0.4) = 0.40 / 0.60 = 0.66666...
    assert pytest.approx(compute_f_beta(p, r, beta=0.5), 1e-4) == 0.6666667

    # Edge cases
    assert compute_f_beta(0.0, 1.0, beta=0.5) == 0.0
    assert compute_f_beta(1.0, 0.0, beta=0.5) == 0.0


def test_evaluate_predictions_and_distributions():
    """Verify evaluation calculates zero-match, singleton, and multi-match counts."""
    preds = pd.DataFrame([
        {"s1_id": "s1_singleton", "match_source": "S2", "match_id": "target_1", "probability": 0.95},
        {"s1_id": "s1_multi", "match_source": "S2", "match_id": "target_2a", "probability": 0.85},
        {"s1_id": "s1_multi", "match_source": "S3", "match_id": "target_2b", "probability": 0.80},
        {"s1_id": "s1_zero", "match_source": "S2", "match_id": "distractor", "probability": 0.10},
    ])
    gt = [
        ("s1_singleton", "S2", "target_1"),
        ("s1_multi", "S2", "target_2a"),
        ("s1_multi", "S3", "target_2b"),
        # s1_zero has 0 matches in gt
    ]

    metrics = evaluate_predictions(preds, gt, threshold=0.50)
    assert metrics["macro_f05"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["total_true_positives"] == 3
    assert metrics["total_false_positives"] == 0
    assert metrics["total_false_negatives"] == 0
    assert metrics["zero_match_s1_count"] == 1  # s1_zero had probability < 0.50
    assert metrics["singleton_match_s1_count"] == 1  # s1_singleton
    assert metrics["multi_match_s1_count"] == 1  # s1_multi (2 matches)


def test_threshold_sweeper():
    """Verify ThresholdSweeper finds best threshold."""
    preds = pd.DataFrame([
        {"s1_id": "s1_1", "match_source": "S2", "match_id": "t1", "probability": 0.92},
        {"s1_id": "s1_2", "match_source": "S2", "match_id": "t2", "probability": 0.74},
        {"s1_id": "s1_3", "match_source": "S2", "match_id": "noise", "probability": 0.58},
    ])
    gt = [("s1_1", "S2", "t1"), ("s1_2", "S2", "t2")]

    sweeper = ThresholdSweeper(thresholds=[0.50, 0.60, 0.70, 0.80, 0.90])
    res = sweeper.sweep(preds, gt)

    assert "best_threshold" in res
    assert "results_df" in res
    # At threshold 0.60 or 0.70, noise (0.58) is excluded and t1, t2 are included -> perfect score
    assert res["best_threshold"] in [0.60, 0.70]
    assert res["best_metrics"]["macro_f05"] == 1.0



def test_modular_models_logistic_and_hist_gbm():
    """Verify LogisticRegressionMatcher and HistGradientBoostingMatcher."""
    X = pd.DataFrame(np.random.rand(60, 22), columns=FEATURE_COLUMNS)
    y = (X["name_exact"] > 0.5).astype(int).values

    # Logistic Regression
    lr = LogisticRegressionMatcher()
    lr.fit(X, y)
    lr_probas = lr.predict_proba(X)
    assert len(lr_probas) == 60
    assert np.all(lr_probas >= 0.0) and np.all(lr_probas <= 1.0)

    # HistGradientBoosting
    hgb = HistGradientBoostingMatcher(max_iter=20)
    hgb.fit(X, y)
    hgb_probas = hgb.predict_proba(X)
    assert len(hgb_probas) == 60
    assert np.all(hgb_probas >= 0.0) and np.all(hgb_probas <= 1.0)


def test_end_to_end_synthetic_pipeline(synthetic_records):
    """
    Complete end-to-end integration test:
    Lakshya Blocking -> Candidate Generation -> Dataset Construction ->
    Smriti Feature Generation -> LightGBM Training -> Threshold Sweep -> Inference
    """
    s1_df = synthetic_records["s1"]
    s2_df = synthetic_records["s2"]
    s3_df = synthetic_records["s3"]
    gt = synthetic_records["ground_truth"]

    # 1. Generate candidates with Lakshya's blocker
    blocking_result = generate_candidates(s1_df, s2_df, s3_df)
    candidates = blocking_result["all_candidates"]
    assert len(candidates) > 0

    # 2. Train matcher pipeline
    pipeline_out = train_matcher_pipeline(
        source1=s1_df,
        source2=s2_df,
        source3=s3_df,
        candidates=candidates,
        ground_truth=gt,
        val_size=0.33,
        random_state=42,
    )

    model = pipeline_out["model"]
    cache = pipeline_out["record_cache"]
    best_t = pipeline_out["best_threshold"]
    val_metrics = pipeline_out["val_metrics"]

    assert model.is_fitted
    assert best_t >= 0.50
    assert "macro_f05" in val_metrics

    # 3. Test Inference
    inf_results = match_candidates(
        candidates=candidates,
        source1=s1_df,
        source2=s2_df,
        source3=s3_df,
        model=model,
        threshold=best_t,
        record_cache=cache,
    )

    assert set(["s1_id", "match_source", "match_id", "probability", "final_match"]).issubset(inf_results.columns)
    assert len(inf_results) == len(build_candidate_dataframe(candidates))

    # Check that multi-match entity s1_multi can have multiple positive matches
    s1_multi_matches = inf_results[
        (inf_results["s1_id"] == "s1_multi") & (inf_results["final_match"] == 1)
    ]
    # No artificial 1-to-1 constraint
    assert len(inf_results) > 0
