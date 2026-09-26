"""High-level training pipeline orchestrating candidate labelling, entity splitting, and model fitting."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np
import pandas as pd

from src.matching.dataset import (
    NormalizedRecordCache,
    build_candidate_dataframe,
    label_candidate_pairs,
    create_entity_split,
    split_pairs_by_entity,
    generate_pair_features_matrix,
    sample_negatives,
)
from src.matching.model import BaseMatcher, LightGBMMatcher
from src.matching.decision import evaluate_predictions, ThresholdSweeper
from src.blocking.candidate_generator import CandidatePair

logger = logging.getLogger(__name__)


def train_matcher_pipeline(
    source1: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    source2: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    source3: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    candidates: Union[pd.DataFrame, List[CandidatePair], List[Dict[str, Any]], Dict[str, Any]],
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
    model: Optional[BaseMatcher] = None,
    val_size: float = 0.2,
    negative_to_positive_ratio: Optional[float] = None,
    thresholds: Optional[List[float]] = None,
    random_state: int = 42,
    s1_id_col: str = "id",
    s2_id_col: str = "id",
    s3_id_col: str = "id",
) -> Dict[str, Any]:
    """
    Complete end-to-end training and evaluation pipeline for entity matching.
    
    Pipeline:
        1. Cache and normalize all raw records once via NormalizedRecordCache.
        2. Label real candidate pairs from Lakshya blocking output.
        3. Partition pairs strictly by S1 entity into Train and Val sets (0 leakage).
        4. Generate 22 pair features using Smriti's frozen feature engine.
        5. Fit model (e.g. LightGBMMatcher) with validation monitoring.
        6. Execute threshold sweep on validation predictions to maximize Macro F0.5.
        7. Compute full validation metrics and feature importances.
        
    Returns:
        Dict containing:
            - 'model': Fitted matcher
            - 'record_cache': Populated NormalizedRecordCache
            - 'best_threshold': Float threshold maximizing Macro F0.5
            - 'val_metrics': Comprehensive metrics at best threshold
            - 'sweep_results': Threshold sweep DataFrame
            - 'feature_importance': Feature importance ranking
            - 'train_stats': Data sizes and ratios
    """
    pipeline_start = time.time()

    # 1. Normalize and cache records exactly once
    cache = NormalizedRecordCache()
    cache.register_dataset("S1", source1, id_col=s1_id_col)
    cache.register_dataset("S2", source2, id_col=s2_id_col)
    cache.register_dataset("S3", source3, id_col=s3_id_col)

    # 2. Label candidates
    labeled_df = label_candidate_pairs(candidates, ground_truth)
    total_positives = int((labeled_df["label"] == 1).sum())
    total_negatives = int((labeled_df["label"] == 0).sum())

    logger.info(
        "Candidate dataset: %d pairs (%d positive, %d negative, ratio %.2f:1)",
        len(labeled_df),
        total_positives,
        total_negatives,
        total_negatives / total_positives if total_positives > 0 else 0.0,
    )

    # 3. Entity-level split
    train_s1_ids, val_s1_ids = create_entity_split(
        labeled_df,
        test_size=val_size,
        random_state=random_state,
        s1_col="s1_id",
    )

    train_pairs, val_pairs = split_pairs_by_entity(
        labeled_df,
        train_s1_ids=train_s1_ids,
        val_s1_ids=val_s1_ids,
        s1_col="s1_id",
    )

    # Optional negative downsampling on train set only (val set is never downsampled)
    if negative_to_positive_ratio is not None:
        train_pairs = sample_negatives(
            train_pairs,
            negative_to_positive_ratio=negative_to_positive_ratio,
            random_state=random_state,
        )

    # 4. Feature generation
    X_train = generate_pair_features_matrix(train_pairs, cache)
    y_train = train_pairs["label"].values
    X_val = generate_pair_features_matrix(val_pairs, cache)
    y_val = val_pairs["label"].values

    # 5. Model fitting
    matcher = model if model is not None else LightGBMMatcher(random_state=random_state)
    matcher.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)] if len(val_pairs) > 0 else None,
    )

    # 6. Predict on validation set & sweep thresholds
    val_probas = matcher.predict_proba(X_val)
    val_preds_df = val_pairs[["s1_id", "match_source", "match_id", "label"]].copy()
    val_preds_df["probability"] = val_probas

    sweeper = ThresholdSweeper(thresholds=thresholds)
    sweep_out = sweeper.sweep(
        predictions_df=val_preds_df,
        ground_truth=ground_truth,
        all_s1_ids=val_s1_ids,
    )

    best_threshold = sweep_out["best_threshold"]
    val_metrics = sweep_out["best_metrics"]
    feature_imp = matcher.get_feature_importance()

    train_stats = {
        "total_pairs": len(labeled_df),
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "pos_neg_ratio": total_negatives / total_positives if total_positives > 0 else 0.0,
        "train_pairs": len(train_pairs),
        "train_positives": int((train_pairs["label"] == 1).sum()),
        "train_negatives": int((train_pairs["label"] == 0).sum()),
        "val_pairs": len(val_pairs),
        "val_positives": int((val_pairs["label"] == 1).sum()),
        "val_negatives": int((val_pairs["label"] == 0).sum()),
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "total_pipeline_time_sec": time.time() - pipeline_start,
    }

    return {
        "model": matcher,
        "record_cache": cache,
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "sweep_results": sweep_out["results_df"],
        "feature_importance": feature_imp,
        "train_stats": train_stats,
        "val_predictions": val_preds_df,
        "X_val": X_val,
        "X_train": X_train,
    }

