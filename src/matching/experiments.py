"""Experiment tracking and model comparison logging framework."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Union, Optional
import pandas as pd

from src.matching.model import (
    BaseMatcher,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    HistGradientBoostingMatcher,
)
from src.matching.training import train_matcher_pipeline

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Structured tracker for ML entity matching experiments.
    """

    def __init__(self):
        self.experiments: List[Dict[str, Any]] = []

    def log_experiment(
        self,
        experiment_id: str,
        model_name: str,
        features: List[str],
        training_size: int,
        positive_count: int,
        negative_count: int,
        positive_negative_ratio: float,
        threshold: float,
        precision: float,
        recall: float,
        macro_f05: float,
        false_positives: int,
        false_negatives: int,
        inference_time: float,
        notes: str = "",
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a single experiment run.
        """
        entry = {
            "experiment_id": experiment_id,
            "model": model_name,
            "features_count": len(features),
            "features": ", ".join(features) if len(features) <= 5 else f"{len(features)} features",
            "training_size": training_size,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_negative_ratio": round(positive_negative_ratio, 2),
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "macro_f0.5": round(macro_f05, 4),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "inference_time_sec": round(inference_time, 4),
            "notes": notes,
        }
        if extra_metrics:
            entry.update(extra_metrics)
        self.experiments.append(entry)
        return entry

    def to_dataframe(self) -> pd.DataFrame:
        """Export all logged experiments as a pandas DataFrame."""
        return pd.DataFrame(self.experiments)

    def summary(self) -> str:
        """Formatted summary table of experiments."""
        df = self.to_dataframe()
        if df.empty:
            return "No experiments recorded."
        return df.to_string(index=False)


def run_model_comparison(
    source1: Any,
    source2: Any,
    source3: Any,
    candidates: Any,
    ground_truth: Any,
    models: Optional[Dict[str, BaseMatcher]] = None,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ExperimentTracker, Dict[str, Any]]:
    """
    Evaluates and compares multiple model architectures (e.g. LightGBM, LogisticRegression, HistGradientBoosting).
    """
    if models is None:
        models = {
            "LightGBM": LightGBMMatcher(random_state=random_state),
            "HistGradientBoosting": HistGradientBoostingMatcher(random_state=random_state),
            "LogisticRegression": LogisticRegressionMatcher(random_state=random_state),
        }

    tracker = ExperimentTracker()
    results: Dict[str, Any] = {}

    for idx, (model_name, model_instance) in enumerate(models.items(), 1):
        exp_id = f"EXP_{idx:03d}_{model_name.upper()}"
        logger.info("Running experiment %s with %s...", exp_id, model_name)

        start_time = time.time()
        train_res = train_matcher_pipeline(
            source1=source1,
            source2=source2,
            source3=source3,
            candidates=candidates,
            ground_truth=ground_truth,
            model=model_instance,
            val_size=val_size,
            random_state=random_state,
        )
        total_time = time.time() - start_time

        val_metrics = train_res["val_metrics"]
        train_stats = train_res["train_stats"]

        # Inference timing on val set
        val_pairs = train_res["val_predictions"]
        X_val_feats = train_res.get("X_val")
        if X_val_feats is None:
            from src.matching.dataset import generate_pair_features_matrix
            X_val_feats = generate_pair_features_matrix(val_pairs, train_res["record_cache"])
        
        inf_start = time.time()
        _ = model_instance.predict_proba(X_val_feats)
        inf_time = time.time() - inf_start


        tracker.log_experiment(
            experiment_id=exp_id,
            model_name=model_name,
            features=model_instance.feature_names,
            training_size=train_stats["train_pairs"],
            positive_count=train_stats["train_positives"],
            negative_count=train_stats["train_negatives"],
            positive_negative_ratio=train_stats["pos_neg_ratio"],
            threshold=train_res["best_threshold"],
            precision=val_metrics["macro_precision"],
            recall=val_metrics["macro_recall"],
            macro_f05=val_metrics["macro_f05"],
            false_positives=val_metrics["total_false_positives"],
            false_negatives=val_metrics["total_false_negatives"],
            inference_time=inf_time,
            notes=f"Trained in {total_time:.2f}s, optimal threshold {train_res['best_threshold']:.2f}",
        )
        results[model_name] = train_res

    return tracker, results
