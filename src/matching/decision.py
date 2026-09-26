"""Decision logic, metric calculation (Macro F0.5 per S1), and threshold sweep framework."""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple, Any, Union, Optional, Iterable
from collections import defaultdict
import numpy as np
import pandas as pd

from src.matching.dataset import normalize_ground_truth_set

logger = logging.getLogger(__name__)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5, eps: float = 1e-9) -> float:
    """
    Computes F-beta score.
    
    For beta = 0.5:
        F0.5 = (1 + 0.25) * (P * R) / (0.25 * P + R)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom <= eps:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
    threshold: Optional[float] = None,
    all_s1_ids: Optional[Iterable[str]] = None,
    s1_col: str = "s1_id",
    source_col: str = "match_source",
    match_id_col: str = "match_id",
    proba_col: str = "probability",
    match_col: str = "final_match",
) -> Dict[str, Any]:
    """
    Evaluates entity matching predictions against ground truth.
    
    Computes competition metric: Macro F0.5 per S1 entity, precision, recall,
    micro metrics, confusion counts, and predicted match count distribution.
    
    Args:
        predictions_df: DataFrame with candidate predictions.
        ground_truth: Ground truth pairs.
        threshold: Optional probability threshold. If provided, overrides/derives match_col.
        all_s1_ids: Full universe of S1 entity IDs in the validation set.
        
    Returns:
        Dict containing comprehensive evaluation metrics.
    """
    gt_set = normalize_ground_truth_set(ground_truth)

    # Determine matched pairs
    df = predictions_df.copy()
    if threshold is not None and proba_col in df.columns:
        df["_is_match"] = (df[proba_col] >= threshold).astype(int)
    elif match_col in df.columns:
        df["_is_match"] = df[match_col].astype(int)
    elif proba_col in df.columns:
        df["_is_match"] = (df[proba_col] >= 0.5).astype(int)
    else:
        df["_is_match"] = 1

    matched_df = df[df["_is_match"] == 1]

    # Collect predicted pairs: (s1_id, match_source, match_id)
    pred_set: Set[Tuple[str, str, str]] = set()
    for row in matched_df.itertuples(index=False):
        s1 = str(getattr(row, s1_col))
        src = str(getattr(row, source_col)).upper()
        m_id = str(getattr(row, match_id_col))
        pred_set.add((s1, src, m_id))

    # Build per-S1 lookups
    gt_by_s1: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for s1, src, m_id in gt_set:
        gt_by_s1[s1].add((src, m_id))

    pred_by_s1: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for s1, src, m_id in pred_set:
        pred_by_s1[s1].add((src, m_id))

    # Determine complete set of S1 entities to evaluate
    if all_s1_ids is not None:
        eval_s1_set = {str(x) for x in all_s1_ids}
    else:
        eval_s1_set = set(df[s1_col].astype(str).unique()) | set(gt_by_s1.keys())

    # Per-S1 Macro metrics
    s1_precisions: List[float] = []
    s1_recalls: List[float] = []
    s1_f05s: List[float] = []
    match_counts: List[int] = []

    for s1_id in eval_s1_set:
        s1_gt = gt_by_s1.get(s1_id, set())
        s1_pred = pred_by_s1.get(s1_id, set())
        match_counts.append(len(s1_pred))

        if len(s1_gt) == 0 and len(s1_pred) == 0:
            # Non-matching entity correctly predicted with 0 matches
            s1_precisions.append(1.0)
            s1_recalls.append(1.0)
            s1_f05s.append(1.0)
        elif len(s1_gt) > 0 and len(s1_pred) == 0:
            # False negative (missed match)
            s1_precisions.append(0.0)
            s1_recalls.append(0.0)
            s1_f05s.append(0.0)
        elif len(s1_gt) == 0 and len(s1_pred) > 0:
            # False positive for non-matching entity
            s1_precisions.append(0.0)
            s1_recalls.append(0.0)
            s1_f05s.append(0.0)
        else:
            tp = len(s1_pred.intersection(s1_gt))
            p = tp / len(s1_pred) if len(s1_pred) > 0 else 0.0
            r = tp / len(s1_gt) if len(s1_gt) > 0 else 0.0
            f05 = compute_f_beta(p, r, beta=0.5)
            s1_precisions.append(p)
            s1_recalls.append(r)
            s1_f05s.append(f05)

    macro_precision = float(np.mean(s1_precisions)) if s1_precisions else 0.0
    macro_recall = float(np.mean(s1_recalls)) if s1_recalls else 0.0
    macro_f05 = float(np.mean(s1_f05s)) if s1_f05s else 0.0

    # Global Micro metrics (evaluated on the active S1 subset)
    relevant_gt = {gt for gt in gt_set if gt[0] in eval_s1_set}
    total_tp = len(pred_set.intersection(relevant_gt))
    total_fp = len(pred_set - relevant_gt)
    total_fn = len(relevant_gt - pred_set)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f05 = compute_f_beta(micro_precision, micro_recall, beta=0.5)


    # Match distribution statistics
    counts_arr = np.array(match_counts, dtype=np.int32) if match_counts else np.array([0])
    zero_matches = int(np.sum(counts_arr == 0))
    singleton_matches = int(np.sum(counts_arr == 1))
    multi_matches = int(np.sum(counts_arr > 1))
    total_s1 = len(counts_arr)

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_s1_entities": total_s1,
        "total_predictions": len(pred_set),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "zero_match_s1_count": zero_matches,
        "singleton_match_s1_count": singleton_matches,
        "multi_match_s1_count": multi_matches,
        "zero_match_rate": zero_matches / total_s1 if total_s1 > 0 else 0.0,
        "singleton_match_rate": singleton_matches / total_s1 if total_s1 > 0 else 0.0,
        "multi_match_rate": multi_matches / total_s1 if total_s1 > 0 else 0.0,
        "mean_matches_per_s1": float(np.mean(counts_arr)),
        "max_matches_per_s1": int(np.max(counts_arr)),
        "threshold": threshold,
    }


class ThresholdSweeper:
    """
    Evaluates probability thresholds and selects the optimal operating point.
    """

    DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        target_metric: str = "macro_f05",
    ):
        self.thresholds = sorted(thresholds or self.DEFAULT_THRESHOLDS)
        self.target_metric = target_metric

    def sweep(
        self,
        predictions_df: pd.DataFrame,
        ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
        all_s1_ids: Optional[Iterable[str]] = None,
        s1_col: str = "s1_id",
        source_col: str = "match_source",
        match_id_col: str = "match_id",
        proba_col: str = "probability",
    ) -> Dict[str, Any]:
        """
        Runs threshold sweep across the candidate predictions.
        
        Returns:
            Dict containing:
                - "results_df": DataFrame of metrics per threshold
                - "best_threshold": Float threshold maximizing target_metric
                - "best_metrics": Dict of metrics at best threshold
        """
        rows = []
        for t in self.thresholds:
            metrics = evaluate_predictions(
                predictions_df=predictions_df,
                ground_truth=ground_truth,
                threshold=t,
                all_s1_ids=all_s1_ids,
                s1_col=s1_col,
                source_col=source_col,
                match_id_col=match_id_col,
                proba_col=proba_col,
            )
            rows.append(metrics)

        results_df = pd.DataFrame(rows)
        best_idx = results_df[self.target_metric].idxmax()
        best_row = results_df.loc[best_idx].to_dict()
        best_t = float(best_row["threshold"])

        logger.info(
            "Threshold sweep completed: best threshold %.2f achieved %s = %.4f",
            best_t,
            self.target_metric,
            best_row[self.target_metric],
        )

        return {
            "results_df": results_df,
            "best_threshold": best_t,
            "best_metrics": best_row,
        }
