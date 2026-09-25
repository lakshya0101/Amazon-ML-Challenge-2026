"""
Decision Logic and Evaluation Metrics for Business Entity Resolution.

Component: Lakshya (ML Matching Model)
Responsibilities:
- Post-processing probability predictions into binary match decisions.
- Configurable thresholding, top-1/top-2 margin enforcement, and match-count caps.
- Supports 0 matches, 1 match, and multiple matches per S1 entity.
- Metric calculation: Precision, Recall, Macro/Micro F0.5, and S1 match statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. DECISION LOGIC & MATCH DECIDER
# ==============================================================================

class MatchDecider:
    """
    Decider that converts continuous match probabilities into discrete match decisions.
    
    Features:
    - Probability thresholding (P >= threshold).
    - Zero-match support (never forces a match if all candidates are below threshold).
    - Multiple-match support (entity may match candidates in both S2 and S3 or multiple records).
    - Optional margin filter (requires top-1 candidate to exceed top-2 candidate by a margin).
    - Optional max_matches_per_s1 cap.
    - Source priority / preference (optional).
    
    Attributes:
        threshold (float): Minimum probability required to qualify as a match.
        margin (float or None): Minimum probability gap required between top-1 and top-2 candidates.
        max_matches_per_s1 (int or None): Maximum matches allowed per S1 entity (None = unlimited).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        margin: Optional[float] = None,
        max_matches_per_s1: Optional[int] = None,
    ) -> None:
        """
        Initialize MatchDecider with configurable decision criteria.
        
        Args:
            threshold: Probability threshold in (0.0, 1.0). Default: 0.5.
            margin: Optional minimum probability gap between top-1 and subsequent candidates.
            max_matches_per_s1: Optional limit on positive matches per S1 query.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if margin is not None and not (0.0 <= margin <= 1.0):
            raise ValueError(f"margin must be between 0.0 and 1.0, got {margin}")

        self.threshold = threshold
        self.margin = margin
        self.max_matches_per_s1 = max_matches_per_s1

    def decide(
        self,
        candidate_df: pd.DataFrame,
        proba_col: str = "match_probability",
        s1_col: str = "s1_entity_id",
        candidate_col: str = "candidate_entity_id",
        source_col: str = "candidate_source",
        output_match_col: str = "is_match",
    ) -> pd.DataFrame:
        """
        Apply decision logic to candidate predictions DataFrame.
        
        Args:
            candidate_df: DataFrame containing candidate pairs and predicted probabilities.
            proba_col: Column name containing predicted probabilities.
            s1_col: Column name for S1 entity ID.
            candidate_col: Column name for candidate entity ID.
            source_col: Column name for candidate source (e.g. S2, S3).
            output_match_col: Column name for binary match decision output.
            
        Returns:
            DataFrame with all original columns plus `output_match_col` (1 for match, 0 for non-match).
        """
        if candidate_df.empty:
            result = candidate_df.copy()
            result[output_match_col] = pd.Series(dtype=np.int32)
            return result

        for col in [proba_col, s1_col, candidate_col]:
            if col not in candidate_df.columns:
                raise ValueError(f"candidate_df must contain column '{col}'.")

        df = candidate_df.copy()
        
        # Step 1: Base thresholding
        # A candidate is initially considered a candidate match if proba >= threshold
        is_above_threshold = df[proba_col] >= self.threshold
        df[output_match_col] = np.where(is_above_threshold, 1, 0).astype(np.int32)

        # If no margin or max_matches constraint is set, return vectorized result directly
        if self.margin is None and self.max_matches_per_s1 is None:
            return df

        # Step 2: S1-level grouped logic for margin and max_matches filtering
        # Sort candidates per S1 by probability descending
        df_sorted = df.sort_values(by=[s1_col, proba_col], ascending=[True, False]).copy()
        df_sorted["_rank_s1"] = df_sorted.groupby(s1_col).cumcount() + 1

        # Margin check: if margin is set and multiple candidates exist
        if self.margin is not None:
            top1_proba = df_sorted.groupby(s1_col)[proba_col].transform("first")
            # Calculate top-2 proba per S1
            top2_series = df_sorted[df_sorted["_rank_s1"] == 2].set_index(s1_col)[proba_col]
            top2_proba = df_sorted[s1_col].map(top2_series).fillna(-1.0)
            
            # If top-1 doesn't clear top-2 by margin, disqualify candidates ranked > 1
            fails_margin = (top1_proba - top2_proba) < self.margin
            mask_disqualify = fails_margin & (df_sorted["_rank_s1"] > 1)
            df_sorted.loc[mask_disqualify, output_match_col] = 0

        # Max matches cap per S1
        if self.max_matches_per_s1 is not None and self.max_matches_per_s1 > 0:
            cum_matches = df_sorted.groupby(s1_col)[output_match_col].cumsum()
            df_sorted.loc[cum_matches > self.max_matches_per_s1, output_match_col] = 0

        df_sorted.drop(columns=["_rank_s1"], inplace=True)
        # Restore original index order
        df_result = df_sorted.loc[df.index]
        return df_result

    def filter_matches(
        self,
        candidate_df: pd.DataFrame,
        proba_col: str = "match_probability",
        s1_col: str = "s1_entity_id",
        candidate_col: str = "candidate_entity_id",
        source_col: str = "candidate_source",
    ) -> pd.DataFrame:
        """
        Apply decision logic and return only the positive matches (is_match == 1).
        
        Returns:
            Filtered DataFrame containing only positive matches.
        """
        decided = self.decide(
            candidate_df=candidate_df,
            proba_col=proba_col,
            s1_col=s1_col,
            candidate_col=candidate_col,
            source_col=source_col,
            output_match_col="is_match",
        )
        return decided[decided["is_match"] == 1].reset_index(drop=True)


def apply_decision_logic(
    candidate_df: pd.DataFrame,
    threshold: float = 0.5,
    margin: Optional[float] = None,
    max_matches_per_s1: Optional[int] = None,
    proba_col: str = "match_probability",
    s1_col: str = "s1_entity_id",
    candidate_col: str = "candidate_entity_id",
    source_col: str = "candidate_source",
) -> pd.DataFrame:
    """
    Functional helper to apply decision logic to a candidate DataFrame.
    
    Args:
        candidate_df: DataFrame with predictions.
        threshold: Decision threshold.
        margin: Optional top-1 margin.
        max_matches_per_s1: Optional match count cap.
        proba_col: Probability column name.
        s1_col: S1 entity ID column.
        candidate_col: Candidate entity ID column.
        source_col: Candidate source column.
        
    Returns:
        DataFrame with 'is_match' column added.
    """
    decider = MatchDecider(
        threshold=threshold,
        margin=margin,
        max_matches_per_s1=max_matches_per_s1,
    )
    return decider.decide(
        candidate_df=candidate_df,
        proba_col=proba_col,
        s1_col=s1_col,
        candidate_col=candidate_col,
        source_col=source_col,
    )


# ==============================================================================
# 2. METRICS & EVALUATION UTILITIES
# ==============================================================================

def compute_f_beta(precision: float, recall: float, beta: float = 0.5, eps: float = 1e-9) -> float:
    """
    Compute F-beta score.
    
    For beta = 0.5:
        F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R) = 1.25 * (P * R) / (0.25 * P + R)
    Precision is weighted twice as heavily as recall.
    
    Args:
        precision: Precision in [0.0, 1.0].
        recall: Recall in [0.0, 1.0].
        beta: Beta parameter (default: 0.5 for F0.5).
        eps: Small epsilon to avoid division by zero.
        
    Returns:
        F-beta score in [0.0, 1.0].
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * (precision * recall)
    denominator = (beta_sq * precision) + recall + eps
    return float(numerator / denominator)


def evaluate_entity_resolution(
    predictions_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Set[Tuple[Any, Any]]],
    s1_col: str = "s1_entity_id",
    candidate_col: str = "candidate_entity_id",
    match_col: str = "is_match",
    all_s1_ids: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate entity resolution predictions against ground truth.
    
    Calculates:
    - Competition metric: Macro F0.5 per S1 entity
    - Macro Precision and Macro Recall per S1
    - Micro Precision, Micro Recall, and Micro F0.5
    - Per-S1 match count statistics (min, max, mean, zero-match %, multi-match %)
    
    Args:
        predictions_df: DataFrame with predictions and binary match_col (or filtered matches).
        ground_truth: DataFrame with [s1_col, candidate_col] or set of (s1_id, candidate_id) ground-truth pairs.
        s1_col: S1 entity column name.
        candidate_col: Candidate entity column name.
        match_col: Column indicating predicted match (1/0 or boolean). If not present, all rows in predictions_df are treated as positive matches.
        all_s1_ids: Optional set of all evaluated S1 IDs (including those with 0 candidate matches).
        
    Returns:
        Dictionary of comprehensive evaluation metrics and match statistics.
    """
    # Build ground truth set
    if isinstance(ground_truth, set):
        gt_set = ground_truth
    elif isinstance(ground_truth, pd.DataFrame):
        gt_set = set(zip(ground_truth[s1_col], ground_truth[candidate_col]))
    else:
        raise TypeError("ground_truth must be a DataFrame or set of (s1_id, candidate_id) tuples.")

    # Filter predicted matches
    if match_col in predictions_df.columns:
        pred_matches = predictions_df[predictions_df[match_col] == 1]
    else:
        pred_matches = predictions_df

    pred_set = set(zip(pred_matches[s1_col], pred_matches[candidate_col]))

    # Determine unique S1 entities in scope
    if all_s1_ids is not None:
        evaluated_s1_set = set(all_s1_ids)
    else:
        # Combined S1s from predictions and ground truth
        evaluated_s1_set = set(predictions_df[s1_col]).union({s1 for s1, _ in gt_set})

    # Group true pairs and predicted pairs by S1
    gt_by_s1: Dict[Any, Set[Any]] = {}
    for s1, cand in gt_set:
        gt_by_s1.setdefault(s1, set()).add(cand)

    pred_by_s1: Dict[Any, Set[Any]] = {}
    for s1, cand in pred_set:
        pred_by_s1.setdefault(s1, set()).add(cand)

    # Per-S1 Macro metrics
    s1_precisions: List[float] = []
    s1_recalls: List[float] = []
    s1_f05s: List[float] = []
    match_counts: List[int] = []

    total_tp = len(pred_set.intersection(gt_set))
    total_fp = len(pred_set - gt_set)
    total_fn = len(gt_set - pred_set)

    for s1 in evaluated_s1_set:
        s1_gt = gt_by_s1.get(s1, set())
        s1_pred = pred_by_s1.get(s1, set())
        match_counts.append(len(s1_pred))

        if len(s1_pred) == 0 and len(s1_gt) == 0:
            # S1 has no true match and correctly predicted no match -> perfect precision & recall
            s1_precisions.append(1.0)
            s1_recalls.append(1.0)
            s1_f05s.append(1.0)
        elif len(s1_pred) == 0 and len(s1_gt) > 0:
            # Missed ground truth
            s1_precisions.append(0.0)
            s1_recalls.append(0.0)
            s1_f05s.append(0.0)
        elif len(s1_pred) > 0 and len(s1_gt) == 0:
            # False positive matches for non-matching S1
            s1_precisions.append(0.0)
            s1_recalls.append(0.0)
            s1_f05s.append(0.0)
        else:
            tp = len(s1_pred.intersection(s1_gt))
            p = tp / len(s1_pred)
            r = tp / len(s1_gt)
            f05 = compute_f_beta(p, r, beta=0.5)
            s1_precisions.append(p)
            s1_recalls.append(r)
            s1_f05s.append(f05)

    # Aggregations
    macro_precision = float(np.mean(s1_precisions)) if s1_precisions else 0.0
    macro_recall = float(np.mean(s1_recalls)) if s1_recalls else 0.0
    macro_f05 = float(np.mean(s1_f05s)) if s1_f05s else 0.0

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f05 = compute_f_beta(micro_precision, micro_recall, beta=0.5)

    # Match count statistics
    match_arr = np.array(match_counts, dtype=np.int32)
    n_total_s1 = len(match_arr)
    n_zero_matches = int(np.sum(match_arr == 0))
    n_single_matches = int(np.sum(match_arr == 1))
    n_multi_matches = int(np.sum(match_arr > 1))

    stats: Dict[str, Any] = {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "match_count_stats": {
            "total_s1_entities": n_total_s1,
            "zero_match_count": n_zero_matches,
            "zero_match_pct": (n_zero_matches / n_total_s1 * 100) if n_total_s1 > 0 else 0.0,
            "single_match_count": n_single_matches,
            "single_match_pct": (n_single_matches / n_total_s1 * 100) if n_total_s1 > 0 else 0.0,
            "multi_match_count": n_multi_matches,
            "multi_match_pct": (n_multi_matches / n_total_s1 * 100) if n_total_s1 > 0 else 0.0,
            "mean_matches_per_s1": float(np.mean(match_arr)) if n_total_s1 > 0 else 0.0,
            "max_matches_per_s1": int(np.max(match_arr)) if n_total_s1 > 0 else 0,
        },
    }

    return stats
