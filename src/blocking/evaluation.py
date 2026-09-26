"""Comprehensive candidate generation evaluation and ablation benchmarking."""

from typing import Dict, List, Set, Tuple, Any, Union, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import time

from src.blocking.config import BlockingConfig
from src.blocking.candidate_generator import CandidatePair, CandidateGenerator
from src.blocking.retrieval import (
    RULE_EXACT_NAME,
    RULE_NAME_CORE,
    RULE_COUNTRY_TOKEN,
    RULE_POSTAL_CODE,
    RULE_HOUSE_ADDRESS_TOKEN,
    RULE_COUNTRY_ADDRESS_TOKENS,
)



def _normalize_ground_truth(
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], List[Dict[str, Any]], Set[Tuple[str, str, str]]]
) -> Set[Tuple[str, str, str]]:
    """Normalizes ground truth into a set of (s1_id, match_source, match_id) tuples."""
    normalized = set()
    if isinstance(ground_truth, pd.DataFrame):
        for _, row in ground_truth.iterrows():
            s1_id = str(row.get("s1_id", row.get("source1_id", "")))
            src = str(row.get("match_source", row.get("source", "")))
            match_id = str(row.get("match_id", row.get("target_id", "")))
            if s1_id and match_id:
                normalized.add((s1_id, src, match_id))
    elif isinstance(ground_truth, (list, set, tuple)):
        for item in ground_truth:
            if isinstance(item, (tuple, list)):
                if len(item) == 3:
                    normalized.add((str(item[0]), str(item[1]), str(item[2])))
                elif len(item) == 2:
                    normalized.add((str(item[0]), "", str(item[1])))
            elif isinstance(item, dict):
                s1_id = str(item.get("s1_id", item.get("source1_id", "")))
                src = str(item.get("match_source", item.get("source", "")))
                match_id = str(item.get("match_id", item.get("target_id", "")))
                if s1_id and match_id:
                    normalized.add((s1_id, src, match_id))
    return normalized


def calculate_metrics_for_subset(
    candidates: List[CandidatePair],
    ground_truth: Set[Tuple[str, str, str]],
    s1_count: int,
    target_count: int,
    target_source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculates recall and volume metrics for a specific target source or combined."""
    # Filter ground truth for this target source if source name is given
    if target_source_name:
        relevant_gt = {
            gt for gt in ground_truth if not gt[1] or gt[1].upper() == target_source_name.upper()
        }
    else:
        relevant_gt = ground_truth

    # Build candidate lookup: (s1_id, match_source, match_id) or (s1_id, match_id)
    retrieved_pairs_with_src = {(c.s1_id, c.match_source, c.match_id) for c in candidates}
    retrieved_pairs_no_src = {(c.s1_id, c.match_id) for c in candidates}

    # Count retrieved true pairs
    retrieved_gt = set()
    missed_gt = []

    for gt in relevant_gt:
        s1_id, src, m_id = gt
        if (s1_id, src, m_id) in retrieved_pairs_with_src or (s1_id, m_id) in retrieved_pairs_no_src:
            retrieved_gt.add(gt)
        else:
            missed_gt.append(gt)

    total_gt = len(relevant_gt)
    recall = len(retrieved_gt) / total_gt if total_gt > 0 else 1.0

    # Calculate per-S1 candidate distribution
    s1_candidate_counts: Dict[str, int] = defaultdict(int)
    for c in candidates:
        s1_candidate_counts[c.s1_id] += 1

    # Include all S1 records (even those with 0 candidates) in distribution
    counts_list = list(s1_candidate_counts.values())
    if s1_count > len(counts_list):
        counts_list.extend([0] * (s1_count - len(counts_list)))

    if not counts_list:
        counts_list = [0]

    avg_candidates = float(np.mean(counts_list))
    median_candidates = float(np.median(counts_list))
    p95_candidates = float(np.percentile(counts_list, 95))
    max_candidates = int(np.max(counts_list))
    total_candidates = len(candidates)

    # Reduction ratio
    total_possible_pairs = s1_count * target_count if (s1_count > 0 and target_count > 0) else 0
    if total_possible_pairs > 0:
        reduction_ratio = 1.0 - (total_candidates / total_possible_pairs)
    else:
        reduction_ratio = 1.0

    return {
        "candidate_recall": recall,
        "true_positives": len(retrieved_gt),
        "total_true_pairs": total_gt,
        "missed_true_pairs": missed_gt,
        "total_candidate_pairs": total_candidates,
        "avg_candidates_per_s1": avg_candidates,
        "median_candidates_per_s1": median_candidates,
        "p95_candidates_per_s1": p95_candidates,
        "max_candidates_per_s1": max_candidates,
        "reduction_ratio": reduction_ratio,
    }


def evaluate_candidates(
    candidates_result: Union[Dict[str, Any], List[CandidatePair]],
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
    s1_count: Optional[int] = None,
    s2_count: Optional[int] = None,
    s3_count: Optional[int] = None,
    runtime_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Evaluates candidate generation performance with S2 and S3 separation."""
    gt_set = _normalize_ground_truth(ground_truth)

    if isinstance(candidates_result, dict):
        s2_candidates = candidates_result.get("s2_candidates", [])
        s3_candidates = candidates_result.get("s3_candidates", [])
        all_candidates = candidates_result.get("all_candidates", s2_candidates + s3_candidates)
        s1_count = s1_count or candidates_result.get("s1_count", 0)
        s2_count = s2_count or candidates_result.get("s2_count", 0)
        s3_count = s3_count or candidates_result.get("s3_count", 0)
        runtime_seconds = runtime_seconds or candidates_result.get("runtime_seconds", 0.0)
    else:
        all_candidates = candidates_result
        s2_candidates = [c for c in all_candidates if c.match_source.upper() == "S2"]
        s3_candidates = [c for c in all_candidates if c.match_source.upper() == "S3"]
        s1_count = s1_count or len({c.s1_id for c in all_candidates})
        s2_count = s2_count or len({c.match_id for c in s2_candidates})
        s3_count = s3_count or len({c.match_id for c in s3_candidates})

    target_total = (s2_count or 0) + (s3_count or 0)

    overall_metrics = calculate_metrics_for_subset(
        all_candidates, gt_set, s1_count or 1, target_total or 1
    )
    s2_metrics = calculate_metrics_for_subset(
        s2_candidates, gt_set, s1_count or 1, s2_count or 1, target_source_name="S2"
    )
    s3_metrics = calculate_metrics_for_subset(
        s3_candidates, gt_set, s1_count or 1, s3_count or 1, target_source_name="S3"
    )

    return {
        "overall": {
            **overall_metrics,
            "runtime_seconds": runtime_seconds,
            "s1_count": s1_count,
            "total_target_count": target_total,
        },
        "s2": {
            **s2_metrics,
            "s1_count": s1_count,
            "s2_count": s2_count,
        },
        "s3": {
            **s3_metrics,
            "s1_count": s1_count,
            "s3_count": s3_count,
        },
    }


def evaluate_strategy_ablation(
    source1: Union[pd.DataFrame, List[Dict[str, Any]]],
    source2: Union[pd.DataFrame, List[Dict[str, Any]]],
    source3: Union[pd.DataFrame, List[Dict[str, Any]]],
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
    base_config: Optional[BlockingConfig] = None,
) -> Dict[str, Any]:
    """Runs ablation study measuring both individual rule contributions and cumulative union."""
    config = base_config or BlockingConfig()

    individual_rules = [
        ("exact_name", BlockingConfig(
            exact_name=True,
            exact_name_core=False,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("name_core", BlockingConfig(
            exact_name=False,
            exact_name_core=True,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("country_name_token", BlockingConfig(
            exact_name=False,
            exact_name_core=False,
            country_name_token=True,
            postal_code=False,
            house_number_address_token=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("postal_code", BlockingConfig(
            exact_name=False,
            exact_name_core=False,
            country_name_token=False,
            postal_code=True,
            house_number_address_token=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("house_number_address_token", BlockingConfig(
            exact_name=False,
            exact_name_core=False,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=True,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("country_address_tokens", BlockingConfig(
            exact_name=False,
            exact_name_core=False,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=False,
            country_address_tokens=True,
            country_address_token_min_shared=config.country_address_token_min_shared,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
    ]

    cumulative_stages = [
        ("exact_name", BlockingConfig(
            exact_name=True,
            exact_name_core=False,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=False,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("exact_name + name_core", BlockingConfig(
            exact_name=True,
            exact_name_core=True,
            country_name_token=False,
            postal_code=False,
            house_number_address_token=False,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("exact_name + name_core + country_token", BlockingConfig(
            exact_name=True,
            exact_name_core=True,
            country_name_token=True,
            postal_code=False,
            house_number_address_token=False,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("exact_name + name_core + country_token + postal", BlockingConfig(
            exact_name=True,
            exact_name_core=True,
            country_name_token=True,
            postal_code=True,
            house_number_address_token=False,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("Phase 1 Union (+ house_addr)", BlockingConfig(
            exact_name=True,
            exact_name_core=True,
            country_name_token=True,
            postal_code=True,
            house_number_address_token=True,
            country_address_tokens=False,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
        ("Full Union (+ country_address_tokens)", BlockingConfig(
            exact_name=True,
            exact_name_core=True,
            country_name_token=True,
            postal_code=True,
            house_number_address_token=True,
            country_address_tokens=True,
            country_address_token_min_shared=config.country_address_token_min_shared,
            name_stopwords=config.name_stopwords,
            address_stopwords=config.address_stopwords,
            max_bucket_size=config.max_bucket_size,
        )),
    ]

    generator = CandidateGenerator()
    generator.build_indexes(source2, source3)

    individual_results = {}
    for name, cfg in individual_rules:
        generator.config = cfg
        t0 = time.time()
        res = generator.generate(source1)
        res["runtime_seconds"] = time.time() - t0
        eval_res = evaluate_candidates(res, ground_truth)
        individual_results[name] = eval_res

    cumulative_results = {}
    for name, cfg in cumulative_stages:
        generator.config = cfg
        t0 = time.time()
        res = generator.generate(source1)
        res["runtime_seconds"] = time.time() - t0
        eval_res = evaluate_candidates(res, ground_truth)
        cumulative_results[name] = eval_res

    return {
        "individual_strategies": individual_results,
        "cumulative_stages": cumulative_results,
    }
