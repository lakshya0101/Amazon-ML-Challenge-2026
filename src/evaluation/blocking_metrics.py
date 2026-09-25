"""
Blocking / Candidate-Generation Evaluator
==========================================

Reusable evaluator that compares a set of generated S1 → {S2/S3} candidate
pairs against the TRAINING ground truth.

Usage
-----
    from src.evaluation.blocking_metrics import (
        load_ground_truth,
        evaluate_blocking,
        print_report,
    )

    ground_truth = load_ground_truth("student_resource/dataset/train/train_ground_truth.tsv")
    results = evaluate_blocking(candidates, ground_truth)
    print_report(results)

Candidate format
----------------
    candidates: dict[str, set[str]]
        Mapping of S1 entity_id → set of candidate S2/S3 entity_ids.

Ground truth format
-------------------
    ground_truth: dict[str, set[str]]
        Mapping of S1 entity_id → set of true-match S2/S3 entity_ids.

IMPORTANT
---------
This evaluator is for development/training evaluation ONLY.
The test split has NO ground truth file — never fabricate or leak one.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Set


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(tsv_path: str) -> Dict[str, Set[str]]:
    """Parse train_ground_truth.tsv into {s1_id: {matched_ids…}}.

    The file has two tab-separated columns:
        source1_entity_id    matched_entity_ids
    where matched_entity_ids is a comma-separated list of S2/S3 IDs.
    """
    gt: Dict[str, Set[str]] = {}
    csv.field_size_limit(10 ** 7)

    with open(tsv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw_matches = row["matched_entity_ids"].strip()
            if raw_matches:
                gt[s1_id] = {m.strip() for m in raw_matches.split(",") if m.strip()}
            else:
                gt[s1_id] = set()
    return gt


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BlockingEvalResult:
    """All metrics produced by evaluate_blocking()."""

    # Core recall
    candidate_pair_recall: float = 0.0          # retrieved true pairs / all true pairs
    total_true_pairs: int = 0                   # denominator
    total_retrieved_true_pairs: int = 0         # numerator
    total_candidate_pairs: int = 0              # total generated pairs (true + false)

    # S1-level coverage
    pct_s1_all_matches_retrieved: float = 0.0   # S1 entities with ALL true matches found
    pct_s1_at_least_one_retrieved: float = 0.0  # S1 entities with ≥1 true match found
    total_s1_entities_in_gt: int = 0

    # Candidate volume stats
    avg_candidates_per_s1: float = 0.0
    median_candidates_per_s1: float = 0.0
    p95_candidates_per_s1: float = 0.0
    max_candidates_per_s1: int = 0

    # Diagnostics
    completely_missed_s1_count: int = 0         # 0 true matches retrieved
    partially_recovered_s1_count: int = 0       # some but not all retrieved
    completely_recovered_s1_count: int = 0       # ALL true matches retrieved

    # Lists of entity IDs for drill-down (capped to save memory)
    completely_missed_s1_ids: list = field(default_factory=list)
    partially_recovered_s1_ids: list = field(default_factory=list)

    # Timing
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | None = None, indent: int = 4) -> str:
        blob = json.dumps(self.to_dict(), indent=indent)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(blob)
        return blob


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

# Maximum number of entity IDs stored in the diagnostic lists to avoid
# blowing up memory when millions of entities are missed.
_MAX_DIAGNOSTIC_IDS = 500


def evaluate_blocking(
    candidates: Dict[str, Set[str]] | Tuple[Dict[str, Set[str]], str],
    ground_truth: Dict[str, Set[str]],
) -> BlockingEvalResult:
    """Compare generated candidate pairs against ground truth.

    Parameters
    ----------
    candidates : dict[str, set[str]] or tuple(dict[str, set[str]], str)
        Generated mapping: S1 entity_id → set of candidate S2/S3 entity_ids.
        If a blocker return tuple (candidate_map, blocker_name) is passed,
        it is defensively unwrapped.
    ground_truth : dict[str, set[str]]
        True mapping:     S1 entity_id → set of true-match S2/S3 entity_ids.

    Returns
    -------
    BlockingEvalResult
        Dataclass with every requested metric.
    """
    # Defensive unwrap for (candidate_map, blocker_name) tuple
    if (
        isinstance(candidates, tuple)
        and len(candidates) == 2
        and isinstance(candidates[0], dict)
        and isinstance(candidates[1], str)
    ):
        candidates = candidates[0]

    t0 = time.perf_counter()
    res = BlockingEvalResult()

    res.total_s1_entities_in_gt = len(ground_truth)

    # Accumulate per-S1 candidate counts for distribution stats.
    candidates_per_s1: list[int] = []

    for s1_id, true_matches in ground_truth.items():
        n_true = len(true_matches)
        res.total_true_pairs += n_true

        cand_set = candidates.get(s1_id, set())
        n_cands = len(cand_set)
        candidates_per_s1.append(n_cands)
        res.total_candidate_pairs += n_cands

        retrieved_true = true_matches & cand_set
        n_retrieved = len(retrieved_true)
        res.total_retrieved_true_pairs += n_retrieved

        # Classify this S1 entity
        if n_retrieved == 0:
            res.completely_missed_s1_count += 1
            if len(res.completely_missed_s1_ids) < _MAX_DIAGNOSTIC_IDS:
                res.completely_missed_s1_ids.append(s1_id)
        elif n_retrieved < n_true:
            res.partially_recovered_s1_count += 1
            if len(res.partially_recovered_s1_ids) < _MAX_DIAGNOSTIC_IDS:
                res.partially_recovered_s1_ids.append(s1_id)
        else:
            res.completely_recovered_s1_count += 1

    # Pair recall
    if res.total_true_pairs > 0:
        res.candidate_pair_recall = res.total_retrieved_true_pairs / res.total_true_pairs

    # S1-level coverage percentages
    if res.total_s1_entities_in_gt > 0:
        res.pct_s1_all_matches_retrieved = (
            res.completely_recovered_s1_count / res.total_s1_entities_in_gt
        )
        res.pct_s1_at_least_one_retrieved = (
            (res.completely_recovered_s1_count + res.partially_recovered_s1_count)
            / res.total_s1_entities_in_gt
        )

    # Candidate volume distribution
    if candidates_per_s1:
        res.avg_candidates_per_s1 = statistics.mean(candidates_per_s1)
        res.median_candidates_per_s1 = statistics.median(candidates_per_s1)
        res.max_candidates_per_s1 = max(candidates_per_s1)
        sorted_cands = sorted(candidates_per_s1)
        p95_idx = int(len(sorted_cands) * 0.95)
        res.p95_candidates_per_s1 = sorted_cands[min(p95_idx, len(sorted_cands) - 1)]

    res.runtime_seconds = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_report(result: BlockingEvalResult, title: str = "Blocking Evaluation") -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Candidate Pair Recall:           {result.candidate_pair_recall:.6f}  "
          f"({result.total_retrieved_true_pairs:,} / {result.total_true_pairs:,})")
    print(f"  Total Candidate Pairs Generated: {result.total_candidate_pairs:,}")
    print()
    print(f"  S1 Entities in Ground Truth:     {result.total_s1_entities_in_gt:,}")
    print(f"  % S1 ALL matches retrieved:      {result.pct_s1_all_matches_retrieved:.4%}")
    print(f"  % S1 >=1 match retrieved:        {result.pct_s1_at_least_one_retrieved:.4%}")
    print()
    print(f"  Completely recovered S1:          {result.completely_recovered_s1_count:,}")
    print(f"  Partially recovered S1:           {result.partially_recovered_s1_count:,}")
    print(f"  Completely missed S1:             {result.completely_missed_s1_count:,}")
    print()
    print(f"  Avg  candidates / S1:            {result.avg_candidates_per_s1:.2f}")
    print(f"  Median candidates / S1:          {result.median_candidates_per_s1:.1f}")
    print(f"  P95  candidates / S1:            {result.p95_candidates_per_s1:.0f}")
    print(f"  Max  candidates / S1:            {result.max_candidates_per_s1:,}")
    print()
    print(f"  Evaluation runtime:              {result.runtime_seconds:.4f}s")
    print(sep)
    print()
