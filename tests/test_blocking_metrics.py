"""
Synthetic test for blocking_metrics evaluator
==============================================

Verifies all metrics against hand-calculated expected values using
a small, deterministic synthetic dataset.

Run:
    python -m tests.test_blocking_metrics
    # or
    python tests/test_blocking_metrics.py
"""

from __future__ import annotations

import sys
import os
import math

# Ensure project root is on sys.path so we can import src.evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.blocking_metrics import evaluate_blocking, print_report, BlockingEvalResult


def _assert_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise AssertionError(f"[FAIL] {label}: expected {expected}, got {actual}")
    print(f"  [PASS] {label} = {actual}")


def _assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"[FAIL] {label}: expected {expected}, got {actual}")
    print(f"  [PASS] {label} = {actual}")


def test_perfect_recall():
    """Every true pair is retrieved — recall should be 1.0."""
    print("\n--- Test: Perfect Recall ---")
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": {"S3-40", "S3-50", "S2-60"},
    }
    # Candidates contain all true matches plus some extras
    cands = {
        "S1-1": {"S2-10", "S3-20", "S2-99"},
        "S1-2": {"S2-30", "S3-88"},
        "S1-3": {"S3-40", "S3-50", "S2-60"},
    }
    r = evaluate_blocking(cands, gt)
    print_report(r, title="Perfect Recall Test")

    # 2 + 1 + 3 = 6 true pairs, all retrieved
    _assert_eq(r.total_true_pairs, 6, "total_true_pairs")
    _assert_eq(r.total_retrieved_true_pairs, 6, "total_retrieved_true_pairs")
    _assert_close(r.candidate_pair_recall, 1.0, "candidate_pair_recall")

    # total candidate pairs: 3 + 2 + 3 = 8
    _assert_eq(r.total_candidate_pairs, 8, "total_candidate_pairs")

    # All 3 S1 entities fully recovered
    _assert_eq(r.completely_recovered_s1_count, 3, "completely_recovered_s1_count")
    _assert_eq(r.partially_recovered_s1_count, 0, "partially_recovered_s1_count")
    _assert_eq(r.completely_missed_s1_count, 0, "completely_missed_s1_count")

    _assert_close(r.pct_s1_all_matches_retrieved, 1.0, "pct_s1_all_matches_retrieved")
    _assert_close(r.pct_s1_at_least_one_retrieved, 1.0, "pct_s1_at_least_one_retrieved")


def test_zero_recall():
    """No candidates generated at all — recall should be 0.0."""
    print("\n--- Test: Zero Recall ---")
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
    }
    cands: dict[str, set[str]] = {}  # empty
    r = evaluate_blocking(cands, gt)
    print_report(r, title="Zero Recall Test")

    _assert_eq(r.total_true_pairs, 3, "total_true_pairs")
    _assert_eq(r.total_retrieved_true_pairs, 0, "total_retrieved_true_pairs")
    _assert_close(r.candidate_pair_recall, 0.0, "candidate_pair_recall")

    _assert_eq(r.completely_missed_s1_count, 2, "completely_missed_s1_count")
    _assert_eq(r.partially_recovered_s1_count, 0, "partially_recovered_s1_count")
    _assert_eq(r.completely_recovered_s1_count, 0, "completely_recovered_s1_count")

    _assert_close(r.pct_s1_all_matches_retrieved, 0.0, "pct_s1_all_matches_retrieved")
    _assert_close(r.pct_s1_at_least_one_retrieved, 0.0, "pct_s1_at_least_one_retrieved")

    # Volume stats: all S1s get 0 candidates
    _assert_close(r.avg_candidates_per_s1, 0.0, "avg_candidates_per_s1")
    _assert_close(r.median_candidates_per_s1, 0.0, "median_candidates_per_s1")
    _assert_eq(r.max_candidates_per_s1, 0, "max_candidates_per_s1")


def test_partial_recall():
    """Mixed scenario — some hits, some misses, some partials."""
    print("\n--- Test: Partial Recall ---")
    gt = {
        "S1-A": {"S2-1", "S2-2", "S3-3"},       # 3 true matches
        "S1-B": {"S3-4"},                          # 1 true match
        "S1-C": {"S2-5", "S3-6"},                  # 2 true matches
        "S1-D": {"S2-7", "S2-8", "S3-9", "S3-10"},# 4 true matches
    }
    # Total true pairs = 3 + 1 + 2 + 4 = 10

    cands = {
        "S1-A": {"S2-1", "S3-3", "S2-99"},        # retrieved 2/3 → partial
        "S1-B": {"S3-4"},                           # retrieved 1/1 → complete
        # S1-C: not present at all                  → missed
        "S1-D": {"S2-7", "S2-8", "S3-9", "S3-10", "S2-77"},  # 4/4 → complete
    }
    # Retrieved true = 2 + 1 + 0 + 4 = 7
    # Total candidate pairs = 3 + 1 + 0 + 5 = 9

    r = evaluate_blocking(cands, gt)
    print_report(r, title="Partial Recall Test")

    _assert_eq(r.total_true_pairs, 10, "total_true_pairs")
    _assert_eq(r.total_retrieved_true_pairs, 7, "total_retrieved_true_pairs")
    _assert_close(r.candidate_pair_recall, 0.7, "candidate_pair_recall")

    _assert_eq(r.total_candidate_pairs, 9, "total_candidate_pairs")
    _assert_eq(r.total_s1_entities_in_gt, 4, "total_s1_entities_in_gt")

    # S1-B complete, S1-D complete → 2 fully recovered
    _assert_eq(r.completely_recovered_s1_count, 2, "completely_recovered_s1_count")
    _assert_close(r.pct_s1_all_matches_retrieved, 0.5, "pct_s1_all_matches_retrieved")

    # S1-A partial → 1
    _assert_eq(r.partially_recovered_s1_count, 1, "partially_recovered_s1_count")

    # S1-C missed → 1
    _assert_eq(r.completely_missed_s1_count, 1, "completely_missed_s1_count")

    # ≥1 retrieved: S1-A, S1-B, S1-D → 3/4 = 0.75
    _assert_close(r.pct_s1_at_least_one_retrieved, 0.75, "pct_s1_at_least_one_retrieved")

    # Volume distribution: [3, 1, 0, 5] → sorted [0, 1, 3, 5]
    _assert_close(r.avg_candidates_per_s1, 9 / 4, "avg_candidates_per_s1")
    _assert_close(r.median_candidates_per_s1, 2.0, "median_candidates_per_s1")  # median of [0,1,3,5]
    _assert_eq(r.max_candidates_per_s1, 5, "max_candidates_per_s1")

    # Diagnostic IDs
    _assert_eq("S1-C" in r.completely_missed_s1_ids, True, "S1-C in missed list")
    _assert_eq("S1-A" in r.partially_recovered_s1_ids, True, "S1-A in partial list")


def test_extra_candidates_ignored():
    """Candidates for S1 IDs not in ground truth should not affect metrics."""
    print("\n--- Test: Extra Candidates Ignored ---")
    gt = {
        "S1-X": {"S2-100"},
    }
    cands = {
        "S1-X": {"S2-100", "S2-200"},
        "S1-UNKNOWN": {"S2-300", "S3-400"},  # not in GT
    }
    r = evaluate_blocking(cands, gt)
    print_report(r, title="Extra Candidates Ignored Test")

    _assert_eq(r.total_true_pairs, 1, "total_true_pairs")
    _assert_eq(r.total_retrieved_true_pairs, 1, "total_retrieved_true_pairs")
    _assert_close(r.candidate_pair_recall, 1.0, "candidate_pair_recall")

    # total_candidate_pairs only counts candidates for S1 IDs in the GT
    _assert_eq(r.total_candidate_pairs, 2, "total_candidate_pairs")
    _assert_eq(r.total_s1_entities_in_gt, 1, "total_s1_entities_in_gt")


def test_serialisation():
    """to_dict() and to_json() should not crash."""
    print("\n--- Test: Serialisation ---")
    r = BlockingEvalResult(candidate_pair_recall=0.42, total_true_pairs=100)
    d = r.to_dict()
    _assert_eq(d["candidate_pair_recall"], 0.42, "dict round-trip recall")
    _assert_eq(d["total_true_pairs"], 100, "dict round-trip total_true_pairs")

    j = r.to_json()
    _assert_eq('"candidate_pair_recall": 0.42' in j, True, "json contains recall")
    print("  [PASS] to_json() produces valid string")


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Blocking Metrics — Synthetic Test Suite")
    print("=" * 60)

    test_perfect_recall()
    test_zero_recall()
    test_partial_recall()
    test_extra_candidates_ignored()
    test_serialisation()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
