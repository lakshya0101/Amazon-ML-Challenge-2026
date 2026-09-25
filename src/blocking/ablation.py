"""
Ablation Study Module for Candidate Generation.
Evaluates individual blockers and progressive unions to measure recall vs pair volume trade-offs.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Set, Tuple

from src.evaluation.blocking_metrics import evaluate_blocking, BlockingEvalResult


def run_ablation_study(
    blocker_results: List[Tuple[Dict[str, Set[str]], str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str],
    out_path: str | None = None,
) -> dict:
    """
    Run two ablation experiments:
    1. Standalone: each blocker evaluated on its own.
    2. Progressive union: B1 -> B1+B2 -> B1+B2+B3 -> ...
    """
    standalone_metrics = {}
    progressive_metrics = {}

    # 1. Standalone evaluations
    for cmap, name in blocker_results:
        res = evaluate_blocking(cmap, ground_truth)
        standalone_metrics[name] = {
            "recall": res.candidate_pair_recall,
            "pct_s1_all_matches": res.pct_s1_all_matches_retrieved,
            "total_candidate_pairs": res.total_candidate_pairs,
            "avg_candidates_per_s1": res.avg_candidates_per_s1,
            "p95_candidates_per_s1": res.p95_candidates_per_s1,
        }

    # 2. Progressive Union
    accumulated_candidates: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    stage_name = []

    for cmap, name in blocker_results:
        stage_name.append(name)
        current_stage = " + ".join(stage_name)

        for s1, cset in cmap.items():
            accumulated_candidates.setdefault(s1, set()).update(cset)

        res = evaluate_blocking(accumulated_candidates, ground_truth)
        progressive_metrics[current_stage] = {
            "recall": res.candidate_pair_recall,
            "pct_s1_all_matches": res.pct_s1_all_matches_retrieved,
            "total_candidate_pairs": res.total_candidate_pairs,
            "avg_candidates_per_s1": res.avg_candidates_per_s1,
            "p95_candidates_per_s1": res.p95_candidates_per_s1,
            "max_candidates_per_s1": res.max_candidates_per_s1,
        }

    ablation_report = {
        "standalone": standalone_metrics,
        "progressive_union": progressive_metrics,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(ablation_report, fh, indent=4)
        print(f"Ablation report saved to: {out_path}")

    return ablation_report
