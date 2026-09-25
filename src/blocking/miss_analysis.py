"""
Miss Analysis Module for Amazon ML Challenge 2026.
Analyzes ground truth pairs missed by candidate generation to diagnose recall gaps.
Uses canonical EntityRecord representation.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Set, Any, Optional

from src.data.record import EntityRecord
from src.normalization.name_normalizer import normalize_name, informative_tokens


def _to_entity_record(info: Any, fallback_id: str = "") -> EntityRecord:
    """Helper to convert dictionary, EntityRecord, 3-tuple or 4-tuple to EntityRecord."""
    if isinstance(info, EntityRecord):
        return info
    if isinstance(info, (tuple, list)) and len(info) == 3:
        return EntityRecord(
            entity_id=fallback_id,
            business_name=str(info[0] or "").strip(),
            business_address=str(info[1] or "").strip(),
            country=str(info[2] or "").strip(),
        )
    return EntityRecord.from_any(info)


def analyze_misses(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s1_dict: Optional[Dict[str, Any]] = None,
    target_dict: Optional[Dict[str, Any]] = None,
    max_sample_display: int = 50,
) -> dict:
    """
    Examine completely missed S1 entities and missed pairs.
    s1_dict and target_dict map entity_id -> EntityRecord (or tuple/dict).
    """
    total_true_pairs = 0
    missed_pairs_count = 0
    completely_missed_s1 = []
    partially_missed_s1 = []

    miss_reasons = Counter()
    sample_misses = []

    for s1_id, true_set in ground_truth.items():
        n_true = len(true_set)
        total_true_pairs += n_true

        cands = candidates.get(s1_id, set())
        retrieved = true_set & cands
        missed = true_set - cands
        missed_pairs_count += len(missed)

        if not retrieved and n_true > 0:
            completely_missed_s1.append(s1_id)
        elif len(retrieved) < n_true:
            partially_missed_s1.append(s1_id)

        # Inspect reasons if entity metadata provided
        if s1_dict and target_dict and missed:
            s1_info = s1_dict.get(s1_id, ("", "", "", ""))
            s1_rec = _to_entity_record(s1_info, fallback_id=s1_id)
            s1_name_norm = normalize_name(s1_rec.business_name)
            s1_tokens = set(informative_tokens(s1_rec.business_name))

            for m_id in missed:
                t_info = target_dict.get(m_id, ("", "", "", ""))
                t_rec = _to_entity_record(t_info, fallback_id=m_id)
                t_name_norm = normalize_name(t_rec.business_name)
                t_tokens = set(informative_tokens(t_rec.business_name))

                # Diagnostic checks
                if not s1_tokens or not t_tokens:
                    miss_reasons["empty_or_stopword_name"] += 1
                elif not (s1_tokens & t_tokens):
                    miss_reasons["zero_token_overlap"] += 1
                else:
                    miss_reasons["token_capped_or_filtered"] += 1

                if len(sample_misses) < max_sample_display:
                    sample_misses.append({
                        "s1_id": s1_id,
                        "s1_name": s1_rec.business_name,
                        "s1_norm": s1_name_norm,
                        "matched_id": m_id,
                        "matched_name": t_rec.business_name,
                        "matched_norm": t_name_norm,
                        "s1_country": s1_rec.country,
                        "matched_country": t_rec.country,
                        "common_tokens": list(s1_tokens & t_tokens),
                    })

    report = {
        "total_true_pairs": total_true_pairs,
        "missed_true_pairs": missed_pairs_count,
        "pair_recall": (total_true_pairs - missed_pairs_count) / total_true_pairs if total_true_pairs else 0.0,
        "completely_missed_s1_count": len(completely_missed_s1),
        "partially_missed_s1_count": len(partially_missed_s1),
        "miss_reasons": dict(miss_reasons),
        "sample_misses": sample_misses[:max_sample_display],
    }
    return report


def save_miss_analysis(report: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)
    print(f"Miss analysis saved to: {out_path}")
