"""
Candidate Generator — Blocking-based candidate generation
==========================================================

Implements scalable, recall-oriented blocking strategies and a
unified runner that merges their outputs, evaluates against training ground
truth, and writes results to disk.

Canonical Record Representation:
All blockers and loaders use `src.data.record.EntityRecord`, a lightweight
NamedTuple supporting attribute access (.business_name, .entity_id, etc.),
dictionary-style access (.get()), and tuple indexing without memory overhead.

Supported Stages:
- PROFILE
- BASELINE_BLOCKING
- ADDRESS_BLOCKING
- TFIDF_RETRIEVAL
- MISS_ANALYSIS
- ABLATION
- FINAL_CANDIDATE_GENERATION

All paths are configurable via src.config or CLI arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

from src.config import Config, default_config
from src.data.record import EntityRecord, ensure_records
from src.normalization.name_normalizer import (
    normalize_name,
    first_token,
    informative_tokens,
)
from src.evaluation.blocking_metrics import (
    load_ground_truth,
    evaluate_blocking,
    print_report,
)
from src.data.loader import load_source_df, load_source_tuples, load_ground_truth_fast

# Optional advanced blockers
try:
    from src.blocking.address_blocker import blocker_country_postal_initial
except ImportError:
    blocker_country_postal_initial = None

try:
    from src.blocking.tfidf_blocker import blocker_tfidf_retrieval
except ImportError:
    blocker_tfidf_retrieval = None

try:
    from src.blocking.miss_analysis import analyze_misses, save_miss_analysis
except ImportError:
    analyze_misses = None

try:
    from src.blocking.ablation import run_ablation_study
except ImportError:
    run_ablation_study = None

csv.field_size_limit(10 ** 7)

# Type alias
CandidateMap = Dict[str, Set[str]]   # s1_id → set of s2/s3 candidate ids


# ---------------------------------------------------------------------------
# Data loader helper (backward compatible + fast)
# ---------------------------------------------------------------------------

def load_source(path: str) -> List[EntityRecord]:
    """
    Load a source TSV into a list of canonical EntityRecord instances.
    Guarantees standard column alignment (entity_id, business_name, business_address, country)
    while maintaining tuple efficiency.
    """
    return load_source_tuples(path)


# ---------------------------------------------------------------------------
# Index builders with disk caching
# ---------------------------------------------------------------------------

def _build_index_norm_name(
    rows: List[Any],
    cache_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """norm_name → [entity_id, …]"""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    records = ensure_records(rows)
    idx: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        key = normalize_name(rec.business_name)
        if key:
            idx[key].append(rec.entity_id)

    idx_dict = dict(idx)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(idx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return idx_dict


def _build_index_country_norm_name(
    rows: List[Any],
    cache_path: Optional[str] = None,
) -> Dict[Tuple[str, str], List[str]]:
    """(country, norm_name) → [entity_id, …]"""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    records = ensure_records(rows)
    idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for rec in records:
        c = rec.country.strip()
        key = normalize_name(rec.business_name)
        if key and c:
            idx[(c, key)].append(rec.entity_id)

    idx_dict = dict(idx)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(idx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return idx_dict


def _build_index_first_token(
    rows: List[Any],
    cache_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """first_token → [entity_id, …]"""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    records = ensure_records(rows)
    idx: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        tok = first_token(rec.business_name)
        if tok:
            idx[tok].append(rec.entity_id)

    idx_dict = dict(idx)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(idx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return idx_dict


def _build_index_token_set(
    rows: List[Any],
    cache_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Each informative_token → [entity_id, …] (inverted index)."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass

    records = ensure_records(rows)
    idx: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        for tok in informative_tokens(rec.business_name):
            idx[tok].append(rec.entity_id)

    idx_dict = dict(idx)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(idx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return idx_dict


# ---------------------------------------------------------------------------
# Individual blockers (Keeping existing API signature)
# ---------------------------------------------------------------------------

def blocker_exact_norm_name(
    s1_rows: List[Any],
    s2_rows: List[Any],
    s3_rows: List[Any],
    cache_dir: Optional[str] = None,
) -> Tuple[CandidateMap, str]:
    """B1 — Exact normalized-name match across S2 and S3."""
    name = "B1_exact_norm_name"
    c2 = os.path.join(cache_dir, "idx_s2_norm_name.pkl") if cache_dir else None
    c3 = os.path.join(cache_dir, "idx_s3_norm_name.pkl") if cache_dir else None

    idx_s2 = _build_index_norm_name(s2_rows, cache_path=c2)
    idx_s3 = _build_index_norm_name(s3_rows, cache_path=c3)

    s1_records = ensure_records(s1_rows)
    candidates: CandidateMap = defaultdict(set)
    for rec in s1_records:
        key = normalize_name(rec.business_name)
        if not key:
            continue
        for cid in idx_s2.get(key, []):
            candidates[rec.entity_id].add(cid)
        for cid in idx_s3.get(key, []):
            candidates[rec.entity_id].add(cid)
    return dict(candidates), name


def blocker_country_norm_name(
    s1_rows: List[Any],
    s2_rows: List[Any],
    s3_rows: List[Any],
    cache_dir: Optional[str] = None,
) -> Tuple[CandidateMap, str]:
    """B2 — Country + exact normalized-name match."""
    name = "B2_country_norm_name"
    c2 = os.path.join(cache_dir, "idx_s2_country_norm_name.pkl") if cache_dir else None
    c3 = os.path.join(cache_dir, "idx_s3_country_norm_name.pkl") if cache_dir else None

    idx_s2 = _build_index_country_norm_name(s2_rows, cache_path=c2)
    idx_s3 = _build_index_country_norm_name(s3_rows, cache_path=c3)

    s1_records = ensure_records(s1_rows)
    candidates: CandidateMap = defaultdict(set)
    for rec in s1_records:
        c = rec.country.strip()
        key = normalize_name(rec.business_name)
        if not key or not c:
            continue
        lookup = (c, key)
        for cid in idx_s2.get(lookup, []):
            candidates[rec.entity_id].add(cid)
        for cid in idx_s3.get(lookup, []):
            candidates[rec.entity_id].add(cid)
    return dict(candidates), name


def blocker_first_token(
    s1_rows: List[Any],
    s2_rows: List[Any],
    s3_rows: List[Any],
    cache_dir: Optional[str] = None,
) -> Tuple[CandidateMap, str]:
    """B3 — First normalized-name token match."""
    name = "B3_first_token"
    c2 = os.path.join(cache_dir, "idx_s2_first_token.pkl") if cache_dir else None
    c3 = os.path.join(cache_dir, "idx_s3_first_token.pkl") if cache_dir else None

    idx_s2 = _build_index_first_token(s2_rows, cache_path=c2)
    idx_s3 = _build_index_first_token(s3_rows, cache_path=c3)

    s1_records = ensure_records(s1_rows)
    candidates: CandidateMap = defaultdict(set)
    for rec in s1_records:
        tok = first_token(rec.business_name)
        if not tok:
            continue
        for cid in idx_s2.get(tok, []):
            candidates[rec.entity_id].add(cid)
        for cid in idx_s3.get(tok, []):
            candidates[rec.entity_id].add(cid)
    return dict(candidates), name


def blocker_token_overlap(
    s1_rows: List[Any],
    s2_rows: List[Any],
    s3_rows: List[Any],
    max_candidates_per_token: int = 2000,
    cache_dir: Optional[str] = None,
) -> Tuple[CandidateMap, str]:
    """B4 — Informative-token overlap (inverted index)."""
    name = "B4_token_overlap"
    c2 = os.path.join(cache_dir, "idx_s2_token_set.pkl") if cache_dir else None
    c3 = os.path.join(cache_dir, "idx_s3_token_set.pkl") if cache_dir else None

    idx_s2 = _build_index_token_set(s2_rows, cache_path=c2)
    idx_s3 = _build_index_token_set(s3_rows, cache_path=c3)

    s1_records = ensure_records(s1_rows)
    candidates: CandidateMap = defaultdict(set)
    for rec in s1_records:
        for tok in informative_tokens(rec.business_name):
            for cid in idx_s2.get(tok, [])[:max_candidates_per_token]:
                candidates[rec.entity_id].add(cid)
            for cid in idx_s3.get(tok, [])[:max_candidates_per_token]:
                candidates[rec.entity_id].add(cid)
    return dict(candidates), name


# ---------------------------------------------------------------------------
# Union / deduplication
# ---------------------------------------------------------------------------

def union_candidates(
    blocker_outputs: List[Tuple[CandidateMap, str]],
    all_s1_ids: List[str],
) -> Tuple[CandidateMap, Dict[str, Dict[str, int]]]:
    """Union all blocker outputs and track per-blocker contribution."""
    merged: CandidateMap = {s1_id: set() for s1_id in all_s1_ids}
    contrib: Dict[str, Dict[str, int]] = {}

    for cmap, bname in blocker_outputs:
        contrib[bname] = {}
        for s1_id, cand_set in cmap.items():
            before = len(merged.get(s1_id, set()))
            merged.setdefault(s1_id, set()).update(cand_set)
            after = len(merged[s1_id])
            contrib[bname][s1_id] = after - before

    return merged, contrib


# ---------------------------------------------------------------------------
# Blocker attribution analysis
# ---------------------------------------------------------------------------

def analyze_blocker_attribution(
    blocker_outputs: List[Tuple[CandidateMap, str]],
    ground_truth: CandidateMap,
) -> Dict[str, dict]:
    """For each blocker, count how many unique TRUE pairs it recovered."""
    result: Dict[str, dict] = {}
    for cmap, bname in blocker_outputs:
        recovered = 0
        unique_recovered = 0
        for s1_id, true_set in ground_truth.items():
            cset = cmap.get(s1_id, set())
            blocker_true = true_set & cset
            recovered += len(blocker_true)

            others_union: set[str] = set()
            for other_cmap, other_bname in blocker_outputs:
                if other_bname != bname:
                    others_union.update(other_cmap.get(s1_id, set()))
            unique_recovered += len(blocker_true - others_union)

        result[bname] = {
            "recovered_true_pairs": recovered,
            "unique_true_pairs_only_this_blocker": unique_recovered,
        }
    return result


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------

def write_candidate_pairs_tsv(
    merged: CandidateMap,
    all_s1_ids: List[str],
    out_path: str,
) -> None:
    """Write candidate_pairs.tsv in the exact submission format."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = merged.get(s1_id, set())
            fh.write(f"{s1_id}\t{','.join(sorted(cands))}\n")


# ---------------------------------------------------------------------------
# Unified Execution with Stage Control
# ---------------------------------------------------------------------------

def run(
    train_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    max_token_cap: int = 2000,
    config: Optional[Config] = None,
    stages: Optional[List[str]] = None,
) -> dict:
    """
    Main candidate generation and evaluation runner.
    Supports stage toggles:
    ['BASELINE_BLOCKING', 'ADDRESS_BLOCKING', 'TFIDF_RETRIEVAL', 'MISS_ANALYSIS', 'ABLATION']
    """
    cfg = config or default_config
    if train_dir:
        cfg.train_dir = train_dir
        cfg.train_source1 = os.path.join(train_dir, "train_source1.tsv")
        cfg.train_source2 = os.path.join(train_dir, "train_source2.tsv")
        cfg.train_source3 = os.path.join(train_dir, "train_source3.tsv")
        cfg.train_ground_truth = os.path.join(train_dir, "train_ground_truth.tsv")
    if out_dir:
        cfg.output_dir = out_dir

    active_stages = set(stages) if stages else {
        "BASELINE_BLOCKING",
        "ADDRESS_BLOCKING",
        "MISS_ANALYSIS",
        "ABLATION",
        "FINAL_CANDIDATE_GENERATION"
    }

    t_start = time.perf_counter()
    print("\n" + "=" * 65)
    print("  Candidate Generator — Amazon ML Challenge 2026")
    print(f"  Dataset: {cfg.train_dir}")
    print(f"  Outputs: {cfg.output_dir}")
    print(f"  Active Stages: {', '.join(sorted(active_stages))}")
    print("=" * 65)

    # 1. Load data
    print("\n[1/5] Loading data files ...")
    t0 = time.perf_counter()
    s1_rows = load_source(cfg.train_source1)
    s2_rows = load_source(cfg.train_source2)
    s3_rows = load_source(cfg.train_source3)
    ground_truth = load_ground_truth_fast(cfg.train_ground_truth)
    all_s1_ids = [r.entity_id for r in s1_rows]

    print(f"    S1: {len(s1_rows):>10,} rows")
    print(f"    S2: {len(s2_rows):>10,} rows")
    print(f"    S3: {len(s3_rows):>10,} rows")
    print(f"    GT: {len(ground_truth):>10,} entries")
    print(f"    Data loaded in {time.perf_counter() - t0:.1f}s")

    blocker_outputs: List[Tuple[CandidateMap, str]] = []

    # 2. Baseline Blockers
    if "BASELINE_BLOCKING" in active_stages:
        print("\n[2/5] Running Baseline Blockers ...")
        for label, fn, kwargs in [
            ("B1 — Exact normalized name", blocker_exact_norm_name, {"cache_dir": cfg.cache_dir}),
            ("B2 — Country + norm name", blocker_country_norm_name, {"cache_dir": cfg.cache_dir}),
            ("B3 — First token prefix", blocker_first_token, {"cache_dir": cfg.cache_dir}),
            ("B4 — Informative token overlap", blocker_token_overlap, {"max_candidates_per_token": max_token_cap, "cache_dir": cfg.cache_dir}),
        ]:
            t0 = time.perf_counter()
            cmap, bname = fn(s1_rows, s2_rows, s3_rows, **kwargs)
            n_pairs = sum(len(v) for v in cmap.values())
            print(f"    {label}: {n_pairs:>12,} pairs ({time.perf_counter() - t0:.1f}s)")
            blocker_outputs.append((cmap, bname))

    # 3. Address Blocker
    if "ADDRESS_BLOCKING" in active_stages and blocker_country_postal_initial:
        print("\n[*] Running Address Blocker ...")
        t0 = time.perf_counter()
        cmap, bname = blocker_country_postal_initial(s1_rows, s2_rows, s3_rows)
        n_pairs = sum(len(v) for v in cmap.values())
        print(f"    {bname}: {n_pairs:>12,} pairs ({time.perf_counter() - t0:.1f}s)")
        blocker_outputs.append((cmap, bname))

    # 4. TF-IDF Retrieval
    if "TFIDF_RETRIEVAL" in active_stages and blocker_tfidf_retrieval:
        print("\n[*] Running TF-IDF Candidate Retrieval ...")
        t0 = time.perf_counter()
        tfidf_cache = os.path.join(cfg.cache_dir, "tfidf_model")
        cmap, bname = blocker_tfidf_retrieval(
            s1_rows, s2_rows, s3_rows,
            top_k=cfg.tfidf_top_k,
            cache_path=tfidf_cache
        )
        n_pairs = sum(len(v) for v in cmap.values())
        print(f"    {bname}: {n_pairs:>12,} pairs ({time.perf_counter() - t0:.1f}s)")
        blocker_outputs.append((cmap, bname))

    # Union + Deduplicate
    print("\n[3/5] Merging candidate pools ...")
    t0 = time.perf_counter()
    merged, _ = union_candidates(blocker_outputs, all_s1_ids)
    total_pairs = sum(len(v) for v in merged.values())
    print(f"    Total unique candidate pairs: {total_pairs:>12,} ({time.perf_counter() - t0:.1f}s)")

    # Evaluate
    print("\n[4/5] Evaluating recall on training ground truth ...")
    eval_result = evaluate_blocking(merged, ground_truth)
    print_report(eval_result, title="Blocking Evaluation")

    attribution = analyze_blocker_attribution(blocker_outputs, ground_truth)
    print("  Blocker True Pair Attribution:")
    for bname, stats in sorted(attribution.items(), key=lambda kv: kv[1]["recovered_true_pairs"], reverse=True):
        print(f"    {bname:<35} recovered={stats['recovered_true_pairs']:>9,} unique_only_this={stats['unique_true_pairs_only_this_blocker']:>7,}")

    # Miss Analysis
    if "MISS_ANALYSIS" in active_stages and analyze_misses:
        print("\n[*] Running Miss Analysis ...")
        s1_dict = {r.entity_id: r for r in s1_rows}
        target_dict = {r.entity_id: r for r in s2_rows}
        for r in s3_rows:
            target_dict[r.entity_id] = r
        miss_rep = analyze_misses(merged, ground_truth, s1_dict, target_dict)
        save_miss_analysis(miss_rep, os.path.join(cfg.output_dir, "miss_analysis.json"))

    # Ablation
    if "ABLATION" in active_stages and run_ablation_study:
        print("\n[*] Running Ablation Study ...")
        ablation_rep = run_ablation_study(
            blocker_outputs, ground_truth, all_s1_ids,
            out_path=os.path.join(cfg.output_dir, "ablation_study.json")
        )

    # 5. Write outputs
    if "FINAL_CANDIDATE_GENERATION" in active_stages or not stages:
        print("\n[5/5] Writing output files ...")
        cand_path = os.path.join(cfg.output_dir, "candidate_pairs.tsv")
        write_candidate_pairs_tsv(merged, all_s1_ids, cand_path)
        print(f"    Saved: {cand_path}")

        metrics_path = os.path.join(cfg.output_dir, "blocking_metrics.json")
        metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_root": cfg.dataset_root,
            "evaluation": eval_result.to_dict(),
            "attribution": attribution,
        }
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=4)
        print(f"    Saved: {metrics_path}")

    total_time = time.perf_counter() - t_start
    print(f"\nCompleted in {total_time:.1f}s")
    print("=" * 65 + "\n")
    return {"eval": eval_result, "attribution": attribution}


def _parse_args():
    p = argparse.ArgumentParser(description="Candidate generator with stage control.")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-token-cap", type=int, default=2000)
    p.add_argument("--stages", nargs="*", default=None, help="Specific stages to run")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        max_token_cap=args.max_token_cap,
        stages=args.stages,
    )
