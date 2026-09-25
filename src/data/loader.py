"""
Fast, memory-efficient data loader for Amazon ML Challenge 2026.
Uses Polars lazy scanning and Parquet caching where available.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Try importing polars
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

csv.field_size_limit(10 ** 7)


def get_parquet_cache_path(tsv_path: str, cache_dir: str) -> str:
    """Generate deterministic parquet cache file path."""
    stem = Path(tsv_path).stem
    return os.path.join(cache_dir, f"{stem}.parquet")


def convert_tsv_to_parquet(tsv_path: str, parquet_path: str) -> bool:
    """Convert a large TSV to compressed Parquet format using Polars streaming."""
    if not HAS_POLARS:
        return False
    try:
        os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
        # Scan lazily and stream out to parquet
        lazy_df = pl.scan_csv(
            tsv_path,
            separator="\t",
            has_header=True,
            quote_char=None,
            truncate_ragged_lines=True,
            infer_schema_length=10000,
        )
        lazy_df.sink_parquet(parquet_path, compression="zstd")
        return True
    except Exception as e:
        print(f"[Warning] Failed to convert {tsv_path} to parquet: {e}")
        return False


def load_source_df(
    tsv_path: str,
    columns: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load a source TSV into a Polars DataFrame (or fallback).
    Uses lazy scanning and column projection to avoid loading unused columns into RAM.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"File not found: {tsv_path}")

    if HAS_POLARS:
        if cache_dir and use_cache:
            p_path = get_parquet_cache_path(tsv_path, cache_dir)
            # Check if cache is valid and fresh
            if os.path.exists(p_path) and os.path.getmtime(p_path) >= os.path.getmtime(tsv_path):
                # Lazy scan parquet and select only requested columns
                scan = pl.scan_parquet(p_path)
                if columns:
                    avail = scan.columns
                    cols = [c for c in columns if c in avail]
                    scan = scan.select(cols)
                return scan.collect()
            else:
                # TSV scan
                scan = pl.scan_csv(
                    tsv_path,
                    separator="\t",
                    has_header=True,
                    quote_char=None,
                    truncate_ragged_lines=True,
                )
                if columns:
                    avail = scan.columns
                    cols = [c for c in columns if c in avail]
                    scan = scan.select(cols)
                df = scan.collect()
                # Optionally cache to parquet in background
                if cache_dir:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        df.write_parquet(p_path, compression="zstd")
                    except Exception:
                        pass
                return df
        else:
            scan = pl.scan_csv(
                tsv_path,
                separator="\t",
                has_header=True,
                quote_char=None,
                truncate_ragged_lines=True,
            )
            if columns:
                avail = scan.columns
                cols = [c for c in columns if c in avail]
                scan = scan.select(cols)
            return scan.collect()
    else:
        # Fallback when Polars is not available: return list of tuples
        return load_source_tuples(tsv_path, columns=columns)


def load_source_tuples(
    tsv_path: str,
    columns: Optional[List[str]] = None,
) -> List[Tuple[Any, ...]]:
    """
    Memory-efficient stream loading returning compact tuples instead of dictionaries.
    Avoids dict overhead for millions of records.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"File not found: {tsv_path}")

    results = []
    with open(tsv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            headers = next(reader)
        except StopIteration:
            return []

        col_indices = []
        if columns:
            for c in columns:
                if c in headers:
                    col_indices.append(headers.index(c))
                else:
                    col_indices.append(-1)
        else:
            col_indices = list(range(len(headers)))

        for row in reader:
            item = tuple(
                (row[idx] if 0 <= idx < len(row) else "")
                for idx in col_indices
            )
            results.append(item)
    return results


def load_ground_truth_fast(
    gt_path: str,
) -> Dict[str, Set[str]]:
    """
    Fast loader for train_ground_truth.tsv.
    If Polars is available, parses columns in Rust and splits matches.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"File not found: {gt_path}")

    gt: Dict[str, Set[str]] = {}

    if HAS_POLARS:
        df = pl.read_csv(
            gt_path,
            separator="\t",
            has_header=True,
            columns=["source1_entity_id", "matched_entity_ids"],
            quote_char=None,
            truncate_ragged_lines=True,
        )
        s1_ids = df["source1_entity_id"].to_list()
        m_ids = df["matched_entity_ids"].to_list()

        for s1, m in zip(s1_ids, m_ids):
            if m and str(m).strip() and str(m).lower() != "null":
                gt[s1] = set(m.split(","))
            else:
                gt[s1] = set()
    else:
        with open(gt_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                s1_id = row["source1_entity_id"].strip()
                matches = row["matched_entity_ids"].strip()
                if matches and matches.lower() != "null":
                    gt[s1_id] = set(matches.split(","))
                else:
                    gt[s1_id] = set()

    return gt
