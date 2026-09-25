"""
Fast, memory-efficient data loader for Amazon ML Challenge 2026.
Uses Polars lazy scanning and Parquet caching where available,
and yields canonical EntityRecord tuples for consistent in-memory representation.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from src.data.record import EntityRecord

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
    Load a source TSV into a Polars DataFrame (or fallback to List[EntityRecord]).
    Uses lazy scanning and column projection to avoid loading unused columns into RAM.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"File not found: {tsv_path}")

    if HAS_POLARS:
        if cache_dir and use_cache:
            p_path = get_parquet_cache_path(tsv_path, cache_dir)
            if os.path.exists(p_path) and os.path.getmtime(p_path) >= os.path.getmtime(tsv_path):
                scan = pl.scan_parquet(p_path)
                if columns:
                    avail = scan.columns
                    cols = [c for c in columns if c in avail]
                    scan = scan.select(cols)
                return scan.collect()
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
                df = scan.collect()
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
        return load_source_tuples(tsv_path, columns=columns)


def load_source_tuples(
    tsv_path: str,
    columns: Optional[List[str]] = None,
) -> List[EntityRecord]:
    """
    Memory-efficient stream loading returning canonical EntityRecord objects.
    Preserves tuple interface (isinstance(r, tuple) is True) while guaranteeing
    standard column alignment (entity_id, business_name, business_address, country)
    and attribute/.get() access with zero extra memory overhead.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"File not found: {tsv_path}")

    results: List[EntityRecord] = []
    with open(tsv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            headers = next(reader)
        except StopIteration:
            return []

        header_map = {col.strip().lower(): idx for idx, col in enumerate(headers)}
        eid_idx = header_map.get("entity_id", 0)
        name_idx = header_map.get("business_name", 1)
        addr_idx = header_map.get("business_address", 2)
        country_idx = header_map.get("country", 3)

        for row in reader:
            n = len(row)
            item = EntityRecord(
                entity_id=row[eid_idx] if 0 <= eid_idx < n else "",
                business_name=row[name_idx] if 0 <= name_idx < n else "",
                business_address=row[addr_idx] if 0 <= addr_idx < n else "",
                country=row[country_idx] if 0 <= country_idx < n else "",
            )
            results.append(item)
    return results


def load_source_records(
    tsv_path: str,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> List[EntityRecord]:
    """Alias for load_source_tuples, returning canonical EntityRecord objects."""
    return load_source_tuples(tsv_path)


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
