"""Dataset construction, caching, feature generation, and entity-level splitting for ML matching."""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple, Any, Union, Optional, Iterable, Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.normalization import normalize_record
from src.matching.features import FEATURE_COLUMNS, generate_pair_features
from src.blocking.candidate_generator import CandidatePair

logger = logging.getLogger(__name__)


class NormalizedRecordCache:
    """
    In-memory cache for normalized entity records.
    
    Guarantees every raw record is normalized exactly once using Smriti's normalize_record().
    Supports multi-source registration (S1, S2, S3) and key-based lookup.
    """

    def __init__(self):
        # Maps (source, record_id) -> normalized_record_dict
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(
        self,
        source: str,
        record_id: str,
        default: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a normalized record from the cache."""
        return self._cache.get((str(source).upper(), str(record_id)), default)

    def put(
        self,
        source: str,
        record_id: str,
        normalized_record: Dict[str, Any],
    ) -> None:
        """Store a normalized record in the cache."""
        self._cache[(str(source).upper(), str(record_id))] = normalized_record

    def contains(self, source: str, record_id: str) -> bool:
        """Check if record is in the cache."""
        return (str(source).upper(), str(record_id)) in self._cache

    def register_record(
        self,
        source: str,
        record_id: str,
        business_name: Any,
        business_address: Any,
        country: Any,
    ) -> Dict[str, Any]:
        """
        Normalize a record if not already cached and store it.
        """
        key = (str(source).upper(), str(record_id))
        if key in self._cache:
            return self._cache[key]

        norm = normalize_record(
            business_name=business_name if business_name is not None and not pd.isna(business_name) else "",
            business_address=business_address if business_address is not None and not pd.isna(business_address) else "",
            country=country if country is not None and not pd.isna(country) else "",
        )
        self._cache[key] = norm
        return norm

    def register_dataset(
        self,
        source: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        id_col: str = "id",
        name_col: str = "business_name",
        address_col: str = "business_address",
        country_col: str = "country",
    ) -> int:
        """
        Bulk normalize and cache an entire dataset for a given source (e.g. S1, S2, S3).
        
        Returns the number of records registered.
        """
        source_tag = str(source).upper()
        count = 0

        if isinstance(data, pd.DataFrame):
            # Fast iteration
            for row in data.itertuples(index=False):
                row_dict = row._asdict()
                rec_id = str(row_dict.get(id_col, count))
                key = (source_tag, rec_id)
                if key not in self._cache:
                    name_val = row_dict.get(name_col, "")
                    addr_val = row_dict.get(address_col, "")
                    country_val = row_dict.get(country_col, "")
                    self._cache[key] = normalize_record(
                        business_name="" if name_val is None or pd.isna(name_val) else str(name_val),
                        business_address="" if addr_val is None or pd.isna(addr_val) else str(addr_val),
                        country="" if country_val is None or pd.isna(country_val) else str(country_val),
                    )
                count += 1

        elif isinstance(data, dict):
            for rec_id, row in data.items():
                key = (source_tag, str(rec_id))
                if key not in self._cache:
                    if isinstance(row, dict):
                        name_val = row.get(name_col, "")
                        addr_val = row.get(address_col, "")
                        country_val = row.get(country_col, "")
                    else:
                        name_val, addr_val, country_val = "", "", ""
                    self._cache[key] = normalize_record(
                        business_name="" if name_val is None or pd.isna(name_val) else str(name_val),
                        business_address="" if addr_val is None or pd.isna(addr_val) else str(addr_val),
                        country="" if country_val is None or pd.isna(country_val) else str(country_val),
                    )
                count += 1

        elif isinstance(data, (list, tuple)):
            for idx, row in enumerate(data):
                if isinstance(row, dict):
                    rec_id = str(row.get(id_col, idx))
                    key = (source_tag, rec_id)
                    if key not in self._cache:
                        name_val = row.get(name_col, "")
                        addr_val = row.get(address_col, "")
                        country_val = row.get(country_col, "")
                        self._cache[key] = normalize_record(
                            business_name="" if name_val is None or pd.isna(name_val) else str(name_val),
                            business_address="" if addr_val is None or pd.isna(addr_val) else str(addr_val),
                            country="" if country_val is None or pd.isna(country_val) else str(country_val),
                        )
                count += 1

        logger.info("Registered %d records for source %s into NormalizedRecordCache.", count, source_tag)
        return count

    def __len__(self) -> int:
        return len(self._cache)


def normalize_ground_truth_set(
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], List[Dict[str, Any]], Set[Tuple[str, str, str]]]
) -> Set[Tuple[str, str, str]]:
    """
    Standardizes ground truth pairs into a set of (s1_id, match_source, match_id) tuples.
    """
    gt_set: Set[Tuple[str, str, str]] = set()

    if isinstance(ground_truth, pd.DataFrame):
        for row in ground_truth.itertuples(index=False):
            row_dict = row._asdict()
            s1_id = str(row_dict.get("s1_id", row_dict.get("source1_id", "")))
            src = str(row_dict.get("match_source", row_dict.get("source", ""))).upper()
            match_id = str(row_dict.get("match_id", row_dict.get("target_id", "")))
            if s1_id and match_id:
                gt_set.add((s1_id, src, match_id))
    elif isinstance(ground_truth, (list, set, tuple)):
        for item in ground_truth:
            if isinstance(item, (tuple, list)):
                if len(item) == 3:
                    gt_set.add((str(item[0]), str(item[1]).upper(), str(item[2])))
                elif len(item) == 2:
                    gt_set.add((str(item[0]), "", str(item[1])))
            elif isinstance(item, dict):
                s1_id = str(item.get("s1_id", item.get("source1_id", "")))
                src = str(item.get("match_source", item.get("source", ""))).upper()
                match_id = str(item.get("match_id", item.get("target_id", "")))
                if s1_id and match_id:
                    gt_set.add((s1_id, src, match_id))

    return gt_set


def build_candidate_dataframe(
    candidates: Union[pd.DataFrame, List[CandidatePair], List[Dict[str, Any]], Dict[str, Any]],
) -> pd.DataFrame:
    """
    Converts candidate generator output to a canonical DataFrame with columns [s1_id, match_source, match_id].
    """
    if isinstance(candidates, pd.DataFrame):
        df = candidates.copy()
        # Rename if necessary
        rename_map = {}
        if "source1_id" in df.columns:
            rename_map["source1_id"] = "s1_id"
        if "source" in df.columns and "match_source" not in df.columns:
            rename_map["source"] = "match_source"
        if "target_id" in df.columns and "match_id" not in df.columns:
            rename_map["target_id"] = "match_id"
        if rename_map:
            df = df.rename(columns=rename_map)

        df["s1_id"] = df["s1_id"].astype(str)
        df["match_source"] = df["match_source"].astype(str).str.upper()
        df["match_id"] = df["match_id"].astype(str)
        return df[["s1_id", "match_source", "match_id"]].drop_duplicates().reset_index(drop=True)

    if isinstance(candidates, dict) and "all_candidates" in candidates:
        candidates = candidates["all_candidates"]

    rows = []
    if isinstance(candidates, list):
        for c in candidates:
            if isinstance(c, CandidatePair):
                rows.append((c.s1_id, c.match_source.upper(), c.match_id))
            elif isinstance(c, dict):
                s1 = str(c.get("s1_id", c.get("source1_id", "")))
                src = str(c.get("match_source", c.get("source", ""))).upper()
                m_id = str(c.get("match_id", c.get("target_id", "")))
                if s1 and m_id:
                    rows.append((s1, src, m_id))
            elif isinstance(c, (tuple, list)) and len(c) >= 3:
                rows.append((str(c[0]), str(c[1]).upper(), str(c[2])))

    df = pd.DataFrame(rows, columns=["s1_id", "match_source", "match_id"])
    return df.drop_duplicates().reset_index(drop=True)


def label_candidate_pairs(
    candidates: Union[pd.DataFrame, List[CandidatePair], List[Dict[str, Any]]],
    ground_truth: Union[pd.DataFrame, List[Tuple[str, str, str]], Set[Tuple[str, str, str]]],
) -> pd.DataFrame:
    """
    Constructs labelled ML training pairs from candidate pairs.
    
    label = 1 if candidate exists in ground truth, else 0.
    Preserves S2/S3 match_source identity.
    """
    cand_df = build_candidate_dataframe(candidates)
    gt_set = normalize_ground_truth_set(ground_truth)

    # Fast set membership lookup
    labels = np.zeros(len(cand_df), dtype=np.int32)
    for idx, row in enumerate(cand_df.itertuples(index=False)):
        pair_key = (row.s1_id, row.match_source, row.match_id)
        if pair_key in gt_set:
            labels[idx] = 1
        elif (row.s1_id, "", row.match_id) in gt_set:
            # Fallback if source was omitted in ground truth
            labels[idx] = 1

    cand_df["label"] = labels
    return cand_df


def create_entity_split(
    pairs_or_s1_ids: Union[pd.DataFrame, pd.Series, Iterable[Any]],
    test_size: float = 0.2,
    random_state: int = 42,
    s1_col: str = "s1_id",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits strictly at the S1 entity level.
    
    Guarantees that no S1 entity appears in both train and validation sets (0 overlap).
    """
    if isinstance(pairs_or_s1_ids, pd.DataFrame):
        if s1_col not in pairs_or_s1_ids.columns:
            raise ValueError(f"DataFrame missing S1 entity column '{s1_col}'.")
        unique_s1 = pairs_or_s1_ids[s1_col].astype(str).unique()
    elif isinstance(pairs_or_s1_ids, pd.Series):
        unique_s1 = pairs_or_s1_ids.astype(str).unique()
    else:
        unique_s1 = np.unique([str(x) for x in pairs_or_s1_ids])

    if len(unique_s1) == 0:
        return np.array([], dtype=object), np.array([], dtype=object)

    if len(unique_s1) == 1:
        return unique_s1, np.array([], dtype=object)

    train_s1, val_s1 = train_test_split(
        unique_s1,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    overlap = set(train_s1).intersection(set(val_s1))
    assert len(overlap) == 0, f"Entity leakage detected: {len(overlap)} overlapping S1 IDs."

    return train_s1, val_s1


def split_pairs_by_entity(
    pairs_df: pd.DataFrame,
    train_s1_ids: Union[Iterable[Any], Set[Any]],
    val_s1_ids: Union[Iterable[Any], Set[Any]],
    s1_col: str = "s1_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions pairs DataFrame into train and validation sets based on S1 entity sets.
    """
    train_set = {str(x) for x in train_s1_ids}
    val_set = {str(x) for x in val_s1_ids}

    s1_series = pairs_df[s1_col].astype(str)
    train_mask = s1_series.isin(train_set)
    val_mask = s1_series.isin(val_set)

    return pairs_df[train_mask].reset_index(drop=True), pairs_df[val_mask].reset_index(drop=True)


def generate_pair_features_matrix(
    pairs_df: pd.DataFrame,
    record_cache: NormalizedRecordCache,
    feature_columns: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes Smriti's 22 features for each pair in pairs_df using cached normalized records.
    
    Args:
        pairs_df: DataFrame with columns [s1_id, match_source, match_id].
        record_cache: Pre-populated NormalizedRecordCache.
        feature_columns: Expected feature column order (defaults to FEATURE_COLUMNS).
        batch_size: Optional batch size for processing.
        
    Returns:
        DataFrame with 22 feature columns of float32.
    """
    expected_cols = feature_columns or FEATURE_COLUMNS
    n_pairs = len(pairs_df)

    if n_pairs == 0:
        return pd.DataFrame(columns=expected_cols, dtype=np.float32)

    # Empty dummy record fallback
    dummy_record = normalize_record("", "", "")

    feature_rows: List[List[float]] = []

    for row in pairs_df.itertuples(index=False):
        s1_id = str(getattr(row, "s1_id"))
        match_src = str(getattr(row, "match_source")).upper()
        match_id = str(getattr(row, "match_id"))

        rec_a = record_cache.get("S1", s1_id)
        if rec_a is None:
            rec_a = dummy_record

        rec_b = record_cache.get(match_src, match_id)
        if rec_b is None:
            rec_b = dummy_record

        feat_dict = generate_pair_features(rec_a, rec_b)
        # Fast ordered list extraction
        row_feats = [float(feat_dict[col]) for col in expected_cols]
        feature_rows.append(row_feats)

    features_df = pd.DataFrame(feature_rows, columns=expected_cols, dtype=np.float32)
    return features_df


def sample_negatives(
    labeled_df: pd.DataFrame,
    negative_to_positive_ratio: Optional[float] = None,
    max_negatives_per_s1: Optional[int] = None,
    random_state: int = 42,
    label_col: str = "label",
    s1_col: str = "s1_id",
) -> pd.DataFrame:
    """
    Optionally downsamples negative candidate pairs for balanced training.
    
    All positives (label=1) are always preserved.
    """
    positives = labeled_df[labeled_df[label_col] == 1]
    negatives = labeled_df[labeled_df[label_col] == 0]

    if negative_to_positive_ratio is None or len(positives) == 0 or len(negatives) == 0:
        return labeled_df.copy()

    target_n_neg = int(round(len(positives) * negative_to_positive_ratio))
    if len(negatives) <= target_n_neg:
        return labeled_df.copy()

    rng = np.random.RandomState(random_state)

    if max_negatives_per_s1 is not None:
        # Grouped negative sampling per S1
        sampled_neg_list = []
        for s1_id, grp in negatives.groupby(s1_col):
            if len(grp) <= max_negatives_per_s1:
                sampled_neg_list.append(grp)
            else:
                sampled_idx = rng.choice(grp.index, size=max_negatives_per_s1, replace=False)
                sampled_neg_list.append(grp.loc[sampled_idx])
        sampled_neg = pd.concat(sampled_neg_list, ignore_index=True)
        if len(sampled_neg) > target_n_neg:
            sampled_idx = rng.choice(sampled_neg.index, size=target_n_neg, replace=False)
            sampled_neg = sampled_neg.loc[sampled_idx]
    else:
        sampled_idx = rng.choice(negatives.index, size=target_n_neg, replace=False)
        sampled_neg = negatives.loc[sampled_idx]

    sampled_df = pd.concat([positives, sampled_neg], ignore_index=True)
    return sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
