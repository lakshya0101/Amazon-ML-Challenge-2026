"""
TF-IDF Candidate Retrieval for Amazon ML Challenge 2026.
Uses character/word n-gram TF-IDF and sparse inverted index or batched matrix multiplication.
Can be trained and cached to disk for reusable execution.
Uses canonical EntityRecord representation.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

from src.data.record import EntityRecord, get_field
from src.normalization.name_normalizer import normalize_name

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import scipy.sparse as sp
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def blocker_tfidf_retrieval(
    s1_data: List[Any],
    s2_data: List[Any],
    s3_data: List[Any],
    top_k: int = 15,
    min_similarity: float = 0.35,
    ngram_range: Tuple[int, int] = (3, 3),
    max_features: int = 50000,
    cache_path: Optional[str] = None,
) -> Tuple[Dict[str, Set[str]], str]:
    """
    TF-IDF character n-gram blocker.
    Partitions by country to remain scalable without OOM.
    Caches vectorizer model if cache_path is specified.
    """
    name = "B_tfidf_retrieval"
    if not HAS_SKLEARN:
        print("[Warning] scikit-learn not available. Skipping TF-IDF blocker.")
        return {}, name

    # Group targets (S2 + S3) by country
    targets_by_country: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in s2_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        country = get_field(row, 'country', 3)
        bname = normalize_name(business_name)
        c = str(country).strip() or "UNKNOWN"
        if bname:
            targets_by_country[c].append((entity_id, bname))

    for row in s3_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        country = get_field(row, 'country', 3)
        bname = normalize_name(business_name)
        c = str(country).strip() or "UNKNOWN"
        if bname:
            targets_by_country[c].append((entity_id, bname))

    # Group queries (S1) by country
    queries_by_country: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in s1_data:
        entity_id = get_field(row, 'entity_id', 0)
        business_name = get_field(row, 'business_name', 1)
        country = get_field(row, 'country', 3)
        bname = normalize_name(business_name)
        c = str(country).strip() or "UNKNOWN"
        if bname:
            queries_by_country[c].append((entity_id, bname))

    candidates: Dict[str, Set[str]] = defaultdict(set)

    for country, q_list in queries_by_country.items():
        t_list = targets_by_country.get(country, [])
        if not t_list or not q_list:
            continue

        vec_cache = f"{cache_path}_{country}.pkl" if cache_path else None
        vectorizer = None

        if vec_cache and os.path.exists(vec_cache):
            try:
                with open(vec_cache, "rb") as fh:
                    vectorizer, t_matrix, t_ids = pickle.load(fh)
            except Exception:
                vectorizer = None

        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=ngram_range,
                max_features=max_features,
                dtype=np.float32,
            )
            t_names = [t[1] for t in t_list]
            t_ids = [t[0] for t in t_list]
            t_matrix = vectorizer.fit_transform(t_names)

            if vec_cache:
                os.makedirs(os.path.dirname(vec_cache) or ".", exist_ok=True)
                try:
                    with open(vec_cache, "wb") as fh:
                        pickle.dump((vectorizer, t_matrix, t_ids), fh, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass

        # Query in batches of 10,000 to keep memory under control
        batch_size = 10000
        for i in range(0, len(q_list), batch_size):
            chunk = q_list[i : i + batch_size]
            q_names = [q[1] for q in chunk]
            q_ids = [q[0] for q in chunk]

            q_matrix = vectorizer.transform(q_names)
            sim_matrix = q_matrix.dot(t_matrix.T)

            for row_idx, q_id in enumerate(q_ids):
                row = sim_matrix.getrow(row_idx)
                if row.nnz == 0:
                    continue
                col_indices = row.indices
                scores = row.data

                mask = scores >= min_similarity
                valid_cols = col_indices[mask]
                valid_scores = scores[mask]

                if len(valid_scores) > top_k:
                    top_indices = np.argpartition(valid_scores, -top_k)[-top_k:]
                    valid_cols = valid_cols[top_indices]

                for target_idx in valid_cols:
                    candidates[q_id].add(t_ids[target_idx])

    return dict(candidates), name
