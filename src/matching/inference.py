"""Production inference interface for entity matching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np
import pandas as pd

from src.matching.model import BaseMatcher, LightGBMMatcher
from src.matching.dataset import (
    NormalizedRecordCache,
    build_candidate_dataframe,
    generate_pair_features_matrix,
)
from src.blocking.candidate_generator import CandidatePair

logger = logging.getLogger(__name__)


class EntityMatchInference:
    """
    End-to-end inference engine executing the ML matching layer on candidate pairs.
    """

    def __init__(
        self,
        model: Union[BaseMatcher, str, Path],
        threshold: float = 0.5,
        record_cache: Optional[NormalizedRecordCache] = None,
    ):
        """
        Args:
            model: Fitted BaseMatcher instance or filepath to saved model.
            threshold: Probability decision boundary.
            record_cache: Optional shared NormalizedRecordCache.
        """
        if isinstance(model, (str, Path)):
            self.model = LightGBMMatcher.load(model)
        elif isinstance(model, BaseMatcher):
            self.model = model
        else:
            raise TypeError(f"Expected BaseMatcher or file path, got {type(model)}")

        self.threshold = float(threshold)
        self.record_cache = record_cache or NormalizedRecordCache()

    def register_sources(
        self,
        source1: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        source2: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        source3: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        s1_id_col: str = "id",
        s2_id_col: str = "id",
        s3_id_col: str = "id",
    ) -> None:
        """
        Pre-caches and normalizes all source records exactly once.
        """
        if source1 is not None:
            self.record_cache.register_dataset("S1", source1, id_col=s1_id_col)
        if source2 is not None:
            self.record_cache.register_dataset("S2", source2, id_col=s2_id_col)
        if source3 is not None:
            self.record_cache.register_dataset("S3", source3, id_col=s3_id_col)

    def match(
        self,
        candidates: Union[pd.DataFrame, List[CandidatePair], List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
        return_only_matches: bool = False,
    ) -> pd.DataFrame:
        """
        Classifies candidate pairs into match decisions.
        
        Outputs canonical DataFrame with columns:
            ['s1_id', 'match_source', 'match_id', 'probability', 'final_match']
            
        Supports:
            - 0 matches per S1 entity
            - 1 match per S1 entity
            - Multiple matches per S1 entity
            
        Does not apply any 1-to-1 constraint or Hungarian assignment.
        """
        t = self.threshold if threshold is None else float(threshold)
        cand_df = build_candidate_dataframe(candidates)

        if len(cand_df) == 0:
            cols = ["s1_id", "match_source", "match_id", "probability", "final_match"]
            return pd.DataFrame(columns=cols)

        # Generate 22 pair features
        features_df = generate_pair_features_matrix(
            pairs_df=cand_df,
            record_cache=self.record_cache,
            feature_columns=self.model.feature_names,
            batch_size=batch_size,
        )

        # Predict probabilities
        probabilities = self.model.predict_proba(features_df, batch_size=batch_size)

        # Format output
        output_df = cand_df.copy()
        output_df["probability"] = probabilities.astype(np.float32)
        output_df["final_match"] = (probabilities >= t).astype(np.int32)

        if return_only_matches:
            output_df = output_df[output_df["final_match"] == 1].reset_index(drop=True)

        return output_df


def match_candidates(
    candidates: Union[pd.DataFrame, List[CandidatePair], List[Dict[str, Any]], Dict[str, Any]],
    source1: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
    source2: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
    source3: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
    model: Optional[Union[BaseMatcher, str, Path]] = None,
    threshold: float = 0.5,
    record_cache: Optional[NormalizedRecordCache] = None,
    return_only_matches: bool = False,
) -> pd.DataFrame:
    """
    Convenience functional interface for inference.
    """
    if model is None:
        raise ValueError("A fitted BaseMatcher or model path must be provided.")

    engine = EntityMatchInference(model=model, threshold=threshold, record_cache=record_cache)
    if source1 is not None or source2 is not None or source3 is not None:
        engine.register_sources(source1=source1, source2=source2, source3=source3)

    return engine.match(
        candidates=candidates,
        threshold=threshold,
        return_only_matches=return_only_matches,
    )
