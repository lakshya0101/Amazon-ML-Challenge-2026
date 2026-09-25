"""Candidate generation engine coordinating multi-source blocking and deduplication."""

from dataclasses import dataclass
from typing import Dict, List, Set, Any, Union, Optional
import pandas as pd
import time

from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex, ensure_normalized_record
from src.blocking.retrieval import retrieve_candidates_with_provenance


@dataclass(frozen=True)
class CandidatePair:
    """Represents a generated candidate pair with rule provenance."""
    s1_id: str
    match_source: str
    match_id: str
    blocking_rules: Set[str]

    def to_dict(self, include_provenance: bool = False) -> Dict[str, Any]:
        data = {
            "s1_id": self.s1_id,
            "match_source": self.match_source,
            "match_id": self.match_id,
        }
        if include_provenance:
            data["blocking_rules"] = list(self.blocking_rules)
        return data


class CandidateGenerator:
    """Coordinates indexing and candidate generation across multiple sources."""

    def __init__(self, config: BlockingConfig = None):
        self.config = config or BlockingConfig()
        self.s2_index: Optional[BlockingIndex] = None
        self.s3_index: Optional[BlockingIndex] = None

    def build_indexes(
        self,
        source2: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        source3: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        s2_id_col: str = "id",
        s3_id_col: str = "id",
    ):
        """Builds separate inverted indexes for S2 and S3."""
        self.s2_index = BlockingIndex(config=self.config)
        self.s2_index.build_from_records(source2, id_column=s2_id_col)

        self.s3_index = BlockingIndex(config=self.config)
        self.s3_index.build_from_records(source3, id_column=s3_id_col)

    def retrieve_for_record(
        self,
        s1_id: str,
        s1_record: Dict[str, Any],
        target_index: BlockingIndex,
        target_source: str,
    ) -> List[CandidatePair]:
        """Retrieves and deduplicates candidates for a single S1 record against a target index."""
        match_dict = retrieve_candidates_with_provenance(s1_record, target_index, self.config)
        candidates: List[CandidatePair] = []
        for match_id, rules in match_dict.items():
            candidates.append(
                CandidatePair(
                    s1_id=str(s1_id),
                    match_source=target_source,
                    match_id=str(match_id),
                    blocking_rules=rules,
                )
            )
        return candidates

    def generate_candidates_for_source(
        self,
        source1: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        target_index: BlockingIndex,
        target_source_name: str = "S2",
        s1_id_col: str = "id",
    ) -> List[CandidatePair]:
        """Generates candidate pairs for Source 1 against a single target index."""
        candidates: List[CandidatePair] = []

        if isinstance(source1, pd.DataFrame):
            records = source1.to_dict(orient="records")
            for idx, row in enumerate(records):
                s1_id = str(row.get(s1_id_col, row.get("id", idx)))
                norm_row = ensure_normalized_record(row)
                candidates.extend(self.retrieve_for_record(s1_id, norm_row, target_index, target_source_name))
        elif isinstance(source1, dict):
            for s1_id, row in source1.items():
                norm_row = ensure_normalized_record(row)
                candidates.extend(self.retrieve_for_record(str(s1_id), norm_row, target_index, target_source_name))
        elif isinstance(source1, (list, tuple)):
            for idx, row in enumerate(source1):
                s1_id = str(row.get(s1_id_col, row.get("id", idx)))
                norm_row = ensure_normalized_record(row)
                candidates.extend(self.retrieve_for_record(s1_id, norm_row, target_index, target_source_name))

        return candidates

    def generate(
        self,
        source1: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        source2: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        source3: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
        s1_id_col: str = "id",
        s2_id_col: str = "id",
        s3_id_col: str = "id",
    ) -> Dict[str, Any]:
        """Generates candidates across S1 -> S2 and S1 -> S3 with full separation."""
        start_time = time.time()

        if source2 is not None or self.s2_index is None:
            self.s2_index = BlockingIndex(config=self.config)
            if source2 is not None:
                self.s2_index.build_from_records(source2, id_column=s2_id_col)

        if source3 is not None or self.s3_index is None:
            self.s3_index = BlockingIndex(config=self.config)
            if source3 is not None:
                self.s3_index.build_from_records(source3, id_column=s3_id_col)

        s2_candidates = self.generate_candidates_for_source(
            source1, self.s2_index, target_source_name="S2", s1_id_col=s1_id_col
        )
        s3_candidates = self.generate_candidates_for_source(
            source1, self.s3_index, target_source_name="S3", s1_id_col=s1_id_col
        )

        all_candidates = s2_candidates + s3_candidates
        s1_count = len(source1) if hasattr(source1, "__len__") else 0

        return {
            "s2_candidates": s2_candidates,
            "s3_candidates": s3_candidates,
            "all_candidates": all_candidates,
            "s1_count": s1_count,
            "s2_count": len(self.s2_index),
            "s3_count": len(self.s3_index),
            "runtime_seconds": time.time() - start_time,
        }

    @staticmethod
    def to_dataframe(candidates: List[CandidatePair], include_provenance: bool = False) -> pd.DataFrame:
        """Converts a list of CandidatePair objects to a pandas DataFrame."""
        rows = [c.to_dict(include_provenance=include_provenance) for c in candidates]
        if not rows:
            cols = ["s1_id", "match_source", "match_id"]
            if include_provenance:
                cols.append("blocking_rules")
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)

    @staticmethod
    def export_candidate_pairs_tsv(candidates: Union[List[CandidatePair], pd.DataFrame], output_path: str):
        """Exports the final candidate pairs to TSV format with standard s1_id, match_source, match_id columns."""
        if isinstance(candidates, list):
            df = CandidateGenerator.to_dataframe(candidates, include_provenance=False)
        else:
            df = candidates[["s1_id", "match_source", "match_id"]].copy()

        df.to_csv(output_path, sep="\t", index=False)


def generate_candidates(
    source1: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    source2: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    source3: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    config: BlockingConfig = None,
    s1_id_col: str = "id",
    s2_id_col: str = "id",
    s3_id_col: str = "id",
) -> Dict[str, Any]:
    """High-level entry point for candidate generation."""
    generator = CandidateGenerator(config=config)
    return generator.generate(
        source1=source1,
        source2=source2,
        source3=source3,
        s1_id_col=s1_id_col,
        s2_id_col=s2_id_col,
        s3_id_col=s3_id_col,
    )
