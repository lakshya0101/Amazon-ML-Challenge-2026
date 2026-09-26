"""Inverted index structures for fast candidate blocking."""

from collections import defaultdict
from typing import Dict, List, Any, Union, Set
import pandas as pd

from src.preprocessing.normalization import normalize_record
from src.blocking.config import BlockingConfig


def ensure_normalized_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures a record has the canonical normalized fields from Smriti's normalization contract."""
    if "normalized_name" in record and "name_core" in record:
        return record

    raw_name = record.get("business_name", record.get("raw_name", ""))
    raw_address = record.get("business_address", record.get("raw_address", ""))
    raw_country = record.get("country", record.get("raw_country", ""))

    return normalize_record(raw_name, raw_address, raw_country)


class BlockingIndex:
    """Inverted index supporting exact name, name core, country+token, postal code, and house+address tokens."""

    def __init__(self, config: BlockingConfig = None):
        self.config = config or BlockingConfig()

        self.exact_name_index: Dict[str, List[str]] = defaultdict(list)
        self.name_core_index: Dict[str, List[str]] = defaultdict(list)
        self.country_token_index: Dict[tuple, List[str]] = defaultdict(list)
        self.postal_index: Dict[str, List[str]] = defaultdict(list)
        self.house_address_token_index: Dict[tuple, List[str]] = defaultdict(list)
        self.country_address_token_index: Dict[tuple, List[str]] = defaultdict(list)

        self.records: Dict[str, Dict[str, Any]] = {}
        self.record_ids: List[str] = []

    def build_from_records(self, records: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame], id_column: str = "id"):
        """Populates the inverted index from records."""
        if isinstance(records, pd.DataFrame):
            record_iter = records.to_dict(orient="records")
            for row in record_iter:
                rec_id = str(row.get(id_column, row.get("id", len(self.record_ids))))
                norm_rec = ensure_normalized_record(row)
                self._add_record(rec_id, norm_rec)
        elif isinstance(records, dict):
            for rec_id, row in records.items():
                rec_id = str(rec_id)
                norm_rec = ensure_normalized_record(row)
                self._add_record(rec_id, norm_rec)
        elif isinstance(records, (list, tuple)):
            for idx, row in enumerate(records):
                rec_id = str(row.get(id_column, row.get("id", idx)))
                norm_rec = ensure_normalized_record(row)
                self._add_record(rec_id, norm_rec)
        else:
            raise ValueError(f"Unsupported record format: {type(records)}")

    def _add_record(self, record_id: str, record: Dict[str, Any]):
        """Indexes a single normalized record."""
        self.records[record_id] = record
        self.record_ids.append(record_id)

        # 1. Exact Name Index
        norm_name = record.get("normalized_name", "")
        if norm_name:
            self.exact_name_index[norm_name].append(record_id)

        # 2. Name Core Index
        name_core = record.get("name_core", "")
        if name_core:
            self.name_core_index[name_core].append(record_id)

        # 3. Country + Name Token Index
        country = record.get("normalized_country", "")
        tokens = record.get("name_tokens", [])
        if country and tokens:
            seen_tokens: Set[str] = set()
            for token in tokens:
                if (
                    len(token) >= self.config.min_name_token_length
                    and token not in self.config.name_stopwords
                    and token not in seen_tokens
                ):
                    seen_tokens.add(token)
                    self.country_token_index[(country, token)].append(record_id)

        # 4. Postal Code Index
        postal_code = record.get("postal_code", "")
        if postal_code:
            self.postal_index[postal_code].append(record_id)

        # 5. House Number + Informative Address Token Index
        house_number = record.get("house_number", "")
        norm_address = record.get("normalized_address", "")
        if house_number and norm_address:
            addr_tokens = norm_address.split()
            seen_addr_tokens: Set[str] = set()
            for token in addr_tokens:
                if (
                    len(token) >= self.config.min_address_token_length
                    and token != house_number
                    and token not in self.config.address_stopwords
                    and token not in seen_addr_tokens
                ):
                    seen_addr_tokens.add(token)
                    self.house_address_token_index[(house_number, token)].append(record_id)

        # 6. Country + Informative Address Token Index
        if country and norm_address:
            addr_tokens = norm_address.split()
            seen_country_addr_tokens: Set[str] = set()
            for token in addr_tokens:
                if (
                    len(token) >= self.config.min_address_token_length
                    and token not in self.config.address_stopwords
                    and token not in seen_country_addr_tokens
                ):
                    seen_country_addr_tokens.add(token)
                    self.country_address_token_index[(country, token)].append(record_id)

    def __len__(self) -> int:
        return len(self.records)

