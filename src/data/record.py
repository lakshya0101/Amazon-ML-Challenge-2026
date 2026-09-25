"""
Canonical Record Representation for Amazon ML Challenge 2026.
Provides EntityRecord, a lightweight NamedTuple supporting both
tuple indexing and dictionary-style attribute/.get() access with zero memory overhead.
"""

from __future__ import annotations

from typing import Any, Iterable, List, NamedTuple, Optional


class EntityRecord(NamedTuple):
    """
    Canonical in-memory representation for an entity record.
    Memory footprint is identical to a standard Python tuple (~72 bytes).
    Supports:
        - Attributes: rec.entity_id, rec.business_name, rec.business_address, rec.country
        - Tuple indexing & unpacking: eid, name, addr, country = rec
        - Dict-style access: rec.get('business_name', '')
    """
    entity_id: str
    business_name: str
    business_address: str
    country: str

    def get(self, key: str, default: Any = "") -> Any:
        """Provide dictionary-style .get(key, default) for seamless compatibility."""
        val = getattr(self, key, None)
        return default if val is None else val

    def to_dict(self) -> dict[str, str]:
        """Convert to standard dictionary."""
        return {
            "entity_id": self.entity_id,
            "business_name": self.business_name,
            "business_address": self.business_address,
            "country": self.country,
        }

    @classmethod
    def from_any(cls, row: Any) -> "EntityRecord":
        """
        Convert any raw input (dict, tuple, list, or EntityRecord) into a canonical EntityRecord.
        If row is already an EntityRecord, returns it unchanged with zero overhead.
        """
        if isinstance(row, cls):
            return row

        if isinstance(row, dict):
            return cls(
                entity_id=str(row.get("entity_id", "") or "").strip(),
                business_name=str(row.get("business_name", "") or "").strip(),
                business_address=str(row.get("business_address", "") or "").strip(),
                country=str(row.get("country", "") or "").strip(),
            )

        if isinstance(row, (tuple, list)):
            return cls(
                entity_id=str(row[0] if len(row) > 0 and row[0] is not None else "").strip(),
                business_name=str(row[1] if len(row) > 1 and row[1] is not None else "").strip(),
                business_address=str(row[2] if len(row) > 2 and row[2] is not None else "").strip(),
                country=str(row[3] if len(row) > 3 and row[3] is not None else "").strip(),
            )

        # Generic object attribute access fallback
        return cls(
            entity_id=str(getattr(row, "entity_id", "") or "").strip(),
            business_name=str(getattr(row, "business_name", "") or "").strip(),
            business_address=str(getattr(row, "business_address", "") or "").strip(),
            country=str(getattr(row, "country", "") or "").strip(),
        )


def ensure_records(rows: Iterable[Any]) -> List[EntityRecord]:
    """
    Ensure all elements in rows are canonical EntityRecord instances.
    If rows is already a list of EntityRecord, returns rows directly without copying.
    """
    if not rows:
        return []
    if isinstance(rows, list):
        if all(isinstance(r, EntityRecord) for r in rows):
            return rows
    return [EntityRecord.from_any(r) for r in rows]

def get_field(row: Any, field: str, index: int, default: Any = "") -> Any:
    """
    Safely access a field from a row that might be a dict, a tuple/list, or an object (like EntityRecord).
    Does NOT construct any new objects, avoiding memory allocation overhead on large datasets.
    """
    if isinstance(row, dict):
        return row.get(field, default)
    if isinstance(row, (tuple, list)):
        return row[index] if len(row) > index else default
    return getattr(row, field, default)
