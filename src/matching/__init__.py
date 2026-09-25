"""
ML Matching Model Package for Amazon ML Challenge 2026 - Business Entity Resolution.

Submodules:
- model: LightGBM EntityMatcher class and feature schema contract.
- training: Training pairs construction, hard-negative sampling, entity-level validation split.
- decision: MatchDecider, decision logic, thresholding, and macro/micro F0.5 metrics.
"""

from src.matching.model import (
    DEFAULT_FEATURE_COLUMNS,
    FEATURE_COLUMNS_CONTRACT,
    EntityMatcher,
)
from src.matching.training import (
    construct_labeled_pairs,
    create_batches,
    create_entity_split,
    optimize_memory_usage,
    sample_negatives,
    split_pairs_by_entity,
)
from src.matching.decision import (
    MatchDecider,
    apply_decision_logic,
    compute_f_beta,
    evaluate_entity_resolution,
)

__all__ = [
    # Model
    "EntityMatcher",
    "FEATURE_COLUMNS_CONTRACT",
    "DEFAULT_FEATURE_COLUMNS",
    # Training & Splits
    "construct_labeled_pairs",
    "sample_negatives",
    "create_entity_split",
    "split_pairs_by_entity",
    "optimize_memory_usage",
    "create_batches",
    # Decision & Metrics
    "MatchDecider",
    "apply_decision_logic",
    "compute_f_beta",
    "evaluate_entity_resolution",
]
