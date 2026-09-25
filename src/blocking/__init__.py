"""Candidate Generation and Blocking module."""

from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex
from src.blocking.candidate_generator import CandidateGenerator, CandidatePair, generate_candidates
from src.blocking.evaluation import evaluate_candidates, evaluate_strategy_ablation

__all__ = [
    "BlockingConfig",
    "BlockingIndex",
    "CandidateGenerator",
    "CandidatePair",
    "generate_candidates",
    "evaluate_candidates",
    "evaluate_strategy_ablation",
]
