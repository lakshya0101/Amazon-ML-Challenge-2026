"""
Configuration module for Amazon ML Challenge 2026.
Supports local execution, Colab, and custom environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Project-wide paths and hyperparameter configuration."""

    # Base directories
    base_dir: str = field(
        default_factory=lambda: os.getenv(
            "AMAZON_ML_BASE_DIR",
            str(Path(__file__).resolve().parent.parent)
        )
    )

    dataset_root: str = field(
        default_factory=lambda: os.getenv(
            "AMAZON_ML_DATASET_DIR",
            # Check student_resource/dataset first, then dataset/
            "student_resource/dataset"
            if os.path.exists("student_resource/dataset")
            else "dataset"
        )
    )

    output_dir: str = field(
        default_factory=lambda: os.getenv("AMAZON_ML_OUTPUT_DIR", "outputs")
    )

    cache_dir: str = field(
        default_factory=lambda: os.getenv("AMAZON_ML_CACHE_DIR", "cache")
    )

    # Sub-paths derived from dataset_root
    train_dir: str = ""
    test_dir: str = ""

    # Ground truth & split files
    train_source1: str = ""
    train_source2: str = ""
    train_source3: str = ""
    train_ground_truth: str = ""

    test_source1: str = ""
    test_source2: str = ""
    test_source3: str = ""

    # Blocking hyperparameters
    max_token_cap: int = int(os.getenv("MAX_TOKEN_CAP", "2000"))
    max_candidates_per_s1: int = int(os.getenv("MAX_CANDIDATES_PER_S1", "500"))
    tfidf_top_k: int = int(os.getenv("TFIDF_TOP_K", "30"))

    # Caching flags
    use_cache: bool = True

    def __post_init__(self):
        # Resolve dataset_root
        if not os.path.isabs(self.dataset_root):
            candidate_abs = os.path.join(self.base_dir, self.dataset_root)
            if os.path.exists(candidate_abs):
                self.dataset_root = candidate_abs

        self.train_dir = os.path.join(self.dataset_root, "train")
        self.test_dir = os.path.join(self.dataset_root, "test")

        self.train_source1 = os.path.join(self.train_dir, "train_source1.tsv")
        self.train_source2 = os.path.join(self.train_dir, "train_source2.tsv")
        self.train_source3 = os.path.join(self.train_dir, "train_source3.tsv")
        self.train_ground_truth = os.path.join(self.train_dir, "train_ground_truth.tsv")

        self.test_source1 = os.path.join(self.test_dir, "test_source1.tsv")
        self.test_source2 = os.path.join(self.test_dir, "test_source2.tsv")
        self.test_source3 = os.path.join(self.test_dir, "test_source3.tsv")

        # Resolve output and cache dirs
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(self.base_dir, self.output_dir)
        if not os.path.isabs(self.cache_dir):
            self.cache_dir = os.path.join(self.base_dir, self.cache_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# Global default configuration instance
default_config = Config()
