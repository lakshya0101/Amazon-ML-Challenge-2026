"""ML Entity Matching module."""

from src.matching.features import FEATURE_COLUMNS, generate_pair_features
from src.matching.dataset import (
    NormalizedRecordCache,
    build_candidate_dataframe,
    label_candidate_pairs,
    create_entity_split,
    split_pairs_by_entity,
    generate_pair_features_matrix,
    sample_negatives,
    normalize_ground_truth_set,
)
from src.matching.model import (
    BaseMatcher,
    LightGBMMatcher,
    LogisticRegressionMatcher,
    HistGradientBoostingMatcher,
)
from src.matching.decision import (
    compute_f_beta,
    evaluate_predictions,
    ThresholdSweeper,
)
from src.matching.inference import (
    EntityMatchInference,
    match_candidates,
)
from src.matching.training import (
    train_matcher_pipeline,
)
from src.matching.experiments import (
    ExperimentTracker,
    run_model_comparison,
)

__all__ = [
    # Feature generation (Smriti)
    "FEATURE_COLUMNS",
    "generate_pair_features",
    # Dataset & Cache
    "NormalizedRecordCache",
    "build_candidate_dataframe",
    "label_candidate_pairs",
    "create_entity_split",
    "split_pairs_by_entity",
    "generate_pair_features_matrix",
    "sample_negatives",
    "normalize_ground_truth_set",
    # Models
    "BaseMatcher",
    "LightGBMMatcher",
    "LogisticRegressionMatcher",
    "HistGradientBoostingMatcher",
    # Evaluation & Decisions
    "compute_f_beta",
    "evaluate_predictions",
    "ThresholdSweeper",
    # Inference & Training
    "EntityMatchInference",
    "match_candidates",
    "train_matcher_pipeline",
    # Experiment Tracking
    "ExperimentTracker",
    "run_model_comparison",
]
