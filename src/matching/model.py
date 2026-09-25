"""
ML Matching Model for Amazon ML Challenge 2026 - Business Entity Resolution.

Component: Lakshya (ML Matching Model)
Model: LightGBM Binary Classifier for P(candidate S2/S3 record matches S1 record).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


# ==============================================================================
# FEATURE CONTRACT
# ==============================================================================
# This contract defines the feature schema expected by the matching model.
# The feature set remains fully configurable so Smriti's normalization/similarity
# module can map seamlessly into this interface.
# ==============================================================================

FEATURE_COLUMNS_CONTRACT: Dict[str, Dict[str, str]] = {
    "NAME_FEATURES": {
        "name_exact": "Exact string equality between normalized names (0.0 or 1.0)",
        "name_fuzzy_similarity": "Fuzzy string similarity score in [0.0, 1.0]",
        "name_token_similarity": "Token set / token sort similarity score in [0.0, 1.0]",
        "name_jaccard": "Token-level Jaccard similarity score in [0.0, 1.0]",
        "name_core_similarity": "Similarity on core business name after legal suffix removal",
        "name_length_difference": "Normalized difference in name string lengths",
    },
    "ADDRESS_FEATURES": {
        "address_exact": "Exact string equality between normalized addresses (0.0 or 1.0)",
        "address_fuzzy_similarity": "Fuzzy string similarity score in [0.0, 1.0]",
        "address_token_similarity": "Token sort / set similarity score in [0.0, 1.0]",
        "address_jaccard": "Token-level Jaccard similarity score in [0.0, 1.0]",
        "postal_match": "Postal / ZIP code match indicator (0.0 or 1.0)",
        "house_number_match": "Street number / building number match indicator (0.0 or 1.0)",
        "address_missing": "Flag indicating if address is missing in either record (0.0 or 1.0)",
    },
    "ENTITY_FEATURES": {
        "country_match": "Country code exact match indicator (0.0 or 1.0)",
        "source_indicator": "Source dataset indicator (e.g. 0 for S2, 1 for S3, or categorical code)",
    },
}

DEFAULT_FEATURE_COLUMNS: List[str] = [
    # Name features
    "name_exact",
    "name_fuzzy_similarity",
    "name_token_similarity",
    "name_jaccard",
    "name_core_similarity",
    "name_length_difference",
    # Address features
    "address_exact",
    "address_fuzzy_similarity",
    "address_token_similarity",
    "address_jaccard",
    "postal_match",
    "house_number_match",
    "address_missing",
    # Entity features
    "country_match",
    "source_indicator",
]


class EntityMatcher(BaseEstimator, ClassifierMixin):
    """
    LightGBM-based binary classification model for business entity matching.
    
    Predicts the posterior probability:
        P(candidate S2/S3 record matches S1 record | engineered features)
        
    Attributes:
        params (dict): LightGBM hyperparameters.
        feature_names (list of str): List of feature columns used for training and inference.
        model (lgb.LGBMClassifier): Underlying LightGBM classifier instance.
        is_fitted (bool): Whether the model has been fitted.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 300,
        "importance_type": "gain",
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the EntityMatcher.
        
        Args:
            params: Optional dictionary of LightGBM hyperparameter overrides.
            feature_names: Optional explicit list of feature names to use.
                           If None, will be inferred from the training DataFrame.
            random_state: Random seed for reproducibility.
        """
        self.params = dict(self.DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)
        self.params["random_state"] = random_state

        self.feature_names = list(feature_names) if feature_names is not None else None
        self.random_state = random_state
        self.model: Optional[lgb.LGBMClassifier] = None
        self.is_fitted = False

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit_mode: bool = False,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Validate and format input features according to the feature contract.
        
        Args:
            X: Input feature matrix (DataFrame or numpy array).
            fit_mode: If True, allow setting self.feature_names from DataFrame columns.
            
        Returns:
            Prepared DataFrame or ndarray.
        """
        if isinstance(X, pd.DataFrame):
            if fit_mode and self.feature_names is None:
                self.feature_names = list(X.columns)
            elif self.feature_names is not None:
                missing_cols = [c for c in self.feature_names if c not in X.columns]
                if missing_cols:
                    raise ValueError(
                        f"Input DataFrame is missing required feature columns: {missing_cols}"
                    )
                X = X[self.feature_names]
        return X

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]] = None,
        categorical_feature: Union[str, List[str]] = "auto",
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> "EntityMatcher":
        """
        Train the LightGBM matching classifier.
        
        Args:
            X: Training features (DataFrame or numpy array of engineered pair features).
            y: Binary target labels (1 for true match, 0 for negative candidate).
            eval_set: Optional validation set(s) [(X_val, y_val)] for monitoring / early stopping.
            categorical_feature: Categorical feature column names or indices ('auto' by default).
            sample_weight: Optional array of sample weights.
            early_stopping_rounds: Optional early stopping rounds.
            callbacks: Optional list of LightGBM callbacks.
            **kwargs: Additional keyword arguments passed to LGBMClassifier.fit.
            
        Returns:
            Self (fitted instance).
        """
        X_clean = self._prepare_features(X, fit_mode=True)
        y_clean = np.asarray(y, dtype=np.int32)

        prepared_eval_set = None
        if eval_set is not None:
            prepared_eval_set = []
            for ev_x, ev_y in eval_set:
                ev_x_clean = self._prepare_features(ev_x, fit_mode=False)
                ev_y_clean = np.asarray(ev_y, dtype=np.int32)
                prepared_eval_set.append((ev_x_clean, ev_y_clean))

        self.model = lgb.LGBMClassifier(**self.params)

        fit_kwargs: Dict[str, Any] = dict(kwargs)
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks
        elif early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]

        self.model.fit(
            X_clean,
            y_clean,
            eval_set=prepared_eval_set,
            categorical_feature=categorical_feature,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

        self.is_fitted = True
        logger.info(
            "EntityMatcher fitted successfully with %d features and %d trees.",
            len(self.feature_names) if self.feature_names else (X_clean.shape[1] if hasattr(X_clean, "shape") else -1),
            self.model.n_estimators if self.model else -1,
        )
        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict match probabilities for candidate pairs.
        
        Returns 1D array of P(match = 1).
        
        Args:
            X: Feature matrix of candidate pairs.
            batch_size: Optional batch size for memory-efficient inference on large datasets.
            
        Returns:
            1D numpy array of float match probabilities in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("EntityMatcher is not fitted yet. Call fit() before predict_proba().")

        X_clean = self._prepare_features(X, fit_mode=False)
        n_samples = len(X_clean) if hasattr(X_clean, "__len__") else X_clean.shape[0]

        if batch_size is not None and batch_size > 0 and n_samples > batch_size:
            probas = np.zeros(n_samples, dtype=np.float32)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                if isinstance(X_clean, pd.DataFrame):
                    batch_X = X_clean.iloc[start_idx:end_idx]
                else:
                    batch_X = X_clean[start_idx:end_idx]
                
                batch_proba = self.model.predict_proba(batch_X)[:, 1]
                probas[start_idx:end_idx] = batch_proba.astype(np.float32)
            return probas
        else:
            probas_2d = self.model.predict_proba(X_clean)
            return probas_2d[:, 1].astype(np.float32)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict binary match labels (0 or 1) using a fixed probability threshold.
        
        Args:
            X: Feature matrix of candidate pairs.
            threshold: Probability threshold for positive classification (default: 0.5).
            batch_size: Optional batch size for memory-efficient prediction.
            
        Returns:
            1D numpy array of binary integers (0 or 1).
        """
        probas = self.predict_proba(X, batch_size=batch_size)
        return (probas >= threshold).astype(np.int32)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Get feature importances sorted in descending order.
        
        Args:
            importance_type: 'gain' (default) or 'split'.
            
        Returns:
            DataFrame with columns ['feature', 'importance', 'normalized_importance'].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("EntityMatcher is not fitted yet.")

        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        features = (
            self.feature_names
            if self.feature_names is not None
            else [f"feature_{i}" for i in range(len(importances))]
        )

        total_gain = np.sum(importances)
        normalized = importances / total_gain if total_gain > 0 else np.zeros_like(importances)

        df = pd.DataFrame(
            {
                "feature": features,
                "importance": importances,
                "normalized_importance": normalized,
            }
        ).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted EntityMatcher model and metadata to disk.
        
        Args:
            filepath: Destination file path (e.g. 'models/entity_matcher.joblib').
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted EntityMatcher.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "params": self.params,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "model": self.model,
        }
        joblib.dump(payload, str(path), compress=3)
        logger.info("EntityMatcher saved successfully to %s", str(path))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EntityMatcher":
        """
        Load a saved EntityMatcher model from disk.
        
        Args:
            filepath: Path to the saved model file.
            
        Returns:
            Loaded EntityMatcher instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")

        payload = joblib.load(str(path))
        matcher = cls(
            params=payload.get("params"),
            feature_names=payload.get("feature_names"),
            random_state=payload.get("random_state", 42),
        )
        matcher.model = payload.get("model")
        matcher.is_fitted = payload.get("is_fitted", True)
        logger.info("EntityMatcher loaded successfully from %s", str(path))
        return matcher
