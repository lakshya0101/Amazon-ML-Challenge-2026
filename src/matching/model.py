"""ML Matcher models for entity resolution (LightGBM baseline and modular classifiers)."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.matching.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class BaseMatcher(abc.ABC):
    """Abstract base class for all entity matching models."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.random_state = random_state
        self.is_fitted: bool = False

    @abc.abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]] = None,
        **kwargs: Any,
    ) -> "BaseMatcher":
        """Fit the matching model on feature matrix X and binary labels y."""
        pass

    @abc.abstractmethod
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict match probability P(match = 1) for candidate pairs.
        
        Returns 1D numpy array of floats in [0.0, 1.0].
        """
        pass

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict binary match decisions (0 or 1) by thresholding probabilities.
        """
        probas = self.predict_proba(X, batch_size=batch_size)
        return (probas >= threshold).astype(np.int32)

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        fit_mode: bool = False,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Validates feature columns and shapes."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X.columns]
                if missing:
                    raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")
                return X[self.feature_names]
        return X

    @abc.abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model to disk."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseMatcher":
        """Deserialize model from disk."""
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame if supported by the model."""
        return pd.DataFrame(columns=["feature", "importance", "normalized_importance"])


class LightGBMMatcher(BaseMatcher):
    """
    Production-grade LightGBM entity resolution classifier.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        super().__init__(feature_names=feature_names, random_state=random_state)
        self.params = dict(self.DEFAULT_PARAMS)
        self.params["random_state"] = random_state
        if params is not None:
            self.params.update(params)

        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> "LightGBMMatcher":
        """
        Train LightGBM model on candidate pair features.
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

        fit_kwargs = dict(kwargs)
        active_callbacks = list(callbacks) if callbacks is not None else []
        if early_stopping_rounds is not None and not any(isinstance(cb, lgb.callback._EarlyStoppingCallback) for cb in active_callbacks):
            active_callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        if active_callbacks:
            fit_kwargs["callbacks"] = active_callbacks

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(
                X_clean,
                y_clean,
                eval_set=prepared_eval_set,
                **fit_kwargs,
            )


        self.is_fitted = True
        logger.info("LightGBMMatcher fitted successfully.")
        return self


    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict match probabilities P(match = 1).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LightGBMMatcher is not fitted yet. Call fit() first.")

        X_clean = self._prepare_features(X, fit_mode=False)
        n_samples = len(X_clean) if hasattr(X_clean, "__len__") else X_clean.shape[0]

        if n_samples == 0:
            return np.array([], dtype=np.float32)

        if batch_size is not None and batch_size > 0 and n_samples > batch_size:
            probas = np.zeros(n_samples, dtype=np.float32)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X_clean.iloc[start_idx:end_idx] if isinstance(X_clean, pd.DataFrame) else X_clean[start_idx:end_idx]
                batch_proba = self.model.predict_proba(batch_X)[:, 1]
                probas[start_idx:end_idx] = batch_proba.astype(np.float32)
            return probas
        else:
            probas_2d = self.model.predict_proba(X_clean)
            return probas_2d[:, 1].astype(np.float32)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importances sorted descending.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LightGBMMatcher is not fitted.")

        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        features = (
            self.feature_names
            if self.feature_names is not None
            else [f"feature_{i}" for i in range(len(importances))]
        )
        total_gain = float(np.sum(importances))
        normalized = importances / total_gain if total_gain > 0 else np.zeros_like(importances, dtype=float)

        df = pd.DataFrame({
            "feature": features,
            "importance": importances,
            "normalized_importance": normalized,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted LightGBMMatcher.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "LightGBMMatcher",
            "params": self.params,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "model": self.model,
        }
        joblib.dump(payload, str(path), compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LightGBMMatcher":
        """Load model from disk."""
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
        return matcher


class LogisticRegressionMatcher(BaseMatcher):
    """
    L2-regularized Logistic Regression matcher with standard feature scaling.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        super().__init__(feature_names=feature_names, random_state=random_state)
        self.C = C
        self.class_weight = class_weight
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=1000,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionMatcher":
        X_clean = self._prepare_features(X, fit_mode=True)
        y_clean = np.asarray(y, dtype=np.int32)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionMatcher is not fitted.")
        X_clean = self._prepare_features(X, fit_mode=False)
        if len(X_clean) == 0:
            return np.array([], dtype=np.float32)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)[:, 1].astype(np.float32)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionMatcher is not fitted.")
        coefs = np.abs(self.model.coef_[0])
        total = float(np.sum(coefs))
        norm = coefs / total if total > 0 else np.zeros_like(coefs)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": coefs,
            "normalized_importance": norm,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "LogisticRegressionMatcher",
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "scaler": self.scaler,
            "model": self.model,
        }
        joblib.dump(payload, str(path), compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LogisticRegressionMatcher":
        payload = joblib.load(str(filepath))
        matcher = cls(
            feature_names=payload.get("feature_names"),
            random_state=payload.get("random_state", 42),
        )
        matcher.scaler = payload.get("scaler")
        matcher.model = payload.get("model")
        matcher.is_fitted = payload.get("is_fitted", True)
        return matcher


class HistGradientBoostingMatcher(BaseMatcher):
    """
    Scikit-learn HistGradientBoostingClassifier matcher.
    """

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        class_weight: str = "balanced",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        super().__init__(feature_names=feature_names, random_state=random_state)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]] = None,
        **kwargs: Any,
    ) -> "HistGradientBoostingMatcher":
        X_clean = self._prepare_features(X, fit_mode=True)
        y_clean = np.asarray(y, dtype=np.int32)
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HistGradientBoostingMatcher is not fitted.")
        X_clean = self._prepare_features(X, fit_mode=False)
        if len(X_clean) == 0:
            return np.array([], dtype=np.float32)
        return self.model.predict_proba(X_clean)[:, 1].astype(np.float32)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "HistGradientBoostingMatcher",
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "model": self.model,
        }
        joblib.dump(payload, str(path), compress=3)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HistGradientBoostingMatcher":
        payload = joblib.load(str(filepath))
        matcher = cls(
            feature_names=payload.get("feature_names"),
            random_state=payload.get("random_state", 42),
        )
        matcher.model = payload.get("model")
        matcher.is_fitted = payload.get("is_fitted", True)
        return matcher
