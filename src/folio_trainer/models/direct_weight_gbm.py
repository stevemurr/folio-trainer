"""GBM-based direct weight prediction model (spec 2.3)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from folio_trainer.config.schema import DirectWeightModelConfig
from folio_trainer.labels.distill_labels import masked_softmax

logger = logging.getLogger(__name__)


class DirectWeightGBM:
    """Cross-sectional scoring model using LightGBM.

    Trains one GBM that outputs a raw score per asset row.
    At inference, scores are collected per date and converted
    to weights via masked softmax.
    """

    def __init__(self, config: DirectWeightModelConfig):
        self.config = config
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        w_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Train the GBM model.

        Parameters
        ----------
        X_train, y_train, w_train
            Training features, targets (per-asset soft target weight), sample weights.
        X_val, y_val, w_val
            Validation data.
        feature_names
            Feature column names.

        Returns
        -------
        dict with training metadata (best_iteration, etc.)
        """
        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        train_data = lgb.Dataset(
            X_train, label=y_train, weight=w_train,
            feature_name=self.feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, weight=w_val,
            feature_name=self.feature_names,
            reference=train_data,
        )

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "num_leaves": self.config.num_leaves,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )

        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
        }

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict raw scores for each asset row.

        Returns
        -------
        (n_rows,) raw scores.
        """
        if self.model is None:
            msg = "Model not trained. Call train() first."
            raise RuntimeError(msg)
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict_weights(
        self,
        X: np.ndarray,
        date_indices: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict portfolio weights by date.

        Groups rows by date, applies masked softmax to convert
        per-asset scores to weights.

        Parameters
        ----------
        X
            (n_rows, n_features) feature matrix (rows = assets across dates).
        date_indices
            (n_rows,) index mapping each row to a date group.
        mask
            (n_rows,) binary mask (1 = available asset). If None, all available.

        Returns
        -------
        (n_rows,) predicted weights (sum to 1 within each date group).
        """
        scores = self.predict_scores(X)

        if mask is None:
            mask = np.ones(len(scores))

        weights = np.zeros(len(scores))
        unique_dates = np.unique(date_indices)

        for d in unique_dates:
            idx = date_indices == d
            date_scores = scores[idx]
            date_mask = mask[idx]
            date_weights = masked_softmax(date_scores, date_mask, self.config.temperature)
            weights[idx] = date_weights

        return weights

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance.tolist()))

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "config": self.config,
                "feature_names": self.feature_names,
            }, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> DirectWeightGBM:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(data["config"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance
