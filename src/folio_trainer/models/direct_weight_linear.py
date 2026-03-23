"""Simple linear direct-weight scorer for interpretable baselines."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from folio_trainer.config.schema import DirectWeightModelConfig
from folio_trainer.labels.distill_labels import masked_softmax

logger = logging.getLogger(__name__)


class DirectWeightLinear:
    """Weighted ridge regression scorer with softmax-normalized inference."""

    def __init__(self, config: DirectWeightModelConfig):
        self.config = config
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
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
        **_: dict,
    ) -> dict:
        """Fit a weighted ridge regressor."""
        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        sqrt_w = np.sqrt(np.clip(w_train, 1e-8, None))[:, None]
        X_design = np.concatenate([X_train, np.ones((len(X_train), 1), dtype=X_train.dtype)], axis=1)
        Xw = X_design * sqrt_w
        yw = y_train * sqrt_w[:, 0]

        ridge = max(float(self.config.l2_weight_change_penalty), 1e-6)
        reg = np.eye(Xw.shape[1], dtype=np.float32) * ridge
        reg[-1, -1] = 0.0

        params = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
        self.coef_ = params[:-1]
        self.intercept_ = float(params[-1])

        val_pred = self.predict_scores(X_val)
        val_loss = float(np.average((val_pred - y_val) ** 2, weights=np.clip(w_val, 1e-8, None)))
        return {"best_iteration": 1, "best_score": {"val": {"weighted_mse": val_loss}}}

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict raw per-asset scores."""
        if self.coef_ is None:
            msg = "Model not trained. Call train() first."
            raise RuntimeError(msg)
        return X @ self.coef_ + self.intercept_

    def predict_weights(
        self,
        X: np.ndarray,
        date_indices: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert scores to weights within each date group."""
        scores = self.predict_scores(X)
        if mask is None:
            mask = np.ones(len(scores))

        weights = np.zeros(len(scores), dtype=np.float32)
        for date_idx in np.unique(date_indices):
            date_mask = date_indices == date_idx
            weights[date_mask] = masked_softmax(
                scores[date_mask],
                mask[date_mask],
                self.config.temperature,
            )
        return weights

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return absolute coefficient magnitude as importance."""
        if self.coef_ is None:
            return {}
        return dict(zip(self.feature_names, np.abs(self.coef_).tolist()))

    def save(self, path: str | Path) -> None:
        """Persist model state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "coef_": self.coef_,
                    "intercept_": self.intercept_,
                    "config": self.config,
                    "feature_names": self.feature_names,
                },
                f,
            )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DirectWeightLinear":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(data["config"])
        instance.coef_ = data["coef_"]
        instance.intercept_ = data["intercept_"]
        instance.feature_names = data["feature_names"]
        return instance
