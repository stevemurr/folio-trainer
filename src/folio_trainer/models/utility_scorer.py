"""Utility scorer model (spec 2.4).

Predicts teacher_objective_total from (state_features, candidate_weights).
Used to score candidate portfolios at inference time.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class UtilityScorer:
    """Scores candidate portfolios by predicted utility."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model: lgb.Booster | None = None

    def train(
        self,
        state_features: np.ndarray,
        candidate_weights: np.ndarray,
        objectives: np.ndarray,
        val_state: np.ndarray | None = None,
        val_weights: np.ndarray | None = None,
        val_objectives: np.ndarray | None = None,
    ) -> dict:
        """Train utility scorer.

        Parameters
        ----------
        state_features
            (n_samples, n_state_features) market/portfolio state features.
        candidate_weights
            (n_samples, n_assets) candidate weight vectors.
        objectives
            (n_samples,) teacher objective scores (target).

        Returns
        -------
        dict with training metadata.
        """
        if not HAS_LGB:
            raise ImportError("LightGBM required for utility scorer.")

        # Concatenate state features and candidate weights as input
        X = np.concatenate([state_features, candidate_weights], axis=1)
        y = objectives

        train_data = lgb.Dataset(X, label=y)

        valid_sets = [train_data]
        valid_names = ["train"]
        if val_state is not None and val_weights is not None and val_objectives is not None:
            X_val = np.concatenate([val_state, val_weights], axis=1)
            val_data = lgb.Dataset(X_val, label=val_objectives)
            valid_sets.append(val_data)
            valid_names.append("val")

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "verbose": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )

        return {"best_iteration": self.model.best_iteration}

    def predict(self, state_features: np.ndarray, candidate_weights: np.ndarray) -> np.ndarray:
        """Predict utility scores for candidates."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        X = np.concatenate([state_features, candidate_weights], axis=1)
        return self.model.predict(X)

    def evaluate(
        self,
        state_features: np.ndarray,
        candidate_weights: np.ndarray,
        true_objectives: np.ndarray,
        group_indices: np.ndarray | None = None,
    ) -> dict:
        """Evaluate with ranking metrics.

        Parameters
        ----------
        group_indices
            If provided, compute rank metrics within groups (per-date).

        Returns
        -------
        dict with spearman, kendall, top_k_recall.
        """
        preds = self.predict(state_features, candidate_weights)

        # Overall correlation
        spearman, _ = spearmanr(preds, true_objectives)
        kendall, _ = kendalltau(preds, true_objectives)

        # Per-group top-k recall
        top_k_recalls = []
        if group_indices is not None:
            for g in np.unique(group_indices):
                mask = group_indices == g
                g_preds = preds[mask]
                g_true = true_objectives[mask]
                k = max(1, len(g_true) // 5)
                true_top_k = set(np.argsort(-g_true)[:k])
                pred_top_k = set(np.argsort(-g_preds)[:k])
                recall = len(true_top_k & pred_top_k) / k
                top_k_recalls.append(recall)

        return {
            "spearman": float(spearman),
            "kendall": float(kendall),
            "top_k_recall": float(np.mean(top_k_recalls)) if top_k_recalls else None,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str | Path) -> UtilityScorer:
        instance = cls()
        with open(path, "rb") as f:
            instance.model = pickle.load(f)
        return instance
