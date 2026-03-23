"""Tests for model and execution simplex constraints."""

from __future__ import annotations

import numpy as np
import pytest

from folio_trainer.backtest.execution_policy import apply_execution_policy
from folio_trainer.config.schema import DirectWeightModelConfig, ExecutionConfig
from folio_trainer.models.direct_weight_linear import DirectWeightLinear


def test_linear_model_predict_weights_stay_on_simplex():
    model = DirectWeightLinear(DirectWeightModelConfig(kind="linear", temperature=0.5))

    X_train = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    y_train = np.array([1.0, 0.0, 0.8, 0.2], dtype=np.float32)
    w_train = np.ones(4, dtype=np.float32)
    date_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    target_weights = np.array([1.0, 0.0, 0.8, 0.2], dtype=np.float32)

    model.train(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_train,
        y_val=y_train,
        w_val=w_train,
        feature_names=["f0", "f1"],
        train_date_indices=date_indices,
        val_date_indices=date_indices,
        target_weights_train=target_weights,
        target_weights_val=target_weights,
    )

    pred_weights = model.predict_weights(X_train, date_indices)

    for group in np.unique(date_indices):
        group_weights = pred_weights[date_indices == group]
        assert np.all(group_weights >= 0.0)
        assert group_weights.sum() == pytest.approx(1.0)

    exec_weights, _ = apply_execution_policy(
        pred_target=pred_weights[date_indices == 0],
        prev_live_weights=np.array([0.4, 0.6], dtype=float),
        config=ExecutionConfig(rebalance_band=0.0, partial_rebalance_alpha=1.0),
    )
    assert np.all(exec_weights >= 0.0)
    assert exec_weights.sum() == pytest.approx(1.0)
