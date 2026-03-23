"""Tabular neural network for direct weight prediction (spec 2.3.2).

Optional — requires torch. Install with: pip install folio-trainer[nn]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required for neural net model. Install with: pip install folio-trainer[nn]")


class AssetScorerNet(nn.Module if HAS_TORCH else object):
    """Per-asset scoring network with shared encoder + market context."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 3, dropout: float = 0.1):
        _check_torch()
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns (batch_size, 1) raw scores."""
        h = self.encoder(x)
        return self.score_head(h).squeeze(-1)


class DirectWeightNN:
    """Neural net wrapper matching the GBM model interface."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 3,
                 dropout: float = 0.1, temperature: float = 1.0, lr: float = 1e-3):
        _check_torch()
        self.net = AssetScorerNet(n_features, hidden_dim, n_layers, dropout)
        self.temperature = temperature
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def train_model(
        self,
        X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
        date_indices_train: np.ndarray, date_indices_val: np.ndarray,
        n_epochs: int = 100, batch_size: int = 2048, patience: int = 15,
    ) -> dict:
        """Train the neural network."""
        self.net.train()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(w_train, dtype=torch.float32, device=self.device)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None

        for epoch in range(n_epochs):
            # Simple full-batch for now (tabular data is typically small enough)
            optimizer.zero_grad()
            scores = self.net(X_t)
            # MSE to soft target weights (per-row regression)
            loss = torch.mean(w_t * (scores - y_t) ** 2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Validation
            if epoch % 5 == 0:
                val_loss = self._eval_loss(X_val, y_val, w_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                elif epoch - best_epoch > patience:
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}

    def _eval_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            w_t = torch.tensor(w, dtype=torch.float32, device=self.device)
            scores = self.net(X_t)
            loss = torch.mean(w_t * (scores - y_t) ** 2)
        self.net.train()
        return float(loss.item())

    def predict_weights(self, X: np.ndarray, date_indices: np.ndarray) -> np.ndarray:
        """Predict weights with masked softmax grouping by date."""
        from folio_trainer.labels.distill_labels import masked_softmax

        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            scores = self.net(X_t).cpu().numpy()

        weights = np.zeros(len(scores))
        for d in np.unique(date_indices):
            idx = date_indices == d
            date_scores = scores[idx]
            mask = np.ones(len(date_scores))
            weights[idx] = masked_softmax(date_scores, mask, self.temperature)

        return weights

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "temperature": self.temperature,
            "config": {
                "n_features": self.net.encoder[0].in_features,
                "hidden_dim": self.net.encoder[0].out_features,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> DirectWeightNN:
        _check_torch()
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        instance = cls(cfg["n_features"], cfg["hidden_dim"], temperature=data["temperature"])
        instance.net.load_state_dict(data["state_dict"])
        return instance
