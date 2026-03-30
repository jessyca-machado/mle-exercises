"""torch_mlp — Implementação de um MLP usando PyTorch, com wrapper para scikit-learn.

Implementação de um MLP usando PyTorch, com wrapper para scikit-learn.

Permite usar o MLP dentro de um Pipeline, e habilita métodos fit/predict/predict_proba para avaliação com AUC e PR-AUC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TorchMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Tuple[int, ...] = (64, 32),
            dropout: float = 0.1
        ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Estimador scikit-learn para usar um MLP PyTorch dentro de Pipeline.
    Implementa fit / predict / predict_proba para habilitar AUC e PR-AUC.
    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
    ):
        self.random_state = random_state

        self.hidden_dims = (64, 32)
        self.dropout = 0.1
        self.lr = 1e-3
        self.batch_size = 64
        self.epochs = 30
        self.weight_decay = 0.0
        self.device: Optional[str] = None
        self.verbose = 0

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_features_in_ = int(X.shape[1])
        n_classes = int(len(self.classes_))
        self._class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.vectorize(self._class_to_index_.get)(y).astype(np.int64)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        device = self._get_device()

        self.model_ = TorchMLP(
            input_dim=self.n_features_in_,
            hidden_dims=self.hidden_dims,
            output_dim=n_classes,
            dropout=self.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n = 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            if self.verbose:
                logger.info("[TorchMLP] epoch %d/%d loss=%.4f", epoch + 1, self.epochs, total_loss / max(n, 1))

        return self

    @torch.no_grad()
    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X)

        device = self._get_device()
        self.model_.eval()

        xb = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
        logits = self.model_(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]
