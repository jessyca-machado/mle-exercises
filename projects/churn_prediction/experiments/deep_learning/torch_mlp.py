"""
torch_mlp.py — Arquitetura de MLP em PyTorch para uso com skorch (+ Optuna, CV, etc.)

Este módulo agora contém apenas:
- TorchMLP: nn.Module que produz 1 logit (para BCEWithLogitsLoss)
- build_skorch_mlp: helper opcional para criar um NeuralNetBinaryClassifier (skorch)

O wrapper sklearn manual (TorchMLPClassifier) foi removido porque o skorch já fornece
integração sklearn (fit/predict/predict_proba), compatível com Pipeline e busca/tuning.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier


HiddenDims = Union[str, Tuple[int, ...], Iterable[int]]

class TorchMLP(nn.Module):
    """
    MLP simples para classificação binária:
        - camadas densas + ReLU + Dropout (opcional)
        - saída: 1 logit (sem sigmoid no forward)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: HiddenDims = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, str):
            # ex: "256,128" -> (256, 128)
            hidden_dims = tuple(int(x.strip()) for x in hidden_dims.split(",") if x.strip())

        hidden_dims = tuple(hidden_dims)

        layers = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # retorna logits shape (N,)
        return self.net(x).squeeze(1)


def build_skorch_mlp(
    input_dim: int,
    *,
    hidden_dims: HiddenDims = (128, 64),
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 30,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    train_split=None,
    callbacks=None,
    verbose: int = 0,
):
    """
    Factory para criar um estimador skorch (NeuralNetBinaryClassifier) pronto para entrar em Pipeline/CV/Optuna.

    Parâmetros importantes:
        - input_dim: número de features após o preprocessing (OHE/Scaler/FS etc.)
        - train_split: default None (sem validação interna). Se você quiser EarlyStopping do skorch,
        passe um ValidSplit(...) e callbacks=[EarlyStopping(...)].

    Retorna:
        - instancia de NeuralNetBinaryClassifier
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # BCEWithLogitsLoss combina sigmoid + BCE de forma estável
    net = NeuralNetBinaryClassifier(
        module=TorchMLP,
        module__input_dim=int(input_dim),
        module__hidden_dims=tuple(hidden_dims),
        module__dropout=float(dropout),
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=float(weight_decay),
        lr=float(lr),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        iterator_train__shuffle=True,
        train_split=train_split,   # None = sem validação interna
        callbacks=callbacks,
        device=device,
        verbose=verbose,
    )

    return net
