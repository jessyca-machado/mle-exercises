"""torch_mlp — Implementação de um MLP utilizando PyTorch, com wrapper para scikit-learn.

Implementação de um MLP utilizando PyTorch, com wrapper para scikit-learn.

Permite usar o MLP dentro de um Pipeline, e habilita métodos fit/predict/predict_proba para avaliação com AUC e PR-AUC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import io

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from src.utils.constants import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 32), dropout: float = 0.1):
        """
        Constrói a arquitetura do MLP (camadas densas + ReLU + Dropout) para classificação binária,
        produzindo 1 logit na saída.

        Args:
            input_dim: Número de features de entrada.
            hidden_dims: Tamanho das camadas escondidas.
            dropout: Probabilidade de dropout (0 desativa).

        Returns:
            None: Inicializa a rede em self.net.
        """
        super().__init__()
        logger.info(
            "Inicializando TorchMLP: input_dim=%s hidden_dims=%s dropout=%s",
            input_dim, hidden_dims, dropout
        )

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h

        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza o forward pass do MLP e retorna logits (antes de sigmoid) para cada amostra.

        Args:
            x: Tensor de entrada shape (N, input_dim).

        Returns:
            torch.Tensor: Logits shape (N,).
        """
        logger.info("Forward: x.shape=%s x.dtype=%s x.device=%s", tuple(x.shape), x.dtype, x.device)
        return self.net(x).squeeze(1)


@dataclass
class EarlyStoppingState:
    best: float = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve: int = 0


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, random_state: Optional[int] = RANDOM_STATE):
        """
        Wrapper scikit-learn para um MLP em PyTorch, com suporte a:
            - batching (DataLoader)
            - validação interna (train/val split)
            - early stopping por val_loss
            - predict/predict_proba compatíveis com sklearn

        Args:
            random_state: Semente para reprodutibilidade.

        Returns:
            None: Inicializa hiperparâmetros.
        """
        self.random_state = random_state

        # modelo/otimizacao
        self.hidden_dims = (64, 32)
        self.dropout = 0.1
        self.lr = 1e-3
        self.weight_decay = 0.0

        # treino
        self.batch_size = 64
        self.epochs = 50
        self.val_size = 0.2

        # early stopping
        self.patience = 7
        self.min_delta = 0.0

        # imbalance (opcional)
        self.pos_weight = None

        # infra
        self.device = None
        self.verbose = 0

        logger.info("TorchMLPClassifier inicializado: random_state=%s", random_state)


    def __sklearn_is_fitted__(self) -> bool:
        """
        Indica para o scikit-learn se o estimador já foi ajustado.

        Args:
            None

        Returns:
            bool: True se o modelo foi treinado e possui classes_.
        """
        fitted = hasattr(self, "model_") and self.model_ is not None and hasattr(self, "classes_")
        logger.info("Estimator fitted? %s", fitted)
        return fitted
    

    def _get_device(self) -> torch.device:
        """
        Define o device (CPU/GPU) para treinamento e inferência.

        Args:
            None

        Returns:
            torch.device: Device selecionado.
        """
        dev = torch.device(self.device) if self.device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Device selecionado: %s", dev)
        return dev
    

    def _map_labels_binary(self, y: np.ndarray) -> np.ndarray:
        """
        Mapeia os rótulos originais para {0,1} e armazena self.classes_ para compatibilidade sklearn.

        Args:
            y: Array-like com labels originais.
        
        Returns:
            np.ndarray: Labels mapeados para float32 com valores 0.0/1.0.
        """
        self.classes_ = unique_labels(y)
        logger.info("Classes detectadas: %s", self.classes_)

        if len(self.classes_) != 2:
            raise ValueError("Este classificador está configurado para problema binário.")

        self._class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        y01 = np.vectorize(self._class_to_index_.get)(y).astype(np.float32)

        logger.info("Mapeamento labels: exemplos=%s", y01[:10])
        return y01
    

    def _make_criterion(self, device: torch.device) -> nn.Module:
        """
        Constrói a função de loss (BCEWithLogitsLoss), com suporte opcional a pos_weight.
        
        Args:
            device: Device onde pos_weight será alocado (se usado).
        
        Returns:
            nn.Module: Critério (loss function).
        """
        if self.pos_weight is None:
            logger.info("Loss: BCEWithLogitsLoss (sem pos_weight)")
            return nn.BCEWithLogitsLoss()

        pos_w = torch.tensor([float(self.pos_weight)], dtype=torch.float32, device=device)
        logger.info("Loss: BCEWithLogitsLoss (pos_weight=%s)", float(self.pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)


    def _make_loaders(self, X_tr, y_tr, X_val, y_val):
        """
        Cria DataLoaders para treino e validação.
        
        Args:
            X_tr: Features de treino.
            y_tr: Labels de treino.
            X_val: Features de validação.
            y_val: Labels de validação.
        
        Returns:
            tuple: (dl_tr, dl_val) DataLoaders de treino e validação.
        """
        ds_tr = torch.utils.data.TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        ds_val = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)

        logger.info("Loaders criados: batches_train=%s batches_val=%s", len(dl_tr), len(dl_val))
        return dl_tr, dl_val


    def _train_one_epoch(
        self,
        dl_tr,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> float:
        """
        Executa uma época de treinamento em minibatches.
        
        Args:
            dl_tr: DataLoader de treino.
            device: Device para tensores/modelo.
            optimizer: Otimizador.
            criterion: Função de loss.
            epoch: Índice da época (1-based).
        
        Returns:
            float: Loss média de treino na época.
        """
        self.model_.train()
        loss_sum, n = 0.0, 0

        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model_(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = loss_sum / max(n, 1)
        logger.info("Treino: epoch=%s train_loss=%.6f", epoch, train_loss)
        return train_loss


    @torch.no_grad()
    def _validate_one_epoch(self, dl_val, device: torch.device, criterion: nn.Module, epoch: int) -> float:
        """
        Executa validação em minibatches (sem gradiente).
        
        Args:
            dl_val: DataLoader de validação.
            device: Device para tensores/modelo.
            criterion: Função de loss.
            epoch: Índice da época (1-based).
        
        Returns:
            float: Loss média de validação na época.
        """
        self.model_.eval()
        loss_sum, n = 0.0, 0

        for xb, yb in dl_val:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = self.model_(xb)
            loss = criterion(logits, yb)

            loss_sum += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val_loss = loss_sum / max(n, 1)
        logger.info("Validação: epoch=%s val_loss=%.6f", epoch, val_loss)
        return val_loss


    def _early_stopping_step(self, es: EarlyStoppingState, current_val_loss: float, epoch: int):
        """
        Atualiza o estado de early stopping e indica se deve interromper o treinamento.
        
        Args:
            es: Estado atual do early stopping.
            current_val_loss: Loss de validação da época atual.
            epoch: Índice da época (1-based).
        
        Returns:
            tuple: (es_atualizado, should_stop).
        """
        improved = (es.best - current_val_loss) > self.min_delta

        if improved:
            es.best = current_val_loss
            es.best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            es.epochs_no_improve = 0
            logger.info("EarlyStopping: epoch=%s improved=True best_val_loss=%.6f", epoch, es.best)
            return es, False

        es.epochs_no_improve += 1
        should_stop = es.epochs_no_improve >= self.patience
        logger.info(
            "EarlyStopping: epoch=%s improved=False no_improve=%s/%s should_stop=%s",
            epoch, es.epochs_no_improve, self.patience, should_stop
        )
        return es, should_stop


    def _restore_best(self, es: EarlyStoppingState, device: torch.device) -> None:
        """
        Restaura os melhores pesos do modelo (menor val_loss) obtidos durante o treino.
        Args:
            es: Estado do early stopping contendo best_state.
            device: Device destino para carregar os pesos.
        Returns:
            None: Atualiza self.model_ in-place.
        """
        if es.best_state is None:
            logger.info("Restore best: nenhum best_state encontrado (treino sem melhoria registrada).")
            return
        self.model_.load_state_dict({k: v.to(device) for k, v in es.best_state.items()})
        logger.info("Restore best: modelo restaurado para best_val_loss=%.6f", es.best)


    def fit(self, X, y) -> "TorchMLPClassifier":
        """
        Treina o MLP para classificação binária com:
            - split interno treino/val
            - batching com DataLoader
            - BCEWithLogitsLoss
            - early stopping por val_loss
        
        Args:
            X: Array-like shape (N, D) com features.
            y: Array-like shape (N,) com labels.
        
        Returns:
            TorchMLPClassifier: self ajustado.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = int(X.shape[1])
        logger.info("Fit: X.shape=%s n_features_in_=%s", X.shape, self.n_features_in_)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        logger.info("Fit: seeds configuradas random_state=%s", self.random_state)

        y01 = self._map_labels_binary(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            np.asarray(X, dtype=np.float32),
            y01,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y01,
        )
        logger.info("Fit: split treino/val feito X_tr=%s X_val=%s", X_tr.shape, X_val.shape)

        device = self._get_device()

        self.model_ = TorchMLP(
            input_dim=self.n_features_in_,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)
        logger.info("Fit: modelo criado hidden_dims=%s dropout=%s", self.hidden_dims, self.dropout)

        criterion = self._make_criterion(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        logger.info("Fit: optimizer Adam lr=%s weight_decay=%s", self.lr, self.weight_decay)

        dl_tr, dl_val = self._make_loaders(X_tr, y_tr, X_val, y_val)

        es = EarlyStoppingState()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(dl_tr, device, optimizer, criterion, epoch=epoch)
            val_loss = self._validate_one_epoch(dl_val, device, criterion, epoch=epoch)

            if self.verbose:
                logger.info("Fit: epoch=%s/%s train_loss=%.6f val_loss=%.6f", epoch, self.epochs, train_loss, val_loss)

            es, should_stop = self._early_stopping_step(es, val_loss, epoch=epoch)
            if should_stop:
                logger.info("Fit: early stopping acionado na epoch=%s", epoch)
                break

        self._restore_best(es, device)
        logger.info("Fit: finalizado best_val_loss=%.6f", es.best)
        return self


    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """
        Retorna probabilidades no formato scikit-learn:
            - coluna 0: P(classe 0)
            - coluna 1: P(classe 1)
        
        Args:
            X: Array-like shape (N, D).
        
        Returns:
            np.ndarray: Probabilidades shape (N, 2).
        """
        check_is_fitted(self, "model_")
        X = check_array(X)

        device = self._get_device()
        self.model_.eval()

        xb = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
        logits = self.model_(xb)
        p1 = torch.sigmoid(logits).cpu().numpy()
        p0 = 1.0 - p1

        proba = np.c_[p0, p1]
        logger.info("Predict_proba: proba.shape=%s exemplos=%s", proba.shape, proba[:5])
        return proba


    def predict(self, X) -> np.ndarray:
        """
        Realiza predição de classe usando o modelo ajustado (self.model_) no conjunto de features X.
            - O método usa predict_proba para obter P(y=1)
            - Aplica threshold 0.5 para decidir a classe prevista
            - Retorna os rótulos originais de self.classes_ (compatível com sklearn)
        
        Args:
            X: Array-like de features para predição.
        
        Returns:
            np.ndarray: Array com as predições (labels originais).
        """
        proba1 = self.predict_proba(X)[:, 1]
        idx = (proba1 >= 0.5).astype(int)
        y_pred = self.classes_[idx]

        logger.info("Predictions: %s", y_pred[:10])
        return y_pred


    def __getstate__(self) -> dict:
        """
        Serializa o estado do estimador para permitir persistência via pickle.
            - Copia o __dict__ do objeto
            - Se existir um modelo PyTorch treinado (self.model_), salva apenas o state_dict em bytes
            - Remove a referência direta ao nn.Module do dicionário (evita problemas com pickle)
        
        Args:
            None
        
        Returns:
            dict: Estado serializável do objeto, contendo opcionalmente "_model_state_bytes_".
        """
        logger.info("Serialização (__getstate__): iniciando")

        state = self.__dict__.copy()
        has_model = ("model_" in state) and (state["model_"] is not None)
        logger.info("Serialização (__getstate__): has_model=%s", has_model)

        if has_model:
            buffer = io.BytesIO()
            torch.save(state["model_"].state_dict(), buffer)
            state["_model_state_bytes_"] = buffer.getvalue()
            state["model_"] = None

            logger.info(
                "Serialização (__getstate__): state_dict salvo em bytes (%s bytes)",
                len(state["_model_state_bytes_"])
            )
        else:
            logger.info("Serialização (__getstate__): nenhum modelo para serializar")

        logger.info("Serialização (__getstate__): finalizando")
        return state


    def __setstate__(self, state) -> None:
        """
        Deserializa o estado do estimador reconstruindo o nn.Module e carregando os pesos salvos.
            - Restaura atributos a partir do dict 'state'
            - Se existir "_model_state_bytes_", recria a arquitetura TorchMLP
            - Carrega o state_dict salvo e coloca o modelo em modo eval
        
        Args:
            state: dict com o estado serializado (ex.: vindo de pickle.load).
        
        Returns:
            None: Atualiza o objeto in-place, recriando self.model_ quando aplicável.
        """
        logger.info("Deserialização (__setstate__): iniciando")
        logger.info("Deserialização (__setstate__): keys=%s", list(state.keys())[:20])

        self.__dict__.update(state)

        has_bytes = state.get("_model_state_bytes_") is not None
        logger.info("Deserialização (__setstate__): has_model_bytes=%s", has_bytes)

        if not has_bytes:
            logger.info("Deserialização (__setstate__): nenhum peso encontrado para restaurar")
            return

        device = self._get_device()
        logger.info("Deserialização (__setstate__): device=%s", device)

        self.model_ = TorchMLP(
            input_dim=int(self.n_features_in_),
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)
        logger.info(
            "Deserialização (__setstate__): arquitetura recriada input_dim=%s hidden_dims=%s dropout=%s",
            int(self.n_features_in_), self.hidden_dims, self.dropout
        )

        buffer = io.BytesIO(state["_model_state_bytes_"])
        sd = torch.load(buffer, map_location=device)
        self.model_.load_state_dict(sd)
        self.model_.eval()

        logger.info("Deserialização (__setstate__): pesos carregados e modelo setado para eval()")
