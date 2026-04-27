"""Experiment runner — Criação de toda a lógica de experimentação que dará suporte para diferentes modelos de implementação

O que este script faz:
- Abre um run `pai` no MLflow (com tags e parâmetros globais do job/execução).
- Para cada modelo:
    - Cria um runner/pipeline via build_runner.
    - Opcionalmente roda cross-validation no treino.
    - Opcionalmente gera predições OOF (out-of-fold) para calcular e logar trade-off de custo por threshold.
    - Treina (fit) o modelo (com ou sem tuning via GridSearchCV, dependendo do modo).
    - Avalia no teste e loga métricas no MLflow.
    - Opcionalmente loga artefatos (matriz de confusão, modelo no MLflow, curva de custo, scores de CV etc.).
    - Ao final, consolida tudo em um DataFrame com as métricas/informações de cada execução.
- suporta três modos principais:
    - tuning_mode="none": roda tudo sem tuning
    - tuning_mode="all": roda todos com tuning
    - tuning_mode="best_only": roda todos sem tuning para escolher o melhor e tuna só o vencedor (em uma segunda fase)
- E pode também selecionar o melhor modelo por métrica de teste (spec.best_metric) ou por mean_cv (se select_by_cv=True), além de salvar o melhor modelo em joblib se configurado.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.ml.cost_utils import CostSpec, bootstrap_ci, sweep_thresholds_cost
from src.ml.logging_utils import ModelLogger
from src.ml.metrics_utils import save_confusion_matrix_artifacts
from src.ml.mlflow_utils import (
    _cleanup_dir,
    _mlflow_set_tags,
    _safe_log_metrics,
    _safe_log_params,
)
from src.ml.persistence import save_and_log_joblib

TuneMode = str  # "none" | "all" | "best_only"


@dataclass(frozen=True)
class ExperimentSpec:
    parent_run_name: str
    job_tag: str
    best_metric: str

    log_oof_cost_tradeoff: bool = True
    cost_fp: float = 1.0
    cost_fn: float = 5.0
    cost_thresholds_n: int = 201

    save_best: bool = False
    best_output_path_tpl: str = "models/best_{model}.joblib"

    log_best_confusion_matrix: bool = False
    cm_normalize: Optional[str] = None
    cm_labels: Optional[tuple] = None

    log_model: bool = True
    model_artifact_name: str = "model"
    register_model_name: Optional[str] = None
    skops_trusted_types: Optional[Tuple[type, ...]] = None

    cv_best_metric: str = "cv_mean_f1"


@dataclass
class SklearnModelsExperiment:
    """
    Orquestrador de experimento genérico (SklearnPipelineRunner + MLflow).
    """

    logger: Any
    spec: ExperimentSpec
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    models: Dict[str, Any]
    build_runner: Callable[..., Any]

    parent_tags: Optional[Dict[str, str]] = None
    parent_params: Optional[Dict[str, Any]] = None

    tuning_mode: TuneMode = "none"  # "none" | "all" | "best_only"
    do_cv: bool = False
    select_by_cv: bool = False
    cv_refit_only_best: bool = True
    tuned_suffix: str = "_OPT"

    artifacts_dir: str | Path = "artifacts/tmp"

    def __post_init__(self) -> None:
        """
        Normalizar valores default, validar combinações de flags e preparar diretório de artefatos.

        Args:
            - Nenhum argumento explícito (usa atributos do dataclass)

        Returns:
            - None
        """
        self.parent_tags = self.parent_tags or {}
        self.parent_params = self.parent_params or {}
        self.artifacts_dir = Path(self.artifacts_dir)

        self.logger.info(
            "Experiment init | mode=%s | do_cv=%s | select_by_cv=%s | cv_refit_only_best=%s | n_models=%d",
            self.tuning_mode,
            self.do_cv,
            self.select_by_cv,
            self.cv_refit_only_best,
            len(self.models),
        )

        if self.select_by_cv and not self.do_cv:
            raise ValueError("select_by_cv=True requer do_cv=True.")


    def run(self) -> pd.DataFrame:
        """
        Executar o experimento no MLflow (run pai) e orquestrar a execução conforme o modo de tuning.

        Args:
            - Nenhum argumento explícito (usa atributos do objeto: spec, models, tuning_mode, do_cv etc.)

        Returns:
            - DataFrame com uma linha por execução de modelo (métricas, infos de CV/OOF, etc.)
        """
        with mlflow.start_run(run_name=self.spec.parent_run_name):
            self.logger.info("MLflow parent run started: %s", self.spec.parent_run_name)

            mlflow.set_tag("job_tag", self.spec.job_tag)
            _mlflow_set_tags(self.logger, self.parent_tags)
            _safe_log_params(self.logger, self.parent_params)

            if self.tuning_mode == "best_only":
                return self._run_best_only()

            if self.tuning_mode not in ("all", "none"):
                raise ValueError(f"tuning_mode inválido: {self.tuning_mode}")

            return self._run_single_phase(tune_for_all=(self.tuning_mode == "all"))


    def _run_best_only(self) -> pd.DataFrame:
        """
        Rodar um experimento em 2 fases: (1) avalia todos sem tuning para escolher o melhor;
        (2) aplica tuning apenas no vencedor e avalia/loga artefatos do melhor.

        Args:
            - Nenhum argumento explícito (usa atributos do objeto)

        Returns:
            - DataFrame com resultados da fase 1 (todos modelos) + fase 2 (modelo tunado vencedor)
        """
        self.logger.info("Mode best_only: phase 1 (no tuning for all)")

        rows_phase1: list[Dict[str, Any]] = []
        fitted_runners_phase1: Dict[str, Any] = {}

        for name, model in self.models.items():
            run_cv_here = self.do_cv and self.cv_refit_only_best
            row, runner = self._fit_eval_one(
                model_name=name,
                model_obj=model,
                enable_tuning=False,
                log_confusion=False,
                extra_tags={"phase": "1"},
                run_cv=run_cv_here,
                run_oof_cost=run_cv_here,
                cm_name=name,
            )
            rows_phase1.append(row)
            fitted_runners_phase1[name] = runner

        df1 = pd.DataFrame(rows_phase1)
        self.logger.info("Phase 1 done. Rows=%d", len(df1))

        if self.select_by_cv and ("n_folds" in df1.columns) and (df1["n_folds"].fillna(0).sum() == 0):
            raise ValueError("CV falhou para todos os modelos (n_folds=0). Verifique colunas do pipeline.")

        best_name = self._select_best(df1)
        mlflow.set_tag("selected_model", best_name)

        tuned_name = f"{best_name}{self.tuned_suffix}"
        self.logger.info("Mode best_only: phase 2 (grid only on winner): %s", tuned_name)

        row2, best_runner = self._fit_eval_one(
            model_name=tuned_name,
            model_obj=self.models[best_name],
            enable_tuning=True,
            log_confusion=self.spec.log_best_confusion_matrix,
            extra_tags={"phase": "2", "selected_model": best_name},
            run_cv=False,
            run_oof_cost=True,         # <-- AQUI
            runner_key=best_name,
            cm_name=best_name,
        )

        if self.spec.save_best:
            self._save_best_joblib(best_name=best_name, best_runner=best_runner)

        if self.spec.log_best_confusion_matrix:
            self._log_best_confusion_matrix(best_name=best_name, best_runner=best_runner)

        return pd.concat([df1, pd.DataFrame([row2])], ignore_index=True)


    def _run_single_phase(self, tune_for_all: bool) -> pd.DataFrame:
        """
        Rodar todos os modelos em uma única fase, com ou sem tuning para todos, e opcionalmente salvar o melhor.

        Args:
            - tune_for_all: habilita tuning (GridSearchCV) para todos os modelos

        Returns:
            - DataFrame com uma linha por modelo executado
        """
        self.logger.info(
            "Mode %s: single phase for all models | enable_tuning_for_all=%s",
            self.tuning_mode,
            tune_for_all,
        )

        rows: list[Dict[str, Any]] = []
        fitted_runners: Dict[str, Any] = {}

        for name, model in self.models.items():
            run_cv_here = self.do_cv and self.cv_refit_only_best
            row, runner = self._fit_eval_one(
                model_name=name,
                model_obj=model,
                enable_tuning=tune_for_all,
                log_confusion=self.spec.log_best_confusion_matrix,
                extra_tags={"phase": "single"},
                run_cv=run_cv_here,
                run_oof_cost=run_cv_here,
                cm_name=name,
            )
            rows.append(row)
            fitted_runners[name] = runner

        df = pd.DataFrame(rows)
        self.logger.info("Experiment finished. Rows=%d", len(df))

        if self.spec.save_best and not df.empty:
            best_name = self._select_best(df)
            mlflow.set_tag("selected_model", best_name)
            self._save_best_joblib(best_name=best_name, best_runner=fitted_runners[best_name])

        return df


    def _make_runner(self, model_name: str, model_obj: Any) -> Any:
        """
        Construir um runner de pipeline compatível com duas assinaturas:
        build_runner(model) ou build_runner(model_name, model).

        Args:
            - model_name: nome lógico do modelo (usado por alguns builders)
            - model_obj: estimador/configuração do modelo

        Returns:
            - Runner construído (ex.: SklearnPipelineRunner ou compatível)
        """
        try:
            sig = inspect.signature(self.build_runner)
            if len(sig.parameters) >= 2:
                return self.build_runner(model_name, model_obj)
        except Exception:
            pass

        return self.build_runner(model_obj)


    def _set_runner_tuning(self, mlogger: Any, runner: Any, enabled: bool) -> Any:
        """
        Habilitar/desabilitar tuning no runner, limpando distribuições quando desabilitado.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: objeto runner do pipeline
            - enabled: True para habilitar GridSearchCV; False para desabilitar

        Returns:
            - Runner configurado
        """
        if hasattr(runner, "use_grid_search"):
            mlogger.debug("Config runner.use_grid_search=%s", enabled)
            runner.use_grid_search = enabled
            if not enabled:
                runner.grid_param_grid = {}

        return runner


    def _prepare_runner(
        self,
        mlogger: Any,
        model_name: str,
        model_obj: Any,
        enable_tuning: bool,
        runner_key: str | None,
    ) -> Any:
        """
        Preparar o runner correto para um modelo, resolvendo o id (runner_key) e aplicando configuração de tuning.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - model_name: nome da execução no MLflow
            - model_obj: objeto do modelo/estimador
            - enable_tuning: ativa/desativa tuning no runner
            - runner_key: identificador alternativo para construir o runner (ex.: sem suffix "_OPT")

        Returns:
            - Runner pronto para fit/eval
        """
        runner_id = runner_key or model_name
        runner = self._make_runner(runner_id, model_obj)
        runner = self._set_runner_tuning(mlogger, runner, enable_tuning)

        return runner


    def _start_child_run_and_log_context(
        self,
        mlogger: Any,
        model_name: str,
        runner: Any,
        enable_tuning: bool,
        extra_tags: Optional[Dict[str, str]],
        extra_params: Optional[Dict[str, Any]],
    ) -> None:
        """
        Logar contexto inicial de uma execução filha no MLflow (tags e parâmetros do runner).

        Args:
            - mlogger: logger contextual (ModelLogger)
            - model_name: nome do run filho (MLflow)
            - runner: runner do pipeline (fonte de parâmetros)
            - enable_tuning: indica se tuning está habilitado
            - extra_tags: tags adicionais para o run filho
            - extra_params: parâmetros adicionais para log

        Returns:
            - None
        """
        tags = {"model": model_name, "gridsearch_tuning": "1" if enable_tuning else "0"}
        if extra_tags:
            tags.update(extra_tags)

        _mlflow_set_tags(mlogger, tags)
        _safe_log_params(
            mlogger,
            {
                "model_name": model_name,
                "use_grid_search": enable_tuning,
                "cv": getattr(runner, "cv", None),
                "scoring": getattr(runner, "scoring", None),
                "use_feature_engineering": getattr(runner, "use_feature_engineering", None),
                "use_feature_selection": getattr(runner, "use_feature_selection", None),
                "k_best": getattr(runner, "k_best", None),
            },
        )
        _safe_log_params(mlogger, extra_params or {})


    def _maybe_run_cv(self, mlogger, model_name, runner, run_cv: bool) -> Dict[str, Any]:
        """
        Executar cross-validation (se habilitado) e registrar resultados no MLflow.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - model_name: nome do modelo (para artefatos/logs)
            - runner: runner com método cross_validate
            - run_cv: controla se o CV será executado

        Returns:
            - Dicionário com mean_cv, std_cv, n_folds (ou vazio se run_cv=False)
        """
        if not run_cv:
            return {}

        mlogger.info("Running CV on train...")
        mlflow.set_tag("phase", "cv_train")
        try:
            cv_info = runner.cross_validate(self.X_train, self.y_train, include_auc=True)
            _safe_log_metrics(mlogger, cv_info)
            # agora o runner retorna cv_n_folds
            return {**cv_info, "cv_n_folds": cv_info.get("cv_n_folds")}
        except Exception as e:
            mlogger.exception("Falha no cross_validate: %s", e)
            mlflow.set_tag("cv_error", str(e)[:200])
            return {"cv_n_folds": 0}
            

    def _coerce_y_for_oof(self, y: Any) -> np.ndarray:
        """
        Padroniza y para OOF (skorch + BCE costuma precisar de float32).
        """
        y_arr = np.asarray(y)
        if y_arr.dtype == bool:
            return y_arr.astype(np.float32)
        if np.issubdtype(y_arr.dtype, np.number):
            return y_arr.astype(np.float32)
        uniq = pd.unique(y_arr)
        if len(uniq) != 2:
            raise ValueError(f"y precisa ser binário; encontrei {len(uniq)} classes: {uniq}")
        uniq_sorted = sorted(list(uniq))
        mapping = {uniq_sorted[0]: 0.0, uniq_sorted[1]: 1.0}
        return np.vectorize(mapping.get)(y_arr).astype(np.float32)


    def _oof_predict_proba(self, runner: Any) -> Tuple[np.ndarray, int]:
        """
        Gerar probabilidades out-of-fold (OOF) no treino via cross_val_predict para análises (ex.: custo x threshold).

        Args:
            - runner: runner do pipeline (fonte do estimator e do cv)

        Returns:
            - Tupla (y_score_oof, n_folds)
        """
        if hasattr(runner, "build_estimator"):
            estimator = runner.build_estimator(self.X_train)
        else:
            runner._filter_existing_columns(self.X_train)
            estimator = runner._build_pipeline()

        cv_obj = getattr(runner, "cv", 5)

        y_train_oof = self._coerce_y_for_oof(self.y_train)
        y_train_oof = np.asarray(y_train_oof, dtype=np.float32).reshape(-1)

        proba_oof = cross_val_predict(
            estimator,
            self.X_train,
            y_train_oof,
            cv=cv_obj,
            method="predict_proba",
            n_jobs=1,
        )
        y_score_oof = proba_oof[:, 1]

        if hasattr(cv_obj, "get_n_splits"):
            n_folds = int(cv_obj.get_n_splits(self.X_train, y_train_oof))
        elif isinstance(cv_obj, int):
            n_folds = int(cv_obj)
        else:
            n_folds = -1

        return y_score_oof, n_folds


    def _log_oof_cost_tradeoff(
        self,
        mlogger: Any,
        model_name: str,
        y_score_oof: np.ndarray,
        n_folds: int,
    ) -> Dict[str, Any]:
        """
        Calcular e logar no MLflow o trade-off de custo por threshold utilizando probabilidades OOF (curva + melhor ponto).

        Args:
            - mlogger: logger contextual (ModelLogger)
            - model_name: nome do modelo (para nomear artefatos)
            - y_score_oof: probabilidades OOF para classe positiva
            - n_folds: número de folds usado para gerar OOF

        Returns:
            - Dicionário com total_cost, best_threshold, net_value do melhor threshold
        """
        spec = CostSpec(cost_fp=float(self.spec.cost_fp), cost_fn=float(self.spec.cost_fn))
        thresholds = np.linspace(0.0, 1.0, int(self.spec.cost_thresholds_n))

        y_true_cost = np.asarray(self.y_train).astype(int).reshape(-1)
        y_score_oof = np.asarray(y_score_oof).astype(float).reshape(-1)

        df_curve, best = sweep_thresholds_cost(y_true_cost, y_score_oof, spec, thresholds=thresholds)
        row_05 = df_curve.iloc[(df_curve["threshold"] - 0.5).abs().argsort()[:1]].iloc[0].to_dict()

        metrics_to_log = {
            "total_cost": float(best["total_cost"]),
            "best_threshold": float(best["threshold"]),
            "cost_at_0p5": float(row_05["total_cost"]),
            "fp_at_best": float(best["fp"]),
            "fn_at_best": float(best["fn"]),
            "oof_n_folds": float(n_folds),
            "net_value": float(best["net_value"]),
        }
        _safe_log_metrics(mlogger, metrics_to_log)

        tmp_dir = self.artifacts_dir / f"oof_cost_{model_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "oof_threshold_cost_curve.csv"
        df_curve.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="oof_cost")

        return {
            "total_cost": float(best["total_cost"]),
            "best_threshold": float(best["threshold"]),
            "net_value": float(best["net_value"]),
            "oof_n_folds": float(n_folds),
        }


    def _maybe_run_oof_cost(self, mlogger: Any, model_name: str, runner: Any, run_oof_cost: bool) -> Dict[str, Any]:
        """
        Executar (condicionalmente) o cálculo de custo por threshold utilizando OOF, somente quando CV estiver habilitado.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - model_name: nome do modelo
            - runner: runner do pipeline
            - run_cv: se False, não calcula OOF/custo (mantém comportamento atual)

        Returns:
            - Dicionário com informações do custo (ou vazio se não executado)
        """
        if not (getattr(self.spec, "log_oof_cost_tradeoff", False) and run_oof_cost):
            return {}
        
        mlogger.info("Running OOF predict_proba for cost trade-off on train...")
        try:
            y_score_oof, n_folds = self._oof_predict_proba(runner)
            return self._log_oof_cost_tradeoff(
                mlogger=mlogger,
                model_name=model_name,
                y_score_oof=y_score_oof,
                n_folds=n_folds,
            )
        
        except Exception as e:
            mlogger.exception("Falha no OOF cost trade-off: %s", e)
            mlflow.set_tag("oof_cost_error", str(e)[:200])
            return {}


    def _fit_runner(self, mlogger: Any, runner: Any, enable_tuning: bool) -> None:
        """
        Treinar o runner no conjunto de treino e, quando houver tuning, logar informações de scoring e melhores parâmetros.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner do pipeline
            - enable_tuning: indica se tuning está ativo (para logar best_params)

        Returns:
            - None
        """
        mlogger.info("Fitting model...")
        mlflow.set_tag("phase", "fit_eval")

        runner.fit(self.X_train, self.y_train)

        if enable_tuning:
            gs_scoring = getattr(runner, "scoring", None)
            if gs_scoring is not None:
                _safe_log_params(mlogger, {"tuning_scoring": str(gs_scoring)})

            pipe = getattr(runner, "pipeline", None)
            if pipe is not None and hasattr(pipe, "best_params_"):
                mlogger.info("GridSearchCV best_params available; logging.")
                _safe_log_params(mlogger, {"best_params": runner.pipeline.best_params_})


    def _get_best_params_if_available(self, runner: Any, enable_tuning: bool) -> Optional[dict]:
        """
        Extrair best_params_ do objeto de tuning (quando disponível).

        Args:
            - runner: runner com atributo pipeline (possivelmente GridSearchCV)
            - enable_tuning: controla se tentará extrair parâmetros

        Returns:
            - Dicionário com best_params_ ou None
        """
        if not enable_tuning:
            return None

        pipe = getattr(runner, "pipeline", None)
        if pipe is not None and hasattr(pipe, "best_params_"):
            return dict(pipe.best_params_)

        return None


    def _evaluate_on_test(self, mlogger: Any, runner: Any) -> Dict[str, Any]:
        """
        Avaliar o modelo no conjunto de teste e logar métricas no MLflow.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner já treinado

        Returns:
            - Dicionário de métricas (ex.: accuracy, precision, recall, auc etc.)
        """
        mlogger.info("Evaluating on test...")
        mlflow.set_tag("phase", "eval_test")

        metrics = runner.evaluate(self.X_test, self.y_test, include_auc=True)
        _safe_log_metrics(mlogger, metrics)

        return metrics


    def _maybe_log_auc_bootstrap_ci(self, mlogger: Any, runner: Any) -> Dict[str, Any]:
        """
        Calcular e logar intervalos de confiança (bootstrap) da AUC no conjunto de teste.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner já treinado (precisa de predict_proba)

        Returns:
            - Dicionário com estatísticas do CI da AUC (ou vazio se falhar)
        """
        try:
            y_score_test = runner.predict_proba(self.X_test)
            auc_ci = bootstrap_ci(
                y_true=np.asarray(self.y_test).astype(int),
                y_score=np.asarray(y_score_test).astype(float),
                metric_fn=lambda yt, ys: float(roc_auc_score(yt, ys)),
                n_bootstrap=1000,
                confidence=0.95,
                random_state=42,
                require_both_classes=True,
            )
            auc_ci_info = {
                "auc_ci_mean": float(auc_ci["mean"]),
                "auc_ci_std": float(auc_ci["std"]),
                "auc_ci_lower": float(auc_ci["lower"]),
                "auc_ci_upper": float(auc_ci["upper"]),
                "auc_ci_n": float(auc_ci["n_samples"]),
            }
            _safe_log_metrics(mlogger, auc_ci_info)
            return auc_ci_info
        except Exception as e:
            mlogger.warning("Falha ao calcular AUC bootstrap CI: %s", e)

            return {}


    def _log_confusion_matrix(
        self,
        mlogger: Any,
        predictor: Any,
        name: str,
        artifact_path: str,
        dir_prefix: str = "cm",
        log_level: str = "info",
    ) -> None:
        """
        Gerar e logar artefatos de matriz de confusão (csv/png) no MLflow para o conjunto de teste.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - predictor: objeto com método predict (runner ou modelo)
            - name: nome usado para diretório/identificação do artefato
            - artifact_path: subdiretório no MLflow para salvar os artefatos
            - dir_prefix: prefixo do diretório local temporário
            - log_level: nível do log inicial (ex.: "info", "debug")

        Returns:
            - None
        """
        msg = f"Logging confusion matrix artifacts ({name})..."
        getattr(mlogger, log_level, mlogger.info)(msg)

        try:
            y_pred = predictor.predict(self.X_test)
            cm_dir = self.artifacts_dir / f"{dir_prefix}_{name}"
            paths = save_confusion_matrix_artifacts(
                y_true=self.y_test,
                y_pred=y_pred,
                out_dir=cm_dir,
                labels=self.spec.cm_labels,
                normalize=self.spec.cm_normalize,
                prefix="confusion_matrix",
            )
            mlflow.log_artifact(str(paths["csv"]), artifact_path=artifact_path)
            mlflow.log_artifact(str(paths["png"]), artifact_path=artifact_path)
            mlogger.debug("Confusion matrix logged to %s: %s, %s", artifact_path, paths["csv"], paths["png"])
        except Exception as e:
            mlogger.warning("Falha ao gerar/logar confusion matrix (%s): %s", name, e)


    def _maybe_log_confusion_matrix(self, mlogger: Any, runner: Any, log_confusion: bool, cm_name: str) -> None:
        """
        Controlar condicionalmente o logging da matriz de confusão para um runner.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner treinado
            - log_confusion: flag que habilita/desabilita logging
            - cm_name: nome usado na identificação dos artefatos

        Returns:
            - None
        """
        if not log_confusion:
            return

        self._log_confusion_matrix(
            mlogger=mlogger,
            predictor=runner,
            name=cm_name,
            artifact_path="confusion_matrix",
            dir_prefix="cm",
            log_level="info",
        )


    def _log_best_confusion_matrix(self, best_name: str, best_runner: Any) -> None:
        """
        Logar a matriz de confusão do melhor modelo em um caminho dedicado (best/confusion_matrix).

        Args:
            - best_name: nome do melhor modelo selecionado
            - best_runner: runner do melhor modelo

        Returns:
            - None
        """
        mlogger = ModelLogger(self.logger, {"model_name": best_name})
        self._log_confusion_matrix(
            mlogger=mlogger,
            predictor=best_runner,
            name=best_name,
            artifact_path="best/confusion_matrix",
            dir_prefix="cm_best",
            log_level="info",
        )


    def _log_model_mlflow(self, mlogger: Any, runner: Any) -> None:
        """
        Logar o modelo treinado no MLflow, com suporte opcional a skops (quando configurado).

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner treinado (precisa expor runner.best_model)

        Returns:
            - None
        """
        mlogger.info("Logging model to MLflow...")
        try:
            if self.spec.skops_trusted_types:
                mlflow.sklearn.log_model(
                    sk_model=runner.best_model,
                    name=self.spec.model_artifact_name,
                    registered_model_name=self.spec.register_model_name,
                    serialization_format="skops",
                    skops_trusted_types=list(self.spec.skops_trusted_types),
                    pip_requirements="requirements.txt",
                )
            else:
                mlogger.info("Logging com pickle/cloudpickle (sem skops_trusted_types).")
                mlflow.sklearn.log_model(
                    sk_model=runner.best_model,
                    name=self.spec.model_artifact_name,
                    registered_model_name=self.spec.register_model_name,
                )
        except TypeError as e:
            mlogger.warning("MLflow sem suporte a skops args (%s). utilizando log_model padrão.", e)
            try:
                import mlflow.sklearn as mlflow_sklearn

                mlflow_sklearn.log_model(
                    sk_model=runner.best_model,
                    name=self.spec.model_artifact_name,
                    registered_model_name=self.spec.register_model_name,
                )
            except Exception as e2:
                mlogger.warning("Falha ao logar modelo no MLflow: %s", e2)
        except Exception as e:
            mlogger.warning("Falha ao logar modelo no MLflow: %s", e)


    def _maybe_log_model(self, mlogger: Any, runner: Any) -> None:
        """
        Controlar condicionalmente o logging do modelo no MLflow.

        Args:
            - mlogger: logger contextual (ModelLogger)
            - runner: runner treinado

        Returns:
            - None
        """
        if self.spec.log_model:
            self._log_model_mlflow(mlogger=mlogger, runner=runner)


    def _build_result_row(
        self,
        model_name: str,
        cv_info: Dict[str, Any],
        oof_cost_info: Dict[str, Any],
        metrics: Dict[str, Any],
        auc_ci_info: Dict[str, Any],
        best_params: Optional[dict],
    ) -> Dict[str, Any]:
        """
        Consolidar todas as informações (CV, OOF cost, métricas de teste, CI, best_params) em uma linha de resultado.

        Args:
            - model_name: nome do modelo (identificador principal)
            - cv_info: dicionário com informações de cross-validation
            - oof_cost_info: dicionário com informações de custo/threshold via OOF
            - metrics: métricas de teste retornadas por runner.evaluate
            - auc_ci_info: informações de intervalo de confiança da AUC (bootstrap)
            - best_params: melhores parâmetros do tuning (ou None)

        Returns:
            - Dicionário pronto para virar uma linha em DataFrame
        """
        row: Dict[str, Any] = {"model": model_name, **cv_info, **oof_cost_info, **metrics, **auc_ci_info}
        if best_params is not None:
            row["best_params"] = str(best_params)

        return row


    def _fit_eval_one(
        self,
        model_name: str,
        model_obj: Any,
        enable_tuning: bool,
        log_confusion: bool,
        runner_key: str | None = None,
        extra_tags: Optional[Dict[str, str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        run_cv: bool = False,
        run_oof_cost: bool = False,   # <-- NOVO
        cm_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Executar uma execução completa de um modelo: configurar runner, (opcional) CV/OOF, treinar, avaliar no teste
        e logar artefatos/métricas no MLflow.

        Args:
            - model_name: nome do run filho no MLflow
            - model_obj: objeto do modelo/estimador a ser executado
            - enable_tuning: se True, habilita tuning no runner
            - log_confusion: se True, loga matriz de confusão
            - runner_key: id alternativo para construir runner (útil quando model_name tem suffix)
            - extra_tags: tags adicionais no MLflow
            - extra_params: parâmetros adicionais no MLflow
            - run_cv: se True, executa cross_validate e também OOF-cost (se habilitado no spec)
            - cm_name: nome base para artefatos de matriz de confusão

        Returns:
            - Tupla (row_dict, runner_fitted)
        """
        mlogger = ModelLogger(self.logger, {"model_name": model_name})
        mlogger.info("Child run start | enable_tuning=%s | run_cv=%s", enable_tuning, run_cv)

        runner = self._prepare_runner(
            mlogger=mlogger,
            model_name=model_name,
            model_obj=model_obj,
            enable_tuning=enable_tuning,
            runner_key=runner_key,
        )
        cm_name = cm_name or model_name

        with mlflow.start_run(run_name=model_name, nested=True):
            self._start_child_run_and_log_context(
                mlogger=mlogger,
                model_name=model_name,
                runner=runner,
                enable_tuning=enable_tuning,
                extra_tags=extra_tags,
                extra_params=extra_params,
            )

            cv_info = self._maybe_run_cv(mlogger, model_name, runner, run_cv=run_cv)
            # oof_cost_info = self._maybe_run_oof_cost(mlogger, model_name, runner, run_oof_cost=run_oof_cost)

            self._fit_runner(mlogger, runner, enable_tuning=enable_tuning)

            if enable_tuning:
                try:
                    mlogger.info("Running CV on train after tuning (best_estimator_)...")
                    # pega o melhor pipeline (GridSearchCV.best_estimator_) se existir
                    est_best = runner._get_estimator_for_cv_after_fit()

                    cv_post = runner.cross_validate(
                        self.X_train,
                        self.y_train,
                        include_auc=True,
                        estimator_override=est_best,
                    )

                    # loga no MLflow
                    _safe_log_metrics(mlogger, cv_post)

                    # garante que o df final tenha essas colunas preenchidas
                    cv_info = {**cv_info, **cv_post}
                    cv_info["n_folds"] = cv_info.get("n_folds", cv_info.get("n_folds"))

                except Exception as e:
                    mlogger.exception("Falha no CV pós-fit (tuned): %s", e)

            oof_cost_info = {}

            if self.spec.log_oof_cost_tradeoff:
                try:
                    mlogger.info("Computing OOF cost on train after fit (uses best params if tuned)...")
                    y_train_oof = self._coerce_y_for_oof(self.y_train)
                    y_score_oof = runner.oof_predict_proba_after_fit(self.X_train, y_train_oof)
                    oof_cost_info = self._log_oof_cost_tradeoff(
                        mlogger=mlogger,
                        model_name=model_name,
                        y_score_oof=y_score_oof,
                        n_folds=getattr(runner, "cv", 5) if isinstance(getattr(runner, "cv", 5), int) else -1,
                    )
                except Exception as e:
                    mlogger.exception("Falha ao calcular OOF cost após fit: %s", e)
                    mlflow.set_tag("oof_cost_after_fit_error", str(e)[:200])
                    oof_cost_info = {}
                    
            best_params = self._get_best_params_if_available(runner, enable_tuning=enable_tuning)

            metrics = self._evaluate_on_test(mlogger, runner)
            auc_ci_info = self._maybe_log_auc_bootstrap_ci(mlogger, runner)

            self._maybe_log_confusion_matrix(mlogger, runner, log_confusion=log_confusion, cm_name=cm_name)
            self._maybe_log_model(mlogger, runner)

            row = self._build_result_row(
                model_name=model_name,
                cv_info=cv_info,
                oof_cost_info=oof_cost_info,
                metrics=metrics,
                auc_ci_info=auc_ci_info,
                best_params=best_params,
            )
            mlogger.info("Child run end")

            return row, runner


    def _select_best(self, df: pd.DataFrame) -> str:
        """
        Selecionar o melhor modelo de acordo com CV (cv_mean_f1) ou uma métrica do teste (spec.best_metric).

        Args:
            - df: DataFrame com resultados dos modelos

        Returns:
            - Nome do melhor modelo (string)
        """
        if df.empty:
            raise ValueError("DataFrame vazio: não há modelos para selecionar.")

        if self.select_by_cv:
            metric = self.spec.cv_best_metric
            if metric not in df.columns:
                raise ValueError(f"select_by_cv=True, mas não há {metric}.")
            best = str(df.sort_values(metric, ascending=False).iloc[0]["model"])
            return best

        if self.spec.best_metric not in df.columns:
            raise ValueError(f"best_metric='{self.spec.best_metric}' não está no DataFrame: {df.columns.tolist()}")

        best = str(df.sort_values(self.spec.best_metric, ascending=False).iloc[0]["model"])
        self.logger.info("Selected best by TEST metric (%s): %s", self.spec.best_metric, best)

        return best


    def _save_best_joblib(self, best_name: str, best_runner: Any) -> None:
        """
        Persistir o melhor modelo (sem retraining) em joblib e logar como artefato no MLflow.

        Args:
            - best_name: nome do melhor modelo
            - best_runner: runner correspondente ao melhor modelo (precisa expor best_model)

        Returns:
            - None
        """
        try:
            out_path = self.spec.best_output_path_tpl.format(model=best_name)
            mlogger = ModelLogger(self.logger, {"model_name": best_name})
            save_and_log_joblib(
                model=best_runner.best_model,
                path=out_path,
                mlflow_module=mlflow,
                artifact_path="saved_model_joblib",
                logger=mlogger,
                log_prefix="Persistindo melhor modelo (sem retraining) via joblib",
            )
        except Exception as e:
            self.logger.exception("Falha ao salvar/logar joblib do melhor: %s", e)


def run_sklearn_models_experiment(
    logger,
    spec: ExperimentSpec,
    X_train,
    y_train,
    X_test,
    y_test,
    models: Dict[str, Any],
    build_runner: Callable[..., Any],
    parent_tags: Optional[Dict[str, str]] = None,
    parent_params: Optional[Dict[str, Any]] = None,
    tuning_mode: TuneMode = "none",
    do_cv: bool = False,
    select_by_cv: bool = False,
    cv_refit_only_best: bool = True,
    tuned_suffix: str = "_GS",
    artifacts_dir: str | Path = "artifacts/tmp",
) -> pd.DataFrame:
    exp = SklearnModelsExperiment(
        logger=logger,
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models,
        build_runner=build_runner,
        parent_tags=parent_tags,
        parent_params=parent_params,
        tuning_mode=tuning_mode,
        do_cv=do_cv,
        select_by_cv=select_by_cv,
        cv_refit_only_best=cv_refit_only_best,
        tuned_suffix=tuned_suffix,
        artifacts_dir=artifacts_dir,
    )

    return exp.run()
