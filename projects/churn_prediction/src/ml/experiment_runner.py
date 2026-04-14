from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

# Mantém seus imports/utilitários originais:
# (ajuste os caminhos conforme seu projeto)
from src.ml.logging_utils import ModelLogger
from src.ml.metrics_utils import save_confusion_matrix_artifacts
from src.ml.persistence import save_and_log_joblib
from src.ml.mlflow_utils import (
    _mlflow_set_tags,
    _safe_log_metrics,
    _safe_log_params,
    _cleanup_dir,
)

TuneMode = str  # "none" | "all" | "best_only"

@dataclass(frozen=True)
class ExperimentSpec:
    parent_run_name: str
    job_tag: str
    best_metric: str
    save_best: bool = False
    best_output_path_tpl: str = "models/best_{model}.joblib"
    log_best_confusion_matrix: bool = False
    cm_normalize: Optional[str] = None
    cm_labels: Optional[tuple] = None
    log_model: bool = True
    model_artifact_name: str = "model"
    register_model_name: Optional[str] = None
    skops_trusted_types: Optional[tuple[type, ...]] = None


@dataclass
class SklearnModelsExperiment:
    """
    Orquestrador de experimento genérico (SklearnPipelineRunner + MLflow).

    Mantém a lógica da antiga run_sklearn_models_experiment, mas com:
    - estado encapsulado
    - métodos menores (testáveis e reutilizáveis)
    - wrapper function para retrocompatibilidade
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

    # -----------------------------
    # API pública
    # -----------------------------
    def run(self) -> pd.DataFrame:
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

    # -----------------------------
    # Core workflow methods
    # -----------------------------
    def _run_best_only(self) -> pd.DataFrame:
        self.logger.info("Mode best_only: phase 1 (no tuning for all)")

        rows_phase1: list[dict[str, Any]] = []
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

        # Se quiser CV só no melhor (modo diagnóstico)
        if self.do_cv and (not self.cv_refit_only_best):
            self.logger.info("Running CV only for best model (diagnostic): %s", best_name)
            cv_info_best = self._run_cv_only_for_best(best_name)
            for col in ("mean_cv", "std_cv", "n_folds"):
                df1.loc[df1["model"] == best_name, col] = cv_info_best.get(col)

        # Phase 2: OptunaSearchCV só no vencedor (com run_name tuned_name)
        tuned_name = f"{best_name}{self.tuned_suffix}"
        self.logger.info("Mode best_only: phase 2 (grid only on winner): %s", tuned_name)

        row2, best_runner = self._fit_eval_one(
            model_name=tuned_name,
            model_obj=self.models[best_name],
            enable_tuning=True,
            log_confusion=self.spec.log_best_confusion_matrix,
            extra_tags={"phase": "2", "selected_model": best_name},
            run_cv=False,
            runner_key=best_name,  # runner_id sem suffix para pegar param_grid correto
            cm_name=best_name,     # confusion matrix com nome do "best"
        )

        # salvar melhor SEM retraining
        if self.spec.save_best:
            self._save_best_joblib(best_name=best_name, best_runner=best_runner)

        if self.spec.log_best_confusion_matrix:
            self._log_best_confusion_matrix(best_name=best_name, best_runner=best_runner)

        return pd.concat([df1, pd.DataFrame([row2])], ignore_index=True)

    def _run_single_phase(self, tune_for_all: bool) -> pd.DataFrame:
        self.logger.info(
            "Mode %s: single phase for all models | enable_tuning_for_all=%s",
            self.tuning_mode,
            tune_for_all,
        )

        rows: list[dict[str, Any]] = []
        fitted_runners: dict[str, Any] = {}

        for name, model in self.models.items():
            row, runner = self._fit_eval_one(
                model_name=name,
                model_obj=model,
                enable_tuning=tune_for_all,
                log_confusion=self.spec.log_best_confusion_matrix,
                extra_tags={"phase": "single"},
                run_cv=self.do_cv,
                cm_name=name,
            )
            rows.append(row)
            fitted_runners[name] = runner

        df = pd.DataFrame(rows)
        self.logger.info("Experiment finished. Rows=%d", len(df))

        # salvar melhor SEM retraining
        if self.spec.save_best and not df.empty:
            best_name = self._select_best(df)
            mlflow.set_tag("selected_model", best_name)
            best_runner = fitted_runners[best_name]
            self._save_best_joblib(best_name=best_name, best_runner=best_runner)

        return df

    # -----------------------------
    # Single-model operations
    # -----------------------------
    def _make_runner(self, model_name: str, model_obj: Any):
        """
        Compatível com build_runner(model) e build_runner(model_name, model).
        """
        try:
            sig = inspect.signature(self.build_runner)
            if len(sig.parameters) >= 2:
                return self.build_runner(model_name, model_obj)
        except Exception:
            pass
        return self.build_runner(model_obj)

    def _set_runner_tuning(self, mlogger: Any, runner: Any, enabled: bool):
        mlogger.debug("Config runner.use_optuna_search=%s", enabled)
        runner.use_optuna_search = enabled
        if not enabled:
            runner.param_grid = {}
        return runner

    def _log_cv_scores(self, mlogger: Any, model_name: str, scores: np.ndarray, scoring: str) -> Dict[str, Any]:
        scores = np.asarray(scores, dtype=float)
        mean_cv = float(scores.mean()) if len(scores) else float("nan")
        std_cv = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

        mlogger.info(
            "CV done | scoring=%s | mean=%.4f | std=%.4f | folds=%d",
            scoring, mean_cv, std_cv, len(scores)
        )

        _safe_log_metrics(mlogger, {
            f"cv_mean_{scoring}": mean_cv,
            f"cv_std_{scoring}": std_cv,
            "cv_folds": int(len(scores)),
        })

        tmp_dir = self.artifacts_dir / f"cv_{model_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        fold_path = tmp_dir / "cv_scores.csv"
        pd.DataFrame({"fold": np.arange(len(scores)), "score": scores}).to_csv(fold_path, index=False)
        mlflow.log_artifact(str(fold_path), artifact_path="cv")
        mlogger.debug("CV folds artifact logged: %s", fold_path)

        _cleanup_dir(mlogger, tmp_dir)

        return {"mean_cv": mean_cv, "std_cv": std_cv, "n_folds": int(len(scores))}

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
        cm_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Executa um child run aninhado:
        - tags/params
        - (opcional) cross_validate
        - fit
        - eval no teste
        - (opcional) confusion matrix
        - (opcional) log do modelo no MLflow
        Retorna: (row, runner_fitted)
        """
        mlogger = ModelLogger(self.logger, {"model_name": model_name})
        mlogger.info("Child run start | enable_tuning=%s | run_cv=%s", enable_tuning, run_cv)

        runner_id = runner_key or model_name
        runner = self._make_runner(runner_id, model_obj)
        runner = self._set_runner_tuning(mlogger, runner, enable_tuning)

        cm_name = cm_name or model_name

        with mlflow.start_run(run_name=model_name, nested=True):
            tags = {"model": model_name, "optuna_tuning": "1" if enable_tuning else "0"}
            if extra_tags:
                tags.update(extra_tags)
            _mlflow_set_tags(mlogger, tags)

            _safe_log_params(mlogger, {
                "model_name": model_name,
                "use_optuna_search": enable_tuning,
                "cv": getattr(runner, "cv", None),
                "scoring": getattr(runner, "scoring", None),
                "use_feature_engineering": getattr(runner, "use_feature_engineering", None),
                "use_feature_selection": getattr(runner, "use_feature_selection", None),
                "k_best": getattr(runner, "k_best", None),
            })
            _safe_log_params(mlogger, extra_params or {})

            # CV
            cv_info: dict[str, Any] = {}
            if run_cv:
                mlogger.info("Running CV on train...")
                mlflow.set_tag("phase", "cv_train")
                try:
                    scores = runner.cross_validate(self.X_train, self.y_train)
                    cv_info = self._log_cv_scores(
                        mlogger=mlogger,
                        model_name=model_name,
                        scores=np.asarray(scores),
                        scoring=getattr(runner, "scoring", "score"),
                    )
                except Exception as e:
                    mlogger.exception("Falha no cross_validate: %s", e)
                    cv_info = {"mean_cv": float("-inf"), "std_cv": float("nan"), "n_folds": 0}

            # Fit
            mlogger.info("Fitting model...")
            mlflow.set_tag("phase", "fit_eval")
            runner.fit(self.X_train, self.y_train)

            if enable_tuning:
                cv_best_score = getattr(runner, "cv_best_score_", None)

                if cv_best_score is None:
                    pipe = getattr(runner, "pipeline", None)
                    if pipe is not None and hasattr(pipe, "best_score_"):
                        cv_best_score = pipe.best_score_

                gs_scoring = getattr(runner, "scoring", None)
                if gs_scoring is not None:
                    _safe_log_params(mlogger, {"tuning_scoring": str(gs_scoring)})

            # Eval
            mlogger.info("Evaluating on test...")
            mlflow.set_tag("phase", "eval_test")
            metrics = runner.evaluate(self.X_test, self.y_test, include_auc=True)
            _safe_log_metrics(mlogger, metrics)

            if enable_tuning and getattr(runner, "pipeline", None) is not None and hasattr(runner.pipeline, "best_params_"):
                mlogger.info("OptunaSearchCV best_params available; logging.")
                _safe_log_params(mlogger, {"best_params": runner.pipeline.best_params_})

            # Confusion matrix (se pedido)
            if log_confusion:
                self._log_confusion_matrix_for_runner(
                    mlogger=mlogger,
                    runner=runner,
                    cm_name=cm_name,
                    artifact_subdir="confusion_matrix",
                )

            # Model logging
            if self.spec.log_model:
                self._log_model_mlflow(mlogger=mlogger, runner=runner)

            best_params = None
            if enable_tuning:
                pipe = getattr(runner, "pipeline", None)
                if pipe is not None and hasattr(pipe, "best_params_"):
                    best_params = pipe.best_params_

            if enable_tuning:
                cv_best_score = getattr(runner, "cv_best_score_", None)
                if cv_best_score is None:
                    pipe = getattr(runner, "pipeline", None)
                    if pipe is not None and hasattr(pipe, "best_score_"):
                        cv_best_score = pipe.best_score_

                if cv_best_score is not None:
                    # n_folds
                    n_folds = None
                    cv_obj = getattr(runner, "cv", None)
                    if hasattr(cv_obj, "get_n_splits"):
                        n_folds = int(cv_obj.get_n_splits())
                    elif isinstance(cv_obj, int):
                        n_folds = int(cv_obj)

                    # Preenche no mesmo formato do resto
                    cv_info = {
                        **cv_info,
                        "mean_cv": float(cv_best_score),
                        "std_cv": float("nan"),   # OptunaSearchCV não expõe std diretamente
                        "n_folds": n_folds,
                    }

            row: Dict[str, Any] = {
                "model": model_name,
                **cv_info,
                **metrics,
            }

            if best_params is not None:
                row["best_params"] = str(best_params)

            mlogger.info("Child run end")

            return row, runner

    # -----------------------------
    # Selection, CV-only, artifacts
    # -----------------------------
    def _select_best(self, df: pd.DataFrame) -> str:
        if df.empty:
            raise ValueError("DataFrame vazio: não há modelos para selecionar.")

        if self.select_by_cv:
            if "mean_cv" not in df.columns:
                raise ValueError("select_by_cv=True, mas não há mean_cv. Rode CV para permitir seleção por CV.")
            best = str(df.sort_values("mean_cv", ascending=False).iloc[0]["model"])
            self.logger.info("Selected best by CV: %s", best)
            return best

        if self.spec.best_metric not in df.columns:
            raise ValueError(f"best_metric='{self.spec.best_metric}' não está no DataFrame: {df.columns.tolist()}")

        best = str(df.sort_values(self.spec.best_metric, ascending=False).iloc[0]["model"])
        self.logger.info("Selected best by TEST metric (%s): %s", self.spec.best_metric, best)
        return best

    def _run_cv_only_for_best(self, best_name: str) -> Dict[str, Any]:
        with mlflow.start_run(run_name=f"{best_name}_cv_only", nested=True):
            mlogger = ModelLogger(self.logger, {"model_name": best_name})
            runner_tmp = self._make_runner(best_name, self.models[best_name])
            runner_tmp = self._set_runner_tuning(mlogger, runner_tmp, False)
            scores = runner_tmp.cross_validate(self.X_train, self.y_train)
            return self._log_cv_scores(
                mlogger=mlogger,
                model_name=best_name,
                scores=np.asarray(scores),
                scoring=getattr(runner_tmp, "scoring", "score"),
            )

    def _save_best_joblib(self, best_name: str, best_runner: Any) -> None:
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

    def _log_best_confusion_matrix(self, best_name: str, best_runner: Any) -> None:
        mlogger = ModelLogger(self.logger, {"model_name": best_name})
        mlogger.info("Gerando e logando confusion matrix do MELHOR modelo (final)...")

        try:
            y_pred_best = best_runner.predict(self.X_test)
            cm_dir = self.artifacts_dir / f"cm_best_{best_name}"
            paths = save_confusion_matrix_artifacts(
                y_true=self.y_test,
                y_pred=y_pred_best,
                out_dir=cm_dir,
                labels=self.spec.cm_labels,
                normalize=self.spec.cm_normalize,
                prefix="confusion_matrix",
            )
            mlflow.log_artifact(str(paths["csv"]), artifact_path="best/confusion_matrix")
            mlflow.log_artifact(str(paths["png"]), artifact_path="best/confusion_matrix")
        except Exception as e:
            mlogger.warning("Falha ao gerar/logar CM do melhor: %s", e)

    def _log_confusion_matrix_for_runner(
        self,
        mlogger: Any,
        runner: Any,
        cm_name: str,
        artifact_subdir: str,
    ) -> None:
        mlogger.info("Logging confusion matrix artifacts...")

        try:
            y_pred = runner.predict(self.X_test)
            cm_dir = self.artifacts_dir / f"cm_{cm_name}"
            paths = save_confusion_matrix_artifacts(
                y_true=self.y_test,
                y_pred=y_pred,
                out_dir=cm_dir,
                labels=self.spec.cm_labels,
                normalize=self.spec.cm_normalize,
                prefix="confusion_matrix",
            )
            mlflow.log_artifact(str(paths["csv"]), artifact_path=artifact_subdir)
            mlflow.log_artifact(str(paths["png"]), artifact_path=artifact_subdir)
            mlogger.debug("Confusion matrix logged: %s, %s", paths["csv"], paths["png"])
        except Exception as e:
            mlogger.warning("Falha ao logar matriz de confusão: %s", e)

    def _log_model_mlflow(self, mlogger: Any, runner: Any) -> None:
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
            # fallback para versões sem args de skops
            mlogger.warning("MLflow sem suporte a skops args (%s). Usando log_model padrão.", e)
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


# ---------------------------------------------------------------------
# Wrapper para manter compatibilidade com seus scripts atuais
# ---------------------------------------------------------------------
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
    tuned_suffix: str = "_OPT",
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
