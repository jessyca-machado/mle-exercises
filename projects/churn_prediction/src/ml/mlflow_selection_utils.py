from typing import List, Optional

from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from rich.console import Console

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    N_FOLDS,
)

console = Console()


def is_cv_run(run: Run) -> bool:
    """
    Verifica se um run do MLflow corresponde a uma execução de cross-validation (CV)
    compatível com a configuração atual do projeto.

    A regra utilizada é checar se o run possui o parâmetro `cv_folds` ou `n_folds`

    Args:
        run: Objeto `mlflow.entities.Run` retornado pelo MLflow, contendo `run.data.params`.

    Returns:
        True se o run aparenta ser de CV com o mesmo número de folds (`N_FOLDS`),
        caso contrário False.
    """
    p = run.data.params

    return (p.get("cv_folds") == str(N_FOLDS)) or (p.get("n_folds") == str(N_FOLDS))


def is_mlp_refit_run(run: Run) -> bool:
    """
    Verifica se um run do MLflow é um run de "refit final" do MLP.

    Um run é considerado refit do MLP quando:
    - params["model_name"] == "mlp"
    - existe o parâmetro "best_config_name"

    Args:
        run: Objeto `mlflow.entities.Run`, contendo `run.data.params`.

    Returns:
        True se o run representa o refit final do MLP, caso contrário False.
    """
    p = run.data.params

    return p.get("model_name") == "mlp" and ("best_config_name" in p)


def get_latest_mlp_refit_run(client: MlflowClient) -> Optional[Run]:
    """
    Busca o run de refit do MLP mais recente no experimento definido por `MLFLOW_EXPERIMENT_NAME`.

    A função:
    - obtém o experimento pelo nome
    - busca runs ordenados por start_time decrescente
    - retorna o primeiro run que satisfaz `is_mlp_refit_run(run)`

    Args:
        client: Instância de `mlflow.tracking.MlflowClient` utilizada para consultar experimentos e
            runs.

    Returns:
        O run (objeto `mlflow.entities.Run`) mais recente que representa o refit do MLP,
        ou None se não existir.
    """
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    for r in runs:
        if is_mlp_refit_run(r):
            return r
    return None


def find_cv_run_by_config_name(client: MlflowClient, config_name: str) -> Optional[Run]:
    """
    Encontra o run de cross-validation (CV) correspondente a uma configuração específica,
    identificado pelo `config_name`.

    A busca tenta duas formas de identificação:
    1) `run.data.params["config_name"] == config_name` (preferível, explícito)
    2) `run.data.tags["mlflow.runName"] == config_name` (fallback)

    Em ambos os casos, o run também precisa satisfazer `is_cv_run(run)`.

    Args:
        client: Instância de `mlflow.tracking.MlflowClient`.
        config_name: Nome da configuração (por exemplo, "mlp_config_3_kbest_all").

    Returns:
        O run de CV (objeto `mlflow.entities.Run`) correspondente à config, ou None se não
            encontrar.
    """
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    for r in runs:
        p = r.data.params
        if p.get("config_name") == config_name and is_cv_run(r):
            return r
        if r.data.tags.get("mlflow.runName") == config_name and is_cv_run(r):
            return r
    return None


def has_any_oof(client: MlflowClient, run_id: str) -> bool:
    """
    Verifica se um run possui artifacts de predição out-of-fold (OOF) suficientes
    para avaliação por folds.

    Aceita dois padrões de nome de artifact:
    - sem prefixo: oof/y_true_fold_0.npy e oof/y_proba_fold_0.npy
    - com prefixo: oof/<qualquer_prefixo>y_true_fold_0.npy e
        oof/<qualquer_prefixo>y_proba_fold_0.npy

    A função não baixa os arquivos, apenas lista os artifacts no caminho "oof/" e checa
    se existe pelo menos o fold 0 para y_true e y_proba.

    Args:
        client: Instância de `mlflow.tracking.MlflowClient`.
        run_id: ID do run no MLflow.

    Returns:
        True se existir pelo menos um par de arquivos OOF do fold 0 (y_true e y_proba),
        caso contrário False.
    """
    try:
        files = client.list_artifacts(run_id, path="oof")
        paths = [f.path for f in files]
    except Exception:
        return False

    has_true0 = any(p.endswith("y_true_fold_0.npy") for p in paths)
    has_proba0 = any(p.endswith("y_proba_fold_0.npy") for p in paths)
    return bool(has_true0 and has_proba0)


def get_latest_runs_with_mlp_from_refit(client: MlflowClient) -> List[Run]:
    """
    Retorna um conjunto de runs "representativos" (no máximo 1 por model_name),
    pegando os runs mais recentes do experimento, com uma regra especial para o MLP:

    - Para o MLP: tenta localizar o run de CV correspondente à config vencedora apontada
        pelo `best_config_name` no run de refit mais recente do MLP.
    - Para os demais modelos: pega o run de CV mais recente por `model_name`
        (ordenado por start_time desc).
    - Em todos os casos, exige que o run:
        - seja de CV (`is_cv_run`)
        - possua artifacts OOF (`has_any_oof`)

    Args:
        client: Instância de `mlflow.tracking.MlflowClient`.

    Returns:
        Lista de runs (objetos `mlflow.entities.Run`) selecionados (um por model_name, quando
            possível).
    """
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    chosen: dict[str, Run] = {}

    # 1) MLP via refit -> best_config_name -> run de CV correspondente
    mlp_refit = get_latest_mlp_refit_run(client)
    if mlp_refit is not None:
        best_config_name = mlp_refit.data.params.get("best_config_name")
        if best_config_name:
            mlp_cv_run = find_cv_run_by_config_name(client, best_config_name)
            if mlp_cv_run is not None and has_any_oof(client, mlp_cv_run.info.run_id):
                chosen["mlp"] = mlp_cv_run
            else:
                console.print(
                    (
                        f"[yellow]Aviso:[/yellow] não achei run de CV para "
                        f"best_config_name='{best_config_name}'. "
                        "Vou cair no fallback do último CV do mlp."
                    )
                )

    # 2) Outros modelos (e fallback do MLP): último run CV por model_name com OOF
    for r in all_runs:
        p = r.data.params
        model_name = p.get("model_name")
        if not model_name:
            continue
        if not is_cv_run(r):
            continue
        if model_name in chosen:
            continue
        if not has_any_oof(client, r.info.run_id):
            continue
        chosen[model_name] = r

    return list(chosen.values())


def run_display_name(run: Run) -> str:
    """
    Gera o nome de exibição de um run, utilizado em tabelas e dicionários de comparação.

    Prioriza o parâmetro "model_name" (padrão do projeto). Se não existir, usa o `run_id`.

    Args:
        run: Objeto `mlflow.entities.Run` (ou similar), contendo `run.data.params` e
            `run.info.run_id`.

    Returns:
        String com o nome do modelo (model_name) ou, em fallback, o run_id.
    """
    return run.data.params.get("model_name") or run.info.run_id
