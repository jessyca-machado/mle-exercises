"""Comparação estatística entre modelos utilizando testes de hipótese.

Uso:
    # comparação normal, utilizando como baleline a regressão logística:
    python experiments/selection/compare_models.py

    # comparação final, levando em consideração o valor líquido (net_value) dos modelos com recall de treino >= 0.8:
    python experiments/selection/compare_models.py \
    --metric business.sweep_foldwise.net_value \
    --gate-best-cv-score 0.8 \
    --gate-metric-name best_cv_score
"""
import argparse
import logging
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

import mlflow
import numpy as np
import scikit_posthocs as sp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats

from typing import Any, Mapping
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    PRIMARY_METRIC,
    ALPHA,
)
from src.ml.mlflow_selection_utils import (
    get_latest_runs_with_mlp_from_refit,
    run_display_name,
)

logger = logging.getLogger(__name__)
console = Console()


def load_fold_metric(client, run_id: str, metric: str) -> np.ndarray:
    """
    Carrega uma métrica logada por fold, utilizando o histórico de métricas do MLflow
    e retorna como um vetor de tamanho fixo, alinhado aos folds da validação cruzada.

    - busca o histórico da métrica `metric` para o `run_id`
    - mantem apenas pontos com steps em [0, N_FOLDS-1]
    - remove duplicadas por step e mantem o último valor por step após ordenar
    - retorna um array vazio se o run não estiver exatamente N_FOLDS steps

    Args:
        client: Instância de `mlflow.tracking.MlflowClient` usada para consultar o histórico de métricas.
        run_id: ID do run no MLflow de onde será lido o histórico da métrica.
        metric: Nome da métrica a ser carregada.

    Returns:
        Um array NumPy com shape (N_FOLDS,) contendo o valor da métrica para cada fold,
        ordenado por step. Se a métrica for ausente ou incompleta, retorna um array vazio.
    """
    hist = sorted(client.get_metric_history(run_id, metric), key=lambda m: (m.step, m.timestamp))
    by_step = {}
    for m in hist:
        if m.step is None:
            continue
        if 0 <= m.step < N_FOLDS:
            by_step[m.step] = m.value
    if len(by_step) != N_FOLDS:
        return np.array([], dtype=float)

    shape_metric_fold = np.array([by_step[i] for i in range(N_FOLDS)], dtype=float)

    return shape_metric_fold


def print_nemenyi_matrix(
        pvals,
        names: List[str],
        alpha: float = ALPHA
) -> None:
    """
    Imprime de forma legível uma matriz de p-valores pareados do teste post-hoc de Nemenyi,
    utilizando uma tabela do Rich.

    Células com p-valores abaixo de `alpha` são destacadas para indicar diferenças estatisticamente
    significativas.

    Args:
        pvals: Matriz quadrada indexada pelos nomes dos modelos, contendo p-valores pareados.
        names: Lista com os nomes.
        alpha: Nível de significância usado para destacar p-valores.
    """
    table = Table(title="Post-hoc Nemenyi (p-valores)")
    table.add_column("", style="bold")
    for n in names:
        table.add_column(n[:22], justify="right")

    for i, ni in enumerate(names):
        row = [ni[:22]]
        for j, nj in enumerate(names):
            if i == j:
                row.append("[dim]—[/dim]")
            else:
                p = float(pvals.loc[ni, nj])
                if p < alpha:
                    row.append(f"[green]{p:.4f}[/green]")
                else:
                    row.append(f"{p:.4f}")
        table.add_row(*row)

    console.print(table)


def friedman_nemenyi(
        all_scores: Dict[str, np.ndarray],
        model_names: List[str]
) -> Dict:
    """
    Executa o teste global de Friedman para 3+ modelos (amostras pareadas por fold),
    e, se for significativo, executa comparações pareadas post-hoc de Nemenyi.

    Serve como diagnóstico para verificar se há evidência de diferenças entre múltiplos modelos
    avaliados nos mesmos folds do CV do treino.

    Args:
        all_scores: Mapeamento, onde cada valor é um array NumPy, de tamanho N_FOLDS com
            os valores da métrica por fold.
        model_names: Lista de nomes de modelos a incluir no teste, devem existir como chaves no arg `all_scores`.

    Returns:
        Um dicionário contendo:
            - "friedman_stat": estatística qui-quadrado do Friedman.
            - "friedman_p": p-valor do Friedman.
            - "significant": indica se p < ALPHA.
            - opcionalmente "nemenyi_pvals": DataFrame de p-valores pareados (se significativo).
    """
    arrays = [all_scores[n] for n in model_names]
    stat, p = stats.friedmanchisquare(*arrays)
    result = {"friedman_stat": float(stat), "friedman_p": float(p), "significant": bool(p < ALPHA)}
    if p < ALPHA:
        data = np.column_stack(arrays)
        nemenyi = sp.posthoc_nemenyi_friedman(data)
        nemenyi.index = model_names
        nemenyi.columns = model_names
        result["nemenyi_pvals"] = nemenyi
        print_nemenyi_matrix(nemenyi, model_names, alpha=ALPHA)

    return result


def print_duel_panel(
        a: str,
        b: str,
        metric: str,
        mean_a: float,
        mean_b: float,
        res: Dict
) -> None:
    """
    Exibe um resumo detalhado e legível de uma comparação estatística pareada entre dois modelos,
    utilizando um Panel do Rich.

    Args:
        a: Nome do modelo A.
        b: Nome do modelo B.
        metric: Nome da métrica sendo comparada.
        mean_a: Média da métrica do modelo A.
        mean_b: Média da métrica do modelo B.
        res: Dicionário de resultado produzido por uma função de teste pareado.
    """
    sig_style = "bold green" if res["significant"] else "dim"
    sig_mark = "✓ Sig." if res["significant"] else "✗ N.S."

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Comparação", f"{a} vs {b}")
    table.add_row("Médias", f"{mean_a:.6f} vs {mean_b:.6f} (Δ={res['mean_diff']:+.6f})")
    if res["shapiro_p"] is not None:
        table.add_row("Normalidade", f"Shapiro p={res['shapiro_p']:.4f}")
    table.add_row("Teste", f"{res['test']}: stat={res['stat']:.4f}, p={res['p_value']:.6f}")
    table.add_row("Cohen's d", f"{res['cohens_d']:.4f}")
    table.add_row("IC 95% (Δ)", f"[{res['ci95'][0]:+.6f}, {res['ci95'][1]:+.6f}]")
    table.add_row("Resultado", f"[{sig_style}]{sig_mark}[/{sig_style}]")
    console.print(Panel(table, title=f"[bold]{metric} — duelo[/bold]"))


def apply_gate_filter(
    scores: Dict[str, np.ndarray],
    baseline_name: str,
    gate_rel: float,
) -> Dict[str, np.ndarray]:
    """
    Aplica um gate de desempenho relativo a um baseline.
    Um modelo é considerado elegível se:
        - média dos scores_modelo >= média dos scores_baseline * (1 + gate_rel)

    Args:
        scores: Mapeamento por nome do nome_modelo e scores_por_fold.
        baseline_name: Chave em `scores` que representa o modelo baseline.
        gate_rel: Melhoria relativa mínima exigida vs média do baseline
            (ex.: 0.02 significa +2% relativo).

    Returns:
        Um dicionário filtrado contendo apenas os modelos elegíveis e seus arrays por fold.
        Também imprime uma tabela indicando quem passou/falhou no gate.
    """
    if baseline_name not in scores:
        raise ValueError(f"Baseline '{baseline_name}' não encontrado entre os runs carregados.")

    baseline_mean = float(np.mean(scores[baseline_name]))
    gate = baseline_mean * (1.0 + gate_rel)

    console.rule("[bold]Gate técnico (elegibilidade)[/bold]")
    console.print(f"[dim]Baseline: {baseline_name} | mean={baseline_mean:.6f}[/dim]")
    console.print(f"[dim]Gate: mean >= {gate:.6f} (rel +{gate_rel*100:.2f}%) [/dim]\n")

    eligible = {k: v for k, v in scores.items() if float(np.mean(v)) >= gate}

    table = Table(title="Elegíveis pelo gate")
    table.add_column("model", style="cyan")
    table.add_column("mean_metric", justify="right")
    table.add_column("pass", justify="center")
    for name, v in sorted(scores.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True):
        mean_v = float(np.mean(v))
        ok = "✓" if mean_v >= gate else "✗"
        style = "bold green" if ok == "✓" else "dim"
        table.add_row(name, f"{mean_v:.6f}", f"[{style}]{ok}[/{style}]")
    console.print(table)

    console.print(f"\n[bold]Total elegíveis:[/bold] {len(eligible)} de {len(scores)}")

    return eligible


def register_winner(
    client: MlflowClient,
    run: Run,
    model_name: str,
    registry_name: str,
    alias: str,
    tags: Mapping[str, Any],
) -> ModelVersion:
    """
    Registra o artefato de modelo logado em um run no MLflow,
    define um alias no registry e adiciona tags na versão.
    Assume que o run possui um artefato de modelo logado em `runs:/{run_id}/model`.

    Args:
        client: Instância de `mlflow.tracking.MlflowClient`.
        run: Objeto do run do MLflow cujo artefato de modelo será registrado.
        model_name: Identificador amigável do modelo, usado apenas na mensagem do console.
        registry_name: Nome do Registered Model no MLflow Model Registry.
        alias: Alias a ser definido para a versão criada (ex.: "candidate", "champion").
        tags: Dicionário de tags a serem adicionadas à versão criada do modelo.

    Returns:
        O objeto `ModelVersion` criado.
    """
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, registry_name)
    client.set_registered_model_alias(registry_name, alias, mv.version)
    for key, value in tags.items():
        client.set_model_version_tag(registry_name, mv.version, key, str(value))
    console.print(f"  [green]✓[/green] {model_name} → {registry_name} v{mv.version} (alias: {alias})")

    return mv


def wilcoxon_vs_baseline(
        scores_model: np.ndarray,
        scores_baseline: np.ndarray
) -> Dict:
    """
    Executa o teste pareado de Wilcoxon (signed-rank) entre os scores por fold de dois modelos.
    O teste é aplicado às diferenças por fold (modelo - baseline), assumindo que os vetores
    estão alinhados fold a fold.

    Args:
        scores_model: Array NumPy de tamanho N_FOLDS com scores por fold do modelo.
        scores_baseline: Array NumPy de tamanho N_FOLDS com scores por fold do baseline.

    Returns:
        Um dicionário com:
            - "test": sempre "Wilcoxon".
            - "p_value": p-valor.
            - "stat": estatística do teste.
            - "mean_diff": média(scores_model - scores_baseline).
            - "cohens_d": tamanho de efeito calculado sobre as diferenças por fold.
            - "significant": indica se p_value < ALPHA.
    """
    diff = scores_model - scores_baseline
    if np.allclose(diff, 0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = stats.wilcoxon(scores_model, scores_baseline)
    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    wilcoxon_dict = {
        "test": "Wilcoxon",
        "p_value": float(p_val),
        "stat": float(stat),
        "mean_diff": float(np.mean(diff)),
        "cohens_d": float(d),
        "significant": bool(p_val < ALPHA),
    }

    return wilcoxon_dict


def decide_winner(
    scores: Dict[str, np.ndarray],
    metric: str,
    baseline: str,
    decision: str,
) -> Dict:
    """
    Decide o modelo vencedor final dado o conjunto de scores por fold, utilizando uma das
    políticas de decisão suportadas.

    Políticas:
    - "baseline_duel": compara o melhor modelo por média contra o `baseline` utilizando Wilcoxon.
        Se o desafiante for significativamente melhor (p < ALPHA e mean_diff > 0),
        o desafiante vence; caso contrário, o baseline vence.
        Se o baseline não estiver presente, faz fallback para um duelo top-2.
    - "top2_duel": compara top-1 vs top-2 por média utilizando Wilcoxon. Se o top-1 não for
        significativamente melhor, escolhe o top-2 (critério de parcimônia em caso de empate).

    Args:
        scores: Mapeamento {nome_modelo: scores_por_fold} (arrays de tamanho N_FOLDS).
        metric: Nome da métrica que está sendo otimizada.
        baseline: Nome do modelo baseline a ser usado quando decision="baseline_duel".
        decision: Política de decisão ("baseline_duel" ou "top2_duel").

    Returns:
        Um dicionário descrevendo a decisão, incluindo:
            - "winner", "loser"
            - "duel_a", "duel_b" (modelos comparados)
            - "result" (resultado do teste) ou None
            - "ranked" (modelos ordenados por média desc)
            - "means" (médias por modelo)
            - "reason" (string explicando o motivo)
    """
    names = list(scores.keys())
    ranked = sorted(names, key=lambda n: float(np.mean(scores[n])), reverse=True)

    if decision == "top2_duel":
        if len(ranked) < 2:
            raise ValueError("Precisa de pelo menos 2 modelos para top2_duel.")
        a, b = ranked[0], ranked[1]
        res = wilcoxon_vs_baseline(scores[a], scores[b])

        if res["significant"] and res["mean_diff"] > 0:
            winner, loser = a, b
            reason = "top1_significantly_better_than_top2"
        else:
            winner, loser = b, a
            reason = "no_sig_diff_choose_top2"
        
        top_duel_dict = {
            "winner": winner,
            "loser": loser,
            "baseline": None,
            "challenger": None,
            "duel_a": a,
            "duel_b": b,
            "metric": metric,
            "decision": decision,
            "result": res,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
        }

        return top_duel_dict

    if baseline not in scores:
        ranked = sorted(scores.keys(), key=lambda n: float(np.mean(scores[n])), reverse=True)
        if len(ranked) < 2:
            baseline_dict = {
                "winner": ranked[0],
                "loser": None,
                "baseline": None,
                "challenger": None,
                "duel_a": None,
                "duel_b": None,
                "metric": metric,
                "decision": "top2_duel_fallback_no_baseline",
                "result": None,
                "ranked": ranked,
                "means": {n: float(np.mean(scores[n])) for n in ranked},
                "reason": "baseline_not_eligible",
            }

            return baseline_dict

        a, b = ranked[0], ranked[1]
        res = wilcoxon_vs_baseline(scores[a], scores[b])
        if res["significant"] and res["mean_diff"] > 0:
            winner, loser = a, b
            reason = "top1_significantly_better_than_top2"
        else:
            winner, loser = b, a
            reason = "no_sig_diff_choose_top2"
        return {
            "winner": winner,
            "loser": loser,
            "baseline": None,
            "challenger": None,
            "duel_a": a,
            "duel_b": b,
            "metric": metric,
            "decision": "top2_duel_fallback_no_baseline",
            "result": res,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
            "reason": reason,
        }

    challenger = ranked[0] if ranked[0] != baseline else (ranked[1] if len(ranked) > 1 else baseline)
    if challenger == baseline:
        return {
            "winner": baseline,
            "loser": None,
            "baseline": baseline,
            "challenger": None,
            "duel_a": None,
            "duel_b": None,
            "metric": metric,
            "decision": decision,
            "result": None,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
        }

    res = wilcoxon_vs_baseline(scores[challenger], scores[baseline])

    if res["significant"] and res["mean_diff"] > 0:
        winner = challenger
        loser = baseline
        reason = "challenger_significantly_better_than_baseline"
    else:
        winner = baseline
        loser = challenger
        reason = "no_sig_gain_over_baseline_choose_baseline"

    return {
        "winner": winner,
        "loser": loser,
        "baseline": baseline,
        "challenger": challenger,
        "duel_a": challenger,
        "duel_b": baseline,
        "metric": metric,
        "decision": decision,
        "result": res,
        "ranked": ranked,
        "means": {n: float(np.mean(scores[n])) for n in ranked},
        "reason": reason,
    }


def get_gate_score_recall(
    client: MlflowClient,
    run: Run,
    n_folds: int,
) -> float | None:
    """
    Calcula um score de gate de recall de treino para um run, com fallbacks para suportar
    diferentes pipelines de treino (ex.: sklearn search vs CV manual).

    Ordem de resolução:
    1) Se o run logou a métrica "best_cv_score", retorna esse valor.
    2) Caso contrário, se o run logou "cv_mean_recall".
    3) Caso contrário, tenta reconstruir a média do recall a partir do histórico por fold da métrica "recall"
        utilizando steps 0, n_folds-1.
    Se nenhuma opção estiver disponível, retorna None.

    Args:
        client: Instância de `mlflow.tracking.MlflowClient` usada para consultar histórico de métricas.
        run: Objeto do run do MLflow.
        n_folds: Número esperado de folds.

    Returns:
        Um float com o score de recall para o gate, se possível, caso contrário None.
    """
    v = run.data.metrics.get("best_cv_score")
    if v is not None:
        return float(v)
    v = run.data.metrics.get("cv_mean_recall")
    if v is not None:
        return float(v)

    hist = client.get_metric_history(run.info.run_id, "recall")
    hist = sorted(hist, key=lambda m: (m.step, m.timestamp))
    by_step = {}
    for m in hist:
        if m.step is None:
            continue
        if 0 <= m.step < n_folds:
            by_step[m.step] = m.value
    if len(by_step) == n_folds:
        return float(np.mean([by_step[i] for i in range(n_folds)]))

    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Comparação estatística de modelos (Friedman diagnóstico + duelo final)"
    )
    parser.add_argument("--metric", default=PRIMARY_METRIC, help="Métrica para comparar (ex.: average_precision)")
    parser.add_argument(
        "--decision",
        choices=["baseline_duel", "top2_duel"],
        default="baseline_duel",
        help="Regra de decisão: baseline_duel (top vs baseline) ou top2_duel (top1 vs top2).",
    )
    parser.add_argument("--baseline-run", default="logreg", help="Baseline (model_name), usado em baseline_duel")

    parser.add_argument("--apply-gate", action="store_true", help="Aplicar gate técnico vs baseline (por média)")
    parser.add_argument("--gate-rel", type=float, default=0.02, help="Ganho relativo mínimo vs baseline (média)")

    parser.add_argument(
        "--gate-best-cv-score",
        type=float,
        default=0.0,
        help="Se >0, filtra runs cujo score de gate (best_cv_score/cv_mean_recall/recall por fold) >= este valor.",
    )
    parser.add_argument(
        "--gate-metric-name",
        default="best_cv_score",
        help="Nome da métrica para gate. Se 'best_cv_score', usa fallback inteligente (inclui MLP).",
    )

    parser.add_argument("--register", action="store_true", help="Registrar campeão no MLflow Model Registry")
    parser.add_argument("--registry-name", default="telco-churn-model", help="Nome do Registered Model")

    parser.add_argument("--models", nargs="+", default=None, help="Lista de model_name para comparar (opcional)")

    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    metric = args.metric

    with mlflow.start_run(run_name=f"model_selection__{metric}") as sel_run:
        mlflow.set_tag("run_stage", "selection")
        mlflow.log_param("metric", metric)
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("decision_rule", args.decision)
        mlflow.log_param("baseline_run", args.baseline_run)
        mlflow.log_param("apply_gate", bool(args.apply_gate))
        mlflow.log_param("gate_rel", float(args.gate_rel))
        mlflow.log_param("gate_best_cv_score", float(args.gate_best_cv_score))
        mlflow.log_param("gate_metric_name", args.gate_metric_name)
        mlflow.log_param("register", bool(args.register))
        mlflow.log_param("registry_name", args.registry_name)
        mlflow.log_param("models_filter", " ".join(args.models) if args.models else "")

        runs = get_latest_runs_with_mlp_from_refit(client)
        if len(runs) < 2:
            console.print("[red]Precisa de pelo menos 2 runs.[/red]")
            return

        if args.gate_best_cv_score and args.gate_best_cv_score > 0:
            kept = []
            gate_table = Table(title=f"Gate: {args.gate_metric_name} >= {args.gate_best_cv_score}")
            gate_table.add_column("model_name", style="cyan")
            gate_table.add_column("gate_score", justify="right")
            gate_table.add_column("pass", justify="center")

            for r in runs:
                name = run_display_name(r)

                if args.gate_metric_name == "best_cv_score":
                    score = get_gate_score_recall(client, r, N_FOLDS)
                else:
                    score = r.data.metrics.get(args.gate_metric_name)
                    score = float(score) if score is not None else None

                ok = (score is not None) and (float(score) >= args.gate_best_cv_score)
                gate_table.add_row(name, f"{score:.5f}" if score is not None else "NA", "✓" if ok else "✗")

                if ok:
                    kept.append(r)

            console.print(gate_table)
            runs = kept

            if len(runs) < 2:
                console.print("[red]Gate deixou menos de 2 modelos para comparar.[/red]")
                mlflow.set_tag("status", "gate_left_less_than_2")
                return

        tech_scores: Dict[str, np.ndarray] = {}
        run_map: Dict[str, Any] = {}

        for r in runs:
            name = run_display_name(r)
            s = load_fold_metric(client, r.info.run_id, metric)
            if len(s) == N_FOLDS:
                tech_scores[name] = s
                run_map[name] = r

        names = sorted(tech_scores.keys())
        if len(names) < 2:
            console.print(f"[red]Nenhum run tem '{metric}' logado por fold.[/red]")
            mlflow.set_tag("status", "missing_fold_metrics")
            return

        if args.models:
            wanted = set(args.models)
            tech_scores = {k: v for k, v in tech_scores.items() if k in wanted}
            run_map = {k: v for k, v in run_map.items() if k in wanted}
            names = sorted(tech_scores.keys())

            if len(names) < 2:
                console.print("[red]Precisa de pelo menos 2 modelos após o filtro --models.[/red]")
                console.print(f"[dim]Encontrados: {names} | Solicitados: {sorted(wanted)}[/dim]")
                mlflow.set_tag("status", "models_filter_left_less_than_2")
                return

        mlflow.log_dict({n: run_map[n].info.run_id for n in names}, "selection/compared_runs.json")

        if args.apply_gate:
            tech_scores = apply_gate_filter(tech_scores, args.baseline_run, args.gate_rel)
            names = sorted(tech_scores.keys())
            run_map = {k: v for k, v in run_map.items() if k in set(names)}
            mlflow.log_param("n_models_after_gate", len(names))

            if len(names) < 2:
                console.print("[red]Gate deixou menos de 2 modelos elegíveis.[/red]")
                mlflow.set_tag("status", "apply_gate_left_less_than_2")
                if len(names) == 1:
                    mlflow.set_tag("winner", names[0])
                return

        console.rule(f"[bold]Comparação — métrica={metric} | folds={N_FOLDS} | α={ALPHA}[/bold]")
        t = Table(title="Resumo técnico (por fold)")
        t.add_column("model_name", style="cyan")
        t.add_column(f"mean_{metric}", justify="right")
        t.add_column(f"std_{metric}", justify="right")
        for n in sorted(names, key=lambda k: float(np.mean(tech_scores[k])), reverse=True):
            v = tech_scores[n]
            t.add_row(n, f"{np.mean(v):.6f}", f"{np.std(v, ddof=1):.6f}")
        console.print(t)

        if len(names) >= 3:
            fr = friedman_nemenyi(tech_scores, names)
            mlflow.log_metric("friedman_stat", float(fr["friedman_stat"]))
            mlflow.log_metric("friedman_p", float(fr["friedman_p"]))
            mlflow.set_tag("friedman_significant", str(bool(fr["significant"])))
            if "nemenyi_pvals" in fr:
                tmp_dir = Path("mlflow_artifacts_tmp")
                tmp_dir.mkdir(exist_ok=True)
                p = tmp_dir / f"nemenyi_pvals_{metric}.csv"
                fr["nemenyi_pvals"].to_csv(p, index=True)
                mlflow.log_artifact(str(p), artifact_path="selection/tests")

        decision_out = decide_winner(
            scores=tech_scores,
            metric=metric,
            baseline=args.baseline_run,
            decision=args.decision,
        )

        winner = decision_out["winner"]
        res = decision_out["result"]

        mlflow.set_tag("winner", winner)
        mlflow.set_tag("decision_reason", decision_out.get("reason", ""))
        mlflow.log_dict(decision_out["means"], "selection/means.json")
        mlflow.log_dict({"ranked": decision_out["ranked"]}, "selection/ranking.json")

        if res is not None:
            mlflow.log_param("duel_a", decision_out["duel_a"])
            mlflow.log_param("duel_b", decision_out["duel_b"])
            mlflow.log_metric("duel_p_value", float(res["p_value"]))
            mlflow.log_metric("duel_stat", float(res["stat"]))
            mlflow.log_metric("duel_mean_diff", float(res["mean_diff"]))
            mlflow.log_metric("duel_cohens_d", float(res["cohens_d"]))
            mlflow.set_tag("duel_significant", str(bool(res["significant"])))

        console.print(f"\n[bold green]Winner:[/bold green] {winner}  [dim]({decision_out.get('reason','')})[/dim]")
        if res is not None:
            console.print(
                f"[dim]Duel: {decision_out['duel_a']} vs {decision_out['duel_b']} | "
                f"p={res['p_value']:.6f} Δ={res['mean_diff']:+.6f} d={res['cohens_d']:.3f}[/dim]"
            )

        if args.register:
            if winner not in run_map:
                raise RuntimeError(f"Não achei o run do winner='{winner}' para registrar.")

            tags = {
                "selection_metric": metric,
                f"mean_{metric}": f"{float(np.mean(tech_scores[winner])):.6f}",
                "decision_rule": args.decision,
                "decision_reason": decision_out.get("reason", ""),
            }
            if res is not None:
                tags.update({
                    "duel_a": decision_out["duel_a"],
                    "duel_b": decision_out["duel_b"],
                    "duel_test": res["test"],
                    "duel_p_value": f"{res['p_value']:.6f}",
                    "duel_mean_diff": f"{res['mean_diff']:+.6f}",
                    "duel_significant": str(bool(res["significant"])),
                })

            register_winner(
                client=client,
                run=run_map[winner],
                model_name=winner,
                registry_name=args.registry_name,
                alias="champion",
                tags=tags,
            )

if __name__ == "__main__":
    main()
