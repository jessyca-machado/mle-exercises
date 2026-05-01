from typing import Sequence, Optional
import argparse

from src.utils.constants import PRIMARY_METRIC, ALLOWED_METRICS


def build_parser() -> argparse.ArgumentParser:
    """Processa argumentos da linha de comando.

    Os argumentos correspondem aos parâmetros do programa.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric",
        choices=ALLOWED_METRICS,
        default=PRIMARY_METRIC,
        help="Métrica usada para ranquear/avaliar.",
    )
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
    parser.add_argument("--register", action="store_true", help="Registrar campeão no MLflow Model Registry")
    parser.add_argument("--registry-name", default="telco-churn-model", help="Nome do Registered Model")
    parser.add_argument("--models", nargs="+", default=None, help="Lista de model_name para comparar (opcional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold para avaliar net_value/roi")
    parser.add_argument("--topk", type=float, default=0.0, help="Se >0, avalia tratando topk% da base")
    parser.add_argument(
        "--sweep-thresholds",
        nargs=3,
        type=float,
        default=None,
        metavar=("START", "END", "STEP"),
        help="Varre thresholds e escolhe o melhor net_value médio por modelo",
    )
    parser.add_argument("--benefit-tp", type=float, default=10.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument(
        "--roi-on-zero-cost",
        choices=["nan", "inf", "zero"],
        default="nan",
        help="Como tratar ROI quando total_cost=0 (default: nan)",
    )
    parser.add_argument(
        "--business-tag",
        default="",
        help="Tag opcional para diferenciar execuções (ex.: 'v1', 'topk10'). Entra no nome das métricas no parent run.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help=(
            "Cria um NESTED RUN 'business_eval' dentro de cada run de modelo e loga"
            "params + net_value/roi por fold. No PARENT run, loga somente métricas"
            "com namespace para evitar conflitos."
        ),
    )
    parser.add_argument(
        "--gate-metric-name",
        default="best_cv_score",
        help="Nome da métrica no MLflow para o gate (default: best_cv_score).",
    )

    return parser

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Faz o parsing dos argumentos de linha de comando (CLI) e aplica validações básicas.

    Esta função centraliza a definição dos parâmetros aceitos pelo script e converte
    os valores para os tipos corretos, além de validar algumas
    restrições que o ``argparse`` não cobre sozinho.

    Args:
        argv:
            Sequência opcional de argumentos, no mesmo formato de `sys.argv[1:]`
            Exemplos:
                - parse_args(["--metric", "roi", "--threshold", "0.7"])`
                - parse_args() -> (usa a linha de comando real)

    Returns:
        argparse.Namespace
            Objeto com os argumentos parseados. Os nomes seguem o padrão do argparse:
            flags com hífen viram atributos com underscore.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold deve estar entre 0 e 1.")
    if args.topk and not (0.0 < args.topk <= 1.0):
        parser.error("--topk deve estar em (0,1]. Ex.: 0.1 = 10%.")

    if args.sweep_thresholds is not None:
        start, end, step = args.sweep_thresholds
        if step <= 0 or start >= end:
            parser.error("--sweep-thresholds exige START < END e STEP > 0.")

    return args
