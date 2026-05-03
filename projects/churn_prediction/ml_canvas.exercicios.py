import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MLCanvas:
    """Representação do ML Canvas para um projeto de ML.

    Attributes:
        project_name: Nome do projeto.
        business_problem: Descrição do problema de negócio.
        ml_task: Tipo de tarefa ML (classificação, regressão, etc.).
        success_metrics_business: Métricas de negócio de sucesso do projeto.
        success_metrics_technical: Métricas técnicas de sucesso do projeto.
        data_sources: Fontes de dados disponíveis.
        features: Features candidatas.
        target: Variável alvo.
        constraints: Restrições do projeto.
        risks: Riscos identificados.
    """

    project_name: str = ""
    business_problem: str = ""
    ml_task: str = ""
    success_metrics_business: list[str] = field(default_factory=list)
    success_metrics_technical: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    target: str = ""
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def data_readiness_score(self) -> float:
        """Calcula score de prontidão dos dados (0.0 a 1.0).

        Returns:
            Score entre 0.0 (sem dados) e 1.0 (totalmente pronto).
        """
        checks = [
            bool(self.data_sources),
            bool(self.features),
            bool(self.target),
            len(self.data_sources) >= 2,
            len(self.features) >= 3,
        ]
        return sum(checks) / len(checks)

    def is_viable(self) -> bool:
        """Verifica se o projeto possui os elementos mínimos definidos.

        Returns:
            True se o projeto está minimamente definido, False caso contrário.
        """
        return all(
            [
                self.project_name,
                self.business_problem,
                self.ml_task,
                self.target,
                self.success_metrics_business,
                self.success_metrics_technical,
            ]
        )

    def display(self) -> None:
        """Exibe o canvas formatado no log."""
        logger.info("=" * 60)
        logger.info("ML CANVAS — %s", self.project_name)
        logger.info("=" * 60)
        logger.info("Problema de negócio: %s", self.business_problem)
        logger.info("Tarefa ML: %s", self.ml_task)
        logger.info("Variável alvo: %s", self.target)
        logger.info("Métricas de sucesso (Negócio): %s", ", ".join(self.success_metrics_business))
        logger.info("Métricas de sucesso (Técnicas): %s", ", ".join(self.success_metrics_technical))
        logger.info("Fontes de dados: %s", ", ".join(self.data_sources))
        logger.info("Features candidatas: %s", ", ".join(self.features))
        logger.info("Restrições: %s", ", ".join(self.constraints) or "Nenhuma")
        logger.info("Riscos: %s", ", ".join(self.risks) or "Nenhum")
        logger.info("-" * 60)
        score = self.data_readiness_score()
        logger.info("Data Readiness Score: %.0f%%", score * 100)
        logger.info("Projeto viável: %s", "✓" if self.is_viable() else "✗")


def create_predition_churn_canvas() -> MLCanvas:
    """Cria ML Canvas para o dataset Telco Customer Churn da IBM..

    Returns:
        MLCanvas preenchido com dados do projeto Telco Customer Churn da IBM.
    """
    return MLCanvas(
        project_name="Previsão de Churn — Telco customer churn",
        business_problem=("Aumento no churn de clientes em uma empresa de telecomunicações."),
        ml_task="Classificação binária (churnou: 0/1)",
        success_metrics_business=[
            "ROI > 150% em 6 meses vs. 12 meses",
            "Maximiazr valor líquido em $500K/ano",
        ],
        success_metrics_technical=["PR-AUC deve ser pelo menos +2% maior que o baseline"],
        data_sources=["Telco-Customer-Churn.csv (IBM)"],
        features=[
            "SeniorCitizen",
            "gender",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ],
        target="Churn",
        constraints=[
            "Dados históricos — sem possibilidade de coletar mais",
            "Latência de predição < 100ms",
        ],
        risks=[
            "Features pouco preditivas e correlações fracas como target",
        ],
    )


if __name__ == "__main__":
    canvas = create_predition_churn_canvas()
    canvas.display()
