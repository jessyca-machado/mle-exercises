"""Feature Engineering - Criação de novas features para melhorar o modelo de churn.

- Cria features derivadas (TotalChargesPerMonth, ltv).
- Carrega dados, pré-processa e aplica engenharia de features.

Uso:
    python src/data/feature_engineering.py
"""
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import YES_NO_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def feature_engineering(
    df: pd.DataFrame,
    stage_name: str,
) -> pd.DataFrame:
    """Realiza a engenharia de features no dataset Telco Customer Churn da IBM.

    Aplica transformação de variáveis numéricas e criação de novas features.

    Args:
        df: Dataframe a ser transformado.

    Returns:
        Dataframe com as novas features criadas.
    """
    df_feature_engineering = df.copy()

    epsilon = 1e-6

    df_feature_engineering['TotalChargesPerMonth'] = df_feature_engineering['TotalCharges'] / (df_feature_engineering['tenure'] + epsilon)

    df_feature_engineering['ltv'] = df_feature_engineering['MonthlyCharges'] * df_feature_engineering['tenure']

    df_feature_engineering["MonthlyCharges_group"] = pd.qcut(
        df_feature_engineering["MonthlyCharges"],
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(int)

    df_feature_engineering["TotalCharges_group"] = pd.qcut(
        df_feature_engineering["TotalCharges"],
        q=10,
        labels=False,
        duplicates="drop",
    ).astype(int)

    df_feature_engineering["onePlusYearCustomer"] = df_feature_engineering["tenure"].apply(lambda x : 1 if x > 12 else 0)

    df_feature_engineering['MonthlyCharges_squared'] = df_feature_engineering['MonthlyCharges'] ** 2

    logger.info("\n--- %s ---", stage_name)
    logger.info("First 10 rows of df_feature_engineering:\n%s", df_feature_engineering.head(10).to_string(index=False))

    return df_feature_engineering


class TelcoFeatureEngineeringBins(BaseEstimator, TransformerMixin):
    """
    Feature engineering para Telco Churn com bins aprendidos no treino.

    - Aprende os cortes (bin edges) no fit via quantis do TREINO
    - Aplica no transform via pd.cut (consistente no teste)
    - Cria as mesmas features da sua função original

    Observações:
    - Garante que valores fora do range do treino caiam no bin 0 ou no último bin.
    - Retorna DataFrame (bom para ColumnTransformer com nomes).
    """

    def __init__(
        self,
        monthlycharges_q=5,
        totalcharges_q=10,
        epsilon=1e-6,
        create_logs=False,
    ):
        self.monthlycharges_q = monthlycharges_q
        self.totalcharges_q = totalcharges_q
        self.epsilon = epsilon
        self.create_logs = create_logs

        self.monthly_edges_ = None
        self.total_edges_ = None

    def _quantile_edges(self, s: pd.Series, q: int):
        """
        Retorna edges monotônicos e únicos para q bins.
        Usa quantis do treino e remove duplicados.

        Args:
            s: Série numérica para calcular os edges.
            q: Número de bins desejados.
        
        Returns:
            Array de edges para os bins, com pelo menos 2 edges distintos.
        """
        s = pd.to_numeric(s, errors="coerce")
        probs = np.linspace(0, 1, q + 1)
        edges = s.quantile(probs).to_numpy(dtype=float)

        edges = edges[~np.isnan(edges)]
        edges = np.unique(edges)

        if edges.size < 2:
            v = float(np.nanmedian(s))
            edges = np.array([v - 1.0, v + 1.0], dtype=float)

        return edges

    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende os cortes (bin edges) no treino via quantis.

        Args:
            X: DataFrame com as colunas MonthlyCharges e TotalCharges.
            y: Não utilizado.

        Returns:
            self: O próprio objeto TelcoFeatureEngineeringBins.
        """
        X = X.copy()

        if "MonthlyCharges" not in X.columns or "TotalCharges" not in X.columns:
            raise ValueError("Esperado colunas MonthlyCharges e TotalCharges no DataFrame.")

        self.monthly_edges_ = self._quantile_edges(X["MonthlyCharges"], self.monthlycharges_q)
        self.total_edges_ = self._quantile_edges(X["TotalCharges"], self.totalcharges_q)

        return self

    def _apply_bins(self, s: pd.Series, edges: np.ndarray):
        """
        Aplica bins fixos via pd.cut e devolve inteiros [0..n_bins-1].
        Inclui clipping para valores fora do range do treino.

        Args:
            s: Série numérica a ser binned.
            edges: Array de edges para os bins (deve ser monotônico e sem duplicados).
        
        Returns:
            Série de inteiros representando os bins, com -1 para valores NaN.
        """
        s = pd.to_numeric(s, errors="coerce")

        lo, hi = edges[0], edges[-1]
        s2 = s.clip(lower=lo, upper=hi)

        b = pd.cut(s2, bins=edges, include_lowest=True, labels=False)

        return b.fillna(-1).astype(int)

    def transform(self, X: pd.DataFrame):
        """
        Aplica a engenharia de features usando os bins aprendidos no fit.
        
        Args:
            X: DataFrame a ser transformado, deve conter as colunas MonthlyCharges e TotalCharges.
        
        Returns:
            DataFrame com as novas features criadas.
        """
        if self.monthly_edges_ is None or self.total_edges_ is None:
            raise RuntimeError("Transformer não foi fitado. Chame fit antes de transform.")

        df = X.copy()

        eps = self.epsilon
        df["TotalChargesPerMonth"] = pd.to_numeric(df["TotalCharges"], errors="coerce") / (
            pd.to_numeric(df["tenure"], errors="coerce") + eps
        )
        df["ltv"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce") * pd.to_numeric(df["tenure"], errors="coerce")

        df["MonthlyCharges_group"] = self._apply_bins(df["MonthlyCharges"], self.monthly_edges_)
        df["TotalCharges_group"] = self._apply_bins(df["TotalCharges"], self.total_edges_)

        df["onePlusYearCustomer"] = pd.to_numeric(df["tenure"], errors="coerce").apply(lambda x: 1 if x > 12 else 0)

        df["MonthlyCharges_squared"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce") ** 2

        return df


def main() -> None:
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")

    fe = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    df_feature_engineering = fe.fit_transform(df_clean)
    print(df_feature_engineering.head())

if __name__ == "__main__":
    main()
