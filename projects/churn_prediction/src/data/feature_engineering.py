"""Feature Engineering - Criação de novas features para melhorar o modelo de churn.

- Cria features derivadas (TotalChargesPerMonth, ltv).
- Carrega dados, pré-processa e aplica engenharia de features.

Uso:
    python src/data/feature_engineering.py
"""
import logging
from operator import le
import pandas as pd

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

def main() -> None:
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")
    df_feature_engineering = feature_engineering(df_clean, "Dataset after feature engineering")

if __name__ == "__main__":
    main()
