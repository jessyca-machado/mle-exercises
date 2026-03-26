import pandas as pd
import numpy as np


def log_class_distribution(
        y: pd.Series,
        name: str
) -> tuple[str, pd.Series, pd.Series]:
    """
    Calcula a distribuição de classes de uma série (target), retornando contagens
    absolutas e proporções (frequências relativas).

    Args:
        y: Série com os rótulos/classes (pode conter NaN).
        name: Nome do conjunto (ex.: "train", "test" ou nome da variável).

    Returns:
        Tupla (name, counts, ratio), onde:
            - name: o mesmo identificador fornecido.
            - counts: contagem absoluta por classe (inclui NaN se existir).
            - ratio: proporção por classe (normalizada para somar 1), arredondada em 4 casas (inclui NaN se existir).
    """

    counts = y.value_counts(dropna=False)
    ratios = y.value_counts(normalize=True, dropna=False).round(4)

    return name, counts, ratios
