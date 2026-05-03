from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def save_confusion_matrix_artifacts(
    *,
    y_true,
    y_pred,
    out_dir: str | Path,
    labels=(0, 1),
    normalize: str | None = None,  # None, "true", "pred", "all"
    prefix: str = "confusion_matrix",
) -> dict[str, Path]:
    """
    Salva matriz de confusão em CSV e PNG.

    Args:
        y_true: rótulos verdadeiros.
        y_pred: rótulos preditos.
        out_dir: diretório onde os arquivos serão salvos.
        labels: lista de rótulos a considerar na matriz de confusão.
        normalize: tipo de normalização para a matriz de confusão (None, "true", "pred", "all").
        prefix: prefixo para os nomes dos arquivos gerados.

    Returns:
        paths gerados.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(labels), normalize=normalize)

    csv_path = out_dir / f"{prefix}.csv"
    df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    df.to_csv(csv_path)

    png_path = out_dir / f"{prefix}.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=list(labels))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format=".3f" if normalize else "d", colorbar=False)
    ax.set_title("Confusion Matrix" + (f" (normalize={normalize})" if normalize else ""))
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {"csv": csv_path, "png": png_path}
