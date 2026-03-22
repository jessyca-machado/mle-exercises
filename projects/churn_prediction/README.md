# churn_predction


## Pré-requisito: instalar o `uv`

O projeto utiliza o [`uv`](https://github.com/astral-sh/uv) para criar o ambiente virtual e instalar dependências.

### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### Verifique a instalação:
```bash
uv --version
```

## 🔧 Configuração
1. **Crie o ambiente virtual e instale as dependências:**
```bash
cd projects/churn_prediction
uv venv .venv
uv pip install -e .
```

2. **Ative o ambiente virtual:**
### Linux/macOS
```bash
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

3. **Instale/garanta suporte ao Jupyter no ambiente (com o .venv ativado):**
```bash
uv pip install -U ipykernel jupyter
```

4. **Registre o kernel do notebook apontando para o .venv (com o .venv ativado):**
```bash
python -m ipykernel install --user --name churn_prediction --display-name "Python (.venv churn_prediction)"
```

5. **Verifique se o kernel foi registrado:**
```bash
jupyter kernelspec list
```

## Rodar notebook
1. Abra o arquivo `notebooks/eda.ipynb`.
1. Com o notebook aberto, use `Ctrl+Shift+P` → **Jupyter: Select Notebook Kernel**.
2. selecione **Python (.venv churn_prediction)**.
