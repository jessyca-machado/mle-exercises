# churn_predction


## Pré-requisito: instalar o `uv`

O projeto utiliza o [`uv`](https://github.com/astral-sh/uv) para criar o ambiente virtual e instalar dependências.

### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verifique a instalação:
```bash
uv --version
```

## 📦 **Sobre dependências e lockfile**
- As dependências do projeto estão em `pyproject.toml`.
- O arquivo `uv.lock` registra versões exatas para reproduzir o ambiente.
- **Observação**: para trabalhar com notebooks, o projeto usa `ipykernel` como dependência de desenvolvimento (extra dev).

## 🔧 Configuração do ambiente
1. **Entrar na pasta do projeto:**
```bash
cd projects/churn_prediction
```

2. **Criar a venv:**
```bash
uv venv .venv
```

3. **Instalar as dependências (incluindo as de desenvolvimento):**
Instale as dependências conforme o `uv.lock` (ou gere/atualize o lock quando necessário):
```bash
uv sync --extra dev
```

## 🧠 **Jupyter / Notebooks no VS Code**
4. **Selecionar o interpretador Python do projeto:**
No VS Code:
`Ctrl+Shift+P` → Python: Select Interpreter
Selecione: `./.venv/bin/python` (ou o equivalente no Windows)

5. **Abrir o notebbok e recarregar a janela:**
No VS Code:
`Ctrl+Shift+P` → Developer: Reload Window

6. **Ative o ambiente virtual:**
### Linux/macOS
```bash
source .venv/bin/activate
```

### Windows (PowerShell)
```bash
.\.venv\Scripts\activate.bat
```

7. **Executar a suíte de testes unitários**
```bash
PYTHONPATH=. uv run pytest -v tests/
```

8. **Checagem de qualidade de código (Lint e Formatação)**
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
 ```

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```
