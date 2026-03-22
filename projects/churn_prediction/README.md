"# churn_predction" 


## 🔧 Configuração
1. **Crie o ambiente virtual e instale as dependências:**
```bash
cd projects/churn_prediction
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

2. **Instale/garanta suporte ao Jupyter no ambiente:**
Com o .venv ativado:
```bash
uv pip install -U ipykernel jupyter
```

3. **Registre o kernel do notebook apontando para o .venv:**
Ainda com o .venv ativado:
```bash
python -m ipykernel install --user --name churn_prediction --display-name "Python (.venv churn_prediction)"
```

4. **Verifique se o kernel foi registrado:**
```bash
jupyter kernelspec list
```

## Rodar notebook
1. Abra o file eda.ipynb
1. Com o notebook aberto, Ctrl+Shift+P → Jupyter: Select Notebook Kernel
2. selecione Python (.venv churn_prediction)
