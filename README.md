"# mle-exercises" 

## Estrutura do projeto

<!-- TREE:START -->
```text
.
├── .github
│   └── workflows
│       └── update-tree.yml
├── .gitignore
├── .vscode
│   └── settings.json
├── README.md
└── projects
    ├── README.md
    └── churn_prediction
        ├── README.md
        ├── experiments
        │   ├── __init__.py
        │   ├── comparison
        │   │   ├── __init__.py
        │   │   └── train_sklearn.py
        │   ├── deep_learning
        │   │   ├── __init__.py
        │   │   └── train_mlp_torch.py
        │   └── selection
        │       ├── __init__.py
        │       ├── compare_models.py
        │       └── cost_toolkit_metrics.py
        ├── ml_canvas.exercicios.py
        ├── notebooks
        │   └── eda.ipynb
        ├── pyproject.toml
        ├── requirements-mlflow.txt
        ├── requirements.txt
        ├── src
        │   ├── __init__.py
        │   ├── data
        │   │   ├── __init__.py
        │   │   ├── feature_engineering.py
        │   │   ├── load_data.py
        │   │   ├── pipelines.py
        │   │   ├── preprocess.py
        │   │   └── transformers.py
        │   ├── data_io.py
        │   ├── jobs
        │   │   ├── __init__.py
        │   │   └── train.py
        │   ├── ml
        │   │   ├── __init__.py
        │   │   ├── churn_pyfunc.py
        │   │   ├── cost_utils.py
        │   │   ├── data_utils.py
        │   │   ├── experiment_runner.py
        │   │   ├── logging_utils.py
        │   │   ├── metrics_utils.py
        │   │   ├── mlflow_selection_utils.py
        │   │   ├── mlflow_utils.py
        │   │   └── persistence.py
        │   ├── principal.py
        │   ├── tests.py
        │   └── utils
        │       ├── __init__.py
        │       ├── constants.py
        │       └── helpers.py
        └── uv.lock

16 directories, 45 files
```
<!-- TREE:END -->
