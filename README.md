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
        │   ├── core
        │   │   ├── __init__.py
        │   │   └── models
        │   ├── data
        │   │   ├── __init__.py
        │   │   ├── feature_engineering.py
        │   │   ├── load_data.py
        │   │   ├── preprocess.py
        │   │   └── transformers.py
        │   ├── data_io.py
        │   ├── entrypoints
        │   │   ├── __init__.py
        │   │   └── cli.py
        │   ├── infra
        │   │   ├── __init__.py
        │   │   └── mlflow
        │   ├── jobs
        │   │   ├── __init__.py
        │   │   ├── predict.py
        │   │   └── train.py
        │   ├── ml
        │   │   ├── __init__.py
        │   │   ├── churn_pyfunc_mlp.py
        │   │   ├── churn_pyfunc_xgb.py
        │   │   ├── cost_utils.py
        │   │   ├── data_utils.py
        │   │   ├── experiment_runner.py
        │   │   ├── logging_utils.py
        │   │   ├── metrics_utils.py
        │   │   ├── mlflow_selection_utils.py
        │   │   ├── mlflow_utils.py
        │   │   └── persistence.py
        │   ├── principal.py
        │   └── utils
        │       ├── __init__.py
        │       ├── constants.py
        │       └── helpers.py
        ├── tests
        │   ├── __init__.py
        │   ├── conftest.py
        │   ├── integration
        │   │   ├── __init__.py
        │   │   ├── test_e2e.py
        │   │   └── test_mlflow_logging.py
        │   └── units
        │       ├── __init__.py
        │       ├── test_load_data.py
        │       ├── test_mlflow_fetch_best_params.py
        │       ├── test_preprocessing.py
        │       ├── test_preprocessor_sanity.py
        │       ├── test_pyfunc_contract_unit.py
        │       ├── test_trainer_pipeline.py
        │       └── test_trainer_predict_pyfunc_mode.py
        └── uv.lock

24 directories, 62 files
```
<!-- TREE:END -->
