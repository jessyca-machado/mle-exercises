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
        │   ├── baselines
        │   │   ├── __init__.py
        │   │   └── baseline_model.py
        │   └── deep_learning
        │       ├── __init__.py
        │       ├── torch_mlp.py
        │       └── torch_mlp_process.py
        ├── ml_canvas.exercicios.py
        ├── notebooks
        │   └── eda.ipynb
        ├── pyproject.toml
        ├── requirements.txt
        ├── src
        │   ├── __init__.py
        │   ├── data
        │   │   ├── __init__.py
        │   │   ├── feature_engineering.py
        │   │   ├── load_data.py
        │   │   ├── pipelines.py
        │   │   └── preprocess.py
        │   ├── data_io.py
        │   ├── jobs
        │   │   ├── __init__.py
        │   │   └── train.py
        │   ├── models
        │   │   ├── __init__.py
        │   │   ├── predict_mlp.py
        │   │   └── train_mlp.py
        │   ├── principal.py
        │   ├── tests.py
        │   └── utils
        │       ├── __init__.py
        │       ├── constants.py
        │       └── helpers.py
        └── uv.lock

15 directories, 34 files
```
<!-- TREE:END -->
