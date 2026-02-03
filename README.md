# MLOps Final Project (Checkpoint 1)

## Project description
This repository is a simple end-to-end baseline ML project built to practice MLOps foundations: reproducible environment with uv, clean project structure, and a runnable training script. [file:1]

## Task definition
- Task: Multiclass classification. [web:106]
- Goal: Predict the Iris flower species from four numeric measurements (sepal length/width, petal length/width). [web:106]
- Baseline metric: Accuracy. [file:1]

## Dataset source
We use the built-in Iris dataset provided by scikit-learn via `sklearn.datasets.load_iris` (150 samples, 3 classes). [web:106]

## Project structure
- `src/myproj/train.py`: Loads data, splits train/test, applies basic preprocessing, trains a baseline model, prints accuracy. [file:1]
- `pyproject.toml` + `uv.lock`: Dependency management and reproducibility with uv. [file:1]

## How to run (Checkpoint 1)
From the repository root:
```bash
uv run python src/myproj/train.py
