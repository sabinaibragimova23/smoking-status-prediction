# Smoking Status Prediction

Data Mining final project.

Team: Ibragimova Sabina (team lead), Aitkazin Edige, Ten Danil

## Problem

We predict if a person smokes or not using health check data: blood tests, blood pressure, height, weight, eyesight, hearing and dental results. This is a binary classification task. A model like this could help to find people who may hide that they smoke, for example in health insurance or medical screening.

## Dataset

Smoking (body signal of smoking), Kaggle: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking

55,692 rows and 27 columns, target is `smoking` (1 = smoker, 0 = non-smoker). 11,140 rows are exact duplicates, so after cleaning we have 44,552 rows. We don't use the `competition_format` split from Kaggle, because almost all of its test rows are copies of train rows.

## Setup

```
pip install -r requirements.txt
```

In Google Colab, add this as the first cell of every notebook:

```
!git clone https://github.com/USERNAME/smoking-prediction.git
%cd smoking-prediction/notebooks
import sys; sys.path.append('../src')
```

## File structure

```
data/raw/smoking.csv     original data
data/processed/          train / val / test files
notebooks/               notebooks, run them in order
src/utils.py             shared functions (load_split, evaluate)
models/                  saved models (.pkl)
figures/                 plots for the report and slides
```

## Run order

Run the notebooks from the `notebooks/` folder in this order:

1. `00_split.ipynb`: duplicates, 70/15/15 split (Student 1)
2. `01_preprocessing.ipynb`: outliers, encoding, scaling, new features (Student 2)
3. `02_eda.ipynb`: exploratory analysis and statistical tests (Student 2)
4. `03_feature_selection.ipynb`: feature selection and PCA (Student 3)
5. `04_models_baseline.ipynb`: Dummy, Logistic Regression, Decision Tree, Random Forest (Student 3)
6. `05_xgboost_tuning.ipynb`: XGBoost, class imbalance, tuning (Student 1)
7. `06_final_evaluation.ipynb`: model comparison and final test (Student 1)

## Team contribution

_to be filled in at the end_
