# Shared helper functions for the whole team.
# Please don't change them without telling Student 1, otherwise our results won't be comparable.
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

RANDOM_STATE = 42
TARGET = 'smoking'


def load_split(name, folder="../data/processed"):
    # name: 'train', 'val', 'test' (or 'train_clean' etc.), returns X and y
    df = pd.read_csv(f"{folder}/{name}.csv")
    return df.drop(columns=[TARGET]), df[TARGET]


def evaluate(model, X, y, name='model'):
    # same metrics for every model, so we can put all results in one table
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    row = {'Model': name,
           'Accuracy': accuracy_score(y, pred),
           'Precision': precision_score(y, pred, zero_division=0),
           'Recall': recall_score(y, pred),
           'F1': f1_score(y, pred),
           'ROC-AUC': roc_auc_score(y, proba),
           'PR-AUC': average_precision_score(y, proba)}
    print(name, "| confusion matrix:\n", confusion_matrix(y, pred))
    return row
