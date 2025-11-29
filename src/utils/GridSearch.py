import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix


def grid_search_setup(model):
    # Hyperparameter Grid for CV search
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 24],
        "min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=76) # K-Fold Validation

    # Load hyperparameter grid search
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    return gs
