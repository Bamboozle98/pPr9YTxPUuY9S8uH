import optuna
from optuna.pruners import MedianPruner
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.data.LoadData import load_data
from src.utils.FeatureImportance import feature_scores
from src.utils.ForestMetrics import Metrics


X_train, X_test, y_train, y_test = load_data()

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=76)


def _ix(X, idx):
    """Index helper for pandas DataFrame or numpy array."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


# Base guess for imbalance to center the search of scale_pos_weight
neg = float((np.array(y_train) == 0).sum())
pos = float((np.array(y_train) == 1).sum())
base_spw = (neg / max(pos, 1.0)) if pos > 0 else 1.0


def objective(trial: optuna.Trial) -> float:
    params = {
        # core training setup
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",          # fast & safe default (CPU). Switch to "gpu_hist" if you have CUDA.
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),

        # tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),

        # sampling
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

        # imbalance handling
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            max(1e-6, base_spw * 0.25),
            max(1.0, base_spw * 4.0),
            log=True,
        ),

        # fixed
        "random_state": 76,
        "n_jobs": -1,
    }

    scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = _ix(X_train, tr_idx), _ix(y_train, tr_idx)
        X_va, y_va = _ix(X_train, va_idx), _ix(y_train, va_idx)

        model = XGBClassifier(**params)

        # Early stopping + pruning; model will stop before n_estimators if no AUC improvement
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        # Best AUC from this fold
        score = model.best_score if hasattr(model, "best_score") else roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        scores.append(float(score))

        # Report fold-mean so far for Optuna dashboard/pruning visibility (optional)
        trial.report(float(np.mean(scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study = optuna.create_study(
    study_name="xgb_optuna_auc",
    direction="maximize",
    pruner=MedianPruner(n_warmup_steps=1),
)
study.optimize(objective, n_trials=60, show_progress_bar=False)

print("Best params:", study.best_params)
print("Best CV AUC:", study.best_value)

# ----- fit best model on full train and evaluate like before -----
best = XGBClassifier(
    **study.best_params,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=76,
    n_jobs=-1,
)
best.fit(X_train, y_train)

# Test metrics (unchanged)
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", matrix)

roc_auc, f1, precision, recall = Metrics(y_test, y_pred, y_proba)

print("\nTest metrics:")
print("ROC AUC:", roc_auc)
print("F1:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Feature scoring for pattern tracking (works with XGB feature_importances_)
features = feature_scores(X_train, best)
print("\nTop features:\n", features.head(10))