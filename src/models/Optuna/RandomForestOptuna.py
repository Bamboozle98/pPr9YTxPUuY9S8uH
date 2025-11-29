import optuna
from optuna.pruners import MedianPruner
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from src.data.LoadData import load_data
from src.utils.FeatureImportance import feature_scores
from src.utils.ForestMetrics import Metrics


X_train, X_test, y_train, y_test = load_data()

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=76)


def _ix(X, idx):
    # Index helper for pandas DataFrame or numpy array.
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 76,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)

    scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = _ix(X_train, tr_idx), _ix(y_train, tr_idx)
        X_va, y_va = _ix(X_train, va_idx), _ix(y_train, va_idx)

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, proba)
        scores.append(score)

        # Report intermediate result for pruning
        trial.report(float(np.mean(scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study = optuna.create_study(
    study_name="rf_optuna_auc",
    direction="maximize",
    pruner=MedianPruner(n_warmup_steps=1),
)
study.optimize(objective, n_trials=60, show_progress_bar=False)

print("Best params:", study.best_params)
print("Best CV AUC:", study.best_value)

# Prepare Random Forest Classifier
best = RandomForestClassifier(
    **study.best_params, random_state=76, n_jobs=-1
).fit(X_train, y_train)

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

# Feature scoring (unchanged)
features = feature_scores(X_train, best)
print("\nTop features:\n", features.head(10))
