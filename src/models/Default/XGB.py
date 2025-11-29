import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from src.data.LoadData import load_data
from src.utils.FeatureImportance import feature_scores
from src.utils.ForestMetrics import Metrics
from src.utils.search_selector import search_selection
from src.utils.class_reports import class_report
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

# Load Data Split
X_train, X_test, y_train, y_test = load_data()


# Base XGB model
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=76,
    n_jobs=-1,
    tree_method="hist",
)

search = search_selection(xgb, selection='gs')  # Grid Search

class_report(y_train, 'train')
class_report(y_test, 'test')

search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

# Get test results for Metrics
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", matrix)

roc_auc, f1, precision, recall = Metrics(y_test, y_pred, y_proba)

print("\nTest metrics:")
print("ROC AUC:", roc_auc)
print("F1:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Feature scoring for pattern tracking.
features = feature_scores(X_train, best)
print("\nTop features:\n", features.head(10))
