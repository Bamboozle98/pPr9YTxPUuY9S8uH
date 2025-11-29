import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src.data.LoadData import load_data
from src.utils.FeatureImportance import feature_scores
from src.utils.ForestMetrics import Metrics
from src.utils.search_selector import search_selection
from src.utils.class_reports import class_report


# Load Data Split
X_train, X_test, y_train, y_test = load_data()

# Load Random Forest Model
rf = RandomForestClassifier(
    random_state=76,  # Static state for reproducibility
    n_jobs=-1,
    bootstrap=True,
    max_features="sqrt"
)

search = search_selection(rf, selection='rs')

class_report(y_train, 'train')
class_report(y_test, 'test')

# GRID SEARCH RESULTS
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

