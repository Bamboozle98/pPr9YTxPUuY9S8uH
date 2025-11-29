from sklearn.neural_network import MLPClassifier
from src.data.LoadData import load_data
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Load Data Split
X_train, X_test, y_train, y_test, encoder = load_data()
feature_names = encoder.get_feature_names_out()


# Load Random Forest Model
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=72,
    early_stopping=True,
)


# Stratified K-Fold setup
skf = StratifiedKFold(
    n_splits=5,        # 5-fold CV
    shuffle=True,
    random_state=72,
)

# Run CV on TRAINING data only
cv_results = cross_validate(
    nn,
    X_train,
    y_train,
    cv=skf,
    scoring=["accuracy", "balanced_accuracy", "roc_auc", "f1"],
    n_jobs=-1,
    return_train_score=False,
)

print("=== Stratified 5-Fold CV Results (Train Split) ===")
print(f"Accuracy:          {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")
print(f"ROC AUC:           {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"F1 Score:          {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print()


nn.fit(X_train, y_train)

# Predictions
y_proba = nn.predict_proba(X_test)[:, 1]

# Example: lower threshold to favor recall
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)


# Probabilities for ROC AUC (assumes binary classification with labels 0/1)
y_proba = nn.predict_proba(X_test)[:, 1]

# If X_test is sparse, make it dense for permutation importance
X_test_dense = X_test.toarray()

# Feature scoring with permutation importance
result = permutation_importance(
    nn,
    X_test_dense,
    y_test,
    n_repeats=10,
    random_state=72,
    n_jobs=-1,
)


# get feature names from data load
feature_names = encoder.get_feature_names_out().tolist()

importances = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("\nTop 15 features by importance (permutation):")
print(importances.head(15))

# Metrics
roc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC AUC: {roc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
