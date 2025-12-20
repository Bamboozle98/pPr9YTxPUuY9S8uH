# train_mlp_and_save.py

import joblib
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance
from scipy.stats import loguniform

from src.data.LoadData import load_data  # your existing loader


def main():
    # Load Data Split
    X_train, X_test, y_train, y_test, encoder = load_data()
    feature_names = encoder.get_feature_names_out()

    # Base MLP
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

    # Hyperparameter search space
    param_distributions = {
        "hidden_layer_sizes": [
            (64, 32),
            (128, 64),
            (128, 64, 32),
            (64, 64),
            (32, 16),
        ],
        "alpha": loguniform(1e-5, 1e-2),
        "learning_rate_init": loguniform(1e-4, 5e-3),
        "batch_size": [32, 64, 128, 256],
    }

    # Stratified K-Fold setup
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=72,
    )

    rand_search = RandomizedSearchCV(
        estimator=nn,
        param_distributions=param_distributions,
        n_iter=30,
        cv=skf,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=72,
        verbose=1,
    )

    rand_search.fit(X_train, y_train)

    print("Best params:", rand_search.best_params_)
    print("Best CV balanced accuracy:", rand_search.best_score_)

    # Best model (already refit on all X_train, y_train)
    best_nn = rand_search.best_estimator_

    # Evaluate on test
    test_proba = best_nn.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, test_proba)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest ROC AUC: {roc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Permutation importance (optional, but you already had it)
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    result = permutation_importance(
        best_nn,
        X_test_dense,
        y_test,
        n_repeats=10,
        random_state=72,
        n_jobs=-1,
    )

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\nTop 15 features by importance (permutation):")
    print(importances.head(15))

    # ---- Save model + encoder ----
    # Choose whatever folder you like, e.g. "models/"
    model_path = r"C:\Users\cbran\PycharmProjects\pPr9YTxPUuY9S8uH\src\models\Default\Saved model states\models\mlp_best.joblib"
    encoder_path = r"C:\Users\cbran\PycharmProjects\pPr9YTxPUuY9S8uH\src\models\Default\Saved model states\encoders\encoder.joblib"

    import os
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_nn, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"\nSaved tuned MLP model to: {model_path}")
    print(f"Saved encoder to: {encoder_path}")


if __name__ == "__main__":
    main()
