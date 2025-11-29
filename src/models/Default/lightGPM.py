from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from src.data.LoadData import load_data
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier
import pandas as pd
from src.data.LoadData import load_data

# Load encoded data + encoder
X_train, X_test, y_train, y_test, encoder = load_data()

# Train LightGBM
gbm = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
)
gbm.fit(X_train, y_train)

# Get feature names from the encoder (OneHot + scale + remainder)
feature_names = encoder.get_feature_names_out()

# Built-in importance (by default: split-based)
importances = pd.DataFrame({
    "feature": feature_names,
    "importance": gbm.feature_importances_,
}).sort_values("importance", ascending=False)

print("Top 15 features by LightGBM built-in importance:")
print(importances.head(15))

y_proba = gbm.predict_proba(X_test)[:, 1]   # matrix
y_pred = (y_proba >= 0.5).astype(int)

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("F1:", f1_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

