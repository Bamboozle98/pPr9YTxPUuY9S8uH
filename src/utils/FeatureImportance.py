import pandas as pd

def feature_scores(X_train, best):
    fi = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return fi
