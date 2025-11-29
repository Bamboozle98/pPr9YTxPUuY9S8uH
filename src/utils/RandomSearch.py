from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform


def random_search_setup(model):
    # Define parameter distributions instead of grids
    param_distributions = {
        "n_estimators": randint(100, 500),      # integers between 100–500
        "max_depth": randint(4, 16),            # integers between 4–16
        # "min_child_weight": randint(1, 6),      # integers between 1–6
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=76)

    # Randomized search setup
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,                              # number of random combos to try
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=76,
        verbose=1,
        return_train_score=False,
    )
    return rs
