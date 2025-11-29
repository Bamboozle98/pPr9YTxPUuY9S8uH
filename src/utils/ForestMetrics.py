import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score)

def Metrics(y_test, y_pred, y_proba):
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return roc_auc, f1, prec, recall