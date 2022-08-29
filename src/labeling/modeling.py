import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from config import config
from config.config import logger


def clf_eval(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    clf = make_pipeline(
        TfidfVectorizer(),
        OneVsRestClassifier(LogisticRegression(random_state=config.SEED)),
    )
    clf.fit(X_train, y_train)

    pred_val = clf.predict(X_val)
    metrics = precision_recall_fscore_support(
        y_val, pred_val, average="weighted"
    )
    print(y_val.sum().sum())
    performance = {
        "Precision": f"{metrics[0]:0.4f}",
        "Recall": f"{metrics[1]:0.4f}",
        "F1": f"{metrics[2]:0.4f}",
        "MCC": f"{matthews_corrcoef(y_val, pred_val):0.4f}",
    }

    logger.info(f"Fit {clf[1]}: {performance}")
    return performance
