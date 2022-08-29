from typing import Dict

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
) -> Dict[str, float]:
    """
    Fit a classifier and log the performance metrics.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    y_train : np.ndarray
        Training labels.
    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    dict
       Precision, recall, F1, MCC
    """
    clf = make_pipeline(
        TfidfVectorizer(),
        OneVsRestClassifier(LogisticRegression(random_state=config.SEED)),
    )
    clf.fit(X_train, y_train)

    pred_val = clf.predict(X_val)
    metrics = precision_recall_fscore_support(
        y_val, pred_val, average="weighted"
    )

    performance = {
        "Precision": np.round(metrics[0], 4),
        "Recall": np.round(metrics[1], 4),
        "F1": np.round(metrics[2], 4),
        "MCC": np.round(matthews_corrcoef(y_val, pred_val), 4),
    }

    logger.info(f"Fit {clf[1]}: {performance}")
    return performance
