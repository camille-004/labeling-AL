import string
from pathlib import Path
from typing import Tuple

import contractions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import config
from config.config import logger


def load_raw_data(
    train_fp: str, test_fp: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from text files as pandas DataFrames.

    Parameters
    ----------
    train_fp : str
        Name of file containing training data.
    test_fp : str
        Name of file containing testing data.

    Returns
    -------
    tuple
        Training and testing DataFrames.
    """
    train_df = pd.read_csv(
        Path(config.DATA_DIR, train_fp),
        delimiter=" ::: ",
        header=None,
        index_col=0,
        engine="python",
        names=["title", config.TARGET_COL, config.TEXT_COL],
    ).reset_index(drop=True)
    test_df = pd.read_csv(
        Path(config.DATA_DIR, test_fp),
        delimiter=" ::: ",
        header=None,
        index_col=0,
        engine="python",
        names=["title", config.TARGET_COL, config.TEXT_COL],
    ).reset_index(drop=True)

    logger.info(
        f"Successfully loaded raw data: training data {train_df.shape}, testing data {test_df.shape}"
    )
    return train_df, test_df


def unlabel_data(
    df: pd.DataFrame, n_each: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove most labels in the input dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_each : int
        How many examples of each genre to keep labeled.

    Returns
    -------
    tuple
        Labeled and unlabeled DataFrames.
    """
    labeled = df.groupby(config.TARGET_COL).head(n_each)
    unlabeled = df[~df.index.isin(labeled.index)]
    del unlabeled[config.TARGET_COL]
    unlabeled = unlabeled.assign(genre=np.nan)

    logger.info(
        f"Generated new dataset with empty labels. Percentage of data points that are labeled: {np.round(len(labeled) / (len(labeled) + len(unlabeled)), 4) * 100}%"
    )
    return labeled, unlabeled


def clean_text(text: str) -> str:
    """
    Perform basic text cleaning.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned text
    """
    text = text.lower()  # Lowercase text
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    text = "".join([i for i in text if not i.isdigit()])  # Remove numbers
    text = " ".join(text.split())  # Remove extra whitespace
    text = contractions.fix(text)  # Remove contractions
    return text


def view_examples(df: pd.DataFrame, n: int = 5) -> None:
    """
    Query n genres and synopses to see whether it makes sense to guess genres from given tokens.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n : int
        Number of examples to view.

    Returns
    -------
    None
    """
    idx = np.random.choice(range(len(df.head(n))), size=n, replace=False)

    for i in idx:
        print(df.iloc[i][config.TARGET_COL])
        print(df.iloc[i][config.TEXT_COL], end="\n\n")


def split_transform_data(
    labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the labeled data into training and validation sets. Transform the target column with a MultiLabelBinarizer. Get the pooling dataset from unlabeled DataFrame.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Input labeled DataFrame.
    unlabeled_df : pd.DataFrame
        Input unlabeled DataFrame.

    Returns
    -------
    tuple
        Training, validation, and pooling sets
    """
    X, y = labeled_df[config.TEXT_COL], labeled_df[config.TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_val = enc.transform(y_val)
    X_pool = unlabeled_df[config.TEXT_COL].values
    logger.info(
        f"Split dataset into training, validation, and pool. # train, validation samples: {len(X_train)}, {len(X_val)}"
    )
    return X_train, X_val, y_train, y_val, X_pool
