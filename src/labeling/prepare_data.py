import string
from pathlib import Path
from typing import Tuple

import contractions
import numpy as np
import pandas as pd

from config import config

logger = config.logger


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
        names=["title", "genre", "synopsis"],
    ).reset_index(drop=True)
    test_df = pd.read_csv(
        Path(config.DATA_DIR, test_fp),
        delimiter=" ::: ",
        header=None,
        index_col=0,
        engine="python",
        names=["title", "genre", "synopsis"],
    ).reset_index(drop=True)

    logger.info(
        f"Successfully loaded raw data: training data {train_df.shape}, testing data {test_df.shape}"
    )
    return train_df, test_df


def unlabel_data(df):
    """
    Remove most labels in the input dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        New DataFrame with mostly empty labels.
    """
    labeled = df.groupby(config.TARGET_COL).head(2)
    unlabeled = df[~df.index.isin(labeled.index)]
    del unlabeled[config.TARGET_COL]
    unlabeled = unlabeled.assign(genre=np.nan)
    combined = pd.concat([labeled, unlabeled])

    logger.info(
        f"Generated new dataset with empty labels. Percentage of data points that are labeled: {np.round(len(labeled) / len(combined), 4) * 100}%"
    )
    return combined


def clean_text(text):
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
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join([i for i in text if not i.isdigit()])
    text = " ".join(text.split())
    text = contractions.fix(text)
    return text
