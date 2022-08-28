from pathlib import Path
from typing import Tuple

import pandas as pd

from config import config


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
    config.logger.info("Successfully loaded raw data")
    return train_df, test_df
