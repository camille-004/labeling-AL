import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import config

sns.set_style(config.PLOT_STYLE)

logger = config.logger


def target_countplot(df: pd.DataFrame):
    """
    Plot the target distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    None
    """
    ax = sns.countplot(
        x=config.TARGET_COL,
        data=df,
        order=df[config.TARGET_COL].value_counts().index,
        palette=config.PLOT_PALETTE,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Genre Distribution")
    plt.show()
    logger.info("Plotted target distribution")
