import random

import numpy as np
import yaml

from config import config


def load_dict(fp):
    """Load a dictionary from a YAML file's path."""
    with open(fp) as f:
        d = yaml.safe_load(f)
    return d


def save_dict(d, fp):
    """Save a dictionary as a YAML file to a specific location."""
    with open(fp, "w") as out:
        yaml.dump(d, out, default_flow_style=False)


def set_seeds(seed=config.SEED):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
