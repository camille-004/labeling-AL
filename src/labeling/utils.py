import random
import yaml

import numpy as np


def load_dict(fp):
    """Load a dictionary from a YAML file's path."""
    with open(fp, "r") as f:
        d = yaml.safe_load(f)
    return d


def save_dict(d, fp):
    """Save a dictionary as a YAML file to a specific location."""
    with open(fp, "w") as out:
        yaml.dump(d, out, default_flow_style=False)


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
