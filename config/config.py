import logging.config
from pathlib import Path

from rich.logging import RichHandler

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
LOGS_DIR = Path(BASE_DIR, "logs")

logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)

TARGET_COL = "genre"
PLOT_PALETTE = "hls"
PLOT_STYLE = "whitegrid"
