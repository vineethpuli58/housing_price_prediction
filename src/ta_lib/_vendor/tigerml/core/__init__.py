import os

from .utils import set_logger

# setting the logger
log_dir = os.path.join(os.getcwd(), "logs")
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
