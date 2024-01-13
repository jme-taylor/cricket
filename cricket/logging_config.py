import logging
import os
import sys

from cricket.constants import DATA_FOLDER

logger = logging.getLogger("cricket")
logger.setLevel(logging.INFO)

# create a file handler - set level and format
log_filepath = DATA_FOLDER.joinpath("cricket.log")
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
file_handler = logging.FileHandler(log_filepath)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(formatter)

# create a stream handler - set level and format
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
