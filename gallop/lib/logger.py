from typing import Type, Optional
import os
import sys
import logging
from datetime import datetime

from colorlog import ColoredFormatter

NoneType = Type[None]

LOGGER = logging.getLogger(os.getenv("LOGGING_NAME", "GalLoP"))


def setup_logger(logfile: Optional[str] = None) -> NoneType:
    LOGGER.handlers.clear()
    formatter = ColoredFormatter(
        '[%(cyan)s%(asctime)s%(reset)s][%(light_blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)

    file_warning = False
    if logfile is None:
        # create file with datetime in the name in the logs directory
        os.makedirs('logs', exist_ok=True)
        logfile = os.path.join('logs', f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    elif not os.path.exists(os.path.dirname(logfile)):
        os.makedirs('logs', exist_ok=True)
        orig_logfile = logfile
        logfile = os.path.join('logs', os.path.basename(logfile))
        file_warning = True

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)

    LOGGER.setLevel(logging.INFO)

    if file_warning:
        LOGGER.warning(f"Logfile {orig_logfile} does not exist, defaulting to {logfile}")
