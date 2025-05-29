from logging import getLogger, INFO
from logging import Formatter, FileHandler, StreamHandler, Logger
import datetime
import sys
import platform

import torch


def log_setup(save_dir: str, name: str, verbose: bool = True) -> Logger:
    """Setting up the logging format and files.

    Parameters
    ----------
    save_dir: str
        The directory where the logfile will be saved.
    name: str
        The name of the operation (train, inference, interpretation).
    verbose: bool
        Whether of now to printout the logs into console.

    Returns
    -------
    logger: logger
        Logger instance.
    logfile: str
        File to save the logs to.
    """

    # Setting up logging
    current_datetime = datetime.datetime.now()
    str_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

    logger = getLogger()
    logger.setLevel(INFO)
    log_format = "%(asctime)s | %(levelname)s |    %(message)s"
    formatter = Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Remove existing handlers if any
    logger.handlers.clear()

    # Logging to the file
    logfile = f"{save_dir}/{name}_{str_datetime}.log"
    file_handler = FileHandler(filename=logfile, mode="w")
    file_handler.setLevel(INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.handlers

    # Logging to console
    if verbose:
        handler_stdout = StreamHandler(sys.stdout)
        handler_stdout.setLevel(INFO)
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)

    return logger


def torch_compile(model):
    if platform.system() != "Windows":
        if sys.version_info.minor < 11:
            model = torch.compile(model, mode="reduce-overhead")
    return model
