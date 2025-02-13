import os, sys
import logging
import inspect
from pathlib import Path
import pytz
from os.path import join, splitext, basename
from datetime import datetime


ROOT_PATH = str(Path(__file__).resolve().parents[1])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
print(f"\nROOT_PATH: {ROOT_PATH}")


def get_caller_filename():
    caller_frame = inspect.stack()[1]
    caller_filename = splitext(basename(caller_frame.filename))[0]
    return caller_filename


def create_logger(logger_nm: str = None, logs_dir: str = join(ROOT_PATH, '_logs'), log_file_nm: str = None, log_level: int = logging.DEBUG, console_level: int = logging.DEBUG):
    # set logger name
    logger_nm = logger_nm or get_caller_filename()

    # create the logger
    logger = logging.getLogger(logger_nm)

    # set logger level
    logger.setLevel(log_level)

    # if the logger already has handlers, return it without adding more handlers
    if logger.handlers:
        return logger
    
    # format the time
    curr_time = datetime.now(pytz.timezone('Israel')).strftime('%H-%M-%S')

    # set formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(FuncName)s] %(levelname)s - %(message)s')

    # create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)

    # set log file name
    log_file_nm = f"{log_file_nm}.log" if log_file_nm else f"{logger_nm}_{curr_time}.log"

    # set log file path and remove log file if it already exists
    log_file_path = join(logs_dir, log_file_nm)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    return logger