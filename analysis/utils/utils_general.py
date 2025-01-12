from os.path import basename, dirname
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple


def order_df(df, first_cols=None, last_cols=None, sort_by=None, ascending=True):
    first_cols = first_cols if first_cols else []
    last_cols = last_cols if last_cols else []
    middle_cols = [c for c in df.columns.values if c not in first_cols + last_cols]
    df = df[first_cols + middle_cols + last_cols]
    return df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True) if sort_by else df


def read_df(file_path, encoding='utf-8'):
    if ".csv" in basename(file_path):
        df = pd.read_csv(file_path, encoding=encoding)
    elif ".pickle" in basename(file_path):
        with open(file_path, 'rb') as handle:
            df = pickle.load(handle)
    elif ".txt" in basename(file_path):
        df = pd.read_csv(file_path, delimiter=" ")
    else:
        raise ValueError(f"{basename(file_path)} NOT SUPPORTED - only .csv OR .txt")
    return df


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def write_df(df, file_path, encoding='utf-8', keep_index=False):
    create_dir_if_not_exists(dirname(file_path))
    if ".csv" in basename(file_path):
        with open(file_path, 'w') as handle:
            df.to_csv(handle, encoding=encoding, index=keep_index)
    else:
        raise ValueError(f"{basename(file_path)} NOT SUPPORTED - only .csv")
    return
