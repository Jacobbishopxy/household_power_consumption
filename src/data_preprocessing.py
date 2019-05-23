"""
@author Jacob
@time 2019/05/22
"""

from typing import Union, List
import numpy as np
import pandas as pd


def read_data_from_csv(path: str):
    return pd.read_csv(path,
                       header=0,
                       infer_datetime_format=True,
                       parse_dates=['datetime'],
                       index_col=['datetime'])


def split_data(data: pd.DataFrame):
    return data.iloc[1:-328, :], data.iloc[-328:-6, :]


def _sliding_window(arr: np.ndarray, window: int, step: int = 1):
    loop = (arr.shape[0] - window) // step + 1
    return np.array([arr[i * step:i * step + window] for i in range(loop)])


def to_supervised(data: pd.DataFrame,
                  n_in: int,
                  n_out: int,
                  feature_cols: Union[List[int], int] = 0,
                  is_train: bool = True):
    """

    :param data:
    :param n_in:
    :param n_out:
    :param feature_cols:
    :param is_train:
    :return:
    """
    cc = [feature_cols] if isinstance(feature_cols, int) else feature_cols
    raw_features_df = data.iloc[:-n_out, cc]
    raw_labels_df = data.iloc[n_in:, 0]

    if is_train:
        n_in_steps = n_out_steps = 1
    else:
        n_in_steps, n_out_steps = n_in, n_out

    features = _sliding_window(raw_features_df.values, window=n_in, step=n_in_steps)
    labels = _sliding_window(raw_labels_df.values, window=n_out, step=n_out_steps)

    return features, labels
