"""
@author Jacob
@time 2019/06/05
"""

from typing import Union, List, Dict
import pandas as pd
import math
from enum import Enum

from data_preprocessing import to_supervised, data_to_tf_record
from utils import files_exist

FACTOR_TYPE_DICT = pd.read_pickle('./data/factor_type.pkl')


class FactorType(Enum):
    INDUSTRY = [i for i, j in FACTOR_TYPE_DICT.items() if j == 'industry']
    STYLE = [i for i, j in FACTOR_TYPE_DICT.items() if j == 'style']
    COUNTRY = [i for i, j in FACTOR_TYPE_DICT.items() if j == 'country']
    ALL = list(FACTOR_TYPE_DICT.keys())


def read_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop('date', axis=1)


def get_data_by_type(data: pd.DataFrame, factor_type: FactorType) -> pd.DataFrame:
    return data[factor_type.value]


def split_data(data: pd.DataFrame, train_size_pct: float):
    loc = math.floor(data.shape[0] * train_size_pct)
    return data.iloc[:loc, :], data.iloc[loc:, :]


def tf_record_preprocessing(n_in: int,
                            n_out: int,
                            raw_data_path: str,
                            file_train_path: str,
                            file_test_path: str,
                            feature_cols: Union[List[int], List[List[int]]],
                            label_col: int,
                            factor_type: FactorType,
                            train_size_pct: float = .8):
    if not files_exist([file_train_path, file_test_path]):
        # read from csv
        d = read_data_from_csv(raw_data_path)
        # select factor type
        d = get_data_by_type(d, factor_type=factor_type)
        # split train & test
        raw_trn_data, raw_tst_data = split_data(d, train_size_pct=train_size_pct)
        # split train/test-x/y
        trn_fea, trn_lbl = to_supervised(raw_trn_data,
                                         n_in,
                                         n_out,
                                         label_col=label_col,
                                         feature_cols=feature_cols,
                                         is_train=True)

        tst_fea, tst_lbl = to_supervised(raw_tst_data,
                                         n_in,
                                         n_out,
                                         label_col=label_col,
                                         feature_cols=feature_cols,
                                         is_train=True)

        data_to_tf_record(trn_fea,
                          trn_lbl,
                          tst_fea,
                          tst_lbl,
                          file_train_path,
                          file_test_path)
    else:
        print('files already exist, if new records pls rename')


if __name__ == '__main__':
    RAW_DATA_PATH = './data/factor_return.csv'
    FILE_TRAIN = './tmp/univariate_style_train.tfrecords'
    FILE_TEST = './tmp/univariate_style_test.tfrecords'

    N_IN, N_OUT, FEATURE_COLS, LABEL_COL = 14, 7, [0], 0

    tf_record_preprocessing(n_in=N_IN,
                            n_out=N_OUT,
                            raw_data_path=RAW_DATA_PATH,
                            file_train_path=FILE_TRAIN,
                            file_test_path=FILE_TEST,
                            feature_cols=FEATURE_COLS,
                            label_col=LABEL_COL,
                            factor_type=FactorType.STYLE)
