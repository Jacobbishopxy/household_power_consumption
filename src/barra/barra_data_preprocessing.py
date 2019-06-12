"""
@author Jacob
@time 2019/06/05
"""

from typing import Union, List, Dict
import numpy as np
import pandas as pd
import math
from enum import Enum
from sklearn.preprocessing import QuantileTransformer

from data_preprocessing import to_supervised, data_to_tf_records
from utils import files_exist, generate_tf_records_path

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


def transform_labels(data: np.ndarray):
    quantile_transformer = QuantileTransformer(random_state=0)
    return quantile_transformer.fit_transform(data)


def tf_records_preprocessing(n_in: int,
                             n_out: int,
                             raw_data_path: str,
                             tf_records_name: str,
                             feature_cols: Union[List[int], List[List[int]]],
                             label_col: int,
                             factor_type: FactorType,
                             train_size_pct: float = .8):
    train_path, test_path = generate_tf_records_path(tf_records_name)

    if not files_exist([train_path, test_path]):
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

        # trn_lbl = transform_labels(trn_lbl)
        # tst_lbl = transform_labels(tst_lbl)

        data_to_tf_records(trn_fea,
                           trn_lbl,
                           tst_fea,
                           tst_lbl,
                           tf_records_name)
    else:
        print('files already exist, if new records pls rename, or delete current records if updated')


if __name__ == '__main__':
    RAW_DATA_PATH = './data/factor_return.csv'
    TF_RECORDS_NAME = 'barra_style_univariate'

    N_IN, N_OUT, FEATURE_COLS, LABEL_COL = 14, 7, [0], 0

    tf_records_preprocessing(n_in=N_IN,
                             n_out=N_OUT,
                             raw_data_path=RAW_DATA_PATH,
                             tf_records_name=TF_RECORDS_NAME,
                             feature_cols=FEATURE_COLS,
                             label_col=LABEL_COL,
                             factor_type=FactorType.STYLE)
