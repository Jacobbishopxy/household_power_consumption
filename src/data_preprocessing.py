"""
@author Jacob
@time 2019/05/22
"""

from typing import Union, List, Dict
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import files_exist, print_features_labels_name, create_file_dir, generate_tf_records_path


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
                  feature_cols: Union[List[int], List[List[int]]],
                  label_col: int,
                  is_train: bool = True) -> (Dict[str, np.ndarray], np.ndarray):
    """

    :param data:
    :param n_in:
    :param n_out:
    :param feature_cols: List[int] -> single head data, List[List[int]] -> multi head data
    :param label_col: 
    :param is_train:
    :return:
    """

    n_steps = 1 if is_train else n_in

    if all(isinstance(i, List) for i in feature_cols):
        raw_features_list = [data.iloc[:-n_out, i].values for i in feature_cols]
        features_value = [_sliding_window(i, window=n_in, step=n_steps) for i in raw_features_list]
        features_key = [f'input_{i}' for i in range(len(feature_cols))]
        features = dict(zip(features_key, features_value))
    elif all(isinstance(i, int) for i in feature_cols):
        raw_features = data.iloc[:-n_out, feature_cols].values
        features_value = _sliding_window(raw_features, window=n_in, step=n_steps)
        features = {'input_0': features_value}
    else:
        raise ValueError("feature_cols has to be List[int] or List[List[int]]")

    raw_labels = data.iloc[n_in:, label_col].values
    labels = _sliding_window(raw_labels, window=n_out, step=n_steps)

    print_features_labels_name(data, feature_cols, label_col)
    return features, labels


def _float_feature(arr: Union[np.ndarray, list]):
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr))


def write_tf_record(filename: str, features: Dict[str, np.ndarray], labels: np.ndarray):
    """
    write features and labels to TFRecord
    :param filename:
    :param features:
    :param labels:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)

    raw_feat_dict = {k: v.reshape([v.shape[0], -1]) for k, v in features.items()}
    for idx in range(len(labels)):
        feat_dict = {k: _float_feature(v[idx]) for k, v in raw_feat_dict.items()}
        feat = tf.train.Features(feature={
            **feat_dict,
            'labels': _float_feature(labels[idx])
        })
        example = tf.train.Example(features=feat)
        writer.write(example.SerializeToString())
    writer.close()


def data_to_tf_records(train_features: Dict[str, np.ndarray],
                       train_labels: np.ndarray,
                       test_features: Dict[str, np.ndarray],
                       test_labels: np.ndarray,
                       tf_records_name: str):
    """
    train & test write to TFRecord
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :param tf_records_name:
    :return:
    """

    train_path, test_path = generate_tf_records_path(tf_records_name)

    create_file_dir(train_path)

    write_tf_record(train_path, train_features, train_labels)
    print('train to TFRecord completed')
    write_tf_record(test_path, test_features, test_labels)
    print('test to TFRecord completed')


def tf_records_preprocessing(n_in: int,
                             n_out: int,
                             raw_data_path: str,
                             tf_records_name: str,
                             feature_cols: Union[List[int], List[List[int]]]):
    """

    :param n_in:
    :param n_out:
    :param raw_data_path:
    :param tf_records_name:
    :param feature_cols:
    :return:
    """

    train_path, test_path = generate_tf_records_path(tf_records_name)

    if not files_exist([train_path, test_path]):
        # read from csv
        d = read_data_from_csv(raw_data_path)
        # split train & test
        raw_trn_data, raw_tst_data = split_data(d)
        # split train/test-x/y
        trn_fea, trn_lbl = to_supervised(raw_trn_data,
                                         n_in,
                                         n_out,
                                         feature_cols=feature_cols,
                                         label_col=0,
                                         is_train=True)
        tst_fea, tst_lbl = to_supervised(raw_tst_data,
                                         n_in,
                                         n_out,
                                         feature_cols=feature_cols,
                                         label_col=0,
                                         is_train=False)
        # write final train & test data to TFRecord
        data_to_tf_records(trn_fea,
                           trn_lbl,
                           tst_fea,
                           tst_lbl,
                           tf_records_name)
    else:
        print('files already exist')


if __name__ == '__main__':
    RAW_DATA_PATH = '../data/household_power_consumption_days.csv'
    TF_RECORDS_NAME = 'multihead'

    N_IN, N_OUT, FEATURE_COLS = 14, 7, [[i] for i in range(8)]

    '''
    write data to TFRecord then read and evaluate 
    '''
    tf_records_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, TF_RECORDS_NAME, feature_cols=FEATURE_COLS)
