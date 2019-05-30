"""
@author Jacob
@time 2019/05/22
"""

from typing import Union, List
import numpy as np
import pandas as pd
import tensorflow as tf


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


def _float_feature(arr: Union[np.ndarray, list]):
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr))


def write_tf_record(filename: str, features: np.ndarray, labels: np.ndarray):
    """
    write features and labels to TFRecord
    :param filename:
    :param features:
    :param labels:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(features)):
        feat = tf.train.Features(feature={
            'features': _float_feature(features[i]),
            'labels': _float_feature(labels[i])
        })
        example = tf.train.Example(features=feat)
        writer.write(example.SerializeToString())
    writer.close()


def data_to_tf_record(train_features: np.ndarray,
                      train_labels: np.ndarray,
                      test_features: np.ndarray,
                      test_labels: np.ndarray,
                      train_path: str,
                      test_path: str):
    """
    train & test write to TFRecord
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :param train_path:
    :param test_path:
    :return:
    """
    train_features_vec = train_features.reshape([train_features.shape[0], -1])
    test_features_vec = test_features.reshape([test_features.shape[0], -1])

    write_tf_record(train_path, train_features_vec, train_labels)
    print('train to TFRecord completed')
    write_tf_record(test_path, test_features_vec, test_labels)
    print('test to TFRecord completed')


def tf_record_preprocessing(n_in: int,
                            n_out: int,
                            raw_data_path: str,
                            file_train_path: str,
                            file_test_path: str,
                            feature_cols: Union[List[int], int] = 0):
    """

    :param n_in:
    :param n_out:
    :param raw_data_path:
    :param file_train_path:
    :param file_test_path:
    :param feature_cols:
    :return:
    """
    # read from csv
    d = read_data_from_csv(raw_data_path)
    # split train & test
    raw_trn_data, raw_tst_data = split_data(d)
    # split train/test-x/y
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)
    # write final train & test data to TFRecord
    data_to_tf_record(trn_fea,
                      trn_lbl,
                      tst_fea,
                      tst_lbl,
                      file_train_path,
                      file_test_path)


if __name__ == '__main__':
    RAW_DATA_PATH = '../data/household_power_consumption_days.csv'
    FILE_TRAIN = '../tmp/uni_var_train.tfrecords'
    FILE_TEST = '../tmp/uni_var_test.tfrecords'

    N_IN, N_OUT, FEATURE_COLS = 7, 7, [0]

    '''
    write data to TFRecord then read and evaluate 
    '''
    # tf_record_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, FILE_TRAIN, FILE_TEST, feature_cols=FEATURE_COLS)
