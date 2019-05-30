"""
@author Jacob
@time 2019/05/27
"""

from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf


def _parse(feature: np.ndarray, label: np.ndarray):
    return {'inputs': feature}, label


def set_input_fn_csv(features: np.ndarray,
                     labels: np.ndarray,
                     batch_size: int,
                     num_epochs: int = 1):
    """

    :param features:
    :param labels:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(lambda f, l: _parse(f, l))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    itr = dataset.make_one_shot_iterator()
    batch_features, batch_labels = itr.get_next()

    return batch_features, batch_labels


def set_input_fn_tf_record(file_name: str,
                           shape_in: Tuple[int, int],
                           shape_out: Tuple[int],
                           batch_size: int,
                           num_epochs: int = 1):
    """

    :param file_name:
    :param shape_in:
    :param shape_out:
    :param batch_size:
    :param num_epochs:
    :return:
    """

    def _data_from_tf_record(example):
        n_in, num_fea = shape_in
        n_dim_in = n_in * num_fea
        feature_def = {'features': tf.FixedLenFeature(n_dim_in, tf.float32),
                       'labels': tf.FixedLenFeature(shape_out[0], tf.float32)}

        features = tf.parse_single_example(example, feature_def)
        fea = tf.reshape(features['features'], shape_in)
        lbl = tf.reshape(features['labels'], shape_out)
        return fea, lbl

    with tf.name_scope("D"):
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.map(_data_from_tf_record)
        dataset = dataset.map(_parse)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)

    return dataset
