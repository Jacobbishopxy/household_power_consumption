"""
@author Jacob
@time 2019/05/27
"""

from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf

from utils import generate_tf_records_path


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


def set_input_fn_tf_record(tf_records_name: str,
                           is_train: bool,
                           shape_in: Union[Tuple[int, int], List[Tuple[int, int]]],
                           shape_out: Tuple[int],
                           batch_size: int,
                           num_epochs: int = 1):
    """

    :param tf_records_name:
    :param is_train:
    :param shape_in: Tuple[int, int] -> single head data, List[Tuple[int, int]] -> multi head data
    :param shape_out:
    :param batch_size:
    :param num_epochs:
    :return:
    """

    train_path, test_path = generate_tf_records_path(tf_records_name)
    dataset_path = train_path if is_train else test_path

    def _data_from_tf_record(example):
        if isinstance(shape_in, Tuple):
            n_in, num_fea = shape_in
            n_dim_in = n_in * num_fea
            feature_def = {'input_0': tf.FixedLenFeature(n_dim_in, tf.float32),
                           'labels': tf.FixedLenFeature(shape_out[0], tf.float32)}

            features = tf.parse_single_example(example, feature_def)
            fea = {'input_0': tf.reshape(features['input_0'], shape_in)}
            lbl = tf.reshape(features['labels'], shape_out)
            return fea, lbl
        elif all([isinstance(i, Tuple) for i in shape_in]):
            n_dim_in_list = [i[0] * i[1] for i in shape_in]
            feat_dict = {f'input_{k}': tf.FixedLenFeature(v, tf.float32)
                         for k, v in enumerate(n_dim_in_list)}
            feature_def = {**feat_dict,
                           'labels': tf.FixedLenFeature(shape_out[0], tf.float32)}

            features = tf.parse_single_example(example, feature_def)
            fea = {f'input_{i}': tf.reshape(features[f'input_{i}'], shape_in[i])
                   for i, v in enumerate(shape_in)}
            lbl = tf.reshape(features['labels'], shape_out)
            return fea, lbl
        else:
            raise ValueError("shape_in has to be Tuple[int, int] or List[Tuple[int, int]]")

    with tf.name_scope("Dataset"):
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(_data_from_tf_record)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)

    return dataset


if __name__ == '__main__':
    FILE_TRAIN = '../tmp/multihead_train_[[0][1][2]].tfrecords'
    FILE_TEST = '../tmp/multihead_test_[[0][1][2]].tfrecords'

    ds = set_input_fn_tf_record(FILE_TRAIN,
                                is_train=True,
                                shape_in=[(7, 1), (7, 1), (7, 1)],
                                shape_out=(7,),
                                batch_size=4)

    with tf.Session() as s:
        foo = s.run(ds.make_one_shot_iterator().get_next())
        print(foo)
