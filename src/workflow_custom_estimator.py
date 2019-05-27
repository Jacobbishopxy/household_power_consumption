"""
@author Jacob
@time 2019/05/16
"""

from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_estimator import estimator as est

from data_preprocessing import read_data_from_csv, split_data, to_supervised
from utils import crash_proof, create_model_dir, launch_tb


def _float_feature(arr: np.ndarray):
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


def model_to_estimator(keras_model, model_dir: Optional[str] = None):
    """
    convert keras model to estimator
    :param keras_model:
    :param model_dir:
    :return:
    """
    return tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)


def _parse(feature: np.ndarray, label: np.ndarray):
    return {'inputs': feature}, label


def set_input_fn_csv(features: np.ndarray, labels: np.ndarray, num_epochs=None):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(lambda f, l: _parse(f, l))
    dataset = dataset.batch(4)
    dataset = dataset.repeat(num_epochs)

    itr = dataset.make_one_shot_iterator()
    batch_features, batch_labels = itr.get_next()

    return batch_features, batch_labels


def build_model(shape_in: Tuple[int, int], shape_out: Tuple[int]):
    n_out = shape_out[0]

    input_layer = tf.keras.layers.Input(shape=shape_in, name='inputs')
    conv = tf.keras.layers.Conv1D(filters=16,
                                  kernel_size=3,
                                  activation='relu')(input_layer)
    maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    fltn = tf.keras.layers.Flatten()(maxp)
    dns1 = tf.keras.layers.Dense(10, activation='relu')(fltn)
    dns2 = tf.keras.layers.Dense(n_out)(dns1)

    model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])
    model.summary()
    return model


def create_model(shape_in: Tuple[int, int], shape_out: Tuple[int]):
    n_out = shape_out[0]

    with tf.name_scope('keras_model'):
        input_layer = tf.keras.layers.Input(shape=shape_in, name='inputs')
        conv = tf.keras.layers.Conv1D(filters=16,
                                      kernel_size=3,
                                      activation='relu')(input_layer)
        maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
        fltn = tf.keras.layers.Flatten()(maxp)
        dns1 = tf.keras.layers.Dense(10, activation='relu')(fltn)
        dns2 = tf.keras.layers.Dense(n_out)(dns1)

        model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    return model


def set_model_fn(features: Dict[str, tf.Tensor],
                 labels: tf.Tensor,
                 mode: est.ModeKeys,
                 params: Dict[str, Any]):
    """

    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    model_fn = params.get('model_fn')
    model_params = params.get('model_params')
    shape_in, shape_out = model_params.get('shape_in'), model_params.get('shape_out')
    model = model_fn(shape_in, shape_out)

    learning_rate = params.get('learning_rate', 1e-4)

    fea = features['inputs']

    if mode == est.ModeKeys.PREDICT:
        result = model(fea, training=False)

        predictions = {
            'prices': tf.squeeze(result, axis=1),
        }
        return est.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'prices': est.export.PredictOutput(predictions)
            }
        )

    # todo: enhance interaction with TensorBoard
    if mode == est.ModeKeys.TRAIN:
        result = model(fea, training=True)

        optimizer = tf.train.AdamOptimizer()
        loss = tf.losses.mean_squared_error(labels=labels, predictions=result)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())

        tf.identity(learning_rate, 'learning_rate')
        tf.identity(loss, 'loss')
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('train_loss', loss)

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    if mode == est.ModeKeys.EVAL:
        result = model(fea, training=False)

        loss = tf.losses.mean_squared_error(labels=labels, predictions=result)
        rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=result)
        # todo: check each shape_out's rmse

        eval_metric_ops = {
            'rmse': rmse
        }

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


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


def _data_from_tf_record(example,
                         shape_in: Tuple[int, int],
                         shape_out: Tuple[int]):
    n_in, num_fea = shape_in
    n_dim_in = n_in * num_fea
    feature_def = {'features': tf.FixedLenFeature(n_dim_in, tf.float32),
                   'labels': tf.FixedLenFeature(shape_out[0], tf.float32)}

    features = tf.parse_single_example(example, feature_def)
    fea = tf.reshape(features['features'], shape_in)
    lbl = tf.reshape(features['labels'], shape_out)
    return fea, lbl


def set_input_fn_tf_record(file_name: str,
                           shape_in: Tuple[int, int],
                           shape_out: Tuple[int],
                           num_epochs: Optional[int] = None):
    """

    :param file_name:
    :param shape_in:
    :param shape_out:
    :param num_epochs:
    :return:
    """
    with tf.name_scope("D"):
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.map(lambda x: _data_from_tf_record(x, shape_in, shape_out))
        dataset = dataset.map(_parse)
        dataset = dataset.batch(4)
        dataset = dataset.repeat(num_epochs)

    return dataset


def eval_from_csv(shape_in: Tuple[int, int],
                  shape_out: Tuple[int],
                  file_csv: str,
                  feature_cols: Union[List[int], int] = 0,
                  num_epochs: Optional[int] = 10,
                  activate_tb: bool = False):
    """
    train & test read from csv
    :param shape_in:
    :param shape_out:
    :param file_csv:
    :param feature_cols:
    :param num_epochs:
    :param activate_tb:
    :return:
    """

    n_in, n_out = shape_in[0], shape_out[0]

    model = build_model(shape_in=shape_in, shape_out=shape_out)
    model_dir = create_model_dir(r'..\tmp')
    classifier = model_to_estimator(model, model_dir=model_dir)

    d = read_data_from_csv(file_csv)
    raw_trn_data, raw_tst_data = split_data(d)
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)

    classifier.train(
        input_fn=lambda: set_input_fn_csv(trn_fea, trn_lbl),
        steps=20
    )

    result = classifier.evaluate(
        input_fn=lambda: set_input_fn_csv(tst_fea, tst_lbl, num_epochs=num_epochs)
    )
    if activate_tb:
        launch_tb(model_dir)
    return result


def eval_from_tf_record(shape_in: Tuple[int, int],
                        shape_out: Tuple[int],
                        file_train: str,
                        file_test: str,
                        num_epochs: Optional[int] = 10,
                        activate_tb: bool = False):
    """
    train & test read from TFRecord
    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param num_epochs:
    :param activate_tb:
    :return:
    """
    model = build_model(shape_in=shape_in, shape_out=shape_out)
    model_dir = create_model_dir(r'..\tmp')
    classifier = model_to_estimator(model, model_dir=model_dir)

    classifier.train(
        input_fn=lambda: set_input_fn_tf_record(file_train,
                                                shape_in=shape_in,
                                                shape_out=shape_out),
        steps=20
    )
    result = classifier.evaluate(
        input_fn=lambda: set_input_fn_tf_record(file_test,
                                                shape_in=shape_in,
                                                shape_out=shape_out,
                                                num_epochs=num_epochs)
    )
    if activate_tb:
        launch_tb(model_dir)
    return result


def ev(shape_in: Tuple[int, int],
       shape_out: Tuple[int],
       file_train: str,
       file_test: str,
       epochs: int = 10,
       steps: int = 20,
       model_dir: str = r'..\tmp\test',
       consistent_model: bool = True,
       activate_tb: bool = True):
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)

    classifier = est.Estimator(
        model_fn=set_model_fn,
        model_dir=model_dir,
        params={
            'model_fn': create_model,
            'model_params': {
                'shape_in': shape_in,
                'shape_out': shape_out
            }
        }
    )

    for _ in range(epochs):
        classifier.train(input_fn=lambda: set_input_fn_tf_record(file_train,
                                                                 shape_in=shape_in,
                                                                 shape_out=shape_out),
                         steps=steps)

        result = classifier.evaluate(input_fn=lambda: set_input_fn_tf_record(file_test,
                                                                             shape_in=shape_in,
                                                                             shape_out=shape_out,
                                                                             num_epochs=20))

        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return classifier


if __name__ == '__main__':
    crash_proof()

    '''
    N_IN: timesteps 
    N_OUT: labels
    FEATURE_COLS: features to use for train_x & test_x
    '''

    RAW_DATA_PATH = '../data/household_power_consumption_days.csv'
    FILE_TRAIN = '../tmp/uni_var_train.tfrecords'
    FILE_TEST = '../tmp/uni_var_test.tfrecords'

    N_IN, N_OUT, FEATURE_COLS = 7, 7, [0]

    SHAPE_IN = (N_IN, len(FEATURE_COLS))
    SHAPE_OUT = (N_OUT,)

    '''
    read data from csv and evaluate model
    '''
    # r1 = eval_from_csv(SHAPE_IN, SHAPE_OUT, feature_cols=FEATURE_COLS, file_csv=RAW_DATA_PATH)
    # print(r1)

    '''
    write data to TFRecord then read and evaluate 
    '''
    # tf_record_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, FILE_TRAIN, FILE_TEST, feature_cols=FEATURE_COLS)

    # r2 = eval_from_tf_record(SHAPE_IN, SHAPE_OUT, file_train=FILE_TRAIN, file_test=FILE_TEST, activate_tb=True)
    # print(r2)

    '''
    use model fn to create an estimator    
    '''

    ev(shape_in=SHAPE_IN,
       shape_out=SHAPE_OUT,
       file_train=FILE_TRAIN,
       file_test=FILE_TEST)
