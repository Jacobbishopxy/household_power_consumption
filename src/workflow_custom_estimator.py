"""
@author Jacob
@time 2019/05/16


In real world, data is better to be separated into three parts: train, evaluate, predict.

"""

from typing import Union, List, Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow_estimator import estimator as est
from tensorflow.python.ops import math_ops

from models import create_compiled_model, model_to_estimator, create_vanilla_model
from input_functions import set_input_fn_csv, set_input_fn_tf_record
from data_preprocessing import read_data_from_csv, split_data, to_supervised
from utils import crash_proof, create_model_dir, launch_tb


def estimator_from_csv(shape_in: Tuple[int, int],
                       shape_out: Tuple[int],
                       file_csv: str,
                       feature_cols: Union[List[int], int] = 0,
                       epochs: Optional[int] = 10,
                       steps: int = 20,
                       model_dir: str = r'..\tmp\test',
                       consistent_model: bool = True,
                       activate_tb: bool = False):
    """
    train & test read from csv
    :param shape_in:
    :param shape_out:
    :param file_csv:
    :param feature_cols:
    :param epochs:
    :param steps:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :return:
    """
    n_in, n_out = shape_in[0], shape_out[0]

    model = create_compiled_model(shape_in=shape_in, shape_out=shape_out)
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)
    classifier = model_to_estimator(model, model_dir=model_dir)

    d = read_data_from_csv(file_csv)
    raw_trn_data, raw_tst_data = split_data(d)
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)

    for _ in range(epochs):
        classifier.train(
            input_fn=lambda: set_input_fn_csv(trn_fea, trn_lbl),
            steps=steps
        )
        result = classifier.evaluate(
            input_fn=lambda: set_input_fn_csv(tst_fea, tst_lbl, num_epochs=20)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return classifier


def estimator_from_tf_record(shape_in: Tuple[int, int],
                             shape_out: Tuple[int],
                             file_train: str,
                             file_test: str,
                             epochs: Optional[int] = 10,
                             steps: int = 20,
                             model_dir: str = r'..\tmp\test',
                             consistent_model: bool = True,
                             activate_tb: bool = False):
    """
    train & test read from TFRecord
    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param epochs:
    :param steps:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :return:
    """
    model = create_compiled_model(shape_in=shape_in, shape_out=shape_out)
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)
    classifier = model_to_estimator(model, model_dir=model_dir)

    for _ in range(epochs):
        classifier.train(
            input_fn=lambda: set_input_fn_tf_record(file_train,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out),
            steps=steps
        )
        result = classifier.evaluate(
            input_fn=lambda: set_input_fn_tf_record(file_test,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    num_epochs=20)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return classifier


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
    model = model_fn(**model_params)
    print(model.summary())

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
        train_mape = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(result, labels), (labels + 1e-10))))
        tf.identity(learning_rate, 'learning_rate')
        tf.identity(loss, 'loss')
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('train_loss', loss)
            tf.summary.scalar('train_mape', train_mape)

        logging_hook = tf.train.LoggingTensorHook({'train_mape': train_mape}, every_n_iter=10)

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook]
        )

    if mode == est.ModeKeys.EVAL:
        result = model(fea, training=False)

        loss = tf.losses.mean_squared_error(labels=labels, predictions=result)
        rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=result)
        mape = tf.metrics.mean(math_ops.abs(math_ops.div_no_nan(math_ops.subtract(labels, result), labels + 1e-10)))

        # todo: check each shape_out's rmse

        eval_metric_ops = {
            'rmse': rmse,
            'mape': mape
        }

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def estimator_from_model_fn(shape_in: Tuple[int, int],
                            shape_out: Tuple[int],
                            file_train: str,
                            file_test: str,
                            epochs: int = 10,
                            steps: int = 20,
                            model_dir: str = r'..\tmp\test',
                            consistent_model: bool = True,
                            activate_tb: bool = True):
    """

    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param epochs:
    :param steps:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :return:
    """
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)

    classifier = est.Estimator(
        model_fn=set_model_fn,
        model_dir=model_dir,
        params={
            'model_fn': create_vanilla_model,
            'model_params': {
                'shape_in': shape_in,
                'shape_out': shape_out
            }
        }
    )

    for _ in range(1):
        classifier.train(
            input_fn=lambda: set_input_fn_tf_record(file_train,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out, num_epochs=epochs),
            steps=steps
        )

        result = classifier.evaluate(
            input_fn=lambda: set_input_fn_tf_record(file_test,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    num_epochs=1)
        )

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

    N_IN, N_OUT, FEATURE_COLS = 14, 7, [0]
    epochs = 100
    SHAPE_IN = (N_IN, len(FEATURE_COLS))
    SHAPE_OUT = (N_OUT,)

    '''
    read data from csv
    '''
    # c1 = estimator_from_csv(SHAPE_IN, SHAPE_OUT, feature_cols=FEATURE_COLS, file_csv=RAW_DATA_PATH)

    '''
    read data from TFRecord
    '''
    # c2 = estimator_from_tf_record(SHAPE_IN, SHAPE_OUT, file_train=FILE_TRAIN, file_test=FILE_TEST, activate_tb=True)

    '''
    use model fn to create an estimator    
    '''
    # from data_preprocessing import tf_record_preprocessing

    # tf_record_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, FILE_TRAIN, FILE_TEST, feature_cols=FEATURE_COLS)
    c3 = estimator_from_model_fn(shape_in=SHAPE_IN,
                                 shape_out=SHAPE_OUT,
                                 file_train=FILE_TRAIN,
                                 file_test=FILE_TEST,
                                 epochs=epochs, steps=275
                                 )
