"""
@author Jacob
@time 2019/05/16


In real world, data is better to be separated into three parts: train, evaluate, predict.

"""

from typing import Union, List, Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow_estimator import estimator as est

from models import create_compiled_model, model_to_estimator, create_vanilla_model
from input_functions import set_input_fn_csv, set_input_fn_tf_record
from data_preprocessing import read_data_from_csv, split_data, to_supervised
from utils import crash_proof, create_model_dir, launch_tb


def estimator_from_csv(shape_in: Tuple[int, int],
                       shape_out: Tuple[int],
                       file_csv: str,
                       feature_cols: Union[List[int], int] = 0,
                       batch_size: int = 10,
                       epochs: Optional[int] = 10,
                       steps: int = 1,
                       model_dir: str = r'..\tmp\test',
                       consistent_model: bool = True,
                       activate_tb: bool = False):
    """
    train & test read from csv
    :param shape_in:
    :param shape_out:
    :param file_csv:
    :param feature_cols:
    :param batch_size:
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
    estimator = model_to_estimator(model, model_dir=model_dir)

    d = read_data_from_csv(file_csv)
    raw_trn_data, raw_tst_data = split_data(d)
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)

    for _ in range(steps):
        estimator.train(
            input_fn=lambda: set_input_fn_csv(trn_fea,
                                              trn_lbl,
                                              batch_size=batch_size,
                                              num_epochs=epochs)
        )
        result = estimator.evaluate(
            input_fn=lambda: set_input_fn_csv(tst_fea,
                                              tst_lbl,
                                              batch_size=batch_size,
                                              num_epochs=1)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return estimator


def estimator_from_tf_record(shape_in: Tuple[int, int],
                             shape_out: Tuple[int],
                             file_train: str,
                             file_test: str,
                             batch_size: int = 10,
                             epochs: Optional[int] = 10,
                             steps: int = 1,
                             model_dir: str = r'..\tmp\test',
                             consistent_model: bool = True,
                             activate_tb: bool = False):
    """
    train & test read from TFRecord
    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param batch_size:
    :param epochs:
    :param steps:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :return:
    """
    model = create_compiled_model(shape_in=shape_in, shape_out=shape_out)
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)
    estimator = model_to_estimator(model, model_dir=model_dir)

    for _ in range(steps):
        estimator.train(
            input_fn=lambda: set_input_fn_tf_record(file_train,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=epochs)
        )
        result = estimator.evaluate(
            input_fn=lambda: set_input_fn_tf_record(file_test,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=1)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return estimator


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
        with tf.name_scope('train'):

            result = model(fea, training=True)

            optimizer = tf.train.AdamOptimizer()
            loss = tf.losses.mean_squared_error(labels=labels, predictions=result)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())

            tf.identity(learning_rate, 'learning_rate')
            tf.identity(loss, 'loss')
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
            'train_loss/rmse': rmse
        }
        tf.summary.scalar('rmse', rmse)

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def estimator_from_model_fn(shape_in: Tuple[int, int],
                            shape_out: Tuple[int],
                            file_train: str,
                            file_test: str,
                            batch_size: int = 10,
                            epochs: int = 10,
                            steps: int = 1,
                            model_dir: str = r'..\tmp\test',
                            consistent_model: bool = True,
                            activate_tb: bool = True):
    """

    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param batch_size:
    :param epochs:
    :param steps:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :return:
    """
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)

    estimator = est.Estimator(
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

    for _ in range(steps):
        estimator.train(
            input_fn=lambda: set_input_fn_tf_record(file_train,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=epochs)
        )

        result = estimator.evaluate(
            input_fn=lambda: set_input_fn_tf_record(file_test,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=1)
        )

        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return estimator


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
    read data from csv
    '''
    # e1 = estimator_from_csv(SHAPE_IN, SHAPE_OUT, feature_cols=FEATURE_COLS, file_csv=RAW_DATA_PATH)

    '''
    read data from TFRecord
    '''
    # e2 = estimator_from_tf_record(SHAPE_IN, SHAPE_OUT, file_train=FILE_TRAIN, file_test=FILE_TEST, activate_tb=True)

    '''
    use model fn to create an estimator    
    '''
    e3 = estimator_from_model_fn(shape_in=SHAPE_IN,
                                 shape_out=SHAPE_OUT,
                                 file_train=FILE_TRAIN,
                                 file_test=FILE_TEST,
                                 epochs=5,
                                 steps=1,
                                 consistent_model=False)
