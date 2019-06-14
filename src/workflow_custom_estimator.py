"""
@author Jacob
@time 2019/05/16


In real world, data is better to be separated into three parts: train, evaluate, predict.

"""

from typing import Union, List, Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow_estimator import estimator as est
from enum import Enum
from functools import partial

from networks import create_compiled_model, model_to_estimator, create_multichannel_model
from input_functions import set_input_fn_csv, set_input_fn_tf_record
from data_preprocessing import read_data_from_csv, split_data, to_supervised
from utils import crash_proof, create_model_dir, launch_tb


def _calc_mse(labels: tf.Tensor, predictions: tf.Tensor, training: bool = True):
    if training:
        return tf.reduce_mean(tf.square(tf.subtract(predictions, labels)))
    else:
        return tf.metrics.mean_squared_error(labels, predictions)


def _calc_rmse(labels: tf.Tensor, predictions: tf.Tensor, training: bool = True):
    if training:
        return tf.sqrt(_calc_mse(labels, predictions, training))
    else:
        return tf.metrics.root_mean_squared_error(labels, predictions)


def _calc_mae(labels: tf.Tensor, predictions: tf.Tensor, training: bool = True):
    if training:
        return tf.reduce_mean(tf.abs(tf.subtract(predictions, labels)))
    else:
        return tf.metrics.mean_absolute_error(labels, predictions)


def _calc_mape(labels: tf.Tensor, predictions: tf.Tensor, training: bool = True):
    if training:
        return tf.reduce_mean(tf.abs(tf.divide(tf.subtract(predictions, labels),
                                               tf.add(labels, tf.constant(1e-10)))))
    else:
        return tf.metrics.mean(tf.abs(tf.divide(tf.subtract(predictions, labels),
                                                tf.add(labels, tf.constant(1e-10)))))


class InspectionIndicator(Enum):
    MSE = partial(_calc_mse)
    RMSE = partial(_calc_rmse)
    MAE = partial(_calc_mae)
    MAPE = partial(_calc_mape)

    def __call__(self, labels, predictions, training: bool = True):
        return self.value(labels, predictions, training=training)


def _get_indicators(labels: tf.Tensor,
                    predictions: tf.Tensor,
                    inspection_indicators: List[str],
                    training: bool = True):
    if training:
        if 'MSE' in inspection_indicators:
            mse = InspectionIndicator.MSE(labels, predictions)
            tf.identity(mse, 'loss_mse')
            tf.summary.scalar('mse', mse)
        if 'RMSE' in inspection_indicators:
            rmse = InspectionIndicator.RMSE(labels, predictions)
            tf.identity(rmse, 'loss_rmse')
            tf.summary.scalar('rmse', rmse)
        if 'MAE' in inspection_indicators:
            mae = InspectionIndicator.MAE(labels, predictions)
            tf.identity(mae, 'loss_mae')
            tf.summary.scalar('mae', mae)
        if 'MAPE' in inspection_indicators:
            mape = InspectionIndicator.MAPE(labels, predictions)
            tf.identity(mape, 'loss_mape')
            tf.summary.scalar('mape', mape)
    else:
        _eval_metric_ops = {}
        if 'MSE' in inspection_indicators:
            mse = InspectionIndicator.MSE(labels, predictions, training=False)
            _eval_metric_ops.update({'mse': mse})
        if 'RMSE' in inspection_indicators:
            rmse = InspectionIndicator.RMSE(labels, predictions, training=False)
            _eval_metric_ops.update({'rmse': rmse})
        if 'MAE' in inspection_indicators:
            mae = InspectionIndicator.MAE(labels, predictions, training=False)
            _eval_metric_ops.update({'mae': mae})
        if 'MAPE' in inspection_indicators:
            mape = InspectionIndicator.MAPE(labels, predictions, training=False)
            _eval_metric_ops.update({'mape': mape})
        return _eval_metric_ops


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
                                              batch_size=batch_size)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return estimator


def estimator_from_tf_record(shape_in: Tuple[int, int],
                             shape_out: Tuple[int],
                             tf_records_name: str,
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
    :param tf_records_name:
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
            input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                    is_train=True,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=epochs)
        )
        result = estimator.evaluate(
            input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                    is_train=False,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size)
        )
        print(result)

    if activate_tb:
        launch_tb(model_dir)
    return estimator


def model_fn_default(features: Dict[str, tf.Tensor],
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
    network_fn = params.get('network_fn')
    network_params = params.get('network_params')
    learning_rate = params.get('learning_rate', 1e-3)
    _ii = ['MSE', 'RMSE', 'MAE', 'MAPE']
    inspection_indicators = [i.upper() for i in params.get('inspection_indicators', _ii)]

    if len(features.keys()) == 1:
        fea = features['input_0']
    else:
        fea = [features[i] for i in features.keys()]

    network = network_fn(**network_params)
    network.summary()

    if mode == est.ModeKeys.PREDICT:
        predictions = network(fea, training=False)

        return est.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'result': est.export.PredictOutput(predictions)
            }
        )

    if mode == est.ModeKeys.TRAIN:
        predictions = network(fea, training=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        loss = tf.losses.mean_squared_error(labels, predictions)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())

        tf.identity(learning_rate, 'learning_rate')

        _get_indicators(labels, predictions, inspection_indicators)

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == est.ModeKeys.EVAL:
        predictions = network(fea, training=False)

        loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

        eval_metric_ops = _get_indicators(labels, predictions, inspection_indicators, training=False)

        return est.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def estimator_from_model_fn(shape_in: Union[Tuple[int, int], List[Tuple[int, int]]],
                            shape_out: Tuple[int],
                            tf_records_name: str,
                            batch_size: int = 10,
                            epochs: int = 10,
                            model_dir: str = r'.\tmp\test',
                            consistent_model: bool = True,
                            activate_tb: bool = True,
                            n_checkpoints: int = 1,
                            model_fn=model_fn_default,
                            network_fn=create_multichannel_model,
                            learning_rate: float = None,
                            batch_norm: bool = False):
    """
    :param batch_norm:
    :param shape_in:
    :param shape_out:
    :param tf_records_name:
    :param batch_size:
    :param epochs:
    :param model_dir:
    :param consistent_model:
    :param activate_tb:
    :param n_checkpoints:
    :param model_fn:
    :param network_fn:
    :param learning_rate:
    :return:
    """
    model_dir = create_model_dir(model_dir, consistent_model=consistent_model)

    params = {
        'network_fn': network_fn,
        'network_params': {
            'shape_in': shape_in,
            'shape_out': shape_out,
            'batch_norm': batch_norm,
        },
    }

    train_epochs = epochs // n_checkpoints

    if learning_rate is not None:
        params['learning_rate'] = learning_rate

    estimator = est.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params
    )

    for _ in range(n_checkpoints):
        estimator.train(
            input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                    is_train=True,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size,
                                                    num_epochs=train_epochs),
        )

        result = estimator.evaluate(
            input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                    is_train=False,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size)
        )

        print(result)

        prd = estimator.predict(
            input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                    is_train=False,
                                                    shape_in=shape_in,
                                                    shape_out=shape_out,
                                                    batch_size=batch_size)
        )

        n = 5
        for i in prd:
            print(i, f' len: {len(i)}')
            n -= 1
            if n == 0:
                break

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
    TF_RECORDS_NAME = 'uni_var_train'

    N_IN, N_OUT, FEATURE_COLS = 14, 7, [0]
    EPOCHS = 100
    SHAPE_IN = (N_IN, len(FEATURE_COLS))
    SHAPE_OUT = (N_OUT,)

    '''
    read data from csv
    '''
    # e1 = estimator_from_csv(SHAPE_IN, SHAPE_OUT, feature_cols=FEATURE_COLS, file_csv=RAW_DATA_PATH)

    '''
    read data from TFRecord
    '''
    # e2 = estimator_from_tf_record(SHAPE_IN, SHAPE_OUT, tf_records_name=TF_RECORDS_NAME, activate_tb=True)

    '''
    use model fn to create an estimator    
    '''
    e3 = estimator_from_model_fn(shape_in=SHAPE_IN,
                                 shape_out=SHAPE_OUT,
                                 tf_records_name=TF_RECORDS_NAME,
                                 epochs=EPOCHS,
                                 consistent_model=False)
