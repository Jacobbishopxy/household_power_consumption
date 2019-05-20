"""
@author Jacob
@time 2019/05/13
"""

from typing import Optional, List, Callable, Union, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_estimator import estimator


def model_to_estimator(keras_model, model_dir=None):
    return tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)


def set_np_input_fn(x: Dict, y: np.ndarray, num_epochs: int):
    return estimator.inputs.numpy_input_fn(x=x,
                                           y=y,
                                           shuffle=False,
                                           num_epochs=num_epochs)


def build_model(shape_in, shape_out):
    input_layer = tf.keras.layers.Input(shape=shape_in, name='feature')

    conv = tf.keras.layers.Conv1D(filters=16,
                                  kernel_size=3,
                                  activation='relu')(input_layer)
    maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    fltn = tf.keras.layers.Flatten()(maxp)
    dns1 = tf.keras.layers.Dense(10, activation='relu')(fltn)
    dns2 = tf.keras.layers.Dense(shape_out)(dns1)

    model = tf.keras.Model(inputs=input_layer, outputs=dns2)

    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data = pd.read_csv('../data/household_power_consumption_days.csv',
                       header=0,
                       infer_datetime_format=True,
                       parse_dates=['datetime'],
                       index_col=['datetime'])


    def split_data(d: pd.DataFrame):
        return d.iloc[1:-328, :], d.iloc[-328:-6, :]


    def to_supervised(train, n_in, n_out):
        features, labels = [], []
        for i in range(len(train) - n_in - n_out):
            in_end = i + n_in
            out_end = in_end + n_out
            features.append(train.iloc[i:in_end, 0].values.reshape((n_in, 1)))
            labels.append(train.iloc[in_end:out_end, 0].values)
        return np.array(features), np.array(labels)


    def to_forecasted(test, n_in, n_out):
        features, labels = [], []
        for i in range(len(test) // (n_in + n_out)):
            in_start = i * (n_in + n_out)
            out_end = (i + 1) * (n_in + n_out)
            features.append(test.iloc[in_start:in_start + n_in, 0].values.reshape((n_in, 1)))
            labels.append(test.iloc[in_start + n_in:out_end, 0].values)
        return np.array(features), np.array(labels)


    # following workflow only works for univariate cnn
    trn, tst = split_data(data)

    n_timesteps, n_features, n_outputs = 7, 1, 7

    mdl = build_model(shape_in=(n_timesteps, n_features), shape_out=n_outputs)

    est = model_to_estimator(mdl, 'tmp')

    trn_fea, trn_lbl = to_supervised(trn, n_timesteps, n_outputs)

    training_input_fn = set_np_input_fn(x={'feature': trn_fea},
                                        y=trn_lbl,
                                        num_epochs=20)


    def test_input():
        with tf.Session() as s:
            bar = s.run(training_input_fn())
            print(bar)


    est.train(input_fn=training_input_fn)

    tst_fea, tst_lbl = to_forecasted(tst, n_timesteps, n_outputs)

    testing_input_fn = set_np_input_fn(x={'feature': tst_fea},
                                       y=tst_lbl,
                                       num_epochs=1)

    result = est.evaluate(input_fn=testing_input_fn)

    print(result)
    '''
    tensorboard --logdir=src/tmp/est
    '''
