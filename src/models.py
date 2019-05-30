"""
@author Jacob
@time 2019/05/27
"""

from typing import Union, List, Tuple, Optional, Dict, Any
import tensorflow as tf


def model_to_estimator(keras_model, model_dir: Optional[str] = None):
    """
    convert keras model to estimator
    :param keras_model:
    :param model_dir:
    :return:
    """
    return tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)


def create_compiled_model(shape_in: Tuple[int, int], shape_out: Tuple[int]):
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
        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.train.AdamOptimizer(),
                      metrics=['accuracy'])
    return model


def create_vanilla_model(shape_in: Tuple[int, int], shape_out: Tuple[int]):
    n_out = shape_out[0]

    with tf.name_scope('keras_model'):
        input_layer = tf.keras.layers.Input(shape=shape_in, name='inputs')
        conv = tf.keras.layers.Conv1D(filters=16,
                                      kernel_size=3,
                                      activation='relu', padding='same')(input_layer)
        maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

        conv = tf.keras.layers.Conv1D(filters=32,
                                      kernel_size=3,
                                      activation='relu', padding='same')(maxp)
        maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

        fltn = tf.keras.layers.Flatten()(maxp)
        dns1 = tf.keras.layers.Dense(28, activation='relu')(fltn)
        dns2 = tf.keras.layers.Dense(n_out)(dns1)

        model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    return model
