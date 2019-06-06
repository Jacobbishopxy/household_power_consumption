"""
@author Jacob
@time 2019/05/27
"""

from typing import List, Tuple, Optional
import tensorflow as tf

from utils_single_layer import create_conv1d, create_fully_connected


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


def create_multichannel_model(shape_in: Tuple[int, int],
                              shape_out: Tuple[int],
                              batch_norm: bool = False):
    n_out = shape_out[0]

    with tf.name_scope('keras_model'):
        # input layer
        input_layer = tf.keras.layers.Input(shape=shape_in, name='input_0')
        if batch_norm:
            norm_layer = tf.keras.layers.BatchNormalization()(input_layer)
        else:
            norm_layer = input_layer

        # conv1d * 2
        conv1 = create_conv1d(norm_layer, 16, 3, 'same', 2, batch_norm)
        conv2 = create_conv1d(conv1, 32, 3, 'same', 2, batch_norm)

        # flatten
        flatten = tf.keras.layers.Flatten()(conv2)

        dns = create_fully_connected(flatten, 56, batch_norm)
        dns1 = create_fully_connected(dns, 28, batch_norm)
        dns2 = create_fully_connected(dns1, n_out, batch_norm)

        model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    return model


def create_multihead_model(shape_in: List[Tuple[int, int]],
                           shape_out: Tuple[int],
                           batch_norm: bool = False):
    n_out = shape_out[0]

    with tf.name_scope('keras_model'):
        in_layers, out_layers = [], []
        for idx, val in enumerate(shape_in):
            inputs = tf.keras.layers.Input(shape=val, name=f'input_{idx}')

            conv1 = create_conv1d(inputs, 16, 3, 'same', 2, batch_norm)
            conv2 = create_conv1d(conv1, 16, 3, 'same', 2, batch_norm)

            flat = tf.layers.Flatten()(conv2)
            in_layers.append(inputs)
            out_layers.append(flat)

        merged = tf.keras.layers.concatenate(out_layers)

        dense1 = create_fully_connected(merged, 56)
        dense2 = create_fully_connected(dense1, 28)
        outputs = create_fully_connected(dense2, n_out)

        model = tf.keras.Model(inputs=in_layers, outputs=outputs)
    return model
