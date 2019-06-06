import tensorflow as tf


def create_conv1d(prev_layer, filters=16, kernel_size=3, padding='same', pool_size=2, batch_norm=False, use_bias=True):
    conv1d_layer = tf.keras.layers.Conv1D(filters=filters,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          activation=None,
                                          use_bias=use_bias)(prev_layer)
    if batch_norm:
        conv1d_layer = tf.keras.layers.BatchNormalization()(conv1d_layer)
    conv1d_layer = tf.keras.layers.ReLU()(conv1d_layer)
    conv1d_layer = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(conv1d_layer)
    return conv1d_layer


def create_fully_connected(prev_layer, num_units, batch_norm=False):
    if batch_norm:
        layer = tf.keras.layers.BatchNormalization()(prev_layer)
    else:
        layer = prev_layer
    return tf.keras.layers.Dense(num_units, activation='relu')(layer)
