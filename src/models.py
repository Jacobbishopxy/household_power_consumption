"""
@author Jacob
@time 2019/05/09
"""

import keras


def build_univariate_model(train_x, train_y):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=16,
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def build_multichannel_model(train_x, train_y):
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=32,
                                  kernel_size=3,
                                  activation='relu',
                                  input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def build_multihead_model(train_x, train_y):
    verbose, epochs, batch_size = 0, 25, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    in_layers, out_layers = [], []
    for i in range(n_features):
        inputs = keras.layers.Input(shape=(n_timesteps, 1))
        conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        pool1 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        flat = keras.layers.Flatten()(pool1)
        in_layers.append(inputs)
        out_layers.append(flat)
    merged = keras.layers.concatenate(out_layers)
    dense1 = keras.layers.Dense(200, activation='relu')(merged)
    dense2 = keras.layers.Dense(100, activation='relu')(dense1)
    outputs = keras.layers.Dense(n_outputs)(dense2)
    model = keras.Model(inputs=in_layers, outputs=outputs)
    model.compile(loss="mse", optimizer="adam")
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model
