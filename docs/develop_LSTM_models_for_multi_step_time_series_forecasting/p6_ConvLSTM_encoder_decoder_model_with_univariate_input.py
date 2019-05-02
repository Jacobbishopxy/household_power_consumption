"""
@author jacob
@time 5/1/2019

ConvLSTM Encoder-Decoder Model With Univariate Input

A further extension of the CNN-LSTM approach is to perform the convolutions of the CNN (e.g. how the
CNN reads the input sequence data) as part of the LSTM for each time step.

This combination is called a Convolutional LSTM, or ConvLSTM for short, and like the CNN-LSTM is also
used for spatio-temporal data.
Unlike an LSTM that reads the data in directly in order to calculate internal state and state
transitions, and unlike the CNN-LSTM that is interpreting the output from CNN models, the ConvLSTM
is using convolutions directly as part of reading input into the LSTM units themselves.

The Keras library provides the ConvLSTM2D class that supports the ConvLSTM model for 2D data. It can
be configured for 1D multivariate time series forecasting.
The ConvLSTM2D class, by default, expects input data to have the shape:
[samples, timesteps, rows, cols, channels]
Where each time step of data is defined as an image of (row * columns) data points.

We are working with a one-dimensional sequence of total power consumption, which we can interpret as
one row with 14 columns, if we assume that we are using two weeks of data as input.

For the ConvLSTM, this would be a single read: that is, the LSTM would read one time step of 14 days
and perform a convolution across those time steps.

This is not ideal.
Instead, we can split the 14 days into two subsequences with a length of seven days. The ConvLSTM can
then read across the two time steps and perform the CNN process on the seven days of data within each.
For this chosen framing of the problem, the input for the ConvLSTM2D would therefore be:
[n, 2, 1, 7, 1]
1. Samples: n, for the number of examples in the training dataset.
2. Time: 2, for the two subsequences that we split a window of 14 days into.
3. Rows: 1, for the one-dimensional shape of each subsequence.
4. Columns: 7, for the seven days in each subsequence.
5. Channels: 1, for the single feature that we are working with as input.

You can explore other configurations, such as providing 21 days of input split into three subsequences
of seven days, and/or providing all eight features or channels as input.

"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from docs.develop_LSTM_models_for_multi_step_time_series_forecasting. \
    p2_LSTM_model_with_univariate_input_and_vector_output import split_dataset, to_supervised, evaluate_forecasts, \
    summarize_scores


def build_model(train, n_steps, n_length, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    model = keras.Sequential()
    model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                                      input_shape=(n_steps, 1, n_length, n_features)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.RepeatVector(n_outputs))
    model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


"""
This model expects five-dimensional data as input.
[samples, time steps, rows, cols, channels]
"""


def forecast(model, history, n_steps, n_length, n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
    yhat = model.predict(input_x, verbose=0)
    return yhat[0]


def evaluate_model(train, test, n_steps, n_length, n_input):
    model = build_model(train, n_steps, n_length, n_input)
    history = [x for x in train]
    predictions = []
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def main():
    dataset = pd.read_csv('../../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_steps, n_length = 2, 7
    n_input = n_length * n_steps
    score, scores = evaluate_model(train, test, n_steps, n_length, n_input)
    summarize_scores('lstm', score, scores)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker='o', label='lstm')
    plt.show()


if __name__ == '__main__':
    main()
