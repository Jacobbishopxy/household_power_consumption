"""
@author jacob
@time 5/1/2019


Encoder-Decoder LSTM Model With Multivariate Input

In this section, we will update the Encoder-Decoder LSTM developed in the previous section to use each
of the eight time series variables to predict the next standard week of daily total power consumption.

We will do this by providing each one-dimensional time series to the model as a separate sequence of
input.

The LSTM will in turn create an internal representation of each input sequence that will together be
interpreted by the decoder.

Using multivariate inputs is helpful for those problems where the output sequence is some function of
the observations at prior time steps from multiple different features, not just (or including) the
feature being forecasted. It is unclear whether this is the case in the power consumption problem,
but we can explore it nonetheless.

************
First, we must update the preparation of the training data to include all og the eight features, not
just the one total daily power consumed.

"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from docs.develop_LSTM_models_for_multi_step_time_series_forecasting. \
    p2_LSTM_model_with_univariate_input_and_vector_output import split_dataset, evaluate_forecasts, \
    summarize_scores


def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)


def forecast(model, history, n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat = model.predict(input_x, verbose=0)
    return yhat[0]


def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    model = keras.Sequential()
    model.add(keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.RepeatVector(n_outputs))
    model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def evaluate_model(train, test, n_input):
    model = build_model(train, n_input)
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
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
    n_input = 14
    score, scores = evaluate_model(train, test, n_input)
    summarize_scores('lstm', score, scores)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker='o', label='lstm')
    plt.show()


if __name__ == '__main__':
    main()
