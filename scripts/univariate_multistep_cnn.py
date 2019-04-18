"""
@author Jacob
@time 2019/04/18
"""

from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keras


def split_dataset(data):
    train, test = data[1:-328], data[-328, -6]
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


def evaluate_forecasts(actual, predicted):
    scores = []

    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ','.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out

        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)


def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)

    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def forecast(model, history, n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x), 1))
    yhat = model.predict(input_x, verbose=0)
    return yhat[0]


def evaluate_model(train, test, n_input):
    model = build_model(train, n_input)

    history = [x for x in train]

    predictions = []
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


if __name__ == '__main__':
    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_input = 7
    score, scores = evaluate_model(train, test, n_input)

    summarize_scores('cnn', score, scores)

    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

    plt.plot(days, scores, marker='o', label='cnn')
    plt.show()
