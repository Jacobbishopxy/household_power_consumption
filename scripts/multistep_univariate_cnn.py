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
import tensorflow as tf


def split_dataset(data):
    """
    split a univariate dataset into train/test sets
    :param data:
    :return:
    """
    # split into standard weeks
    train, test = data[1: -328], data[-328: -6]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


def evaluate_forecasts(actual, predicted):
    """
    evaluate one or more weekly forecasts against expected values
    :param actual:
    :param predicted:
    :return:
    """
    scores = []
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def summarize_scores(name, score, scores):
    """
    summarize scores
    :param name:
    :param score:
    :param scores:
    :return:
    """
    s_scores = ','.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_supervised(train, n_input, n_out=7):
    """
    convert history into inputs and outputs
    :param train:
    :param n_input:
    :param n_out:
    :return:
    """
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


def build_model(train, n_input):
    """
    train the model
    :param train:
    :param n_input:
    :return:
    """
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    model.summary()
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def build_dev_model(train, n_input):

    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    input_layer = tf.keras.layers.Input(shape=(n_timesteps, n_features))
    conv = tf.keras.layers.Conv1D(filters=16,
                                  kernel_size=3,
                                  activation='relu')(input_layer)
    maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    fltn = tf.keras.layers.Flatten()(maxp)
    dns1 = tf.keras.layers.Dense(10, activation='relu')(fltn)
    dns2 = tf.keras.layers.Dense(n_outputs)(dns1)

    model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])

    model.summary()
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def forecast(model, history, n_input):
    """
    make a forecast
    :param model:
    :param history:
    :param n_input:
    :return:
    """
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forevast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    return yhat[0]


def evaluate_model(train, test, n_input):
    """
    evaluate a single model
    :param train:
    :param test:
    :param n_input:
    :return:
    """
    # fit model
    model = build_model(train, n_input)
    # model = build_dev_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = []
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def main():
    # load the new file
    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    # split into train and test
    train, test = split_dataset(dataset.values)
    # evaluate model and get scores
    n_input = 7
    score, scores = evaluate_model(train, test, n_input)
    # summarize scores
    summarize_scores('cnn', score, scores)
    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

    plt.plot(days, scores, marker='o', label='cnn')
    plt.show()


if __name__ == '__main__':
    main()

    """
    Running the example fits and evaluates the model, printing the overall RMSE across all
    seven days, and the per-day RMSE for each lead time.
    
    We can see that in this case, the model was skillful as compared to a naive forecast,
    achieving an overall RMSE of about 404 kilowatts, less than 465 kilowatts achieved by
    a naive model.
    
    The plot shows that perhaps Tuesdays and Fridays are easier days to forecast than the
    other days and that perhaps Saturday at the end of the standard week is the hardest day
    to forecast.
    
    ******
    
    We can increase the number of prior days to use as input from seven to 14 by changing 
    the n_input variable (n_input = 14).
    
    Re-running the example with this change first prints a summary of the performance of 
    the model.
    
    In this case, we can see a further drop in the overall RMSE, suggesting that further
    tuning of the input size and perhaps the kernel size of the model may result in better
    performance.
    
    Comparing the per-day RMSE scores, we see some are better and some are worse than using
    seventh inputs.
    
    This may suggest a benefit in using the two different sized inputs in some way, such as
    an ensemble of the two approaches or perhaps a single model (e.g. a multi-headed model)
    that reads the training data in different ways.
    """
