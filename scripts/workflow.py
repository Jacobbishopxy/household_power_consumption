"""
@author Jacob
@time 2019/05/06
"""

from typing import Optional, Union, List, Callable
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from math import sqrt
from functools import partial

models_type = Union[keras.models.Model, keras.models.Sequential]


def split_dataset(data):
    train, test = data[1: -328], data[-328: -6]
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


def summarize_scores(name, score, scores):
    s_scores = ','.join(['%.1f' % i for i in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_supervised(train: np.ndarray,
                  n_input: int,
                  n_output: int,
                  train_x_variate_col: Optional[int] = None,
                  train_y_target_col: int = 0) -> (np.ndarray, np.ndarray):
    """
    训练数据转换为train_x和train_y
    :param train: 训练数据
    :param n_input:
    :param n_output:
    :param train_x_variate_col: train的第n列数据，None时为multivariate
    :param train_y_target_col: train的第n列数据
    :return:
    """
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    x, y = [], []

    for i in range(len(data) - n_input - n_output):
        in_end = i + n_input
        out_end = in_end + n_output
        x.append(data[i:in_end, :])
        y.append(data[in_end:out_end, train_y_target_col])

    x = np.array(x)
    y = np.array(y)

    if train_x_variate_col is not None:
        x = x[:, :, train_x_variate_col]
        x = x.reshape((x.shape[0], x.shape[1], 1))

    return x, y


def forecast(model: models_type,
             history: np.ndarray,
             n_input: int,
             input_x_col: Optional[int] = None) -> float:
    """
    walk forward validation: sliding window  # todo: expanding window

    test数据放入model对未来进行预测
    :param model:
    :param history:
    :param n_input:
    :param input_x_col:
    :return:
    """
    data = history.reshape((history.shape[0] * history.shape[1], history.shape[2]))

    if input_x_col is None:
        input_x = data[-n_input:, :]
        x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    else:
        input_x = data[-n_input:, input_x_col]
        x = input_x.reshape((1, len(input_x), 1))
    y_hat = model.predict(x, verbose=0)
    return y_hat[0]


def gen_forecast_func(n_input: int,
                      input_x_col: Optional[int] = None) -> Callable:
    return partial(forecast, n_input=n_input, input_x_col=input_x_col)


def evaluate_forecasts(actual: np.ndarray, predicted: np.ndarray) -> (float, List[float]):
    """
    评估预测结果
    :param actual: test数据
    :param predicted: 预测数据
    :return:
    """
    scores = []

    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)

    sc = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            sc += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(sc / (actual.shape[0] * actual.shape[1]))
    return score, scores


def evaluate_model(model: models_type,
                   train: np.ndarray,
                   test: np.ndarray,
                   forecast_func: Callable[[models_type, np.ndarray], float]):
    """

    :param model:
    :param train:
    :param test:
    :param forecast_func:
    :return:
    """
    history = train
    predictions = []
    for i in range(len(test)):
        y_hat_sequence = forecast_func(model, history)
        predictions.append(y_hat_sequence)
        new_his = [test[i, :]]
        history = np.concatenate((history, new_his))
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def build_model(train_x, train_y):
        # define parameters
        verbose, epochs, batch_size = 0, 20, 4
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # define model
        model = keras.Sequential()
        model.add(
            keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(n_outputs))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return model


    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    _train, _test = split_dataset(dataset.values)
    tx, ty = to_supervised(_train, n_input=7, n_output=7, train_x_variate_col=0)
    m = build_model(tx, ty)
    f = gen_forecast_func(7, 0)
    s, ss = evaluate_model(m, _train, _test, f)
    summarize_scores('tmp_model', s, ss)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

    plt.plot(days, ss, marker='o', label='cnn')
    plt.show()
