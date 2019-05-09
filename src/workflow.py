"""
@author Jacob
@time 2019/05/06
"""

from typing import Union, List, Callable
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from math import sqrt
from functools import partial
from collections import namedtuple
from enum import Enum

keras_model = Union[keras.models.Model, keras.models.Sequential]

_params = [
    'input_x_col',
    'train_x_variate_col',
    'train_y_target_col'
]


class ModelType(Enum):
    UNIVARIATE = namedtuple('UNIVARIATE', _params)
    UNIVARIATE.__new__.__defaults__ = (0, 0, 0)

    MULTICHANNEL = namedtuple('MULTICHANNEL', _params)
    MULTICHANNEL.__new__.__defaults__ = (0, 0, 0)

    MULTIHEAD = namedtuple('MULTIHEAD', _params)
    MULTIHEAD.__new__.__defaults__ = (0, 0, 0)


model_types = Union[
    ModelType.UNIVARIATE.value,
    ModelType.MULTICHANNEL.value,
    ModelType.MULTIHEAD.value
]


def split_dataset(data):
    train, test = data[1: -328], data[-328: -6]
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


def summarize_scores(name, score, scores):
    s_scores = ','.join(['%.1f' % i for i in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_supervised(model_type: model_types,
                  train: np.ndarray,
                  n_input: int,
                  n_output: int) -> (np.ndarray, np.ndarray):
    """
    训练数据转换为train_x和train_y
    :param model_type: 带参数的model类型
    :param train: 训练数据
    :param n_input:
    :param n_output:
    :return:
    """
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    x, y = [], []

    train_y_target_col = model_type.train_y_target_col
    train_x_variate_col = model_type.train_x_variate_col

    for i in range(len(data) - n_input - n_output):
        in_end = i + n_input
        out_end = in_end + n_output
        x.append(data[i:in_end, :])
        y.append(data[in_end:out_end, train_y_target_col])

    x = np.array(x)
    y = np.array(y)

    if isinstance(model_type, ModelType.UNIVARIATE.value):
        x = x[:, :, train_x_variate_col]
        x = x.reshape((x.shape[0], x.shape[1], 1))

    return x, y


def forecast(model_type: model_types,
             model: keras_model,
             history: np.ndarray,
             n_input: int) -> float:
    """
    walk forward validation: sliding window

    test数据放入model对未来进行预测
    :param model_type: 带参数的model类型
    :param model:
    :param history:
    :param n_input:
    :return:
    """
    data = history.reshape((history.shape[0] * history.shape[1], history.shape[2]))

    input_x_col = model_type.input_x_col

    if isinstance(model_type, ModelType.UNIVARIATE.value):
        input_x = data[-n_input:, input_x_col]
        x = input_x.reshape((1, len(input_x), 1))
    elif isinstance(model_type, ModelType.MULTICHANNEL.value):
        input_x = data[-n_input:, :]
        x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    elif isinstance(model_type, ModelType.MULTIHEAD.value):
        input_x = data[-n_input:, :]
        x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
    else:
        raise Exception("model_type: UNIVARIATE, MULTICHANNEL, MULTIHEAD")

    y_hat = model.predict(x, verbose=0)
    return y_hat[0]


def gen_forecast_func(model_type: model_types, n_input: int) -> Callable:
    return partial(forecast, model_type=model_type, n_input=n_input)


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


def evaluate_model(model: keras_model,
                   train: np.ndarray,
                   test: np.ndarray,
                   forecast_func: Callable):
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
        y_hat_sequence = forecast_func(model=model, history=history)
        predictions.append(y_hat_sequence)
        new_his = [test[i, :]]
        history = np.concatenate((history, new_his))
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def ev(model_type: model_types,
       data: np.concatenate,
       build_model_func: Callable[[np.ndarray, np.ndarray], keras_model],
       n_input: int,
       n_output: int):
    """

    :param model_type:
    :param data:
    :param build_model_func:
    :param n_input:
    :param n_output:
    :return:
    """

    train, test = split_dataset(data)
    train_x, train_y = to_supervised(model_type,
                                     train,
                                     n_input=n_input,
                                     n_output=n_output)
    model = build_model_func(train_x, train_y)
    forecast_func = gen_forecast_func(model_type=model_type, n_input=n_input)
    score, scores = evaluate_model(model,
                                   train,
                                   test,
                                   forecast_func=forecast_func)
    summarize_scores('eval_model', score, scores)
    return scores


def viz(d):
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, d, marker='o', label='cnn')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.models import *

    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])

    univariate_model = ModelType.UNIVARIATE.value()
    univariate_scores = ev(univariate_model, dataset.values, build_univariate_model, n_input=7, n_output=7)
    viz(univariate_scores)

    multichannel_model = ModelType.MULTICHANNEL.value()
    multichannel_scores = ev(multichannel_model, dataset.values, build_multichannel_model, n_input=14, n_output=7)
    viz(multichannel_scores)

    multihead_model = ModelType.MULTIHEAD.value()
    multihead_scores = ev(multihead_model, dataset.values, build_multihead_model, n_input=14, n_output=7)
    viz(multihead_scores)
