"""
@author jacob
@time 4/30/2019


LSTM Model With Univariate Input and Vector Output


We will start off by developing a simple or vanilla LSTM model that reads in a sequence of days of total
daily power consumption and predicts a vector output of the next standard week of daily power consumption.

This will provide the foundation for the more elaborate models developed in subsequent sections.

The number of prior days used as input defines the one-dimensional (1D) subsequence of data that the LSTM
will read and learn to extract features. Some ideas on the size and nature of this input include:
1. All prior days, up to years worth of data
2. The prior seven days
3. The prior two weeks
4. The prior one month
5. The prior one year
6. The prior week and the week to be predicted from one year ago

There is no right answer; instead, each approach and more can be tested and the performance of the model can
be used to choose the nature of the input that results in the best model performance.

These choices define a few things:
1. How the training data must be prepared in order to fit the model
2. How the test data must be prepared in order to evaluate the model
3. How to use the model to make predictions with a final model in the future

A good starting point would be to use the prior seven days.

An LSTM model expects data to have the shape: [samples, timesteps, features]

One sample will be comprised of seven time steps with one feature for the seven days of total daily power
consumed.

The training dataset has 159 weeks of data, so the shape of the training dataset would be: [159, 7, 1]

The data in this format would use the prior standard week to predict the next standard week.

... (see prev doc)

"""

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import mean_squared_error
from math import sqrt


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
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


"""
This multi-step time series forecasting problem is an autoregression. That means it is likely best modeled
where that the next seven days is some function of observations at prior time steps. This and the relatively
small amount of data means that a small model is required.

We will develop a model with a single hidden LSTM layer with 200 units. The number of units in the hidden 
layer is unrelated to the number of time steps in the input sequences. The LSTM layer is followed by a fully 
connected layer with 200 nodes that will interpret the features learned by the LSTM layer. Finally, an output
layer will directly predict a vector with seven elements, one for each day in the output sequence.

We will use the mean squared error loss function as it is a good match for our chosen error metric of RMSE.
We will use the efficient Adam implementation of stochastic gradient descent and fit the model for 70 epochs
with a batch size of 16.

The small batch size and the stochastic nature of the algorithm means that the same model will learn a slightly
different mapping of inputs to outputs each time it is trained. This means results may vary when the model is
evaluated. You can try running the model multiple times and calculate an average of model performance.
"""


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


"""
Generally, the model expects data to have the same three dimensional shape when making a prediction. In this 
case, the expected shape of an input pattern is one sample, seven days of one feature for the daily power 
consumed: [1, 7, 1].

Data must have this shape when making predictions for the test set and when a final model is being used to make
predictions in the future. If you change the number if input days to 14, then the shape of the training data and
the shape of new samples when making predictions must be changed accordingly to have 14 time steps. It is a 
modeling choice that you must carry forward when using the model.
"""


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    return yhat[0]


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


if __name__ == '__main__':
    pass

    """
    Running the example fits and evaluates the model, printing the overall RMSE across all seven days, and the 
    per-day RMSE for each lead time.
    """
