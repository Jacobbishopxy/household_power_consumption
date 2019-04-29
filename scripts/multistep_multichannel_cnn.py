"""
@author Jacob
@time 2019/04/22

Multi-step Time Series Forecasting With a Multichannel CNN

In this section, we will updated the CNN developed tn the previous section to use each of
the eight time series variables to predict the next standard week of daily total power
consumption.

We will do this by providing each one-dimensional time series to the model as a separate
channel of input.

The CNN will then use a separate kernel and read each input sequence onto a separate set
of filter maps, essentially learning features from each input time series variable.

This is helpful for those problems where the output sequence is some function of the
observations at prior time steps from multiple different features, not just (or including)
the feature being forecasted. It is unclear whether this is the case in the power
consumption problem, but we can explore it nonetheless.

"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from scripts.multistep_univariate_cnn import evaluate_model, split_dataset, summarize_scores

"""
First, we must update the preparation of the training data to include all of the eight 
features, not just the one total daily power consumed.
"""


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
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


"""
We also must update the function used to make forecasts with the fit model to use all eight
features from the prior time steps.
"""


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
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verboase=0)
    return yhat[0]  # we only want the vector forecast


"""
We will use 14 days of prior obsevations across eight of the input variables as we did in the
final section of the prior section that resulted in slightly better performance.
n_input=14

Finally, the model used in the previous section does not perform well on the new framing of
the problem.

The increase in the amount of data requires a larger and more sophisticated model that is
trained for longer.

With a little trial and error, one model that performs well uses two convolutional layers with
32 filter maps followed by pooling, then another convolutional layer with 16 feature maps and 
pooling. The fully connected layer that interprets the features is increased to 100 nodes and
the model is fit for 70 epochs with a batch size of 16 samples.
"""


def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


"""
We now have all of the elements required to develop a multi-channel CNN for multivariate input
data to make multi-step time series forecasts.
"""

if __name__ == '__main__':
    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    train_, test_ = split_dataset(dataset.values)
    n_input_ = 7
    score_, scores_ = evaluate_model(train_, test_, n_input_)
    summarize_scores('cnn', score_, scores_)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores_, marker='o', label='cnn')
    plt.show()

    """
    Running the example fits and evaluates the model, printing the overall RMSE across all seven
    days, and the per-day RMSE for each lead time.
    
    We can see that in this case, the use of all eight input variables does result in another
    small drop in the overall RMSE score.
    
    For the daily RMSE scores, we do see that some are better and some are worse than the univariate
    CNN from the previous section.
    
    The final day, Saturday, remains a challenging day to forecast, and Friday an easy day to 
    forecast. There may be some benefit in designing models to focus specifically on reducing the
    error of the harder to forecast days.
    
    It may be interesting to see if the variance across daily scores could be further reduced with a
    tuned model or perhaps an ensemble of multiple different models. It may also be interesting to
    compare the performance for a model that uses seven or even 21 days of input data to see if 
    further gains can be made.
    """
