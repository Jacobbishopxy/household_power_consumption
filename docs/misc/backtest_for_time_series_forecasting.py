"""
@author Jacob
@time 2019/05/08

How To Backtest Machine Learning Models for Time Series Forecasting

link: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/


The goal of time series forecasting is to make accurate predictions about the future.

The fast and powerful methods that we rely on in machine learning, such as using train-test splits and k-fold
cross validation, do not work in the case of time series data. This is because they ignore the temporal
components inherent in the problem.

After completing this tutorial, you will know:

* The limitations of traditional methods of model evaluation from machine learning and why evaluating models
on out of sample data is required.
* How to create train-test splits and multiple train-test splits of time series data for model evaluation
in Python.
* How walk-forward validation provides the most realistic evaluation of machine learning models on time
series data.

***********
We will look at three different methods that you can use to backtest your machine learning models on time
series problems. They are:

1. Train-Test split that respect temporal order of observations.
2. Multiple Train-Test splits that respect temporal order of observations.
3. Walk-Forward Validation where a model may be updated each time step new data is received.

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def train_test_split_plot(data):
    train_size = int(len(data) * .66)
    train, test = data[0:train_size], data[train_size:]

    plt.plot(train)
    plt.plot([None for _ in train] + test.tolist())
    plt.show()


"""
Multiple Train-Test Splits

We can repeat the process of splitting the time series into train and test sets multiple times.

This will require multiple models to be trained and evaluated, but this additional computational expense will
provide a more robust estimate of the expected performance of the chosen method and configuration on unseen data.

We could do this manually by repeating the process described in the previous section with different split points.

Alternately, the scikit-learn library provides this capability for us in the TimeSeriesSplit object.
You must specify the number of splits to create and the TimeSeriesSplit to return the indexes of the train and
test observations for each requested split.

***********

Using multiple train-test splits will result in more models being trained, and in turn, a more accurate estimate
of the performance of the models on unseen data.

!!! A limitation of the train-test split approach is that the trained models remain fixed as they are evaluated 
on each evaluation in the test set.

This may not be realistic as models can be retrained as new daily or monthly observations are made available.
This concern is addressed in the next section.

"""


def multiple_train_test_split_plot(data, n_splits):
    splits = TimeSeriesSplit(n_splits=n_splits)
    plt.figure(1)
    index = 1
    for train_index, test_index in splits.split(data):
        train = data[train_index]
        test = data[test_index]
        plt.subplot(310 + index)
        plt.plot(train)
        plt.plot([None for _ in train] + test.tolist())
        index += 1
    plt.show()


"""
Walk Forward Validation

In practice, we very likely will retrain our model as new data becomes available. This would give the model the
best opportunity to make good forecasts at each time step. We can evaluate our machine learning models under
this assumption.

There are few decisions to make:

1. Minimum Number of Observations. First, we must select the minimum number of observations required to train 
the model. This may be thought of as the window width if a sliding window is used.

2. Sliding of Expanding Window. Next, we need to decide whether the model will be trained on all data it has 
available or only on the most recent observations. This determines whether a sliding or expanding window will
be used. 

After a sensible configuration is chosen for your test-setup, models can be trained and evaluated.

1. Starting at the beginning of the time series, the minimum number of samples in the window is used to train
a model.
2. The model makes a prediction for the next time step.
3. The prediction is stored or evaluated against the known value.
4. The window is expanded to include the known value and the process is repeated (go to step 1).

Because this methodology involves moving along the time series one-time step at a time, it is often called
Walk Forward Testing or Walk Forward Validation. Additionally, because a sliding or expanding window is used
to train a model, this method is also referred to as Rolling Window Analysis or a Rolling Forecast.

"""


def walk_forward_validation(data, n_train):
    n_records = len(data)
    for i in range(n_train, n_records):
        train, test = data[0: i], data[i: i+1]
        print('train=%d, test=%d' % (len(train), len(test)))


"""
This has the benefit again of providing a much more robust estimation of how the chosen modeling method and
parameters will perform in practice. This improved estimate comes at the computational cost of creating so
many models.

This is not expensive if the modeling method is simple or dataset is small, but could be an issue at scale.
In the above case, 2820 models would be created and evaluated.

As such, careful attention needs to be paid to the window width and window type. These could be adjusted to
contrive a test harness on your problem that is significantly less computationally expensive.
"""


if __name__ == '__main__':
    d = pd.read_csv('../../data/monthly-sunspots.csv',
                    header=0,
                    infer_datetime_format=True,
                    parse_dates=['Month'],
                    index_col=['Month'])
    x = d['Sunspots'].values

    # train_test_split_plot(x)
    multiple_train_test_split_plot(x, 3)

