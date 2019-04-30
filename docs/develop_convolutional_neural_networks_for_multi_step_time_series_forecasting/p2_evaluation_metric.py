"""
@author Jacob
@time 2019/04/16

2. evaluation metric

"""

from math import sqrt
from sklearn.metrics import mean_squared_error

"""
A forecast will be comprised of seven values, one for each day of the week ahead.

It is common with multi-step forecasting problems to evaluate each forecast
time step separately. This is helpful for a few reasons:
* To comment on the skill at a specific lead time (e.g. +1 day vs +3 days).
* To contrast models based on their skills at different lead times (e.g.
models good at +1 day vs models good at days +5).

The units of the total power are kilowatts and it would be useful to have an 
error metric that was also in the same units. Both Root Mean Squared Error (RMSE)
and Mean Absolute Error (MAE) fit this bill, although RMSE is more commonly used
and will be adopted in this tutorial. Unlike MAE, RMSE is more punishing of
forecast errors. 

The performance metric for this problem will be the RMSE for each lead time from 
day 1 to day 7.

As a short cut, it may be useful to summarize the performance of a model using a
single score in order to aide in model selection.

One possible score that could be used would be the RMSE across all forecast days.

"""


def evaluate_forecasts(actual, predicted):
    scores = list()
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


