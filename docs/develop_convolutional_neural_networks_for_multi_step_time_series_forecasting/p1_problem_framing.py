"""
@author Jacob
@time 2019/04/16

1. problem framing

"""

import pandas as pd


"""

>    Given recent power consumption, what is the expected power consumption
>    for the week ahead?

This requires that a predictive model forecast the total active power for
each day over the next seven days.

Technically, this framing of the problem is referred to as multi-step time
series forecasting problem, given the multiple forecast steps. A model that
makes use of multiple input variables may be referred to as a multivariate
multi-step time series forecasting model.

A model of this type could be helpful within the household in planning
expenditures. It could also be helpful on the supply side for planning
electricity demand for a specific household. 

This framing of the dataset also suggests that it would be useful to 
down-sample the per-minute observations of power consumption to daily totals.
This is not required, but makes sense, given that we are interested in total
power per day.

"""


dataset = pd.read_csv('../../data/household_power_consumption.csv',
                      header=0,
                      infer_datetime_format=True,
                      parse_dates=['datetime'],
                      index_col=['datetime'])

daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

print(daily_data.shape)
print(daily_data.head())

daily_data.to_csv('../../data/household_power_consumption_days.csv')


