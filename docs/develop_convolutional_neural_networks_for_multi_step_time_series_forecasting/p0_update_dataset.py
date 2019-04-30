"""
@author Jacob
@time 2019/04/15

0. data pre-processing

"""

import numpy as np
import pandas as pd

dataset = pd.read_csv(
    '../data/household_power_consumption.txt',
    sep=';',
    header=0,
    low_memory=False,
    infer_datetime_format=True,
    parse_dates={'datetime': [0, 1]},
    index_col=['datetime']
)
dataset.replace('?', np.nan, inplace=True)
dataset = dataset.astype('float32')


def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if np.isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]


# fill missing, fill na by last value
fill_missing(dataset.values)

# add a column for the remainder of sub metering
v = dataset.values
dataset['sub_metering_4'] = (v[:, 0] * 1000 / 60) - (v[:, 4] + v[:, 5] + v[:, 6])

# save updated dataset
dataset.to_csv('../data/household_power_consumption.csv')
