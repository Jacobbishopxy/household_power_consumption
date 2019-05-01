"""
@author Jacob
@time 2019/04/17

3. train and test sets

"""

import numpy as np
import pandas as pd

"""

We will use the first three years of data for training predictive models and the
final year for evaluating models.

The data in a given dataset will be divided into standard weeks. These are weeks
that begin on a Sunday and end on a Saturday.

This is a realistic and useful way for using the chosen framing of the model,
where the power consumption for the week ahead can be predicted. It is also 
helpful with modeling, where models can be used to predict a specific day (e.g. 
Wednesday) or the entire sequence.

We will split the data into standard weeks, working backwards from the test dataset.

The final year of the data is in 2010 and the first Sunday for 2010 was January 
3rd. The data ends in mid November 2010 and the closet final Saturday in the data 
is November 20th. This gives 46 weeks of test data.

Organizing the data into standard weeks gives 159 full standard weeks for training
a predictive model.


"""


def split_dataset(data):
    _train, _test = data[1:-328], data[-328:-6]

    _train = np.array(np.split(_train, len(_train) / 7))
    _test = np.array(np.split(_test, len(_test) / 7))
    return _train, _test


dataset = pd.read_csv('../../data/household_power_consumption_days.csv',
                      header=0,
                      infer_datetime_format=True,
                      parse_dates=['datetime'],
                      index_col=['datetime'])

train, test = split_dataset(dataset.values)

print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])

print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])

