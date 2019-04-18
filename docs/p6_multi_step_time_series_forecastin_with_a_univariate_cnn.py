"""
@author Jacob
@time 2019/04/17

6. multi-step time series forecasting with a univariate cnn

"""

import numpy as np
import keras

"""

In this section, we will develop a convolutional neural network for multi-step
time series forecasting using only the univariate sequence of daily power
consumption.

> Given some number of prior days of total daily power consumption, predict the
> next standard week of daily power consumption.

The number of prior days used as input defines the one-dimensional(1D) subsequence
of data that the CNN will read and learn to extract features. Some ideas on the
size and nature of this input include:

* All prior days, up to years worth of data.
* The prior seven days.
* The prior two weeks.
* The prior one month.
* The prior one year.
* The prior week and the week to be predicted from one year ago.

There is no tight answer; instead, each approach and more can be tested and the
performance of the model can be used to choose the nature of the input that 
results in the best model performance.

These choices define a few things about the implementation, such as:
* How the training data must be prepared in order to fit the model.
* How the test data must be prepared in order to evaluate the model.
* How to use the model to make predictions with a final model in the future.


A good starting point would be to use the prior seven days.
A 1D CNN model expects data to have the shape of:
[samples, time-steps, features]

One sample will be comprised of seven time steps with one feature for the
seven days of total daily power consumed.

The training dataset has 159 weeks of data, so the shape of the training dataset
would be: [159, 7, 1]

This is a good start. The data in this format would use the prior standard week
to predict the next standard week. A problem is that 159 instances is not a 
lot for a neural network.

A way to create a lot more training data is to change the problem during training
to predict the next seven days given the prior seven days, regardless of the 
standard week.

This only impacts the training data, the test problem remains the same: predict
the daily power consumption for the next standard week given the prior standard
week.

The training data is provided in standard weeks with eight variables, specifically
in the shape [159, 7, 8]. The first step is to flatten the data so that we have
eight time series sequences.

We then need to iterate over the time steps and divide the data into overlapping
windows; each iteration moves along one time step and predicts the subsequent
seven days.

For example:
-------------------------------------------------------------------------
|               Input                               Output              |
|   [d01,d02,d03,d04,d05,d06,d07],      [d08,d09,d10,d11,d12,d13,d14]   |
|   [d02,d03,d04,d05,d06,d07,d08],      [d09,d10,d11,d12,d13,d14,d15]   |
|   ...                          ,      ...                             |
-------------------------------------------------------------------------

We can fo this by keeping track of start and end indexes for the inputs and 
outputs as we iterate across the length of the flattened data in terms of time
steps.

We can also do this in a way where the number of inputs and outputs are
parameterized (e.g. n_input, n_out) so that you can experiment with different
values or adapt it for your own problem.

"""


def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0

    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out

        if out_end < len(data):
            x_input = data[in_start: in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)


"""
When we run this function on the entire training dataset, we transform 159
samples into 1099; specifically, the transformed dataset has the shapes
X=[1099, 7, 1] and y=[1099, 7].

Next, we can define and fit the CNN model on the training data.

This multi-step time series forecasting problem is an auto-regression.
That means it is likely best modeled where that the next seven days is some
function of observations at prior time steps. This and the relatively small
amount of data means that a small model is required.

We will use a model with one convolution layer with 16 filters and a kernel 
size of 3. This means that the input sequence of seven days will be read with 
a convolutional operation three time steps at a time and this operation will 
be performed 16 times. A pooling layer will reduce these feature maps by 1/4
their size before the internal representation is flattened to one long vector.
This is then interpreted by a fully connected layer before the output layer
predicts the next seven days in the sequence.

We will use the mean squared error loss function as it is a good match for our
chosen error metric of RMSE. We will use the efficient Adam implementation of
stochastic gradient descent and fit the model for 20 epochs with a batch size
of 4.

The small batch size and the stochastic nature of the algorithm means that
the same model will learn a slightly different mapping of inputs to outputs 
each time it is trained. This means results may vary when the model is 
evaluated. You can try running the model multiple times and calculating an 
average of model performance.

"""


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define params
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
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


"""
Now that we know how to fit the model, we can look at how the model can be used 
to make a prediction.

Generally, the model expects data to have the same three dimensional shape when 
making a prediction.

In this case, the expected shape of an input pattern is one sample, seven days of 
one feature for the daily power consumed: [1, 7, 1]

Data must have this shape when making predictions for the test set and when a 
final model is being used to make predictions in the future. If you change the
number of input days to 14, then the shape of the training data and the shape of 
new samples when making predictions must be changed accordingly to have 14 time
steps. It is a modeling choice that you must carry forward when using the model.

We are using walk-forward validation to evaluate the model as described in the
previous section.

This means that we have the observations available for the prior week in order to
predict the coming week. These are collected into an array of standard weeks,
called history.

In order to predict the next standard week, we need to retrieve the last days of
observations. As with the training data, we must flatten the history data to
remove the weekly structure so that we end up with eight parallel time series.

Next, we need to retrieve the last seven days of daily total power consumed 
(feature number 0). We will parameterize as we did for the training data so that
the number of prior days used as input by the model can be modified in the future.

Next, we reshape the input into the expected three-dimensional structure.

We then make a prediction using the fit model and the input data and retrieve
the vector of seven days of output.



"""


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
    yhat = yhat[0]
    return yhat
