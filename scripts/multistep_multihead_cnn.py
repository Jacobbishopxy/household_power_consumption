"""
@author Jacob
@time 2019/04/29

Multi-step time series forecasting with a multihead CNN
"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from scripts.multistep_multichannel_cnn import to_supervised, evaluate_forecasts, split_dataset, summarize_scores

"""
We can further extend the CNN model to have a separate sub-CNN model or head for each input variable,
which we can refer to as a multi-headed CNN model.

This requires a modification to the preparation of the model, and in turn, modification to the
preparation of the training and test datasets.

Starting with the model, we must define a separate CNN model for each of the eight input variables.

The configuration of the model, including the number of layers and their hyperparameters, were also
modified to better suit the new approach. The new configuration is not optimal and was found with a 
little trail and error.

The multi-headed model is specified using the more flexible functional API for defining Keras models.

We can loop over each variable and create a sub-model that takes a one-dimensional sequence of 14 days
of data and outputs a flat vector containing a summary of the learned features from the sequence. Each
of these vectors can be merged via concatenation to make one very long vector that is then interpreted 
by some fully connected layers before a prediction is made.
"""


# plot training history
def plot_history(history):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('loss', y=0, loc='center')
    plt.legend()
    # plot rmse
    plt.subplot(2, 1, 2)
    plt.plot(history.history['rmse'], label='train')
    plt.plot(history.history['val_rmse'], label='test')
    plt.title('rmse', y=0, loc='center')
    plt.legend()

    plt.show()


"""
As we build up the submodels, we keep track of the input layers and flatten layers in lists.
This is so that we can specify the inputs in the definition of the model object and use the list of
flatten layers in the merge layer.
"""


# train the model
def build_model(train, n_inputs):
    # prepare data
    train_x, train_y = to_supervised(train, n_inputs)
    # define parameters
    verbose, epochs, batch_size = 0, 25, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    """
    When the model is used, it will require eight arrays as input: one for each of the submodels.

    This is required when training the model, when evaluating the model, and when making predictions
    with a final model.

    We can achieve this by creating a list of 3D arrays, where each 3D array contains [samples, 
    timesteps, 1], with one feature.

    """
    # create a channel for each variable
    in_layers, out_layers = [], []
    for i in range(n_features):
        inputs = keras.layers.Input(shape=(n_timesteps, 1))
        conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        pool1 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        flat = keras.layers.Flatten()(pool1)
        # store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # merge heads
    merged = keras.layers.concatenate(out_layers)
    # interpretation
    dense1 = keras.layers.Dense(200, activation='relu')(merged)
    dense2 = keras.layers.Dense(100, activation='relu')(dense1)
    outputs = keras.layers.Dense(n_outputs)(dense2)
    model = keras.Model(inputs=in_layers, outputs=outputs)
    # compile model
    model.compile(loss="mse", optimizer="adam")
    # plot the model
    # plot_model(model, show_shapes=True, to_file="../data/multiheaded_cnn.png")
    # fit network
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


"""
When the model is built, a diagram of the structure of the model is created and saved to file.

Note: the call to plot_model() requires that pygraphviz and pydot are installed
"""

"""
Next, we can update the preparation of input samples when making a prediction for the test dataset.

We must perform the same change, where an unput array of [1, 14, 8] must be transformed into a list
of eight 3D arrays each with [1, 14, 1].
"""


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into n input arrays
    input_x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    return yhat[0]


# evaluate a single model
def evaluate_model(train, test, n_inputs):
    # fit model
    model = build_model(train, n_inputs)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = []
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_inputs)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def main():
    dataset = pd.read_csv('../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_input_ = 14
    score, scores = evaluate_model(train, test, n_input_)
    summarize_scores('cnn', score, scores)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker='o', label='cnn')
    plt.show()


if __name__ == '__main__':
    main()

    """
    Running the example fits and evaluates the model, printing the overall RMSE across all seven days,
    and the per-day RMSE for each lead time.
    
    We can see that in this case, the overall RMSE is skillful compared to a naive forecast, but with
    the chosen configuration may not perform better than the multi-channel model in the previous section.
    
    We can also see a different, more pronounced profile for the daily RMSE scores where perhaps
    Mon-Tue and Thu-Fri are easier for the model to predict than the other forecast days.
    
    It may be interesting to explore alternate methods in the architecture for merging the output of each
    sub-model.
    """
