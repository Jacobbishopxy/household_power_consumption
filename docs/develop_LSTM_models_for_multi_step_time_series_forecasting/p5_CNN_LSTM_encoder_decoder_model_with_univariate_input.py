"""
@author jacob
@time 5/1/2019

CNN-LSTM Encoder-Decoder Model With Univariate Input

A convolutional neural network, or CNN, can be used as the encoder in an encoder-decoder architecture.

The CNN does not directly support sequence input; instead a 1D CNN is capable of reading across sequence
input and automatically learning the salient features. These can then be interpreted by an LSTM decoder
as per normal. We refer to hybrid models that use a CNN and LSTM as CNN-LSTM models, and in this case we
are using them together in an encoder-decoder architecture.

The CNN expects the input data to have the same 3D structure as the LSTM model, although multiple features
are read as different channels that ultimately have the same effect.

***************
We will define a simple but effective CNN architecture for the encoder that is comprised of two
convolutional layers followed by a max pooling layer, the results of which are then flattened.

The first convolutional layer reads across the input sequence and projects the results onto feature maps.
The second performs the same operation on the feature maps created by the first layer, attempting to
amplify any salient features. We will use 64 feature maps per convolutional layer and read the input
sequences with a kernel size of three time steps.

The max pooling layer simplifies the feature maps by keeping 1/4 of the values with the largest (max) signal.
The distilled feature maps after the pooling layer are then flattened into one long vector that can then be
used as input to the decoding process.

"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from docs.develop_LSTM_models_for_multi_step_time_series_forecasting. \
    p2_LSTM_model_with_univariate_input_and_vector_output import split_dataset, to_supervised, summarize_scores, \
    forecast, evaluate_forecasts


def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPool1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.RepeatVector(n_outputs))
    model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def evaluate_model(train, test, n_input):
    model = build_model(train, n_input)
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def main():
    dataset = pd.read_csv('../../data/household_power_consumption_days.csv',
                          header=0,
                          infer_datetime_format=True,
                          parse_dates=['datetime'],
                          index_col=['datetime'])
    train, test = split_dataset(dataset.values)
    n_input = 14
    score, scores = evaluate_model(train, test, n_input)
    summarize_scores('lstm', score, scores)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker='o', label='lstm')
    plt.show()


if __name__ == '__main__':
    main()
