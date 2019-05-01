"""
@author jacob
@time 5/1/2019

Encoder-Decoder LSTM Model With Univariate Input

In this section, we can update the vanilla LSTM to use an encoder-decoder model.
This means that the model will not output a vector sequence directly. Instead, the model will be
comprise of two sub models, the encoder to read and encode the input sequence, and the decoder
that will read the encoded input sequence and make a one-step prediction for each element in the
output sequence.

The difference is subtle, as in practice both approaches do in fact predict a sequence output.
The important difference is that an LSTM model is used in the decoder, allowing it to both know
what was predicted for the prior day in the sequence and accumulate internal state while outputting
the sequence.

*****************
As before, we define an LSTM hidden layer with 200 units. This is the decoder model that will read
the input sequence and will output a 200 element vector (one output pre unit) that captures features
from the input sequence. We will use 14 days of total power consumption as input.

```
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
```

We will use a simple encoder-decoder architecture that is easy to implement in Keras, that has a lot
of similarity to the architecture of an LSTM autoencoder.

First, the internal representation of the input sequence is repeated multiple times, once for each
time step in the output sequence. This sequence of vectors will be presented to the LSTM decoder.

```
model.add(RepeatVector(7))
```

We then define the decoder as an LSTM hidden layer with 200 units. Importantly, the decoder will
output the entire sequence, not just the output at the end of the sequence as we did with the encoder.
This means that each of the 200 units will output a value for each of the seven days, representing
the basis for what to predict for each day in the output sequence.

```
model.add(LSTM(200, activation='relu', return_sequences=True))
```

We will then use a fully connected layer to interpret each time step in the output sequence before the
final output layer. Importantly, the output layer predicts a single step in the output sequence, not
all seven days at a time.
This means that we will use the same layers applied to each step in the output sequence. It means that
the same fully connected layer and output layer will be used to process each time step provided by the
decoder. To achieve this, we will wrap the interpretation layer and the output layer in a TimeDistributed
wrapper that allows the wrapped layers to be used for each time step from the decoder.

> TimeDistributed wrapper:
> https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

```
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
```

This allows the LSTM decoder to figure out the context required for each step in the output sequence and
the wrapped dense layers to interpret each time step separately, yet reusing the same weights to perform
the interpretation. An alternative would be to flatten all of the structure created by the LSTM decoder
and to output the vector directly.

The network therefore outputs a three-dimensional vector with the same structure as the input, with the
dimensions [samples, timesteps, features].
There is a single feature, the daily total power consumed, and there are always seven features. A single
one-week prediction will therefore have the size: [1, 7, 1].
Therefore, when training the model, we must restructure the output day (y) to have the three-dimensional
structure instead of the two-dimensional structure of [samples, features] used in the previous section.

```
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
```

"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from docs.develop_LSTM_models_for_multi_step_time_series_forecasting. \
    p2_LSTM_model_with_univariate_input_and_vector_output import to_supervised, split_dataset, summarize_scores, \
    forecast, evaluate_forecasts


def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    model = keras.Sequential()
    model.add(keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
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
    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker='o', label='lstm')
    plt.show()


if __name__ == '__main__':
    main()
