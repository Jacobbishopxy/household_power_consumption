"""
@author Jacob
@time 2019/04/30

LSTMs for Multi-Step Forecasting

Recurrent neural networks, or RNNs, are specifically designed to work, learn, and predict sequence data.

A recurrent neural network is a neural network where the output of the network from one time step is
provided as an input in the subsequent time step. This allows the model to make a decision as to what to
predict based on both the input for the current time step and direct knowledge of what was output in
the prior time step.

Perhaps the most successful and widely used RNN is the long short-term memory network, or LSTM for short.
It is successful because it overcomes the challenges involved in training a recurrent neural network,
resulting in stable models. In addition to harnessing the recurrent connection of the outputs from the
prior time step, LSTMs also have an internal memory that operates like a local variable, allowing them
to accumulate state over the input sequence.

**************
For more information about Recurrent Neural Networks, see the post:
* Crash Course in Recurrent Neural Networks for Deep Learning
link: https://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/

For more information about Long Short-Term Memory networks, see the post:
* A Gentle Introduction to Long Short-Term Memory Networks by the Experts
link: https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/

**************
LSTMs offer a number of benefits when it comes to multi-step time series forecasting; they are:

1. Native Support for Sequences. LSTMs are a type of recurrent network, and as such are designed to
take sequence data as input, unlike other models where lag observations must be presented as input
features.

2. Multivariate Inputs. LSTMs directly support multiple parallel input sequences for multivariate
inputs, unlike other models where multivariate inputs are presented in a flat structure.

3. Vector Output. Like other neural networks, LSTMs are able to map input data directly to an output
vector that may represent multiple output time steps.

Further, specialized architectures have been developed that are specifically designed to make multi-
step sequence predictions, generally referred to as sequence-to-sequence prediction, or seq2seq for
short. This is useful as multi-step time series forecasting is a type of seq2seq prediction.

An example of a recurrent neural network architecture designed for seq2seq problems is the encoder-
decoder LSTM.

An encoder-decoder LSTM is a model comprised of two sub-models: one called the encoder that reads
the input sequences and compresses it to a fixed-length internal representation, and an output model
called the decoder that interprets the internal representation and uses it to predict the output
sequence.

The encoder-decoder approach to sequence prediction has proven much more effective than outputting a
vector directly and is the preferred approach.

Generally, LSTMs have been found to not be very effective at auto-regression type problems. These are
problems where forecasting the next time step is a function of recent time steps.

For more on this issue, see the post:
On the Suitability of LSTMs for Time Series Forecasting
link: https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/

One-dimensional convolutional neural networks, or CNNs, have proven effective at automatically learning
features from input sequences.

A popular approach has been to combine CNNs with LSTMs, where the CNN is as an encoder to learn features
from sub-sequences of input data which are provided as time steps to an LSTM. This architecture is called
a CNN-LSTM.

For more information on this architecture, see the post:
CNN Long Short-Term Memory Networks
link: https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

A power variation on the CNN LSTM architecture is the ConvLSTM that uses the convolutional reading of input
sub-sequences directly within an LSTM's units. This approach has proven very effective for time series
classification and can be adapted for use in multi-step time series forecasting.

In this tutorial, we will explore a suite of LSTM architectures for multi-step time series forecasting.
Specifically, we will look at how to develop the following models:
1. LSTM model with vector output for multi-step forecasting with univariate input data.
2. Encoder-Decoder LSTM model for multi-step forecasting with univariate input data.
3. Encoder-Decoder LSTM model for multi-step forecasting with multivariate input data.
4. CNN-LSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
5. ConvLSTM Encoder-Decoder model for multi-step forecasting with univariate input data.

The models will be developed and demonstrated on the household power prediction problem. A model is considered
skillful if it achieves performance better than a naive model, which is an overall RMSE of about 465 kilowatts
across a seven day forecast.

We will not focus on the tuning of these models to achieve optimal performance; instead, we will stop short at
skillful models as compared to a naive forecast. The chosen structures and hyperparameters are chosen with a
little trial and error. The scores should be taken as just an example rather than a study of the optimal model
or configuration for the problem.

We cannot konw which approach will be the most effective for a given multi-step forecasting problem. It is a
good idea to explore a suite of methods in order to discover what works best on your specific dataset.


"""
