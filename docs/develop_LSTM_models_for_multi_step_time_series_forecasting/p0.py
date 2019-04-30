"""
@author Jacob
@time 2019/04/30

How to Develop LSTM Models for Multi-Step Time Series Forecasting of Household Power Consumption

link: https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

**********
Given the rise of smart electricity meters and the wide adoption of electricity generation technology
like solar panels, there is a wealth of electricity usage data available.

This data represents a multivariate time series of power-related variables that in turn could be used
to model and even forecast future electricity consumption.

Unlike other machine learning algorithms, long short-term memory recurrent neural networks are capable
of automatically learning features from sequence data, support multiple-variate data, and can output a
variable length sequences that can be used for multi-step forecasting.

In this tutorial, you will discover how to develop long short-term memory recurrent neural networks
for multi-step time series forecasting of household power consumption.

After completing this tutorial, you will know:

* How to develop and evaluate Univariate and multivariate Encoder-Decoder LSTMs for multi-step time
series forecasting.

* How to develop and evaluate an CNN-LSTM Encoder-Decoder model for multi-step time series forecasting.

* How to develop and evaluate a ConvLSTM Encoder-Decoder model for multi-step time series forecasting.

**********
Tutorial Overview

1. Problem Description
2. Load and Prepare Dataset
3. Model Evaluation
4. LSTMs for Multi-Step Forecasting
5. LSTM Model With Univariate Input and Vector Output
6. Encoder-Decoder LSTM Model With Univariate Input
7. Encoder-Decoder LSTM Model With Multivariate Input
8. CNN-LSTM Encoder-Decoder Model With Univariate Input
9. ConvLSTM Encoder-Decoder Model With Univariate Input


"""


