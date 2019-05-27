# Household Power Consumption

Check out this 
[tutorial](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)
for starting.

## Extracting Sample data

1. Unzip data/household_power_consumption.rar
2. run docs/develop_convolutional_neural_networks_for_multi_step_time_series_forecasting/p0_update_dataset.py

## Workflow Structure

1. data_preprocessing:
    
    raw data to train and test data

2. input_functions

    data to dataset where containing features and labels
    
    * read from csv
    
    * write/read from `TFRecord`

3. models

    * keras model (compiled)
    
    * keras model (vanilla)

4. workflow_custom_estimator

    * custom estimator using `tf.keras.estimator.model_to_estimator`
    
    * custom estimator using `model_fn`

