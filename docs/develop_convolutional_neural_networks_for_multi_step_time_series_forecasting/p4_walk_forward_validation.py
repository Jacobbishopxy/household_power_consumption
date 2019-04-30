"""
@author Jacob
@time 2019/04/17

4. wald forward validation

"""

import numpy as np

from docs.develop_convolutional_neural_networks_for_multi_step_time_series_forecasting.p2_evaluation_metric import evaluate_forecasts
from docs.develop_convolutional_neural_networks_for_multi_step_time_series_forecasting.p6_multi_step_time_series_forecastin_with_a_univariate_cnn import build_model


"""

Models will be evaluated using a scheme called wald-forward-validation.

This is where a model is required to make a one week prediction, then the 
actual data for that week is made available to the model so that it can be
used as the basis for making a prediction on the subsequent week. This is
both realistic for how the model may be used in practice and beneficial 
to the models, allowing them to make use of the best available data. 

We can demonstrate this below with separation of input data and 
output/predicted data.

-----------------------------------------------------------------
|              Input                       Predict              |
|              [Week1]                     Week2                |
|              [Week1 + Week2]             Week3                |
|              [Week1 + Week2 + Week3]     Week4                |
|              ...                         ...                  |
-----------------------------------------------------------------

The train and test datasets in standard-week format are provided to the
function as arguments. An additional argument, n_input, is provided that
is used to define the number of prior observations that the model will use
as input in order to make a prediction.

Two new functions are called: one to build a model from the training data 
called build_model() and another uses the model to make forecasts for each 
new standard week, called forecast(). These will be covered in subsequent
sections.

We are working with neural networks and as such they are generally slow to 
train but fast to evaluate. This means that the preferred usage of the 
models is to build them once on historical data and to use them to forecast 
each step of the walk-forward validation. The models are static (i.e. not
updated) during their evaluation.

This is different to other models that are faster to train, where a model
may be re-fit or updated each step of the walk-forward validation as new
data is made available. With sufficient resources, it is possible to use
neural networks this way, but we will not in this tutorial.

"""


def evaluate_model(train, test, n_input):
    model = build_model(train, n_input)

    history = [x for x in train]

    predictions = []

    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])

    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ','.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))



