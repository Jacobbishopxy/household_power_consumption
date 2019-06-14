# 4 Strategies for Multi-Step Time Series Forecasting

Predicting multiple time steps into the future is called multi-step time series forecasting. 
There are four main strategies that you can use for multi-step forecasting.

In this post, you will discover the four main strategies for multi-step time series forecasting.

After reading this post, you will know:

* The difference between one-step and multiple-step time series forecasts.
* The traditional direct and recursive strategies for multi-step forecasting.
* The newer direct-recursive hybrid and multiple output strategies for multi-step forecasting.

## 1. Direct Multi-step Forecast Strategy

The direct method involves developing a separate model for each forecast time step.

In the case of predicting the temperature for the next two days, we would develop a model for predicting the 
temperature on day 1 and a separate model for predicting the temperature on day 2.

For example:
```
prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model2(obs(t-2), obs(t-3), ..., obs(t-n))
```

Having one model for each time step is an added computational and maintenance burden, especially as the number
of time steps to be forecasted increases beyond the trivial.

Because separate models are used, it means that there is no opportunity to model the dependencies between the
predictions, such as the prediction on day 2 being dependent on the prediction in day 1, as is often the case
in time series.

## 2. Recursive Multi-step Forecast

The recursive strategy involves using a one-step model multiple times where the prediction for the prior time
step is used as an input for making a prediction on the following time step.

In the case of predicting the temperature for the next two days, we would develop a one-step forecasting model. 
This model would then be used to predict day 1, then this prediction would be used as an observation input in
order to predict day 2.

For example:
```
prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model(prediction(t+1), obs(t-1), obs(t-2), ..., obs(t-n))
```

Because predictions are used in place of observations, the recursive strategy allows prediction errors to 
accumulate such that performance can quickly degrade as the prediction time horizon increases.

## 3. Direct-Recursive Hybrid Strategies

The direct and recursive strategies can be combined to offer the benefits of both methods.

For example, a separate model can be constructed for each time step to be predicted, but each model may use the 
predictions made by models at prior time steps as input values.

We can see how this might work for predicting the temperature for the next two days, where two models are used, 
but the output from the first model is used as an input for the second model.

For example:
```
preiction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
preiction(t+2) = model2(prediction(t+1), obs(t-1), obs(t-2), ..., obs(t-n))
```

Combining the recursive and direct strategies can help to overcome the limitations of each.

## 4. Multiple Output Strategy

The multiple output strategy involves developing one model that is capable of predicting the entire forecast 
sequence in a one-shot manner.

In the case of predicting the temperature for the next two days, we would develop one model and use it to predict
the next two days as one operation.

For example:
```
prediction(t+1), prediction(t+2) = model(obs(t-1), obs(t-2), ..., obs(t-n))
```

Multiple output models are more complex as they can learn the dependence structure between inputs and outputs as 
well as between outputs.

Being more complex may mean that they are slower to train and require more data to avoid overfitting the problem.
