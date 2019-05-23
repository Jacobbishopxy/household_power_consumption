# Loss and Loss Functions for Training Deep Learning Neural Networks

[link](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

## Overview

This tutorial is divided into seven parts:

1. Neural Network Learning as Optimization
2. What is a Loss Function and Loss?
3. Maximum Likelihood
4. Maximum Likelihood and Cross-Entropy
5. What Loss Function to Use?
6. How to Implement Loss Functions
7. Loss Functions and Reported Model Performance

For help choosing and implementing different loss functions, 
see [here](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)


## Neural Network Learning as Optimization

A deep learning neural network learns to map a set of inputs to a set of outputs from training data.

We cannot calculate the perfect weights for a neural network; there are too many unknowns. Instead, the problem of 
is cast as a search or optimization problem and an algorithm is used to navigate the space of possible sets of weights 
the model may use in order to make good predictions.

Typically, a neural network model is trained using the stochastic gradient descent optimization algorithm and weights
are updated using the back-propagation of error algorithm.

The "gradient" in gradient descent refers to an error gradient. The model with a given set of weights is used to make
predictions and the error for those predictions is calculated.

The gradient descent algorithm seeks to change the weights so that the next evaluation reduces the error, meaning the
optimization algorithm is navigating down the gradient (or slope) of error.


## What Is a Loss Function and Loss?

In the context of an optimization algorithm, the function used to evaluate a candidate solution (i.e. a set of weights)
is referred to as the objective function.

We may seek to maximize or minimize the objective function, meaning that we are searching for a candidate solution that 
has the highest or lowest score respectively. 

Typically, with neural networks, we seek to minimize the error. As such, the objective function is often referred to as 
a cost function or a loss function and the value calculated by the loss function is referred to as simply "loss".

The cost or loss function has an important job in that it must faithfully distill all aspects of the model down into a
single number in such a way that improvements in that number are a sign of a better model.

In calculating the error of the model during the optimization process, a loss function must be chosen. This can be a 
challenging problem as the function must capture the properties of the problem and be motivated by concerns that are 
important to the project and stakeholders.


## Maximum Likelihood

There are many functions that could be used to estimate the error of a set of weights in a neural network.

We prefer a function where the space of candidate solutions maps onto a smooth (but high-dimensional) landscape that the
optimization algorithm can reasonably navigate via iterative updates to the model weights.

Maximum likelihood estimation, ot MLE, is a framework for inference for finding the best statistical estimates of 
parameters from historical training data: exactly what we are trying to do with the neural network.

We have a training dataset with one or more input variables and we require a model to estimate model weight parameters 
that best map examples of the inputs to the output or target variable.

Given input, the model is trying to make predictions that match the data distribution of the target variable. Under 
maximum likelihood, a loss function estimates how closely the distribution of predictions made by a model matches the
distribution of target variables in the training data.

A benefit of using maximum likelihood as a framework for estimating the model parameters (weights) for neural networks 
and in machine learning in general is that as the number of examples in the training dataset is increased, the estimate 
of the model parameters improves. This is called the property of "consistency".


## Maximum Likelihood and Cross-Entropy

Under the framework maximum likelihood, the error between two probability distributions is measured using cross-entropy.
 
When modeling a classification problem where we are interested in mapping input variables to a class label, we can model 
the problem as predicting the probability of an example belonging to each class. In a binary classification problem, 
there would be two classes, so we may predict the probability of the example belonging to the first class. In the case 
of multiple-class classification, we can predict a probability for the example belonging to each of the classes.

In the training dataset, the probability of an example belonging to a given class would be 1 or 0, as each sample in the 
training dataset is a known example from the domain. We know the answer.

Therefore, under maximum likelihood estimation, we would seek a set of model weights that minimize the difference 
between the model's predicted probability distribution given the dataset and the distribution of probabilities in the 
training dataset. This is called the cross-entropy.

Technically, cross-entropy comes from the field of information theory and has the unit of "bits". It is used to estimate
the difference between an estimated and predicted probability distributions.

In the case of regression problems where a quantity is predicted, it is common to use the mean squared error (MSE) loss
function instead.

Nevertheless, under the framework of maximum likelihood estimation and assuming a Gaussian distribution for the target 
variable, mean squared error can be considered the cross-entropy between the distribution of the model predictions and 
the distribution of the target variable.

Almost universally, deep learning neural networks are trained under the framework of maximum likelihood using cross-
entropy as the loss function.

In fact, adopting this framework may be considered a milestone in deep learning, as before being fully formalized, it 
was sometimes common for neural networks for classification to use a mean squared error loss function.

The maximum likelihood approach was adopted almost universally not just because of the theoretical framework, but 
primarily because of the results it produces. Specifically, neural networks for classification that use a sigmoid or
softmax activation function in the output layer learn faster and more robustly using a cross-entropy loss function.


## What Loss Function to Use?

We can summarize the previous section and directly suggest the loss functions that you should use under a framework of 
maximum likelihood.

Importantly, the choice of loss function is directly related to the activation function used in the output layer of your
neural network. These two design elements are connected.

Think of the configuration of the output layer as a choice about the framing of your prediction problem, and the choice 
of the loss function as the way to calculate the error for a given framing of your problem.

### Regression Problem

A problem where you predict a real-value quantity.

* **Output Layer Configuration**: One node with a linear activation unit.
* **Loss Function**: Mean Squared Error (MSE).

### Binary Classification Problem

A problem where you classify an example as belonging to one of two classes.

The problem is framed as predicting the likelihood of an example belonging to class one, e.g. the class that you assign 
the integer value 1, whereas the other class is assigned the value 0.

* **Output Layer Configuration**: One node with a sigmoid activation unit.
* **Loss Function**: Cross-Entropy, also referred to as Logarithmic loss.

### Multi-Class Classification Problem

A problem where you classify an example as belonging to one of more than two classes.

The problem is framed as predicting the likelihood of an example belonging to each class.

* **Output Layer Configuration**: One node for each class using the softmax activation function.
* **Loss Function**: Cross-Entropy, also referred to as Logarithmic loss.


## How to Implement Loss Functions

...


### Loss Functions and Reported Model Performance

Given a framework of maximum likelihood, we know that we want to use a cross-entropy or mean squared error loss function 
under stochastic gradient descent.

Nevertheless, we may or may not want to report the performance of the model using the loss function.

For example, logarithmic loss is challenging to interpret, especially for non-machine learning practitioner 
stakeholders. The same can be said for the mean squared error. Instead, it may be more important to report the accuracy 
and root mean squared error for models used for classification and regression respectively.

It may also be desirable to choose models based on these metrics instead of loss. This is an important consideration, as 
the model with the minimum loss may not be the model with best metric that is important to project stakeholders.

A good division to consider is to use the loss to evaluate and diagnose how well the model is learning. This includes 
all of the considerations of the optimization process, such as overfitting, underfitting, and convergence. An alternate 
metric can then be chosen that has meaning to the project stakeholders to both evaluate model performance and perform 
model selection.  

* **Loss**: Used to evaluate and diagnose model optimization only.
* **Metric**: Used to evaluate and choose models in the context of the project.

The same metric can be used for both concerns but it is more likely that the concerns of the optimization process will 
differ from the goals of the project and different scores will be required. Nevertheless, it is often the case that 
improving the loss improves or, at worst, has no effect on the metric of interest.

