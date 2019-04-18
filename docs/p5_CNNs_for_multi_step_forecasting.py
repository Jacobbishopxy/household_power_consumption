"""
@author Jacob
@time 2019/04/17

5. CNNs for multi-step forecasting

"""

"""

Convolutional Neural Network models are a type of deep neural network that was
developed for use with image data, such as handwriting recognition.

They are proven very effective on challenging computer vision problems when
trained at scale for tasks such as identifying and localizing objects in images
and automatically describing the content of images.

They are a model that are comprised of two main types of elements: convolutional
layers and pooling layers.

* Convolutional layers read an input, such as a 2D image or a 1D signal using a
kernel that reads in small segments at a time and steps across the entire input
field. Each read results in an interpretation of the input that is projected
onto a filter map and represents an interpretation of the input.

* Pooling layers take the feature map projections and distill them to the most
essential elements, such as using a signal averaging or signal maximizing
process.

The convolution and pooling layers can be repeated at depth, providing multiple
layers of abstraction of the input signals.

The output of these networks is often one or more fully-connected layers that
interpret what has been read and maps this internal representation to a class
value.


Convolutional neural networks can be used for multi-step time series forecasting.

* The convolutional layers can read sequences of input data and automatically
extract features.

* The pooling layers can distill the extracted features and focus attention on
the most salient elements.

* The fully connected layers can interpret the internal representation and
output a vector representing multiple time steps.

The key benefits of the approach are the automatic feature learning and the
ability of the model to output a multi-step vector directly.

CNNs can be used in either a recursive or direct forecast strategy, where the
model makes one-step predictions and outputs are fed as inputs for subsequent
predictions, and where one model is developed for each time step to be predicted.

An important secondary benefit of using CNNs is that they can support multiple
1D inputs in order to make a prediction. This is useful if the multi-step output
sequence is a function of more than one input sequence. This can be achieved
using two different model configurations.

* Multiple Input Channels. This is where each input sequence is read as a 
separate channel, like the different channels of an image (e.g. red, green and
blue).

* Multiple Input Heads. This is where each input sequence is read by a different
CNN sub-model and the internal representations are combined before being 
interpreted and used to make a prediction.


In this tutorial, we will explore how to develop three different types of CNN
models for multi-step time series forecasting; they are:

* A CNN for multi-step time series forecasting with univariate input data.
* A CNN for multi-step time series forecasting with multivariate input data
via channels.
* A CNN for multi-step time series forecasting with multivariate input data
via submodels.

The models will be developed and demonstrated on the household power prediction 
problem. A model is considered skillful if it achieves performance better than
a naive model, which is an overall RMSE of about 465 kilowatts across a seven
day forecast.

We will not focus on thr tuning of these models to achieve optimal performance;
instead we will sill stop short at skillful models as compared to a naive 
forecast.

"""
