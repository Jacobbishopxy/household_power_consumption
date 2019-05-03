# Keras Model class API

The Keras functional API is the way to go for defining complex models, such as multi-output models,
directed acyclic graphs, or models with shared layers.

In the functional API, given some input tensor(s) and output tensor(s).

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

This model will include all layers required in the computation of 'b' given 'a'.

In the case of multi-input or multi-output models, you can use lists as well:

```
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```

## Methods

### compile

```
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

Configures the model for training.

### fit

```
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
```

Trains the model for a given number of epochs (iterations on a dataset).

### evaluate

```
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
```

Returns the loss value $ metrics values for the model in test mode.

### predict

```
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
```

Generates output predictions for the input samples.

## Intro

### A densely-connected network

The `Sequential` model is probably a better choice to implement such a network, but it helps to
start with something really simple.

* A layer instance is callable (on a tensor), and it returns a tensor
* Input tensor(s) and output tensor(s) can then be used to define a `Model`
* Such a model can be trained just like Keras `Sequential` models

```
from keras.layers import Input, Dense
from keras.models import Model

# this returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# this creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# starts training
model.fit(data, labels)
```

### All models are callable, just like layers

With the functional API, it is easy to reuse trained models: you can treat any model as if it 
were a layer, by calling it on a tensor. Note that by calling a model you aren't just reusing
the architecture of the model, you are also reusing its weights.

```
x = Input(shape=(784,))
# this works, and returns the 10-way softmax we defined above.
y = model(x)
```

This can allow, for instance, to quickly create models that can process sequences of inputs.
You could turn an image classification model into a video classification model, in just one line.

```
from keras. layers import TimeDistributed

# Input tensor for sequences of 20 timesteps, each containing a 7840dimensional vector
input_sequences = Input(shape=(20, 784))

# this applies our previoous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax, so the output of the layer below will
# be a sequence of 20 vectors  of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

### Multi-input and multi-output models

Let's consider the following model. We seek to predict how many re-tweets and likes a new headline
will receive on Twitter. The main input to the model will be the headline itself, as a sequence of
words, but to spice things up, our model will also have an auxiliary input, receiving extra data 
such as the time of day when the headline was posted, etc. The model will also be supervised via
two loss functions. Using the main loss function earlier in a model is a good regularization
mechanism for deep models.

![img](./img/multi-input-multi-output-graph.png)

The main input will receive the headline, as a sequence of integers (each integer encodes a word).
The integers will be between 1 and 10,000 (a vocabulary of 10,000 words) and the sequences will be
100 words long.

```
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model

# headline input: meant to receive sequences of 100 integers, between 1 and 100000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence into a sequence of sense 512-dimensional vectors
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector, containing information about
# the entire sequence
lstm_out = LSTM(32)(x)

"""
Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly
even though the main loss will be much higher in the model.
"""
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

"""
At this point, we feed into the model our auxiliary input data by concatenating it with the
LSTM output:
"""
auxiliary_input = Input(shape=(5,), name='aux_input')
x = concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main Logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

"""
This defines a model with two inputs and two outputs:
"""
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

"""
We compile the model and assign a weight of 0.2 to the auxiliary loss. To specify different
`loss_weights` or `loss` for each different output, you can use a list or a dictionary. Here
we paa a single loss as the `loss` argument, so the same loss will be used on all outputs.
"""
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., .2])

"""
We can train the model by passing it lists of input arrays and target arrays:
"""
model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)

"""
Since our inputs and outputs are named (we passed them a "name" argument), we could also have
compiled the model via:
"""

model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, 
          batch_size=32)

```

### Shared layers

Another good use for the functional API are models that use shared layers.

Let's consider a dataset of tweets. We want to build a model that can tell whether two tweets
are from the same person or not(this can allow us to compare users by the similarity of their
tweets, for instance).

One way to achieve this is to build a model that encodes two tweets into two vectors, concatenates
the vectors and then adds a logistic regression; this outputs a probability that the two tweets
share the same author. The model would then be trained on positive tweet pairs and negative tweet
pairs.

Because the problem is symmetric, the mechanism that encodes the first tweet should be reused 
(weights and all) to encode the second tweet. Here we use a shared LSTM layer to encode the tweets.  

Let's build this with the functional API, We will take as input for a tweet a binary matrix of
shape `(280, 256)`, i.e. a sequence of 280 vectors of size 256, where each dimension in the 256-
dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent
characters). 

```
from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))

"""
To share a layer across different inputs, simply instantiate the layer once, then call it on
as many inputs as you want:
"""
# This layer can take as input a matrix and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance multiple times, the weight of the layer are also being reused
# (it is effectively the same** layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

