# How to use Dataset in TensorFlow

[link](https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428)

[code](https://github.com/FrancescoSaverioZuppichini/Tensorflow-Dataset-Tutorial/blob/master/dataset_tutorial.ipynb)

## Generic Overview

Three steps of using Dataset:

1. **Importing Data**. Create a Dataset instance from some data

2. **Create an Iterator**. By using the created dataset to make an Iterator instance to iterate through
the dataset

3. **Consuming Data**. By using the created iterator we can get the elements from the dataset to feed
the model

## Importing Data

### From numpy

```
x = np.random.sample((100, 2))

dataset = tf.data.Dataset.from_tensor_slices(x)
```

We can also pass more than one numpy array, one classic example is when we have a couple of data divided
into features and labels.

```
features, labels = np.random.sample((100,2)), np.random.sample((100,1))

dataset = tf.data.Dataset.from_tensor_slces((features, labels))
```

### From tensors

```
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100,2]))
```

### From a placeholder

This is useful when we want to dynamically change the data inside the Dataset.

```
x = tf.placeholder(tf.float32, shape=[None,2])

dataset = tf.data.Dataset.from_tensor_slices(x)
```

### From generator

We can also initialise a Dataset from a generator, this is useful when we have an array of different
length (e.g. a sequence):

```
seq = np.array([
    [[1]],
    [[2],[3]],
    [[3],[4],[5]]
])

def generator():
    for el in seq:
        yield el


shape = (tf.TensorShape(None, 1))

dataset = tf.data.Dataset().batch(1).from_generator(generator, output_type=tf.int64, output_shape=shape)

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))

# outputs: [[1]] then [[2],[3]] then [[3],[4],[5]]
```

In this case, you also need to specify the types and the shapes of your data that will be used to create the
correct tensors.

### From csv file

You can directly read a csv file into a dataset. Be aware that the iterator will create a dictionary with key as
the column names and values as Tensor with the correct row value.


## Create an Iterator

We have seen how to create a dataset, but how to get our data back? We have to use an `Iterator`, that will give us
the ability to iterate through the dataset and retrieve the real values of the data. There exist four types of 
iterators.

* **One shot**. It can iterate once through a dataset, you **cannot feed** any value to it.

* **Initializable**. You can dynamically change calling its `initializer` operation and passing the new data with 
`feed_dict`. It's basically a bucket that you can fill with stuff.

* **Reinitializable**. It can be initialised from different `Dataset`. Very useful when you have a training dataset
that needs some additional transformation, eg. shuffle, and a testing dataset. It's like using a tower crane to select
a different container.

* **Feedable**. It can be used to select with iterator to use. Following the previous example, it's like a tower crane
that selects which tower crane to use to select which container to take.

### One shot Iterator

Using the first example

```
x = np,random.sample((100,2))

dataset = tf.data.Dataset.from_tensor_slice(x)

iter = dataset.make_one_shot_iterator()
```

Then you need to call `get_next()` to get the tensor that will contain your data

```
el = iter.get_next()
```

We can run `el` in order to see its value

```
with tf.Session() as sess:
    print(sess.run(el))
```

### Initializable Iterator

In case we want to build a dynamic dataset in which we can change the data source at runtime, we can create a dataset 
with a placeholder. Then we can initialize the placeholder using the common `feed-dict` mechanism. This done with an
*initializable iterator*. Using example three from last section

```
x = tf.placeholder(tf.float32, shape=[None,2])

dataset = tf.data.Dataset.from_tensor_slices(x)

data = np.random.sample((100,2))

iter = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iter.initializer, feed_dict={x: data})
    print(sess.run(el))
```

This time we call `make_initializable_iterator`. Then, inside the `sess` scope, we run the `initializer` operation 
in order to pass our data, in this case a random numpy array.

Imagine that now we have a train set and a test set, a real common scenario:

```
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))
```

Then we would like to train the model and then evaluate it on the test dataset, this can be done by initialising the 
iterator again after training.

```
# initializable iterator to switch between dataset
EPOCHS = 10

x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None,1])

dataset = tf.data.Dataset.from_tensor_slice((x,y))

train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))

iter = dataset.make_initializable_iterator()

feature, labels = iter.get_next()

with tf.Session() as sess:
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1]})
    for _ in range(EPOCHS):
        sess.run([features, labels])
    # switch to test data
    sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1]})
    print(sess.run([features, labels]))
```

### Reinitializable Iterator

The concept is similar to before, we want to dynamic switch between data. But instead of feed new data to the same 
dataset, we switch dataset. As before, we want to have a train dataset and a test dataset.

```
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))

# create two dataset for training and test
train_dataset = tf.data.Dataset.from_tensor_slces(train_data)
test_dataset = tf.data.Dataset.from_tensor_slces(test_data)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

# get the next element as before
features, labels = iter.get_next()

with tf.Session() as sess:
    # switch to train dataset
    sess.run(train_init_op)
    for _ in range(EPOCHS):
        sess.run([features, labels])
    # switch to val dataset
    sess.run(test_init_op)
```

###  Feedable Iterator

This is very similar to the `reinitializable` iterator, but instead of switch between datasets, it switch between 
iterators.

```
# feedable iterator to swtich between iterators
EPOCHS = 10

train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))

# create placeholder
x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])

# create two datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x,y))

# create the iterators from the dataset
train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()

# the out placeholder that can be dynamically changed
handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
next_elements = iter.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    
    # initialise iterators
    sess.run(train_iterator.initializer, feed_dict={x: train_data[0], y: train_data[1]})
    sess.run(test_iterator.initializer, feed_dict={x: test_data[0], y: test_data[1]})
    
    for _ in range(EPOCHS):
        x, y = sess.run(next_elements, feed_dict={handle: train_handle})
        print(x, y)
    
    x, y = sess.run(next_elements, feed_dict={handle: test_handle})
    print(x, y)
```

## Consuming data

In the previous example we have used the session to print the value of the `next` element in the Dataset.

In order to pass the data to a model we have to just pass the tensors generated from `get_next()`

In the following snippet we have a Dataset that contains two numpy arrays, using the same example from the first 
section. Notice that we need to wrap the `.random.sample` in another numpy array to add a dimension that is needed to 
batch the data.

```
EPOCHS = 10
BATCH_SIZE = 16
# using two numpy arrays
features, labels = (np.array([np.random.sample((100,2))]), 
                    np.array([np.random.sample((100,1))]))

dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat().batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()

# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh)  # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, y)  # pass the second value from iter.get_next() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print(f'Iter: {i}, Loss: {loss_value}')
```

## Useful Stuff

### Batch

Usually batching data is a pain, with the `Dataset` API we can use the method `batch(BATCH_SIZE)` that automatically 
batches the dataset with the provided size. The default value is one. In the following example, we use a batch 
size of 4

```
BATCH_SIZE = 4

x = np.random.sample((100,2))

dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()

el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))
```

### Repeat

Using `.repeat()` we can specify the number of times we want the dataset to be iterated. If no parameter is passed it 
will loop forever, usually is good to just loop forever and directly control the number of epochs with a standard loop.

### Shuffle

We can shuffle the Dataset by using the method `shuffle()` that shuffles the dataset by default every epoch.

*Remember: shuffle the dataset is very important to avoid overfitting.*

We can also set the parameter `buffer_size`, a fixed size buffer from which the next element will be uniformly 
chosen from.

```
BATCH_SIZE = 4

x = np.array([[1],[2],[3],[4]])

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.shffle(buffer_size=100)
dataset = dataset.batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))
```

### Map

You can apply a custom function to each member of a dataset using the `map` method. In the following example we multiply 
each element by two:

```
x = np.array([[1],[2],[3],[4]])

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(lambda x: x*2)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    for _ in range(len(x)):
        print(sess.run(el))
```

## Full example

### Initializable iterator

In the example below we train a simple model using batching and we switch between train and test dataset using a 
*Initializable iterator*

```python
# Wrapping all together -> Switch between train and test set using Initailizable iterator
import tensorflow as tf
import numpy as np

EPOCHS = 10
# create a placeholder to dynamically switch between batch sized 
batch_size = tf.placeholder(tf.int64)
BATCH_SIZE = 32

x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()

# using two numpy arrays
train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
test_data = (np.random.sample((20, 2)), np.random.sample((20, 1)))

itr = dataset.make_initializable_iterator()
features, labels = itr.get_next()

# make a simple model
net = tf.layers.dense(features, 8, activation=tf.tanh)  # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, labels)  # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

n_batches = train_data[0].shape[0] // BATCH_SIZE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(itr.initializer, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    print('Training...')
    for i in range(EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
            print("Itr: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
        # initialise iterator with test data
        sess.run(itr.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape})
        print('Test Loss: {:4f}'.format(sess.run(loss)))

```

**Notice that we use a placeholder for the batch size in order to dynamically switch it after training**

### Reinitializable Iterator

In the example below we train a simple model using batching and we switch between train and test dataset using 
*Reinitializable Iterator*

```python
# Wrapping all together -> Switch between train and test set using Reinitializable iterator
import tensorflow as tf
import numpy as np

EPOCHS = 10

# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)

x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)  # always batch even if you want to one shot it

# using two numpy arrays
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((20,2)), np.random.sample((20,1)))

# create an iterator of the correct shape and type
itr = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = itr.get_next()

# create the initialisation operations
train_init_op = itr.make_initializer(train_dataset)
test_init_op = itr.make_initializer(test_dataset)

# make a simple model
net = tf.layers.dense(features, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

n_batches = train_data[0].shape[0] // 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(train_init_op, feed_dict={x: train_data[0], y: test_data[1], batch_size: 16})
    print('Training...')
    for i in range(EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(test_init_op, feed_dict = {x : test_data[0], y: test_data[1], batch_size:len(test_data[0])})
    print('Test Loss: {:4f}'.format(sess.run(loss)))
```


