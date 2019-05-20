"""
@author jacob
@time 5/10/2019

Train Keras model with TensorFlow Estimators and Datasets API

This post is focused on converting Keras model to an Estimator -- if we wanted to improve accuracy we could try tuning
our model's hyperparameters, changing our layer size, or adding dropout to our input layer.

link: http://androidkt.com/train-keras-model-with-tensorflow-estimators-and-datasets-api/

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

'''
Create Convolutional Neural Network Using Keras 

We'll build a custom model and use Keras to do it. But then we'll convert that Keras model to a TensorFlow Estimator 
and feed TFRecord using `tf.data` API. 
'''


def cnn_model():
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    use_bias = False

    # layer 1
    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(input_layer)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(.2)(max_pool)

    # layer 2
    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(.3)(max_pool)

    # layer 3
    conv = tf.keras.layers.Conv2D(128,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPool2D()(activation)
    dropout = tf.keras.layers.Dropout(.4)(max_pool)

    flatten = tf.keras.layers.Flatten()(dropout)

    # output layer: separate outputs for the weather and the ground labels
    output = tf.keras.layers.Dense(10,
                                   activation='softmax',
                                   name='output')(flatten)

    return tf.keras.Model(inputs=input_layer, outputs=output)


'''
Convert Keras model in TensorFlow Estimators

Why we need to do this conversion is because we need to put it on Cloud ML Engine, which, for now at least, only 
accepts TensorFlow SavedModel formats.

Now the typical reason you would export a Keras model, or at least convert a Keras model to an estimator is for the
ability to do better-distributed training. Instead of training it using Keras, you would convert it to TensorFlow 
Estimator and train it as a TensorFlow Estimator. Then you get distribution and GPU scaling for free in terms of there's
no additional work.
'''

model = cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

cifar_est = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='kkt')

'''
`keras.estimator`, this is one of the pieces that were added to Keras in the inside of TensorFlow. Normal Keras does 
not have a `.estimator` module. You have this estimator.model to convert TensorFlow estimator. Basically, it's a 
conversion function. It's a utility function that will take your jars model. Now, I have the model as a Python pointer. 
I just have that in a variable.

The names of feature columns and labels of a keras estimator come from the corresponding compiled keras model. The 
input key names for `train_input_fn` can be obtained from `keras_inception_v3.input_names`.
'''

'''
Use TensorFlow Dataset API to train Keras Model

Input data is the lifeblood of machine learning, current algorithms and hardware are so thirsty for data that we need 
a powerful input pipeline to be able to keep up with them. You can either feed in data from python at each step, which 
was kind of slow or you could set up queue runners to feed your data, and these are a little challenging to use. 
`tf.data` which is a new library that helps you get all of your data into TensorFlow.

Our model is an Estimator, we'll train and evaluate it a bit differently than we did in Keras. Instead of passing our
features and labels to the model directly when we run training, we need to pass it an input function. In TensorFlow, 
input functions prepare data for the model by mapping raw input data to feature columns.
'''

IMG_SIZE = 64


def dataset_input_fn(filenames, num_epochs):
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        featdef = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.string),
        }

        example = tf.parse_single_example(record, featdef)
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (IMG_SIZE, IMG_SIZE, 3))
        lbl = tf.decode_raw(example['labels'], tf.float32)
        return im, lbl

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(128)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


'''
Train Model

To train our model, all we need to do is call train() and pass in the input function we just defined with our training 
data and labels.
'''

tfrecord_path = '...'
train_data = []
train_input = lambda: dataset_input_fn(train_data, None)
cifar_est.train(input_fn=train_input, steps=3000)

'''
Evaluating the accuracy of our Model

Now that we've trained our model, we can evaluate its accuracy on our training data. We'll use the same input function 
as above, this time passing it our test data instead of training data.
'''

test_data = []
test_input = lambda: dataset_input_fn(test_data, 1)
result = cifar_est.evaluate(input_fn=test_input, steps=1)

'''
Predictions on trained model

Next comes the most important part: using our trained model to generate a prediction on data it hasn't seen before. 
We'll use the first n examples from our test dataset. To make a prediction, we can simply can `.predict()`
'''


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded.set_shape([32, 32, 3])
    return {'input_1': image_decoded}


def predict_input_fn(image_path):
    img_filenames = tf.constant(image_path)

    dataset = tf.data.Dataset.from_tensor_slices(img_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    return image


predict_image = '...'
predict_result = list(cifar_est.predict(input_fn=lambda: predict_input_fn(predict_image)))

true_label = []
class_mapping = dict()
pos = 1
for img, lbl, predict_lbl in zip(predict_image, true_label, predict_result):
    output = np.argmax(predict_lbl.get('output'), axis=None)
    plt.subplot(4, 11, pos)
    # img = mpimg.imread(img)
    # plt.imshow(img)
    plt.axis('off')
    if output == lbl:
        plt.title(class_mapping[output])
    else:
        plt.title(class_mapping[output] + '/' + class_mapping[lbl], color='#ff000')
    pos += 1

plt.show()

'''
calling `predict` with our input functions we're able to generate predictions on our trained Estimator model.
'''

'''
Save Keras model to Tensorflow .pb

Next we're going to take the TensorFlow estimator model, and now we need to export it as a .pd file.

Save inputs function

TensorFlow has a number of utilities to help us create this serving input function.
'''

model_input_name = model.input_names[0]


def serving_input_receiver_fn():
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images = tf.map_fn(partial(tf.image.decode_image, channels=1), input_ph, dtype=tf.unit8)
    images = tf.cast(images, tf.float32) / 255.
    images.set_shape([None, 32, 32, 3])

    return tf.estimator.export.ServingInputReceiver({model_input_name: images}, {'bytes': input_ph})


cifar_est.export_saved_model('...', serving_input_receiver_fn=serving_input_receiver_fn)
