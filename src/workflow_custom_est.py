"""
@author Jacob
@time 2019/05/16
"""

from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_estimator import estimator as est


def _float_feature(arr: np.ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(value=arr))


def write_tf_record(filename: str, features: np.ndarray, labels: np.ndarray):
    """
    write features and labels to TFRecord
    :param filename:
    :param features:
    :param labels:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(features)):
        feat = tf.train.Features(feature={
            'features': _float_feature(features[i]),
            'labels': _float_feature(labels[i])
        })
        example = tf.train.Example(features=feat)
        writer.write(example.SerializeToString())
    writer.close()


def model_to_estimator(keras_model, model_dir: Optional[str] = None):
    """
    convert keras model to estimator
    :param keras_model:
    :param model_dir:
    :return:
    """
    return tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_dir)


def split_data(data: pd.DataFrame):
    return data.iloc[1:-328, :], data.iloc[-328:-6, :]


def _sliding_window(arr: np.ndarray, window: int, step: int = 1):
    loop = (arr.shape[0] - window) // step + 1
    return np.array([arr[i * step:i * step + window] for i in range(loop)])


def to_supervised(data: pd.DataFrame,
                  n_in: int,
                  n_out: int,
                  feature_cols: Union[List[int], int] = 0,
                  is_train: bool = True):
    """

    :param data:
    :param n_in:
    :param n_out:
    :param feature_cols:
    :param is_train:
    :return:
    """
    cc = [feature_cols] if isinstance(feature_cols, int) else feature_cols
    raw_features_df = data.iloc[:-n_out, cc]
    raw_labels_df = data.iloc[n_in:, 0]

    if is_train:
        n_in_steps = n_out_steps = 1
    else:
        n_in_steps, n_out_steps = n_in, n_out

    features = _sliding_window(raw_features_df.values, window=n_in, step=n_in_steps)
    labels = _sliding_window(raw_labels_df.values, window=n_out, step=n_out_steps)

    return features, labels


def _parse(feature: np.ndarray, label: np.ndarray):
    return {'X': feature}, label


def set_input_fn_csv(features: np.ndarray, labels: np.ndarray, num_epochs=None):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(lambda f, l: _parse(f, l))
    dataset = dataset.batch(4)
    dataset = dataset.repeat(num_epochs)

    itr = dataset.make_one_shot_iterator()
    batch_features, batch_labels = itr.get_next()

    return batch_features, batch_labels


def build_model(shape_in: Tuple[int, int], shape_out: Tuple[int]):
    n_out = shape_out[0]

    input_layer = tf.keras.layers.Input(shape=shape_in, name='X')
    conv = tf.keras.layers.Conv1D(filters=16,
                                  kernel_size=3,
                                  activation='relu')(input_layer)
    maxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    fltn = tf.keras.layers.Flatten()(maxp)
    dns1 = tf.keras.layers.Dense(10, activation='relu')(fltn)
    dns2 = tf.keras.layers.Dense(n_out)(dns1)

    model = tf.keras.Model(inputs=input_layer, outputs=dns2)
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])
    model.summary()
    return model


def read_data_from_csv(path: str):
    return pd.read_csv(path,
                       header=0,
                       infer_datetime_format=True,
                       parse_dates=['datetime'],
                       index_col=['datetime'])


def data_to_tf_record(train_features: np.ndarray,
                      train_labels: np.ndarray,
                      test_features: np.ndarray,
                      test_labels: np.ndarray,
                      train_path: str,
                      test_path: str):
    """
    train & test write to TFRecord
    :param train_features:
    :param train_labels:
    :param test_features:
    :param test_labels:
    :param train_path:
    :param test_path:
    :return:
    """
    train_features_vec = train_features.reshape([train_features.shape[0], -1])
    test_features_vec = test_features.reshape([test_features.shape[0], -1])

    write_tf_record(train_path, train_features_vec, train_labels)
    print('train to TFRecord completed')
    write_tf_record(test_path, test_features_vec, test_labels)
    print('test to TFRecord completed')


def tf_record_preprocessing(n_in: int,
                            n_out: int,
                            raw_data_path: str,
                            file_train_path: str,
                            file_test_path: str,
                            feature_cols: Union[List[int], int] = 0):
    """

    :param n_in:
    :param n_out:
    :param raw_data_path:
    :param file_train_path:
    :param file_test_path:
    :param feature_cols:
    :return:
    """
    # read from csv
    d = read_data_from_csv(raw_data_path)
    # split train & test
    raw_trn_data, raw_tst_data = split_data(d)
    # split train/test-x/y
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)
    # write final train & test data to TFRecord
    data_to_tf_record(trn_fea,
                      trn_lbl,
                      tst_fea,
                      tst_lbl,
                      file_train_path,
                      file_test_path)


def _data_from_tf_record(example,
                         shape_in: Tuple[int, int],
                         shape_out: Tuple[int]):
    n_in, num_fea = shape_in
    n_dim_in = n_in * num_fea
    feature_def = {'features': tf.FixedLenFeature(n_dim_in, tf.float32),
                   'labels': tf.FixedLenFeature(shape_out[0], tf.float32)}

    features = tf.parse_single_example(example, feature_def)
    fea = tf.reshape(features['features'], shape_in)
    lbl = tf.reshape(features['labels'], shape_out)
    return fea, lbl


def set_input_fn_tf_record(file_name: str,
                           shape_in: Tuple[int, int],
                           shape_out: Tuple[int],
                           num_epochs: Optional[int] = None):
    """

    :param file_name:
    :param shape_in:
    :param shape_out:
    :param num_epochs:
    :return:
    """
    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(lambda x: _data_from_tf_record(x, shape_in, shape_out))
    dataset = dataset.map(_parse)
    dataset = dataset.batch(4)
    dataset = dataset.repeat(num_epochs)

    return dataset


def eval_from_csv(shape_in: Tuple[int, int],
                  shape_out: Tuple[int],
                  file_csv: str,
                  feature_cols: Union[List[int], int] = 0,
                  num_epochs: Optional[int] = 10):
    """
    train & test read from csv
    :param shape_in:
    :param shape_out:
    :param file_csv:
    :param feature_cols:
    :param num_epochs:
    :return:
    """

    n_in, n_out = shape_in[0], shape_out[0]

    model = build_model(shape_in=shape_in, shape_out=shape_out)
    classifier = model_to_estimator(model, 'tmp')

    d = read_data_from_csv(file_csv)
    raw_trn_data, raw_tst_data = split_data(d)
    trn_fea, trn_lbl = to_supervised(raw_trn_data, n_in, n_out, feature_cols=feature_cols, is_train=True)
    tst_fea, tst_lbl = to_supervised(raw_tst_data, n_in, n_out, feature_cols=feature_cols, is_train=False)

    classifier.train(
        input_fn=lambda: set_input_fn_csv(trn_fea, trn_lbl),
        steps=20
    )

    result = classifier.evaluate(
        input_fn=lambda: set_input_fn_csv(tst_fea, tst_lbl, num_epochs=num_epochs)
    )
    return result


def eval_from_tf_record(shape_in: Tuple[int, int],
                        shape_out: Tuple[int],
                        file_train: str,
                        file_test: str,
                        num_epochs: Optional[int] = 10):
    """
    train & test read from TFRecord
    :param shape_in:
    :param shape_out:
    :param file_train:
    :param file_test:
    :param num_epochs:
    :return:
    """
    model = build_model(shape_in=shape_in, shape_out=shape_out)
    classifier = model_to_estimator(model, 'tmp')

    classifier.train(
        input_fn=lambda: set_input_fn_tf_record(file_train,
                                                shape_in=shape_in,
                                                shape_out=shape_out),
        steps=20
    )
    result = classifier.evaluate(
        input_fn=lambda: set_input_fn_tf_record(file_test,
                                                shape_in=shape_in,
                                                shape_out=shape_out,
                                                num_epochs=num_epochs)
    )
    return result


# todo: explore this function
def ev(shape_in, shape_out, file_train, file_test):
    model = build_model(shape_in=shape_in, shape_out=shape_out)
    classifier = model_to_estimator(model, 'tmp')

    train_spec = est.TrainSpec(
        input_fn=lambda: set_input_fn_tf_record(file_train, shape_in=shape_in, shape_out=shape_out),
    )

    eval_spec = est.EvalSpec(
        input_fn=lambda: set_input_fn_tf_record(file_test, shape_in=shape_in, shape_out=shape_out)
    )

    classifier = est.train_and_evaluate(classifier, train_spec, eval_spec)
    return classifier


if __name__ == '__main__':
    '''
    in case of GPU CUDA crashing
    '''
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    '''
    N_IN: timesteps 
    N_OUT: labels
    FEATURE_COLS: features to use for training
    '''

    RAW_DATA_PATH = '../data/household_power_consumption_days.csv'
    FILE_TRAIN = '../data/uni_var_train.tfrecords'
    FILE_TEST = '../data/uni_var_test.tfrecords'

    N_IN, N_OUT, FEATURE_COLS = 7, 7, [0]

    SHAPE_IN = (N_IN, len(FEATURE_COLS))
    SHAPE_OUT = (N_OUT,)

    '''
    read data from csv and evaluate model
    '''
    # r1 = eval_from_csv(SHAPE_IN, SHAPE_OUT, feature_cols=FEATURE_COLS, file_csv=RAW_DATA_PATH)
    # print(r1)

    '''
    write data to TFRecord then read and evaluate 
    '''
    tf_record_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, FILE_TRAIN, FILE_TEST, feature_cols=FEATURE_COLS)

    r2 = eval_from_tf_record(SHAPE_IN, SHAPE_OUT, file_train=FILE_TRAIN, file_test=FILE_TEST)
    print(r2)
