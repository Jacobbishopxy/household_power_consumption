"""
@author Jacob
@time 2019/06/03

"""

from data_preprocessing import tf_record_preprocessing
from workflow_custom_estimator import crash_proof, estimator_from_model_fn
from models import create_multihead_model, create_vanilla_model

RAW_DATA_PATH = '../data/household_power_consumption_days.csv'


def univ_test():
    file_train = '../tmp/uni_var_train.tfrecords'
    file_test = '../tmp/uni_var_test.tfrecords'

    n_in, n_out, feature_cols = 14, 7, [0]
    epochs = 20

    shape_in, shape_out = (n_in, len(feature_cols)), (n_out,)

    tf_record_preprocessing(n_in=n_in,
                            n_out=n_out,
                            raw_data_path=RAW_DATA_PATH,
                            file_train_path=file_train,
                            file_test_path=file_test,
                            feature_cols=feature_cols)

    e = estimator_from_model_fn(shape_in=shape_in,
                                shape_out=shape_out,
                                file_train=file_train,
                                file_test=file_test,
                                epochs=epochs,
                                consistent_model=False,
                                learning_rate=1e-3,
                                network_fn=create_vanilla_model,
                                batch_norm=True
                                )

    return e


def multi_head_test():
    _file_pattern = '../tmp/multihead_X.tfrecords'

    file_train = _file_pattern.replace('X', 'train')
    file_test = _file_pattern.replace('X', 'test')

    n_in, n_out, feature_cols = 14, 7, [[i] for i in range(8)]
    epochs = 20

    shape_in, shape_out = [(n_in, len(i)) for i in feature_cols], (n_out,)

    tf_record_preprocessing(n_in=n_in,
                            n_out=n_out,
                            raw_data_path=RAW_DATA_PATH,
                            file_train_path=file_train,
                            file_test_path=file_test,
                            feature_cols=feature_cols)

    e = estimator_from_model_fn(
        shape_in=shape_in,
        shape_out=shape_out,
        file_train=file_train,
        file_test=file_test,
        epochs=epochs,
        consistent_model=False,
        learning_rate=1e-3,
        network_fn=create_multihead_model,
    )

    return e


if __name__ == '__main__':
    crash_proof()

    e1 = univ_test()

    # e2 = multi_head_test()
