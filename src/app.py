"""
@author Jacob
@time 2019/06/03

"""

from data_preprocessing import tf_records_preprocessing
from workflow_custom_estimator import crash_proof, estimator_from_model_fn
from networks import create_multihead_model, create_multichannel_model

RAW_DATA_PATH = '../data/household_power_consumption_days.csv'


def univ_test():
    n_in, n_out, feature_cols = 21, 7, [0]
    epochs = 200

    shape_in, shape_out = (n_in, len(feature_cols)), (n_out,)
    tf_records_name = f'uni_var-{shape_in}-{shape_out}'

    tf_records_preprocessing(n_in=n_in,
                             n_out=n_out,
                             raw_data_path=RAW_DATA_PATH,
                             tf_records_name=tf_records_name,
                             feature_cols=feature_cols)

    e = estimator_from_model_fn(shape_in=shape_in,
                                shape_out=shape_out,
                                tf_records_name=tf_records_name,
                                epochs=epochs,
                                consistent_model=False,
                                learning_rate=1,
                                network_fn=create_multichannel_model,
                                batch_norm=True,
                                batch_size=80
                                )

    return e


def multi_head_test():
    n_in, n_out, feature_cols = 14, 7, [[i] for i in range(8)]

    tf_records_name = f'multihead-{n_in}-{n_out}'
    shape_in, shape_out = [(n_in, len(i)) for i in feature_cols], (n_out,)

    tf_records_preprocessing(n_in=n_in,
                             n_out=n_out,
                             raw_data_path=RAW_DATA_PATH,
                             tf_records_name=tf_records_name,
                             feature_cols=feature_cols)

    e = estimator_from_model_fn(
        shape_in=shape_in,
        shape_out=shape_out,
        tf_records_name=tf_records_name,
        epochs=20,
        consistent_model=False,
        learning_rate=1,
        network_fn=create_multihead_model,
        batch_size=10,
        batch_norm=False
    )

    return e


if __name__ == '__main__':
    crash_proof()

    # e1 = univ_test()

    e2 = multi_head_test()
