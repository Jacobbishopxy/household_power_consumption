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
                                network_fn=create_multichannel_model,
                                batch_norm=False,
                                batch_size=80)

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


def check_univ_labels_and_preds():
    from utils import read_labels_and_predictions, check_tf_record
    from workflow_custom_estimator import set_input_fn_tf_record, model_fn_default

    tfr_name = f'uni_var-(21, 1)-(7,)'

    n_in, n_out, feature_cols = 21, 7, [0]

    shape_in, shape_out = (n_in, len(feature_cols)), (n_out,)

    params = {
        'network_fn': create_multichannel_model,
        'network_params': {
            'shape_in': shape_in,
            'shape_out': shape_out,
        },
    }

    d = set_input_fn_tf_record(tfr_name,
                               is_train=True,
                               shape_in=shape_in,
                               shape_out=shape_out,
                               batch_size=10)
    dd = d.make_one_shot_iterator().get_next()

    import tensorflow as tf

    with tf.Session() as sess:
        r = sess.run(dd)
        print(r)

    r = check_tf_record(lambda: set_input_fn_tf_record(tfr_name,
                                                       is_train=True,
                                                       shape_in=shape_in,
                                                       shape_out=shape_out,
                                                       batch_size=10))
    print(r)

    l, p = read_labels_and_predictions(input_fn=lambda: set_input_fn_tf_record(tfr_name,
                                                                               is_train=True,
                                                                               shape_in=shape_in,
                                                                               shape_out=shape_out,
                                                                               batch_size=10),
                                       model_fn=model_fn_default,
                                       model_fn_params=params,
                                       checkpoint_path=r'.\tmp\test\20190612-100424',
                                       print_each_batch=True)
    return l, p


if __name__ == '__main__':
    crash_proof()

    e1 = univ_test()

    # check_univ_labels_and_preds()

    # e2 = multi_head_test()
