"""
@author Jacob
@time 2019/06/06
"""

from utils import crash_proof
from workflow_custom_estimator import estimator_from_model_fn
from networks import create_multihead_model, create_multichannel_model

from barra.barra_data_preprocessing import FactorType, tf_records_preprocessing

RAW_DATA_PATH = r'.\data\factor_return.csv'


def multi_channel_test(factor_type: FactorType, predict_factor: str):
    n_in, n_out = 14, 3
    tf_records_name = f'multichannel_i{n_in}-o{n_out}_p{predict_factor}'

    factor_list = factor_type.value
    feature_cols, label_col = list(range(len(factor_list))), factor_list.index(predict_factor)

    tf_records_preprocessing(n_in=n_in,
                             n_out=n_out,
                             raw_data_path=RAW_DATA_PATH,
                             tf_records_name=tf_records_name,
                             feature_cols=feature_cols,
                             label_col=label_col,
                             factor_type=factor_type)

    epochs = 20
    shape_in, shape_out = (n_in, len(factor_list)), (n_out,)

    e = estimator_from_model_fn(shape_in=shape_in,
                                shape_out=shape_out,
                                tf_records_name=tf_records_name,
                                epochs=epochs,
                                network_fn=create_multichannel_model,
                                consistent_model=False,
                                learning_rate=1,
                                batch_norm=False,
                                batch_size=80)

    return e


def check_multi_channel_labels_and_preds():
    from utils import read_labels_and_predictions, check_tf_record
    from workflow_custom_estimator import set_input_fn_tf_record, model_fn_default

    ft = FactorType.STYLE
    pf = ft.value[1]
    fl = ft.value

    n_in, n_out = 14, 3
    tfr_name = f'multichannel_i{n_in}-o{n_out}_p{pf}'
    shape_in, shape_out = (n_in, len(fl)), (n_out,)

    params = {
        'network_fn': create_multichannel_model,
        'network_params': {
            'shape_in': shape_in,
            'shape_out': shape_out,
        },
    }

    # r = check_tf_record(input_fn=lambda: set_input_fn_tf_record(tfr_name,
    #                                                             is_train=True,
    #                                                             shape_in=shape_in,
    #                                                             shape_out=shape_out,
    #                                                             num_epochs=20,
    #                                                             batch_size=80))
    # print(r)

    l, p = read_labels_and_predictions(input_fn=lambda: set_input_fn_tf_record(tfr_name,
                                                                               is_train=False,
                                                                               shape_in=shape_in,
                                                                               shape_out=shape_out,
                                                                               batch_size=10),
                                       model_fn=model_fn_default,
                                       model_fn_params=params,
                                       checkpoint_path=r'.\tmp\test\20190612-131932',
                                       print_each_batch=True)

    return l, p


if __name__ == '__main__':
    crash_proof()

    ft = FactorType.STYLE
    pf = ft.value[1]

    e1 = multi_channel_test(ft, pf)

    # l, p = check_multi_channel_labels_and_preds()
    # print(l, p)
