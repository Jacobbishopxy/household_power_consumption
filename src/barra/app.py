"""
@author Jacob
@time 2019/06/06
"""

from utils import crash_proof
from workflow_custom_estimator import estimator_from_model_fn

from barra.barra_data_preprocessing import FactorType, tf_records_preprocessing

RAW_DATA_PATH = r'.\data\factor_return.csv'


def multi_channel_test(factor_type: FactorType, predict_factor: str):
    n_in, n_out = 14, 1
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
                                consistent_model=False)

    return e


if __name__ == '__main__':
    crash_proof()

    ft = FactorType.STYLE
    print(ft.value)

    e1 = multi_channel_test(ft, 'CNE5S_LIQUIDTY')
