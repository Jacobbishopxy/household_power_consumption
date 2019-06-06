"""
@author Jacob
@time 2019/06/05
"""

from typing import Union, List, Tuple
import tensorflow as tf  # do not remove this line, otherwise launch_tb will fail
from tensorflow_estimator import estimator as est

from workflow_custom_estimator import model_fn_default
from networks import create_multichannel_model
from input_functions import set_input_fn_tf_record
from utils import create_model_dir, launch_tb, crash_proof


def estimator_from_model_fn(shape_in: Union[Tuple[int, int], List[Tuple[int, int]]],
                            shape_out: Tuple[int],
                            tf_records_name: str,
                            batch_size: int = 10,
                            epochs: int = 10,
                            model_dir: str = r'.\tmp\test',
                            consistent_model: bool = True,
                            model_fn=model_fn_default,
                            network_fn=create_multichannel_model):
    model_dir = create_model_dir(model_dir, consistent_model)

    params = {
        'network_fn': network_fn,
        'network_params': {
            'shape_in': shape_in,
            'shape_out': shape_out
        }
    }

    estimator = est.Estimator(model_fn=model_fn,
                              model_dir=model_dir,
                              params=params)

    estimator.train(
        input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                is_train=True,
                                                shape_in=shape_in,
                                                shape_out=shape_out,
                                                batch_size=batch_size,
                                                num_epochs=epochs)
    )

    result = estimator.evaluate(
        input_fn=lambda: set_input_fn_tf_record(tf_records_name,
                                                is_train=False,
                                                shape_in=shape_in,
                                                shape_out=shape_out,
                                                batch_size=batch_size)
    )

    print(result)

    launch_tb(model_dir)
    return estimator


if __name__ == '__main__':
    crash_proof()

    RAW_DATA_PATH = './data/factor_return.csv'
    TF_RECORDS_NAME = 'barra_style_univariate'

    N_IN, N_OUT, FEATURE_COLS, LABEL_COL = 14, 7, [0], 0
    EPOCHS = 10
    SHAPE_IN = (N_IN, len(FEATURE_COLS))
    SHAPE_OUT = (N_OUT,)

    e = estimator_from_model_fn(shape_in=SHAPE_IN,
                                shape_out=SHAPE_OUT,
                                tf_records_name=TF_RECORDS_NAME,
                                epochs=EPOCHS,
                                consistent_model=False)

