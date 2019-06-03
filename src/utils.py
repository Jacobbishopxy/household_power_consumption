"""
@author Jacob
@time 2019/05/21
"""

import os
import sys
from typing import Union, List
import pandas as pd
import tensorflow as tf
from tensorboard import program


def crash_proof():
    """
    in case of GPU CUDA crashing
    """
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


def create_model_dir(root_path: str, consistent_model: bool = True):
    if consistent_model:
        model_dir = root_path
    else:
        model_dir = os.path.join(root_path, pd.datetime.now().strftime('%Y%m%d-%H%M%S'))

    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def launch_tb(dir_path: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', dir_path])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)
    input('press enter to quit TensorBoard')


def files_exist(file_path: Union[str, List[str]]) -> bool:
    if isinstance(file_path, str):
        if os.path.isfile(file_path):
            return True
        else:
            return False
    elif isinstance(file_path, list):
        if all([os.path.isfile(i) for i in file_path]):
            return True
        else:
            return False
    else:
        raise ValueError('file_path should either be str or List[str]')
