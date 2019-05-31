from data_preprocessing import tf_record_preprocessing
from workflow_custom_estimator import crash_proof, estimator_from_model_fn

crash_proof()

RAW_DATA_PATH = '../data/household_power_consumption_days.csv'
FILE_TRAIN = '../tmp/uni_var_train.tfrecords'
FILE_TEST = '../tmp/uni_var_test.tfrecords'

N_IN, N_OUT, FEATURE_COLS = 14, 7, [0]
EPOCHS = 20
BATCH_SIZE = 20

SHAPE_IN = (N_IN, len(FEATURE_COLS))
SHAPE_OUT = (N_OUT,)

tf_record_preprocessing(N_IN, N_OUT, RAW_DATA_PATH, FILE_TRAIN, FILE_TEST, feature_cols=FEATURE_COLS)

e3 = estimator_from_model_fn(
    shape_in=SHAPE_IN,
    shape_out=SHAPE_OUT,
    file_train=FILE_TRAIN,
    file_test=FILE_TEST,
    epochs=EPOCHS,
    consistent_model=False,
    learning_rate=1e-3
)
