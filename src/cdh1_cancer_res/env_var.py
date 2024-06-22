import os

ABSOLUTE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR_PATH = os.path.join(ABSOLUTE_DIR_PATH, '../../artifacts/')
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR_PATH, 'demo_embeddings/')
CHECKPOINT_CALIBRATION_FILE_PATH = os.path.join(ARTIFACTS_DIR_PATH, 'checkpoint_calibration.csv')
DATASET_CSV_PATH = os.path.join(ARTIFACTS_DIR_PATH, 'demo_dataset.csv')
ENSEMBLE_MODEL_PATH = os.path.join(ARTIFACTS_DIR_PATH, 'ensemble.tsckpt')

CHECKPOINT_FILE_EXTENSION = '.ckpt'
TORCHSCRIPTED_CHECKPOINT_FILE_EXTENSION = '.tsckpt'
