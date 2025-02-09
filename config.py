import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MOLECULAR_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "molecular_data.csv")
TRAIN_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
TEST_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")

# Absorption data settings
ABS_THRESHOLD = 0.01
MAX_PEAKS = 1
