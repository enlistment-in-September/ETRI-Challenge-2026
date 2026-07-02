from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR = BASE_DIR / 'data' / 'submissions'

TRAIN_CSV = RAW_DIR / 'ch2026_metrics_train.csv'
TEST_CSV = RAW_DIR / 'ch2026_submission_sample.csv'

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEEDS = [1, 42, 2024, 8765, 9999]
N_FOLDS = 5

# BiLSTM hyperparameters
LOOKBACK = 14
BATCH_SIZE = 64
EPOCHS = 50
LR = 7e-4
WD = 1e-4
HIDDEN = 96
NUM_LAYERS = 2
DROPOUT = 0.25
MAX_SEQ_FEATURES = 256
NUM_WORKERS = 0
