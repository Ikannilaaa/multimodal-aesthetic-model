import time
import numpy as np
from pathlib import Path
import torch

# Path Configuration
# Folder di google drive
BASE_DIR = Path('/content/drive/MyDrive/TA_DIR/baseline')

DATA_DIR = BASE_DIR / 'data'
CSV_PATH = DATA_DIR / 'sample_subset.csv'
CAPTION_DIR = BASE_DIR / 'captions' / 'clean'

FEATURE_DIR = BASE_DIR / 'features'
FEAT_V = FEATURE_DIR / 'vision'
FEAT_T = FEATURE_DIR / 'text'

CKPT_DIR = BASE_DIR / 'ckpt'
RUNS_DIR = BASE_DIR / 'runs' / time.strftime('%Y%m%d-%H%M%S')


# Local Directory
LOCAL_DIR = Path('/content/work_24k')
LOCAL_IMG_ROOT = LOCAL_DIR / 'images'
LOCAL_CAPTION_ROOT = LOCAL_DIR / 'captions'
LOCAL_FEAT_ROOT = LOCAL_DIR / 'features'
LOCAL_CKPT_ROOT = LOCAL_DIR / 'ckpt'
LOCAL_RUNS_ROOT = LOCAL_DIR / 'runs' / time.strftime('%Y%m%d-%H%M%S')

for p in [
    BASE_DIR, DATA_DIR, CKPT_DIR, 
    CAPTION_DIR, FEAT_V, FEAT_T, 
    RUNS_DIR, LOCAL_DIR, LOCAL_IMG_ROOT, 
    LOCAL_FEAT_ROOT, LOCAL_CAPTION_ROOT,
    LOCAL_CKPT_ROOT, LOCAL_RUNS_ROOT]:
        p.mkdir(parents=True, exist_ok=True)
    
# Model, Seed
MODEL_ID = 'openai/clip-vit-large-patch14-336'
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FP16 = True
COPY_DATA_TO_LOCAL = True

# Hyperparameter
# Jumlah sampel data (img-txt pair), diproses dalam model secara bersamaan
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0 # Jumlah sub-process saat data loading
EPOCHS = 30 # Putaran/round
PATIENCE = 6 # Early stopping
LR = 1e-4 # 10^-4
WEIGHT_DECAY = 1e-4 # Regularisasi
PIN_MEMORY = True
VISUAL_SYNC_EVERY = 5000

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]
# Untuk tuning decision threshold / evaluasi similarity
THRESHOLDS = np.arange(0.30, 0.71, 0.02)

IMG_DIRS = {
    'train': BASE_DIR / 'split_images' / 'train',
    'test': BASE_DIR / 'split_images' / 'test',
    'val': BASE_DIR / 'split_images' / 'val',
}

CAPTION_FILES_CONTRASTIVE = {
    'train': CAPTION_DIR / 'llava_train_contrastive_V1.5-7B.json',
    'test': CAPTION_DIR / 'llava_test_contrastive_V1.5-7B.json',
    'val': CAPTION_DIR / 'llava_val_contrastive_V1.5-7B.json',
}

CAPTION_FILES_STRUCTURED = {
    'train': CAPTION_DIR / 'llava_train_structured_V1.5-7B.json',
    'test': CAPTION_DIR / 'llava_test_structured_V1.5-7B.json',
    'val': CAPTION_DIR / 'llava_val_structured_V1.5-7B.json',
}