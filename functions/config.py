import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch

PROMPT_TYPE = 'structured'
SCENARIO = 'multimodal'
RUN_ID = time.strftime('%Y%m%d-%H%M%S')

# Path Configuration
BASE_CLOUD = Path('/content/drive/MyDrive/TA_DIR/baseline')
BASE_LOCAL = Path('/content/work_24k')

# Folder di google drive
DATA_DIR = BASE_CLOUD / 'data'
CSV_PATH = DATA_DIR / 'sample_subset.csv'
CAPTION_DIR = BASE_CLOUD / 'captions' / 'clean'

FEATURE_DIR = BASE_CLOUD / 'features'
FEAT_V = FEATURE_DIR / 'vision'
FEAT_T = FEATURE_DIR / 'text'

CKPT_DIR = BASE_CLOUD / 'ckpt'
RUNS_DIR = BASE_CLOUD / 'runs'


# Local Directory
LOCAL_IMG_ROOT = BASE_LOCAL / 'images'
LOCAL_CAPTION_ROOT = BASE_LOCAL / 'captions'
LOCAL_FEAT_ROOT = BASE_LOCAL / 'features'
LOCAL_CKPT_ROOT = BASE_LOCAL / 'ckpt'
LOCAL_RUNS_ROOT = BASE_LOCAL / 'runs'

for p in [
    BASE_CLOUD, DATA_DIR, CKPT_DIR, 
    CAPTION_DIR, FEAT_V, FEAT_T, 
    RUNS_DIR, BASE_LOCAL, LOCAL_IMG_ROOT, 
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
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0 
EPOCHS = 30 
PATIENCE = 6 
LR = 1e-4 
WEIGHT_DECAY = 1e-4
PIN_MEMORY = True
VISUAL_SYNC_EVERY = 5000
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]
THRESHOLDS = np.arange(0.30, 0.71, 0.02)

IMG_DIRS = {
    'train': BASE_CLOUD / 'split_images' / 'train',
    'test': BASE_CLOUD / 'split_images' / 'test',
    'val': BASE_CLOUD / 'split_images' / 'val',
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

@dataclass
class ExperimentPaths:
    cloud_root:Path
    local_root:Path
    cloud_ckpt:Path
    cloud_features:Path
    cloud_runs:Path
    local_ckpt:Path
    local_features:Path
    local_runs:Path
    local_images:Path | None = None
    local_captions:Path | None = None

def build_paths(scenario: str, prompt_type: str | None = None) -> ExperimentPaths:
    """
    scenario:
        - visual_only
        - text_only
        - multimodal
    
    prompt_type:
        - structured
        - contrastive
        - None (untuk visual_only)
    """

    if scenario not in {'visual_only', 'text_only', 'multimodal'}:
        raise ValueError(f"scenario: {scenario} is invalid")
    
    if scenario == 'visual_only':
        exp_name = scenario
        cloud_root = BASE_CLOUD / exp_name
        local_root = BASE_LOCAL / exp_name
    else:
        if prompt_type not in {'structured', 'contrastive'}:  
            raise ValueError(f"prompt_type: {prompt_type} is invalid for scenario: {scenario}")
        exp_name = BASE_CLOUD / scenario / prompt_type
        cloud_root = exp_name
        local_root = BASE_LOCAL / scenario / prompt_type

    paths = ExperimentPaths(
        cloud_root=cloud_root,
        local_root=local_root,
        cloud_ckpt=cloud_root / 'ckpt',
        cloud_features=cloud_root / 'features',
        cloud_runs=cloud_root / 'runs' / RUN_ID,
        local_ckpt=local_root / 'ckpt',
        local_features=local_root / 'features',
        local_runs=local_root / 'runs' / RUN_ID
    )

    if scenario in {'text_only', 'multimodal'}:
        paths.local_captions = local_root / 'captions'
    if scenario in {'visual_only', 'multimodal'}:
        paths.local_images = local_root / 'images'

    # mkdir
    for p in [
        paths.cloud_ckpt,
        paths.cloud_features,
        paths.cloud_runs,
        paths.local_ckpt,
        paths.local_features,
        paths.local_runs,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    if paths.local_images is not None:
        paths.local_images.mkdir(parents=True, exist_ok=True)
    if paths.local_captions is not None:
        paths.local_captions.mkdir(parents=True, exist_ok=True)

    return paths    