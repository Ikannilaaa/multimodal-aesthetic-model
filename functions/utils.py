# Import Libraries
import random
import gc # Garbage collector
import shutil
import h5py # Act like Py Dict
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer
)

from config import (
    SEED, DEVICE, USE_FP16, MODEL_ID, VALID_EXT, VISUAL_SYNC_EVERY
)

log = logging.getLogger(__name__)

# Helper
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # PyTorch di CPU
    torch.cuda.manual_seed(seed) # pyTorch di GPU
    torch.cuda.manual_seed_all(seed) # pyTorch di semua GPU

# Bersihkan RAM dan Cache GPU
def free_mem():
    gc.collect()
    # kosongkan cache VRAM PyTorch
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

# Tampilkan informasi env
def print_system_info():
    print(f'\n Device: {DEVICE}')
    print(f' PyTorch: {torch.__version__}')

    if torch.cuda.is_available():
        print(f' GPU: {torch.cuda.get_device_name(0)}')
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f' VRAM: {total_mem:.2f} GB')
    else:
        print('WARNING: GPU not detected!')

    print(f'Model: {MODEL_ID}')

# Menentukan batch size berdasarkan GPU
def choose_batch_size():
    if DEVICE != 'cuda':
      return 8 # CPU

    name = torch.cuda.get_device_name(0).lower()
    if 'h100' in name:
      return 128
    if 'a100' in name:
      return 64
    if 'l4' in name:
      return 32
    if 't4' in name:
      return 16
    return 16

# Check sisa ruang penyimpanan lokal
def check_disk_space(path: Path, required_gb: float = 15.0):
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    log.info(f'Free disk space: {free_gb:.2f} GB')

    if free_gb < required_gb:
      raise RuntimeError(f'Not enough disk space: {free_gb:.2f} GB')

# Copy file images dari cloud ke lokal
def copy_images_to_local(img_dirs: dict, local_img_root: Path):
    check_disk_space(local_img_root.parent, required_gb=15.0)
    out = {}

    # Buat folder local per split
    for split, src_dir in img_dirs.items():
        src_dir = Path(src_dir)
        dst_dir = local_img_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_files = [p for p in sorted(src_dir.iterdir()) if p.is_file()]
        to_copy = [p for p in src_files if not (dst_dir / p.name).exists()]

        if to_copy:
          log.info(f'[{split.upper()}] Copying {len(to_copy)} image to local...')
          for p in tqdm(to_copy, desc=f'Copy images {split}'):
              shutil.copy2(p, dst_dir / p.name)
        else:
          log.info(f'[{split.upper()}] No image to copy.')

        out[split] = dst_dir

    return out

# Copy file caption dari cloud ke local
def copy_captions_to_local(caption_files: dict, local_caption_root: Path):
    check_disk_space(local_caption_root.parent, required_gb=15.0)
    out = {}

    for split, src_file in caption_files.items():
        src_path = Path(src_file)
        dst_path = local_caption_root / src_path.name

        if not dst_path.exists():
            log.info(f'[{split.upper()}] Copying {src_path.name} to local...')
            shutil.copy2(src_path, dst_path)
            log.info(f'[{split.upper()}] Copied caption JSON to local: {dst_path}')
        else:
            log.info(f'[{split.upper()}] Caption JSON already exists: {dst_path}')

        out[split] = dst_path

    return out

# Copy file di local ke cloud
def sync_to_drive(local_path: Path, drive_path: Path):
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, drive_path)
    log.info(f'Synced {local_path} to {drive_path}')

# Load model clip sekali
def load_clip_once(model_id=MODEL_ID):
    dtype = torch.float16 if (DEVICE == 'cuda' and USE_FP16) else torch.float32
    log.info(f'Loading CLIP model from {model_id} | dtype={dtype}')

    if DEVICE == 'cuda':
        model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
    else:
        model = CLIPModel.from_pretrained(model_id).to(DEVICE)

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    processor = CLIPImageProcessor.from_pretrained(model_id, use_fast=False)
    model.eval()

    return model, processor, tokenizer

# Helper HDF5
# Cek key yang sudah ada
def _get_processed_key_h5(h5_path: Path) -> set:
    if not h5_path.exists():
        return set()

    try:
        with h5py.File(h5_path, 'r') as f:
            return set(f.keys())
    except:
        return set()

# Save embedding hasil CLIP ke file HDF5
def _write_feature_h5(h5f, key: str, feat: np.ndarray):
    if key in h5f:
        return

    h5f.create_dataset(key, data=feat, compression='gzip', compression_opts=4)

# Memastikan output fitur shape-nya sesuai
def _normalize_torch_features(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        if hasattr(x, 'pooler_output') and x.pooler_output is not None:
            x = x.pooler_output
        elif hasattr(x, 'last_hidden_state'):
            x = x.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f'Unsupported input type: {type(x)}')

    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

# Cek field name di JSON caption
def _find_caption_field(item: dict) -> str:
    for field in ['description', 'caption', 'text']:
        if field in item and item[field] is not None:
            return str(item[field])

    raise KeyError(f'No caption field found in {item}')

# Image file name builder
def _build_candidates(image_id: str):
    candidates = [image_id]

    for ext in VALID_EXT:
        candidates.append(f'{image_id}{ext}')

    return candidates