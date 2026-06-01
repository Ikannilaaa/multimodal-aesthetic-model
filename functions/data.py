from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
    WeightedRandomSampler
)

from config import VALID_EXT
from utils import _get_processed_key_h5, _build_candidates

# Dataset Loader
class HybridFusionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        visual_h5_path,
        text_h5_path,
        target_split
    ):
        self.visual_h5_path = Path(visual_h5_path)
        self.text_h5_path = Path(text_h5_path)
        self.vis_data = None
        self.txt_data = None
        self.data_list = []
        self.target_split = target_split

        print(f'Loading {target_split.upper()} data...')

        # Baca file CSV
        df = pd.read_csv(csv_path)
        # Filter berdasarkan split
        df = df[df['split'] == target_split].copy()

        # Cek keberadaan kolom 'label_like' atau 'label'
        if 'label_like' in df.columns:
            label_col = 'label_like'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError("CSV must contain 'label_like' column.")

        # Ubah tipe data label_like menjadi numerik
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        # Hapus baris dengan label kosong
        df = df.dropna(subset=[label_col])

        # Cek key gambar di file .h5 visual dan text
        vis_keys = _get_processed_key_h5(self.visual_h5_path)
        txt_keys = _get_processed_key_h5(self.text_h5_path)

        skipped = 0
        for _, row in df.iterrows():
            img_id = str(row['image_id'])
            label = float(row[label_col])

            key_to_use = None
            for k in _build_candidates(img_id):
                if k in vis_keys and k in txt_keys:
                    key_to_use = k
                    break

            if key_to_use is None:
                skipped += 1
                continue

            self.data_list.append({
                'key': key_to_use,
                'label': label
            })

        labels = [item['label'] for item in self.data_list]
        n_pos = int(sum(labels))
        n_neg = len(labels) - n_pos
        # Hitung rasio kelas negatif terhadap positif
        # Untuk perhitungan Loss (BCEWithLogitsLoss)
        self.pos_weight = (n_neg / max(n_pos, 1)) if len(labels) > 0 else 1.0

        print(f'{target_split.upper()}: {len(self.data_list)} items. Skipped: {skipped}')
        print(f'Positive: {n_pos}, Negative: {n_neg}')

    def __len__(self):
        return len(self.data_list)

    # Handler open file
    # Lazy loading
    def _ensure_open(self):
        if self.vis_data is None:
            self.vis_data = h5py.File(self.visual_h5_path, 'r')

        if self.txt_data is None:
            self.txt_data = h5py.File(self.text_h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open() # Pastikan file terbuka
        item = self.data_list[idx]
        key = item['key'] # Ambil key dari data_list

        # Baca embs visual dan text -> ubah ke tensor
        visual = torch.tensor(self.vis_data[key][:], dtype=torch.float32)
        text = torch.tensor(self.txt_data[key][:], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)

        # Kalau dimensi > 1, flatten
        if visual.dim() > 1:
            visual = visual.view(-1)
        if text.dim() > 1:
            text = text.view(-1)

        return (
            visual,
            text,
            label
        )

    # Handler close file
    def close(self):
        if self.vis_data is not None:
            self.vis_data.close()
            self.vis_data = None

        if self.txt_data is not None:
            self.txt_data.close()
            self.txt_data = None

class VisualDataset(Dataset):
    def __init__(
        self,
        csv_path,
        visual_h5_path,
        target_split
    ):
        self.visual_h5_path = Path(visual_h5_path)
        self.vis_data = None
        self.data_list = []
        self.target_split = target_split

        print(f'Loading {target_split.upper()} data...')

        # Baca file CSV
        df = pd.read_csv(csv_path)
        # Filter berdasarkan split
        df = df[df['split'] == target_split].copy()

        # Cek keberadaan kolom 'label_like' atau 'label'
        if 'label_like' in df.columns:
            label_col = 'label_like'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError("CSV must contain 'label_like' column.")

        # Ubah tipe data label_like menjadi numerik
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        # Hapus baris dengan label kosong
        df = df.dropna(subset=[label_col])

        # Cek key gambar di file .h5 visual dan text
        vis_keys = _get_processed_key_h5(self.visual_h5_path)

        skipped = 0
        for _, row in df.iterrows():
            img_id = str(row['image_id'])
            label = float(row[label_col])

            key_to_use = None
            for k in _build_candidates(img_id):
                if k in vis_keys:
                    key_to_use = k
                    break

            if key_to_use is None:
                skipped += 1
                continue

            self.data_list.append({
                'key': key_to_use,
                'label': label
            })

        labels = [item['label'] for item in self.data_list]
        n_pos = int(sum(labels))
        n_neg = len(labels) - n_pos
        # Hitung rasio kelas negatif terhadap positif
        # Untuk perhitungan Loss (BCEWithLogitsLoss)
        self.pos_weight = (n_neg / max(n_pos, 1)) if len(labels) > 0 else 1.0

        print(f'{target_split.upper()}: {len(self.data_list)} items. Skipped: {skipped}')
        print(f'Positive: {n_pos}, Negative: {n_neg}')

    def __len__(self):
        return len(self.data_list)

    # Handler open file
    # Lazy loading
    def _ensure_open(self):
        if self.vis_data is None:
            self.vis_data = h5py.File(self.visual_h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open() # Pastikan file terbuka
        item = self.data_list[idx]
        key = item['key'] # Ambil key dari data_list

        # Baca embs visual dan text -> ubah ke tensor
        visual = torch.tensor(self.vis_data[key][:], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)

        # Kalau dimensi > 1, flatten
        if visual.dim() > 1:
            visual = visual.view(-1)

        return (
            visual,
            label
        )

    # Handler close file
    def close(self):
        if self.vis_data is not None:
            self.vis_data.close()
            self.vis_data = None

class CaptionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        text_h5_path,
        target_split
    ):
        self.text_h5_path = Path(text_h5_path)
        self.txt_data = None
        self.data_list = []
        self.target_split = target_split

        print(f'Loading {target_split.upper()} data...')

        # Baca file CSV
        df = pd.read_csv(csv_path)
        # Filter berdasarkan split
        df = df[df['split'] == target_split].copy()

        # Cek keberadaan kolom 'label_like' atau 'label'
        if 'label_like' in df.columns:
            label_col = 'label_like'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError("CSV must contain 'label_like' column.")

        # Ubah tipe data label_like menjadi numerik
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        # Hapus baris dengan label kosong
        df = df.dropna(subset=[label_col])

        # Cek key gambar di file .h5 visual dan text
        txt_keys = _get_processed_key_h5(self.text_h5_path)

        skipped = 0
        for _, row in df.iterrows():
            img_id = str(row['image_id'])
            label = float(row[label_col])

            key_to_use = None
            for k in _build_candidates(img_id):
                if k in txt_keys:
                    key_to_use = k
                    break

            if key_to_use is None:
                skipped += 1
                continue

            self.data_list.append({
                'key': key_to_use,
                'label': label
            })

        labels = [item['label'] for item in self.data_list]
        n_pos = int(sum(labels))
        n_neg = len(labels) - n_pos
        # Hitung rasio kelas negatif terhadap positif
        # Untuk perhitungan Loss (BCEWithLogitsLoss)
        self.pos_weight = (n_neg / max(n_pos, 1)) if len(labels) > 0 else 1.0

        print(f'{target_split.upper()}: {len(self.data_list)} items. Skipped: {skipped}')
        print(f'Positive: {n_pos}, Negative: {n_neg}')

    def __len__(self):
        return len(self.data_list)

    # Handler open file
    # Lazy loading
    def _ensure_open(self):
        if self.txt_data is None:
            self.txt_data = h5py.File(self.text_h5_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open() # Pastikan file terbuka
        item = self.data_list[idx]
        key = item['key'] # Ambil key dari data_list

        # Baca embs visual dan text -> ubah ke tensor
        text = torch.tensor(self.txt_data[key][:], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)

        # Kalau dimensi > 1, flatten
        if text.dim() > 1:
            text = text.view(-1)

        return (
            text,
            label
        )

    # Handler close file
    def close(self):
        if self.txt_data is not None:
            self.txt_data.close()
            self.txt_data = None

# Handling class imbalance
def make_weighted_sampler(labels_list):
    labels_np = np.asarray(labels_list).astype(int)
    class_counts = np.bincount(labels_np, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_np]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )