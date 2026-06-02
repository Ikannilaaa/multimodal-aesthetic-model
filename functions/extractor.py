# extractor.py

from pathlib import Path
from functions.config import (
    MODEL_ID,
    VISUAL_SYNC_EVERY,
    VALID_EXT,
    DEVICE,
    VISUAL_BATCH_SIZE,
)
from functions.utils import (
    load_clip_once,
    _get_processed_key_h5, _normalize_torch_features,
    _write_feature_h5, sync_to_drive, free_mem, _find_caption_field
)
from tqdm import tqdm
import json
import h5py
import torch
import logging
from PIL import Image

log = logging.getLogger(__name__)

# Ekstraktor
def extract_visual_features(
    image_folder,
    output_file,
    model_id=MODEL_ID,
    batch_size=VISUAL_BATCH_SIZE,
    model=None,
    processor=None,
    sync_output_file=None,
    sync_every=VISUAL_SYNC_EVERY,
):
    # Persiapan path dan validasi path output
    IMAGE_FOLDER = Path(image_folder)
    OUTPUT_FILE = Path(output_file)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Cek esktensi file output fitur (.h5)
    if not OUTPUT_FILE.name.endswith('.h5'):
        raise ValueError(f'Output file must be a HDF5 file: {OUTPUT_FILE}')

    # Load model
    own_model = False
    if model is None or processor is None:
        model, processor, _ = load_clip_once(model_id)
        own_model = True

    # Resume logic + skip file yang udah diproses
    done_keys = _get_processed_key_h5(OUTPUT_FILE)
    all_files = [p for p in sorted(IMAGE_FOLDER.iterdir()) if p.suffix.lower() in VALID_EXT]
    remaining = [p for p in all_files if p.name not in done_keys]

    log.info(f'Found {len(all_files)} images to process...')

    # Kalau data sudah diproses semua
    if not remaining:
        print(f'Visual features already complete and saved in: {OUTPUT_FILE}')
        # Hapus model dari GPU
        if own_model:
            del model
            free_mem()
        return

    # Write fitur visual ke file .h5
    with h5py.File(OUTPUT_FILE, 'a') as h:
        for i in tqdm(range(0, len(remaining), batch_size), desc='Extracting Visual Embedding'):
            # Gambar diproses per size batch
            batch_files = remaining[i:i + batch_size]
            batch_images = []
            valid_batch_filenames = [] # Untuk filter yang valid dan error

            # Buka gambar satu per satu dalam batch
            for p in batch_files:
                # Buka gambar
                try:
                    image = Image.open(p).convert('RGB')
                    batch_images.append(image)
                    valid_batch_filenames.append(p.name)
                except Exception as e:
                    # Kalau error, skip gambar
                    print(f'Failed to load {p.name}: {e}')

            if not batch_images:
                continue

            # Preprocessing gambar
            inputs = processor(images=batch_images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(DEVICE) # Simpan pixel values ke GPU

            # Ekstraksi visual emb
            # Hanya inference model
            with torch.no_grad():
                if DEVICE == 'cuda':
                    # Autocast untuk automatic mixed precision
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        image_features = model.get_image_features(pixel_values=pixel_values)
                else:
                    image_features = model.get_image_features(pixel_values=pixel_values)

                # Normalisasi fitur visual
                image_features = _normalize_torch_features(image_features).float().cpu().numpy()

            # Save hasil ekstraksi ke file .h5
            for filename, feature in zip(valid_batch_filenames, image_features):
                _write_feature_h5(h, filename, feature)

            # Save file lokal ke drive secara berkala
            processed_count = min(i + len(batch_files), len(remaining))
            if processed_count % sync_every == 0 and sync_output_file is not None:
                h.flush()
                sync_to_drive(OUTPUT_FILE, Path(sync_output_file))

    # Save file lokal ke drive
    if sync_output_file is not None:
        sync_to_drive(OUTPUT_FILE, Path(sync_output_file))

    print(f'Visual features saved in {OUTPUT_FILE}')

    # Bersihkan GPU
    if own_model:
        del model, processor
        free_mem()

# Handling caption dan ekstraksi fitur semantik

# Apabila token > 77, dipecah
def _get_long_text_embedding(
    model,
    tokenizer,
    text,
    max_len=77, # Max token CLIP
    device=DEVICE,
    batch_size_chunks=8
):
    # Ubah teks -> token
    token_ids = tokenizer(text, add_special_tokens=False)['input_ids']

    # Kalau teks panjang (> 75 token)
    chunk_size = max_len - 2 # per chunk berisi 75 token kata + 2 special token
    chunks = []

    # pecah token_ids ke chunk
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        # Tambahkan special token
        input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
        chunks.append(input_ids) # Save di list chunks

    # Olah chunks sesuai jumlah batch
    all_embs = []
    for j in range(0, len(chunks), batch_size_chunks):
        chunk_batch = chunks[j:j+batch_size_chunks]
        # Tambahkan padding
        padded = tokenizer.pad({'input_ids': chunk_batch}, return_tensors='pt')
        padded = {k: v.to(device) for k, v in padded.items()}

        with torch.no_grad(): # Inference only
            if device == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    emb = model.get_text_features(**padded)
            else:
                emb = model.get_text_features(**padded)

        # Normalisasi hasil embedding
        all_embs.append(_normalize_torch_features(emb))

    # Mean pooling
    stacked = torch.cat(all_embs, dim=0) # Gabung semua chunk jdi satu
    mean_emb = stacked.mean(dim=0, keepdim=True) # Hitung mean dr seluruh chunk

    # Return hasil normalisasi
    return mean_emb / mean_emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

# Handling teks pendek (token < 75)
def _encode_short_text(
    model,
    tokenizer,
    text,
    max_len=77,
    device=DEVICE,
):
    inputs = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        if device == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                emb = model.get_text_features(**inputs)
        else:
            emb = model.get_text_features(**inputs)

    return _normalize_torch_features(emb).float().cpu().numpy()


# Ekstraksi fitur semantik
def extract_semantic_features(
    json_path, # Path caption
    output_file,
    model_id=MODEL_ID,
    model=None,
    tokenizer=None,
    sync_output_file=None,
    sync_every=VISUAL_SYNC_EVERY,
    text_batch_size=256,
    device=DEVICE,
):
    json_path = Path(json_path)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Cek ekstensi output file
    if output_file.suffix.lower() != '.h5':
        raise ValueError(f'Output file must be a HDF5 file: {output_file}')

    # Cek apakah model sudah di load
    own_model = False
    if model is None or tokenizer is None:
        model, _, tokenizer = load_clip_once(model_id)
        own_model = True

    print(f'Device used: {device} for semantic extraction.')

    if not json_path.exists():
        raise FileNotFoundError(f'Caption JSON file not found: {json_path}')

    # Baca isi JSON caption
    with open(json_path, 'r') as f:
        data = json.load(f)

    text_data = {str(item['filename']): _find_caption_field(item) for item in data}
    filenames = list(text_data.keys())
    done_keys = _get_processed_key_h5(output_file)
    remaining = [p for p in filenames if p not in done_keys]

    print(f'Processing {len(filenames)} captions, {len(remaining)} images...')

    if not remaining:
        print(f'Semantic features extraction is complete: {output_file}')
        if own_model:
            del model, tokenizer
            free_mem()
        return

    # Save ke file .h5
    with h5py.File(output_file, 'a') as h:
        for i in tqdm(range(0, len(remaining), text_batch_size), desc='Encoding text'):
              batch_filenames = remaining[i:i + text_batch_size]
              batch_texts = [text_data[f] for f in batch_filenames]

              # Pisahin short dan long caption
              short_items = []
              long_items = []

              for fn, text in zip(batch_filenames, batch_texts):
                  # Cari jumlah token
                  token_ids = tokenizer(text, add_special_tokens=False)['input_ids']

                  # Filter caption sesuai panjang token
                  if len(token_ids) <= 75:
                      short_items.append((fn, text))
                  else:
                      long_items.append((fn, text))

              if short_items:
                  short_names = [x[0] for x in short_items]
                  short_texts = [x[1] for x in short_items]
                  short_embs = _encode_short_text(model, tokenizer, short_texts)

                  # Save nama dan embedding dalam file .h5
                  for fn, emb in zip(short_names, short_embs):
                      _write_feature_h5(h, fn, emb)

              if long_items:
                  for fn, text in long_items:
                      try:
                          embs = _get_long_text_embedding(model, tokenizer, text)
                          _write_feature_h5(h, fn, embs.float().cpu().numpy()[0])
                      except Exception as e:
                          print(f'Error processing {fn}: {e}')

              # Hitung file yang sudah selesai
              processed_count = min(i + text_batch_size, len(remaining))
              if processed_count % sync_every == 0:
                  h.flush()
                  if sync_output_file is not None:
                      sync_to_drive(output_file, Path(sync_output_file))

        h.flush()

    if sync_output_file is not None:
        sync_to_drive(output_file, Path(sync_output_file))

    print(f'Semantic features saved to {output_file}')

    # Hapus model
    if own_model:
        del model, tokenizer
        free_mem()