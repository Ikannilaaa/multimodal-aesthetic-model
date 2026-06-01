from functions.config import *
from functions.utils import *
from functions.data import HybridFusionDataset, make_weighted_sampler
from functions.models import AestheticFusionNetwork
from functions.train import train_model, evaluate_model
from functions.extractor import extract_visual_features, extract_semantic_features
from torch.utils.data import DataLoader

def run(prompt_type):
    print_system_info()

    if prompt_type == 'contrastive':
        CAPTION_FILES = CAPTION_FILES_CONTRASTIVE
    elif prompt_type == 'structured':
        CAPTION_FILES = CAPTION_FILES_STRUCTURED
    else:
        raise ValueError(f'Invalid prompt type: {prompt_type}')

    paths = build_paths(scenario='multimodal', prompt_type=prompt_type)

    # Copy file cloud ke local
    if COPY_DATA_TO_LOCAL:
        local_img_dirs = copy_images_to_local(IMG_DIRS, paths.local_images)
        local_caption_dirs = copy_captions_to_local(CAPTION_FILES, paths.local_captions)
    else:
        local_img_dirs = IMG_DIRS
        local_caption_dirs = CAPTION_FILES

    # Path .h5 files
    LOCAL_VIS_H5 = {s: paths.local_features / f'visual_{s}.h5' for s in SPLITS}
    LOCAL_TXT_H5 = {s: paths.local_features / f'text_{s}.h5' for s in SPLITS}
    DRIVE_VIS_H5 = {s: paths.cloud_features / f'visual_{s}.h5' for s in SPLITS}
    DRIVE_TXT_H5 = {s: paths.cloud_features / f'text_{s}.h5' for s in SPLITS}

    for split in SPLITS:
        if DRIVE_VIS_H5[split].exists() and not LOCAL_VIS_H5[split].exists():
            shutil.copy2(DRIVE_VIS_H5[split], LOCAL_VIS_H5[split])
            log.info(f'Copied {DRIVE_VIS_H5[split]} to {LOCAL_VIS_H5[split]}')

        if DRIVE_TXT_H5[split].exists() and not LOCAL_TXT_H5[split].exists():
            shutil.copy2(DRIVE_TXT_H5[split], LOCAL_TXT_H5[split])
            log.info(f'Copied {DRIVE_TXT_H5[split]} to {LOCAL_TXT_H5[split]}')

    # Load model CLIP
    clip_model, clip_processor, clip_tokenizer = load_clip_once(MODEL_ID)

    print('=== Step 1: Visual Extraction ===')

    for split in SPLITS:
        extract_visual_features(
            image_folder=local_img_dirs[split],
            output_file=LOCAL_VIS_H5[split],
            model=clip_model,
            model_id=MODEL_ID,
            processor=clip_processor,
            sync_output_file=DRIVE_VIS_H5[split],
            batch_size=VISUAL_BATCH_SIZE,
        )
        free_mem()

    print('=== Step 2: Text Extraction ===')
    for split in SPLITS:
        extract_semantic_features(
            json_path=local_caption_dirs[split],
            output_file=LOCAL_TXT_H5[split],
            model=clip_model,
            model_id=MODEL_ID,
            tokenizer=clip_tokenizer,
            text_batch_size=256,
            sync_output_file=DRIVE_TXT_H5[split],
        )
        free_mem()

    # hapus model setelah ekstraksi selesai
    del clip_model, clip_processor, clip_tokenizer
    free_mem()

    print('=== Step 3: Preparing Dataset ===')
    local_csv = paths.local_root / CSV_PATH.name
    if COPY_DATA_TO_LOCAL and not local_csv.exists():
        shutil.copy2(CSV_PATH, local_csv)
    elif not COPY_DATA_TO_LOCAL:
        local_csv = CSV_PATH

    train_dataset = HybridFusionDataset(local_csv, LOCAL_VIS_H5['train'], LOCAL_TXT_H5['train'], 'train')
    test_dataset = HybridFusionDataset(local_csv, LOCAL_VIS_H5['test'], LOCAL_TXT_H5['test'], 'test')
    val_dataset = HybridFusionDataset(local_csv, LOCAL_VIS_H5['val'], LOCAL_TXT_H5['val'], 'val')

    train_labels = [item['label'] for item in train_dataset.data_list]
    train_sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print('=== Step 4: Model Training ===')
    VISUAL_DIM = train_dataset[0][0].shape[0]
    TEXT_DIM = train_dataset[0][1].shape[0]

    model = AestheticFusionNetwork(
        visual_dim=VISUAL_DIM,
        text_dim=TEXT_DIM
    ).to(DEVICE)

    local_ckpt = paths.local_ckpt / f'best_aesthetic_model_{prompt_type}.pth'
    drive_ckpt = paths.cloud_ckpt / f'best_aesthetic_model_{prompt_type}.pth'

    train_model(
        model,
        train_loader,
        val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=local_ckpt
    )
    sync_to_drive(local_ckpt, drive_ckpt)

    print('=== Step 5: Model Evaluation ===')
    evaluate_model(
        model,
        test_loader,
        device=DEVICE,
        ckpt_path=local_ckpt,
        out_dir=paths.cloud_runs
    )

    train_dataset.close()
    test_dataset.close()
    val_dataset.close()

    print('=== Process Completed ===')