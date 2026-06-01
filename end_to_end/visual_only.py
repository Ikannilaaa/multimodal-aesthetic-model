from functions.config import *
from functions.utils import *
from functions.data import VisualDataset, make_weighted_sampler
from functions.models import VisualOnlyNetwork
from functions.train import train_model, evaluate_model
from functions.extractor import extract_visual_features
from torch.utils.data import DataLoader

def run():
    print_system_info()

    # Build Path
    paths = build_paths(scenario='visual_only')

    # Copy file cloud ke lokal
    if COPY_DATA_TO_LOCAL:
        local_img_dirs = copy_images_to_local(IMG_DIRS, paths.local_images)
    else:
        local_img_dirs = IMG_DIRS

    # Path .h5 files
    LOCAL_VIS_H5 = {s: paths.local_features / f'visual_{s}.h5' for s in SPLITS}
    DRIVE_VIS_H5 = {s: paths.cloud_features / f'visual_{s}.h5' for s in SPLITS}

    for split in SPLITS:
        if DRIVE_VIS_H5[split].exists() and not LOCAL_VIS_H5[split].exists():
            shutil.copy2(DRIVE_VIS_H5[split], LOCAL_VIS_H5[split])
            log.info(f'Copied {DRIVE_VIS_H5[split]} to {LOCAL_VIS_H5[split]}')

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

    print('=== Step 2: Preparing Dataset ===')
    local_csv = paths.local_root / CSV_PATH.name
    if COPY_DATA_TO_LOCAL and not local_csv.exists():
        shutil.copy2(CSV_PATH, local_csv)
    elif not COPY_DATA_TO_LOCAL:
        local_csv = CSV_PATH

    train_dataset = VisualDataset(local_csv, LOCAL_VIS_H5['train'], 'train')
    test_dataset = VisualDataset(local_csv, LOCAL_VIS_H5['test'], 'test')
    val_dataset = VisualDataset(local_csv, LOCAL_VIS_H5['val'], 'val')

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

    print('=== Step 3: Model Training ===')
    VISUAL_DIM = train_dataset[0][0].shape[0]

    model = VisualOnlyNetwork(
        visual_dim=VISUAL_DIM
    ).to(DEVICE)

    local_ckpt = paths.local_ckpt / 'best_aesthetic_model_visual.pth'
    drive_ckpt = paths.cloud_ckpt / 'best_aesthetic_model_visual.pth'

    train_model(
        model,
        train_loader,
        val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        save_path=local_ckpt
    )
    sync_to_drive(local_ckpt, drive_ckpt)

    print('=== Step 4: Model Evaluation ===')
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