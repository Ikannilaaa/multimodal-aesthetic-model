from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from functions.config import (
    THRESHOLDS,
    DEVICE,
    TRAIN_BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    EPOCHS,
    RUNS,
    MODEL_ID
)

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

def infer_logits(model, loader, device):
    model.eval() # Evaluation mode
    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for visual, text, label in loader:
            visual, text, label = visual.to(device, non_blocking=True), text.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits = model(visual, text) # Forward Pass
            all_logits.append(logits.detach().cpu())
            all_labels.append(label.detach().cpu())

    logits = torch.cat(all_logits, dim=0).squeeze().numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return logits, labels

def find_best_threshold(logits, labels):
    # Convert logits -> Probability
    # Sigmoid function
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_threshold = {
        'threshold': 0.5,
        'f1': -1.0,
        'bacc': -1.0,
        'score': -1.0
    }

    for threshold in THRESHOLDS:
        # Convert Probability -> Prediction
        preds = (probs >= threshold).astype(np.int32)
        f1 = f1_score(labels, preds, zero_division=0)
        bacc = balanced_accuracy_score(labels, preds)
        # Custom optimization metric
        # F1 > bacc karena F1 lebih sensitif ke pos class
        # Aesthetic classification mungkin imbalance
        score = 0.7 * f1 + 0.3 * bacc

        if score > best_threshold['score']:
            best_threshold = {
                'threshold': float(threshold),
                'f1': float(f1),
                'bacc': float(bacc),
                'score': float(score)
            }

    return best_threshold

def evaluate_model(
    model,
    test_loader,
    device=DEVICE,
    ckpt_path=None,
    out_dir=None
):
    best_threshold = 0.5 # Bisa diganti pake tuned treshold
    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)

        # Kalau ckpt bentuknya dictionary
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            best_threshold = float(ckpt.get('best_threshold', 0.5))
        else: # Kalau hanya state_dict
            model.load_state_dict(ckpt)

    logits, labels = infer_logits(model, test_loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits)) # Sigmoid
    preds = (probs >= best_threshold).astype(np.int32) # Prediksi biner

    print('Classification Report: ')
    print(classification_report(
        labels,
        preds,
        target_names=['Low Aesthetic', 'High Aesthetic'],
        digits=4
    ))

    try:
        # Dihitung dari probabilitas
        auc = roc_auc_score(labels, probs)
        print(f'ROC-AUC: {auc:.4f}')
    except ValueError:
        print('ROC-AUC cannot be computed.')
        auc = None

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Aesthetic', 'High Aesthetic'],
                yticklabels=['Low Aesthetic', 'High Aesthetic'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='white')
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save confusion matrix
        plt.savefig(out_dir / 'confusion_matrix.png', dpi=180, bbox_inches='tight')

        # Simpan prediksi per sampel
        pd.DataFrame({
            'labels': labels.astype(int),
            'probs': probs,
            'preds': preds
        }).to_csv(out_dir / 'test_predictions.csv', index=False)

        # Simpan metrik lainnya
        pd.DataFrame({
            'threshold': [best_threshold],
            'f1': [float(f1_score(labels, preds, zero_division=0))],
            'bacc': [float(balanced_accuracy_score(labels, preds))],
            'roc_auc': [float(auc)] if auc is not None else [None]
        }).to_csv(out_dir / 'test_metrics.csv', index=False)

    plt.show()

# Handler Training
def train_model(
    model,
    train_loader,
    val_loader,
    device=DEVICE,
    epochs=EPOCHS,
    save_path='best_model.pth'
):
    # Load train dataset
    train_dataset = train_loader.dataset
    # Ambil Pos weight untuk perhitungan class imbalance
    pos_weight_value = getattr(train_dataset, 'pos_weight', 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Optimizer untuk update parameter model
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Menurunkan learning rate ketika validasi metrik tidak membaik
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=PATIENCE)
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    patience_count = 0
    history = []

    # Loop training epoch
    print(f'Training for {epochs} epochs...')
    for epoch in range(1, epochs + 1):
        model.train() # Training mode
        train_loss = 0.0

        for visual, text, label in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            # Pindah ke device
            visual, text, label = visual.to(device, non_blocking=True), text.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # Hapus gradient lama sebelum backward baru

            if device == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(visual, text).squeeze(-1)
                    loss = criterion(out, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # Kalau CPU
                out = model(visual, text).squeeze(-1)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() # Akumulasi loss

        # Validation per epoch
        val_logits, val_labels = infer_logits(model, val_loader, device)
        val_best = find_best_threshold(val_logits, val_labels)
        scheduler.step(val_best['f1'])

        # Save history epoch
        avg_train = train_loss / max(len(train_loader), 1)
        history.append({
            'epoch': epoch,
            'train_loss': avg_train,
            'val_f1': val_best['f1'],
            'val_bacc': val_best['bacc'],
            'val_threshold': val_best['threshold'],
            'lr': optimizer.param_groups[0]['lr']
        })

        print(
            f'Loss Train: {avg_train:.4f}, '
            f'Val F1: {val_best["f1"]:.4f}, '
            f'Val BACC: {val_best["bacc"]:.4f}, '
            f'Threshold: {val_best["threshold"]:.2f}, '
        )

        # Save ckpt terbaik
        if val_best['f1'] > best_val_f1:
            best_val_f1 = val_best['f1']
            best_epoch = epoch
            best_threshold = val_best['threshold']
            patience_count = 0

            # Isi ckpt, bentuk dict
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'best_threshold': best_threshold,
                'config': {
                    'model_id': MODEL_ID,
                    'lr': LR,
                    'epochs': epochs,
                    'train_batch_size': TRAIN_BATCH_SIZE
                }
            }, save_path)
        else:
            # Early stopping kalau F1 score tidak membaik
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'Early stopping at epoch {epoch}.')
                break

    # Save history epoch ke csv
    history_df = pd.DataFrame(history)
    history_df.to_csv(RUNS / f'train_history_{MODEL_ID.replace('/', '_')}.csv', index=False)

    print(f'Best epoch: {best_epoch}, Best F1: {best_val_f1:.4f}, Best Threshold: {best_threshold:.2f}')