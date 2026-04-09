
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.audio_utils import SAMPLE_RATE
from src.training.dataset import (ArabicASRDataset, build_vocab,
                                   collate_fn, normalize_arabic_text)
from src.models.cnn_lstm import CNNLSTM
from src.training.evaluate import evaluate

# ── Config ────────────────────────────────────────────────────────────────
TRAIN_SIZE    = 28000    # full training set
VAL_SIZE      = 2000
BATCH_SIZE    = 16       # 4 GB GPU
EPOCHS        = 40
LR            = 1e-3
WARMUP_EPOCHS = 3        # linear LR warmup before cosine decay
N_MELS        = 80
HIDDEN_SIZE   = 512      # increased capacity
LSTM_LAYERS   = 2
DROPOUT       = 0.3
NUM_WORKERS   = 2
MAX_AUDIO_SEC = 7.0

ROOT         = os.path.join(os.path.dirname(__file__), '..', '..')
CKPT_DIR     = os.path.join(ROOT, 'checkpoints')
RESULTS_DIR  = os.path.join(ROOT, 'results')
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
# ─────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc='  train', leave=False):
        features     = batch['features'].to(device)
        labels       = batch['labels'].to(device)
        feat_lengths = batch['feat_lengths'].to(device)
        lbl_lengths  = batch['label_lengths'].to(device)

        optimizer.zero_grad()
        log_probs, out_lengths = model(features, feat_lengths)

        loss = criterion(log_probs, labels, out_lengths, lbl_lengths)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


def val_loss(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='  val  ', leave=False):
            features     = batch['features'].to(device)
            labels       = batch['labels'].to(device)
            feat_lengths = batch['feat_lengths'].to(device)
            lbl_lengths  = batch['label_lengths'].to(device)

            log_probs, out_lengths = model(features, feat_lengths)
            loss = criterion(log_probs, labels, out_lengths, lbl_lengths)
            if torch.isfinite(loss):
                total_loss += loss.item()
                n += 1
    return total_loss / max(n, 1)


def main():
    print(f"Device : {DEVICE}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    print(f"Loading dataset  (train={TRAIN_SIZE}, val={VAL_SIZE})…")
    raw_train = load_dataset('MohamedRashad/common-voice-18-arabic',
                             split=f'train[:{TRAIN_SIZE}]')
    raw_val   = load_dataset('MohamedRashad/common-voice-18-arabic',
                             split=f'validation[:{VAL_SIZE}]')

    # ── Vocabulary ────────────────────────────────────────────────────
    print("Building vocabulary…")
    norm_sentences = [normalize_arabic_text(s) for s in raw_train['sentence']]
    vocab, idx_to_char = build_vocab(norm_sentences)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size} characters")

    vocab_path = os.path.join(CKPT_DIR, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab,
                   'idx_to_char': {str(k): v for k, v in idx_to_char.items()}},
                  f, ensure_ascii=False, indent=2)

    # ── Datasets & loaders ────────────────────────────────────────────
    train_set = ArabicASRDataset(raw_train, vocab, max_audio_sec=MAX_AUDIO_SEC)
    val_set   = ArabicASRDataset(raw_val,   vocab, max_audio_sec=MAX_AUDIO_SEC)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────
    model = CNNLSTM(n_mels=N_MELS, vocab_size=vocab_size,
                    hidden_size=HIDDEN_SIZE, num_lstm_layers=LSTM_LAYERS,
                    dropout=DROPOUT).to(DEVICE)
    print(f"  Parameters : {model.count_parameters():,}")

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Linear warmup for first WARMUP_EPOCHS, then cosine decay to 0
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS          # ramp up
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss = val_loss(model, val_loader, criterion, DEVICE)

        scheduler.step()
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Train loss: {tr_loss:.4f}  |  Val loss: {vl_loss:.4f}  |  LR: {cur_lr:.2e}")

        # Save intermediate loss plot every 5 epochs
        if epoch % 5 == 0:
            _plot_loss_curves(history, RESULTS_DIR)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'vocab':            vocab,
                'idx_to_char':      idx_to_char,
                'config': {
                    'n_mels':         N_MELS,
                    'vocab_size':     vocab_size,
                    'hidden_size':    HIDDEN_SIZE,
                    'num_lstm_layers': LSTM_LAYERS,
                    'dropout':        DROPOUT,
                },
            }, os.path.join(CKPT_DIR, 'best_model.pt'))
            print(f"  ✓ Saved best model  (val_loss={vl_loss:.4f})")

    # ── Save training history ─────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot and save loss curves ─────────────────────────────────────
    _plot_loss_curves(history, RESULTS_DIR)

    # ── Quick evaluation on val set ───────────────────────────────────
    print("\nRunning final evaluation on validation set…")
    ckpt = torch.load(os.path.join(CKPT_DIR, 'best_model.pt'), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    results = evaluate(model, val_loader, idx_to_char, DEVICE, max_batches=20)
    print(f"  WER : {results['wer']*100:.1f}%")
    print(f"  CER : {results['cer']*100:.1f}%")
    print("\nSample predictions:")
    for r, h in zip(results['references'][:5], results['hypotheses'][:5]):
        print(f"  REF: {r}")
        print(f"  HYP: {h}\n")

    with open(os.path.join(RESULTS_DIR, 'cnn_lstm_results.json'), 'w', encoding='utf-8') as f:
        json.dump({**history, **results}, f, ensure_ascii=False, indent=2)

    print("Training complete. Checkpoint saved to checkpoints/best_model.pt")


def _plot_loss_curves(history, results_dir):
    """Save training/validation loss curve PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        train_loss = history['train_loss']
        val_loss   = history['val_loss']
        epochs     = list(range(1, len(train_loss) + 1))
        best_ep    = int(np.argmin(val_loss)) + 1

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_loss, 'o-', color='steelblue', linewidth=2, label='Train Loss')
        ax.plot(epochs, val_loss,   's-', color='coral',     linewidth=2, label='Validation Loss')
        ax.axvline(best_ep, color='green', linestyle='--', linewidth=1.2,
                   label=f'Best epoch ({best_ep},  val={min(val_loss):.3f})')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('CTC Loss', fontsize=12)
        ax.set_title('CNN-LSTM Training & Validation Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(results_dir, 'training_curves.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Loss curve saved → {out}")
    except Exception as e:
        print(f"  (Could not save loss plot: {e})")


if __name__ == '__main__':
    main()
