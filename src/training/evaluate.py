"""WER / CER evaluation and CTC greedy decoding."""
import sys
import os
import json
import torch
import jiwer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ── CTC greedy decoder ────────────────────────────────────────────────────
def ctc_greedy_decode(log_probs, output_lengths, idx_to_char, blank_idx=0):
    """
    Greedy best-path CTC decoding: argmax → collapse repeats → remove blank.

    Parameters
    ----------
    log_probs      : [T, B, vocab_size]
    output_lengths : [B]
    idx_to_char    : dict  int -> char

    Returns
    -------
    list of decoded strings, length B
    """
    pred_ids = log_probs.argmax(dim=-1).permute(1, 0)  # [B, T]
    results = []
    for b in range(pred_ids.shape[0]):
        length = output_lengths[b].item()
        seq = pred_ids[b, :length].cpu().tolist()
        # Collapse repeated tokens, remove blank
        decoded, prev = [], None
        for idx in seq:
            if idx != blank_idx and idx != prev:
                decoded.append(idx_to_char.get(idx, ''))
            prev = idx
        results.append(''.join(decoded))
    return results


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_wer_cer(references, hypotheses):
    """Return WER and CER as floats in [0, 1]."""
    # Filter pairs where reference is non-empty
    pairs = [(r.strip(), h.strip()) for r, h in zip(references, hypotheses)
             if r.strip()]
    if not pairs:
        return 1.0, 1.0
    refs, hyps = zip(*pairs)
    # Replace empty hypotheses with a single space so jiwer doesn't crash
    hyps = [h if h else ' ' for h in hyps]
    try:
        wer = jiwer.wer(list(refs), list(hyps))
        cer = jiwer.cer(list(refs), list(hyps))
    except Exception:
        # Fallback: compute manually
        wer, cer = 1.0, 1.0
    return wer, cer


# ── Full evaluation loop ──────────────────────────────────────────────────
def evaluate(model, loader, idx_to_char, device, max_batches=None):
    """
    Run the model on `loader`, decode with greedy CTC, compute WER/CER.

    Returns a dict with keys: wer, cer, references, hypotheses
    """
    model.eval()
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Evaluating')):
            if max_batches and i >= max_batches:
                break
            features = batch['features'].to(device)
            feat_lengths = batch['feat_lengths'].to(device)

            log_probs, out_lengths = model(features, feat_lengths)
            hyps = ctc_greedy_decode(log_probs, out_lengths, idx_to_char)

            all_refs.extend(batch['texts'])
            all_hyps.extend(hyps)

    wer, cer = compute_wer_cer(all_refs, all_hyps)

    return {
        'wer': round(wer, 4),
        'cer': round(cer, 4),
        'n_samples': len(all_refs),
        'references': all_refs[:10],
        'hypotheses': all_hyps[:10],
    }


# ── Standalone script ─────────────────────────────────────────────────────
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    from src.models.cnn_lstm import CNNLSTM
    from src.training.dataset import ArabicASRDataset, collate_fn

    CHECKPOINT = os.path.join(os.path.dirname(__file__), '../../checkpoints/best_model.pt')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg = ckpt['config']
    idx_to_char = {int(k): v for k, v in ckpt['idx_to_char'].items()}
    vocab = ckpt['vocab']

    model = CNNLSTM(**cfg).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    print("Loading test split (1000 samples)…")
    test_ds = load_dataset('MohamedRashad/common-voice-18-arabic', split='test[:1000]')
    dataset = ArabicASRDataset(test_ds, vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    results = evaluate(model, loader, idx_to_char, DEVICE)
    print(f"\nWER : {results['wer']*100:.1f}%")
    print(f"CER : {results['cer']*100:.1f}%")
    print(f"\nSample predictions (reference → hypothesis):")
    for r, h in zip(results['references'], results['hypotheses']):
        print(f"  REF: {r}")
        print(f"  HYP: {h}")
        print()

    out_path = os.path.join(os.path.dirname(__file__), '../../results/cnn_lstm_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")
