"""PyTorch Dataset, vocabulary builder, and DataLoader collate for Arabic ASR."""
import re
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.audio_utils import load_audio, normalize_audio, SAMPLE_RATE
from src.preprocessing.feature_extraction import extract_log_mel, normalize_features

# ── Arabic text normalisation ──────────────────────────────────────────────
_DIACRITICS = re.compile(
    r'[\u064B-\u065F'          # harakat
    r'\u0610-\u061A'           # extended arabic letters marks
    r'\u06D6-\u06DC'           # Quranic annotation signs
    r'\u06DF-\u06E4'
    r'\u06E7\u06E8'
    r'\u06EA-\u06ED]'
)
_TATWEEL = re.compile(r'\u0640')          # kashida
_NON_ARABIC = re.compile(r'[^\u0600-\u06FF\s]')  # keep Arabic block + spaces


def normalize_arabic_text(text: str) -> str:
    text = _DIACRITICS.sub('', text)
    text = _TATWEEL.sub('', text)
    text = _NON_ARABIC.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Vocabulary ────────────────────────────────────────────────────────────
def build_vocab(sentences):
    """
    Build char vocabulary from a list of *already normalised* sentences.
    Index 0 is reserved for the CTC blank token.

    Returns:
        vocab       : dict  char -> int  (blank=0)
        idx_to_char : dict  int  -> char
    """
    chars = sorted(set(''.join(sentences)))
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab['<blank>'] = 0
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = '<blank>'
    return vocab, idx_to_char


# ── Dataset ───────────────────────────────────────────────────────────────
class ArabicASRDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_audio_sec=10.0,
                 sample_rate=SAMPLE_RATE):
        self.data = hf_dataset
        self.vocab = vocab
        self.max_samples = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Audio
        audio = load_audio(item['audio'], target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        audio = audio[:self.max_samples]

        # Features: [T, n_mels]
        features = extract_log_mel(audio, self.sample_rate)
        features = normalize_features(features)

        # Text → char indices
        text = normalize_arabic_text(item['sentence'])
        label = [self.vocab[c] for c in text if c in self.vocab]

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,
        }


# ── Collate ───────────────────────────────────────────────────────────────
def collate_fn(batch):
    """Pad features; concatenate labels (CTC format)."""
    # Sort by descending feature length
    batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)

    features = [b['features'] for b in batch]
    labels = [b['label'] for b in batch]
    texts = [b['text'] for b in batch]

    feat_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    # Pad features to max length in batch
    max_len = feat_lengths[0].item()
    n_mels = features[0].shape[1]
    padded = torch.zeros(len(features), max_len, n_mels)
    for i, f in enumerate(features):
        padded[i, :f.shape[0]] = f

    # CTC expects a flat label tensor
    flat_labels = torch.cat([l for l in labels if len(l) > 0])

    return {
        'features': padded,            # [B, T_max, n_mels]
        'feat_lengths': feat_lengths,  # [B]
        'labels': flat_labels,         # [sum(label_lengths)]
        'label_lengths': label_lengths,# [B]
        'texts': texts,                # list[str]
    }
