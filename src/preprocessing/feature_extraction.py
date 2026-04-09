"""Log-mel spectrogram feature extraction using torchaudio."""
import numpy as np
import torch
import torchaudio.transforms as T

N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160   # 10 ms at 16 kHz
WIN_LENGTH = 400   # 25 ms at 16 kHz
SAMPLE_RATE = 16000


def extract_log_mel(audio_array, sample_rate=SAMPLE_RATE,
                    n_mels=N_MELS, n_fft=N_FFT,
                    hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Compute log-mel spectrogram from a 1D float32 numpy array.
    Returns: numpy array of shape [T, n_mels]
    """
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # [1, samples]
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel = mel_transform(waveform)           # [1, n_mels, T]
    log_mel = torch.log(mel + 1e-9)        # log scale
    return log_mel.squeeze(0).T.numpy()    # [T, n_mels]


def normalize_features(features):
    """Per-sample mean-variance normalization across time."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-9
    return ((features - mean) / std).astype(np.float32)
