"""Audio loading, resampling, and normalization utilities."""
import numpy as np
import torch
import torchaudio.transforms as T

SAMPLE_RATE = 16000


def load_audio(audio_input, target_sr=SAMPLE_RATE):
    """
    Load audio from:
    - HuggingFace dict: {'array': np.ndarray, 'sampling_rate': int}
    - File path (str)
    Returns mono float32 numpy array at target_sr.
    """
    if isinstance(audio_input, dict):
        array = np.array(audio_input['array'], dtype=np.float32)
        sr = audio_input['sampling_rate']
        if array.ndim > 1:
            array = array.mean(axis=0)
        if sr != target_sr:
            array = _resample(array, sr, target_sr)
        return array
    else:
        import torchaudio
        waveform, sr = torchaudio.load(audio_input)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = T.Resample(sr, target_sr)(waveform)
        return waveform.squeeze().numpy().astype(np.float32)


def _resample(array, orig_sr, target_sr):
    waveform = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    return T.Resample(orig_sr, target_sr)(waveform).squeeze().numpy()


def normalize_audio(audio):
    """Peak-normalize audio to [-1, 1]."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)
