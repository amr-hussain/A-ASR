"""
Gradio demo for the CNN-LSTM Arabic ASR model.

Usage:
    /mnt/D/pip_envs/asr/bin/python src/demo/app.py
"""
import os
import sys
import numpy as np
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.audio_utils import normalize_audio
from src.preprocessing.feature_extraction import extract_log_mel, normalize_features
from src.models.cnn_lstm import CNNLSTM
from src.training.evaluate import ctc_greedy_decode

ROOT   = os.path.join(os.path.dirname(__file__), '..', '..')
CKPT   = os.path.join(ROOT, 'checkpoints', 'best_model.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(path=CKPT):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run  python src/training/train.py  first."
        )
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt['config']
    model = CNNLSTM(
        n_mels          = cfg['n_mels'],
        vocab_size      = cfg['vocab_size'],
        hidden_size     = cfg['hidden_size'],
        num_lstm_layers = cfg['num_lstm_layers'],
        dropout         = cfg.get('dropout', 0.3),
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    idx_to_char = {int(k): v for k, v in ckpt['idx_to_char'].items()}
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch','?')}  device={DEVICE}")
    return model, idx_to_char


model, idx_to_char = load_model()


def transcribe(audio_input):
    """
    Gradio callback: receives (sample_rate, np.ndarray) and returns Arabic text.
    """
    if audio_input is None:
        return "No audio provided."

    sr, audio = audio_input
    audio = np.array(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # int16 → float32
    if np.abs(audio).max() > 1.5:
        audio = audio / 32768.0

    # Resample to 16 kHz if needed
    if sr != 16000:
        import torchaudio.transforms as T
        wav = torch.tensor(audio).unsqueeze(0)
        audio = T.Resample(sr, 16000)(wav).squeeze().numpy()

    audio = normalize_audio(audio)

    features = extract_log_mel(audio, sample_rate=16000)
    features = normalize_features(features)
    feat_t   = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        log_probs, lengths = model(feat_t)
        text = ctc_greedy_decode(log_probs, lengths, idx_to_char)[0]

    return text.strip() if text.strip() else "(Could not recognise speech)"


description = """
## Arabic Speech Recognition — CNN-LSTM Demo

Record or upload an Arabic audio clip and click **Transcribe**.

**Model architecture**
- Feature: 80-dim log-mel spectrogram (10 ms hop)
- CNN front-end: 2× Conv2d blocks with frequency+time pooling
- Temporal encoder: 2-layer Bidirectional LSTM (256 units/direction)
- Output: CTC greedy decoding → character sequence

**Training**: 5 000 utterances from Mozilla Common Voice 18 Arabic, 15 epochs
"""

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy",
                    label="Arabic Audio (record or upload)"),
    outputs=gr.Textbox(label="Arabic Transcription", lines=4),
    title="Arabic ASR — CNN-LSTM",
    description=description,
    allow_flagging="never",
)

if __name__ == '__main__':
    iface.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True,
        inbrowser=False,
    )
