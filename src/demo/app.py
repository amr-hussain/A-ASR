import os
import sys
import numpy as np
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.audio_utils import normalize_audio
from src.preprocessing.feature_extraction import extract_log_mel, normalize_features
from src.models.cnn_lstm import CNNLSTM
from src.models.whisper_baseline import WhisperBaseline
from src.training.evaluate import ctc_greedy_decode

ROOT   = os.path.join(os.path.dirname(__file__), '..', '..')
CKPT   = os.path.join(ROOT, 'checkpoints', 'best_model.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Load Whisper-tiny ─────────────────────────────────────────────────────
whisper_model = WhisperBaseline(model_size='tiny', device=DEVICE)


# ── Load CNN-LSTM ─────────────────────────────────────────────────────────
def load_cnn_lstm(path=CKPT):
    if not os.path.exists(path):
        return None, None
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
    epoch = ckpt.get('epoch', '?')
    print(f"CNN-LSTM loaded  epoch={epoch}  hidden={cfg['hidden_size']}  device={DEVICE}")
    return model, idx_to_char


cnn_lstm_model, idx_to_char = load_cnn_lstm()


# ── Shared audio preprocessing ────────────────────────────────────────────
def prepare_audio(audio_input):
    """Convert Gradio audio tuple → float32 numpy array at 16 kHz."""
    if audio_input is None:
        return None, None
    sr, audio = audio_input
    audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.abs(audio).max() > 1.5:
        audio = audio / 32768.0
    if sr != 16000:
        import torchaudio.transforms as T
        wav = torch.tensor(audio).unsqueeze(0)
        audio = T.Resample(sr, 16000)(wav).squeeze().numpy()
    return audio, 16000


# ── Inference functions ───────────────────────────────────────────────────
def run_whisper(audio_input):
    audio, sr = prepare_audio(audio_input)
    if audio is None:
        return "No audio provided."
    try:
        return whisper_model.transcribe(audio, sr)
    except Exception as e:
        return f"Error: {e}"


def run_cnn_lstm(audio_input):
    if cnn_lstm_model is None:
        return "No checkpoint found. Run training first."
    audio, sr = prepare_audio(audio_input)
    if audio is None:
        return "No audio provided."
    try:
        audio = normalize_audio(audio)
        features = extract_log_mel(audio, sample_rate=sr)
        features = normalize_features(features)
        feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            log_probs, lengths = cnn_lstm_model(feat_t)
            text = ctc_greedy_decode(log_probs, lengths, idx_to_char)[0]
        return text.strip() if text.strip() else "(Could not recognise speech)"
    except Exception as e:
        return f"Error: {e}"


def transcribe_both(audio_input):
    return run_whisper(audio_input), run_cnn_lstm(audio_input)


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Arabic ASR") as demo:

    gr.Markdown("""
# Arabic ASR under supervision of Prof. Ahmed B. Zaki

## Whisper-tiny (pretrained as base model) vs CNN-LSTM approach (trained from scratch on 28k samples in 53 hours)
### Made by Amr Hussain Elsayed
""")

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Arabic Audio Input",
        )

    btn = gr.Button("Transcribe Both", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Whisper-tiny (OpenAI pretrained)")
            gr.Markdown("*Trained on 680,000 hours of multilingual audio. Accurate.*")
            whisper_out = gr.Textbox(
                label="Whisper Transcription",
                lines=4,
                rtl=True,
                show_copy_button=True,
            )

        with gr.Column():
            gr.Markdown("### CNN-LSTM (Custom implementation)")
            gr.Markdown("*Trained from scratch on 28k Arabic utterances with CTC loss.*")
            cnn_lstm_out = gr.Textbox(
                label="CNN-LSTM Transcription",
                lines=4,
                rtl=True,
                show_copy_button=True,
                interactive=True,
            )

    btn.click(fn=transcribe_both, inputs=audio_in,
              outputs=[whisper_out, cnn_lstm_out])

    gr.Markdown("""
---
**CNN-LSTM Architecture**

| Layer | Detail |
|-------|--------|
| Input | 80-dim log-mel spectrogram, 10 ms hop |
| CNN Block 1 | Conv2d(1 -> 32) + batch norm + ReLU + MaxPool(freq÷2) |
| CNN Block 2 | Conv2d(32 -> 64) + batch norm + ReLU + MaxPool(freq÷2, time÷2) |
| Bidirectional LSTM × 2 | 512 units/direction, bidirectional |
| Output | Linear → 45 Arabic chars, CTC loss |
| Dataset | Mozilla Common Voice 18 Arabic (28,000 samples) |
| Metric | Word Error Rate (WER) using jiwer library |
| Date | 7 April 2026 |
""")


if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True,
        inbrowser=False,
        show_api=False,
    )
