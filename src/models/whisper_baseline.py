"""
Whisper-tiny baseline for Arabic ASR.

Wraps OpenAI Whisper with Arabic-forced language decoding.
No training required — uses the pretrained multilingual checkpoint.

Usage:
    model = WhisperBaseline()          # downloads ~72 MB on first run
    text  = model.transcribe(array, sr)
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class WhisperBaseline:
    """
    Whisper-tiny wrapper for Arabic speech recognition.

    Parameters
    ----------
    model_size : str   one of 'tiny', 'base', 'small' (default 'tiny')
    device     : str   'cuda' | 'cpu' | None  (auto-detects)
    """

    def __init__(self, model_size: str = 'tiny', device: str = None):
        import whisper

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Whisper-{model_size} on {self.device} ...")
        self.model = whisper.load_model(model_size, device=self.device)
        self.fp16 = (self.device == 'cuda')
        print(f"Whisper-{model_size} ready  "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy audio array to Arabic text.

        Parameters
        ----------
        audio_array : np.ndarray  mono float32, any sample rate
        sample_rate : int         sample rate of audio_array

        Returns
        -------
        str  Arabic transcription
        """
        import whisper

        audio = np.array(audio_array, dtype=np.float32)

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # int16 -> float32
        if np.abs(audio).max() > 1.5:
            audio = audio / 32768.0

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            import torchaudio.transforms as T
            wav = torch.tensor(audio).unsqueeze(0)
            audio = T.Resample(sample_rate, 16000)(wav).squeeze().numpy()

        # Pad/trim to Whisper's 30-second context window
        audio = whisper.pad_or_trim(audio)

        # Log-mel spectrogram (80 bins, Whisper convention)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Decode with forced Arabic
        options = whisper.DecodingOptions(
            language='ar',
            fp16=self.fp16,
            without_timestamps=True,
        )
        result = whisper.decode(self.model, mel, options)
        return result.text.strip()

    def transcribe_file(self, path: str) -> str:
        """Transcribe directly from a file path."""
        result = self.model.transcribe(path, language='ar', fp16=self.fp16)
        return result['text'].strip()
