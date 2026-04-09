"""
CNN-LSTM model for Arabic ASR with CTC loss.

Architecture
------------
Input  : [B, T, n_mels]
Reshape: [B, 1, n_mels, T]   (treat mel-freq as height, time as width)
CNN Block 1: Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(freq=2, time=1)
CNN Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(freq=2, time=1)
Reshape: [B, T, 64 × (n_mels//4)]
BiLSTM layer 1: hidden=hidden_size
BiLSTM layer 2: hidden=hidden_size
Dropout
Linear: hidden_size×2 → vocab_size
log_softmax → CTC loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNNBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU + MaxPool (frequency-only pooling)."""
    def __init__(self, in_ch, out_ch, kernel=3, pool=(2, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      stride=1, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool),
        )

    def forward(self, x):
        return self.net(x)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM-DNN for Arabic ASR.

    Parameters
    ----------
    n_mels        : number of mel filter banks (default 80)
    vocab_size    : output vocabulary size (including CTC blank at index 0)
    hidden_size   : LSTM hidden units per direction
    num_lstm_layers: stacked BiLSTM layers
    dropout       : dropout between LSTM layers and before projection
    """
    def __init__(self, n_mels=80, vocab_size=50,
                 hidden_size=256, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        self.n_mels = n_mels

        # ── CNN front-end ─────────────────────────────────────────────
        # Block 1: pool freq × 2 only  (time preserved)
        # Block 2: pool freq × 2 AND time × 2  (halves sequence length)
        self.cnn = nn.Sequential(
            _CNNBlock(1, 32,  pool=(2, 1)),  # freq: n_mels   → n_mels//2
            _CNNBlock(32, 64, pool=(2, 2)),  # freq: n_mels//2 → n_mels//4, time ÷ 2
        )

        cnn_feat_dim = 64 * (n_mels // 4)   # e.g. 64*20 = 1280 for n_mels=80

        # ── BiLSTM temporal encoder ───────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Output projection ─────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif isinstance(p, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(p.weight)

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x, input_lengths=None):
        """
        Parameters
        ----------
        x             : [B, T, n_mels]  padded log-mel features
        input_lengths : [B]             actual frame counts before padding

        Returns
        -------
        log_probs      : [T, B, vocab_size]  log-softmax scores
        output_lengths : [B]                 (same as input_lengths; time not pooled)
        """
        B, T, _ = x.shape

        # Reshape for Conv2d: [B, 1, n_mels, T]
        x = x.unsqueeze(1)              # [B, 1, T, n_mels]
        x = x.permute(0, 1, 3, 2)      # [B, 1, n_mels, T]

        x = self.cnn(x)                 # [B, 64, n_mels//4, T]

        _, C, n_freq, T_out = x.shape
        x = x.permute(0, 3, 1, 2)           # [B, T, C, n_freq]
        x = x.reshape(B, T_out, C * n_freq) # [B, T, C*n_freq]

        x, _ = self.lstm(x)             # [B, T, hidden*2]
        x = self.dropout(x)
        x = self.fc(x)                  # [B, T, vocab_size]

        # CTC convention: [T, B, vocab_size]
        log_probs = F.log_softmax(x.permute(1, 0, 2), dim=-1)

        # CNN block 2 pools time by 2 (floor division matches MaxPool2d behaviour)
        if input_lengths is not None:
            output_lengths = (input_lengths // 2).clamp(min=1, max=T_out)
        else:
            output_lengths = torch.full((B,), T_out, dtype=torch.long, device=x.device)

        return log_probs, output_lengths

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
