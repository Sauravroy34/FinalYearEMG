"""2-layer bidirectional LSTM that summarises a sequence of window embeddings."""
import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """Input : (B, S, in_dim) - one ResNet embedding per 500 ms window
    Output: (B, 2 * hidden) - final forward state (has read the whole sequence
            left->right) concatenated with final backward state (read right->left).

    Real-time note: the LSTM only ever sees a sliding buffer of *past* windows,
    so being bidirectional inside that buffer is still causal overall.
    """

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)           # h_n: (num_layers * 2, B, hidden)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)   # top layer fwd || bwd
