"""ResNet18-1D  ->  2-layer BiLSTM  ->  fully-connected regression head (MFI in [0, 1])."""
import torch
import torch.nn as nn

from bilstm import BiLSTMEncoder
from config import Config
from resnet18_1d import ResNet18_1D


class FatigueRegressor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = ResNet18_1D(cfg.in_channels, cfg.resnet_base_width)
        self.temporal = BiLSTMEncoder(self.backbone.out_dim, cfg.lstm_hidden,
                                      cfg.lstm_layers, cfg.lstm_dropout)
        self.head = nn.Sequential(
            nn.Linear(self.temporal.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),                      # regression target is bounded to [0, 1]
        )

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """(N, C, L) -> (N, D). Used live to embed each new window only once."""
        return self.backbone(windows)

    def forward_from_embeddings(self, emb_seq: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B,)"""
        return self.head(self.temporal(emb_seq)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, C, L) sequence of S consecutive windows -> MFI of the last window."""
        b, s, c, l = x.shape
        emb = self.backbone(x.reshape(b * s, c, l)).reshape(b, s, -1)
        return self.forward_from_embeddings(emb)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)
