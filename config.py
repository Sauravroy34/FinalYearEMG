"""Central configuration for the muscle-fatigue (MFI) regression pipeline."""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ---------------- data ----------------
    recordings_dir: str = "recordings"
    fs: int = 1000                      # Hz (row 0 of every .npy is time in s, dt = 1 ms)

    # Windowing: 500 ms windows, 50 % overlap  -> one new prediction every 250 ms
    win_ms: int = 500
    overlap: float = 0.5

    # Number of consecutive windows the BiLSTM sees (context = (seq_len-1)*hop + win)
    # 8 windows -> 2.25 s of history, roughly one curl repetition.
    seq_len: int = 8

    # EMG filter (applied causally to the *raw* ADC signal so training == live).
    # Matches the filter used in the processed recordings / pre-training slides.
    bp_low: float = 74.5
    bp_high: float = 149.5
    bp_order: int = 2                   # scipy band order 2 -> 4th-order band-pass
    notch_hz: float = 50.0
    notch_q: float = 30.0

    # Per-subject baseline normalisation: EMG is divided by the RMS of the first
    # `calib_sec` seconds (live: a short "fresh muscle" calibration phase).
    calib_sec: float = 5.0

    # IMU (row 2, joint angle in degrees) is used for rep counting.
    # Set True to also feed it to the ResNet as a 2nd input channel.
    use_imu_input: bool = False
    rep_prominence_deg: float = 25.0

    # ---------------- model ----------------
    resnet_base_width: int = 64         # 64 = standard ResNet-18 widths (64-128-256-512)
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    head_dropout: float = 0.3

    # ---------------- training ----------------
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12                  # early stopping on inner-validation RMSE
    seed: int = 42
    num_workers: int = 0
    augment: bool = True
    out_dir: str = "runs"

    @property
    def win_len(self) -> int:
        return int(self.fs * self.win_ms / 1000)

    @property
    def hop(self) -> int:
        return int(self.win_len * (1 - self.overlap))

    @property
    def in_channels(self) -> int:
        return 2 if self.use_imu_input else 1

    def to_dict(self):
        return asdict(self)
