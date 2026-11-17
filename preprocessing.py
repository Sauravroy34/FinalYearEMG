"""Signal processing shared by training (offline) and live inference.

Recording format (each .npy is shape (3, N), fs = 1 kHz):
    row 0 : time in seconds (0.000, 0.001, ...)
    row 1 : EMG   (`*_raw.npy` = 12-bit ADC counts 0..4095, `*.npy` = band-passed)
    row 2 : IMU   joint angle in degrees (used for rep counting)

Every recording is one subject performing the task until failure, so the label is
    MFI(t) = t / T_f        (T_f = recording length)  -> 0 fresh ... 1 exhausted.

All filtering here is *causal* (sosfilt with carried state), so the exact same code
runs sample-by-sample in real time. Offline we therefore re-filter the raw files
instead of using the pre-filtered `*.npy` (which was produced by an equivalent
causal filter, corr ~0.97).
"""
from __future__ import annotations

import glob
import os

import numpy as np
from scipy import signal

from config import Config


# --------------------------------------------------------------------------- #
#  Filtering
# --------------------------------------------------------------------------- #
class StreamingEMGFilter:
    """Causal band-pass + notch that keeps its state between calls."""

    def __init__(self, cfg: Config):
        bp = signal.butter(cfg.bp_order, [cfg.bp_low, cfg.bp_high], btype="band",
                           fs=cfg.fs, output="sos")
        b, a = signal.iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.fs)
        self.sos = np.vstack([bp, signal.tf2sos(b, a)])
        self.zi = None

    def reset(self):
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        if self.zi is None:  # start in steady state for the first sample (no DC step)
            self.zi = signal.sosfilt_zi(self.sos) * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


def filter_emg(raw: np.ndarray, cfg: Config) -> np.ndarray:
    return StreamingEMGFilter(cfg).process(raw)


def calibration_scale(emg_filt: np.ndarray, cfg: Config) -> float:
    """RMS of the first `calib_sec` seconds = the subject's fresh-muscle baseline."""
    n = int(cfg.calib_sec * cfg.fs)
    rms = float(np.sqrt(np.mean(emg_filt[:n] ** 2)))
    return max(rms, 1e-6)


def normalise_imu(angle: np.ndarray) -> np.ndarray:
    """Fixed scaling for the joint angle (degrees) so it is subject independent."""
    return (np.asarray(angle) - 90.0) / 45.0


# --------------------------------------------------------------------------- #
#  Rep counting from the IMU angle (streaming, hysteresis based)
# --------------------------------------------------------------------------- #
class RepCounter:
    """Counts one rep per angle peak whose prominence exceeds `prominence` degrees.

    Works sample-by-sample, so it is usable live. A light EMA removes IMU jitter.
    """

    def __init__(self, prominence: float = 30.0, fs: int = 1000, tau_s: float = 0.05):
        self.p = prominence
        self.alpha = 1.0 - np.exp(-1.0 / (tau_s * fs))
        self.reset()

    def reset(self):
        self.reps = 0
        self.state = None
        self.y = None
        self.lo = self.hi = None
        self.rep_sample_idx: list[int] = []
        self.n = 0

    def update(self, angles: np.ndarray) -> int:
        for x in np.atleast_1d(angles):
            self.y = x if self.y is None else self.y + self.alpha * (x - self.y)
            y = self.y
            if self.state is None:
                self.lo = y if self.lo is None else min(self.lo, y)
                self.hi = y if self.hi is None else max(self.hi, y)
                if y > self.lo + self.p:
                    self.state, self.hi = "rising", y
                elif y < self.hi - self.p:
                    self.state, self.lo = "falling", y
            elif self.state == "rising":
                self.hi = max(self.hi, y)
                if y < self.hi - self.p:          # peak confirmed -> one rep
                    self.state, self.lo = "falling", y
                    self.reps += 1
                    self.rep_sample_idx.append(self.n)
            else:  # falling
                self.lo = min(self.lo, y)
                if y > self.lo + self.p:
                    self.state, self.hi = "rising", y
            self.n += 1
        return self.reps


# --------------------------------------------------------------------------- #
#  Loading / windowing
# --------------------------------------------------------------------------- #
def list_subjects(rec_dir: str) -> list[str]:
    """One subject per recording; returns base paths without `_raw.npy`."""
    raws = sorted(glob.glob(os.path.join(rec_dir, "*_raw.npy")))
    if not raws:
        raise FileNotFoundError(f"No *_raw.npy files in {rec_dir}")
    return [r[: -len("_raw.npy")] for r in raws]


def trim_dropout(raw: np.ndarray, min_run: int = 50) -> int:
    """Return the length after removing a trailing flat run (sensor disconnect)."""
    n = len(raw)
    i = n - 1
    while i > 0 and raw[i] == raw[n - 1]:
        i -= 1
    run = n - 1 - i
    return i + 1 if run >= min_run else n


def load_recording(base: str):
    arr = np.load(base + "_raw.npy")
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"{base}: expected shape (3, N), got {arr.shape}")
    t, emg_raw, imu = arr
    n = trim_dropout(emg_raw)
    return t[:n], emg_raw[:n], imu[:n]


def window_starts(n: int, cfg: Config) -> np.ndarray:
    return np.arange(0, n - cfg.win_len + 1, cfg.hop)


def prepare_subject(base: str, cfg: Config) -> dict:
    """Filter, normalise, window and label one recording."""
    t, emg_raw, imu = load_recording(base)
    n = len(emg_raw)

    emg = filter_emg(emg_raw, cfg)
    scale = calibration_scale(emg, cfg)
    chans = [emg / scale]
    if cfg.use_imu_input:
        chans.append(normalise_imu(imu))
    x = np.stack(chans).astype(np.float32)              # (C, N)

    starts = window_starts(n, cfg)
    idx = starts[:, None] + np.arange(cfg.win_len)[None, :]
    windows = x[:, idx].transpose(1, 0, 2).copy()       # (W, C, L)
    end = starts + cfg.win_len                          # sample after window end
    labels = (end / n).astype(np.float32)               # MFI = t / T_f

    rc = RepCounter(cfg.rep_prominence_deg, cfg.fs)
    rc.update(imu)
    rep_idx = np.array(rc.rep_sample_idx, dtype=int)
    reps_at_window = np.searchsorted(rep_idx, end, side="left")

    return dict(
        name=os.path.basename(base),
        windows=windows,
        labels=labels,
        t_end=end / cfg.fs,
        reps=reps_at_window,
        total_reps=rc.reps,
        duration_s=n / cfg.fs,
        calib_scale=scale,
    )
