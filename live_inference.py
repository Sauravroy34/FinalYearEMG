"""Real-time muscle fatigue index (0 = fresh, 1 = exhausted) + rep counting.

A new score is produced every hop (250 ms) from the last `seq_len` 500 ms windows.
The first `calib_sec` seconds (default 5 s) are a calibration phase: the subject
should start the exercise fresh so the EMG baseline can be measured.

Sources
  replay : stream a recorded *_raw.npy at real-time speed (for testing)
      python live_inference.py --model runs/<run>/final_model.pt --replay recordings/emg_imu_20261117_064452_raw.npy
  serial : read lines "emg,imu" (or "t,emg,imu") at 1 kHz from a microcontroller
      python live_inference.py --model runs/<run>/final_model.pt --serial COM5 --baud 115200
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import numpy as np
import torch

from config import Config
from fatigue_model import FatigueRegressor
from preprocessing import RepCounter, StreamingEMGFilter, normalise_imu


class LiveFatigueEstimator:
    def __init__(self, model: FatigueRegressor, cfg: Config, device="cpu", ema: float = 0.3):
        self.model = model.eval().to(device)
        self.cfg = cfg
        self.device = device
        self.filter = StreamingEMGFilter(cfg)
        self.reps = RepCounter(cfg.rep_prominence_deg, cfg.fs)
        self.ctx = (cfg.seq_len - 1) * cfg.hop + cfg.win_len       # samples the LSTM needs
        self.calib_n = int(cfg.calib_sec * cfg.fs)
        self.calib_buf = []
        self.scale = None
        self.emg_buf = collections.deque(maxlen=max(self.ctx, cfg.win_len))
        self.imu_buf = collections.deque(maxlen=max(self.ctx, cfg.win_len))
        self.emb = collections.deque(maxlen=cfg.seq_len)
        self.n = 0
        self.ema_a = ema
        self.smoothed = None

    def _window(self, end_offset: int) -> np.ndarray:
        """Window ending `end_offset` samples before the newest sample -> (C, L)."""
        e = np.asarray(self.emg_buf, dtype=np.float32)
        stop = len(e) - end_offset
        chans = [e[stop - self.cfg.win_len: stop] / self.scale]
        if self.cfg.use_imu_input:
            im = np.asarray(self.imu_buf, dtype=np.float32)
            chans.append(normalise_imu(im[stop - self.cfg.win_len: stop]))
        return np.stack(chans).astype(np.float32)

    @torch.no_grad()
    def _embed(self, windows: list[np.ndarray]):
        x = torch.from_numpy(np.stack(windows)).to(self.device)
        return list(self.model.embed(x).cpu())

    def push(self, emg_raw, imu) -> list[dict]:
        """Feed new samples; returns one result dict per completed hop."""
        cfg = self.cfg
        emg_f = self.filter.process(emg_raw)
        imu = np.asarray(imu, dtype=np.float64)
        out = []
        for i in range(len(emg_f)):
            self.emg_buf.append(emg_f[i])
            self.imu_buf.append(imu[i])
            self.reps.update(imu[i:i + 1])
            self.n += 1

            if self.scale is None:
                self.calib_buf.append(emg_f[i])
                if self.n >= self.calib_n:
                    self.scale = max(float(np.sqrt(np.mean(np.square(self.calib_buf)))), 1e-6)
                    self.calib_buf = None
                continue

            if self.n < cfg.win_len or (self.n - cfg.win_len) % cfg.hop != 0:
                continue
            if not self.emb:  # just calibrated: back-fill history from the buffer
                k = min(cfg.seq_len, (len(self.emg_buf) - cfg.win_len) // cfg.hop + 1)
                self.emb.extend(self._embed([self._window(j * cfg.hop) for j in reversed(range(k))]))
            else:
                self.emb.extend(self._embed([self._window(0)]))
            if len(self.emb) < cfg.seq_len:
                continue

            seq = torch.stack(list(self.emb)).unsqueeze(0).to(self.device)
            mfi = float(self.model.forward_from_embeddings(seq).item())
            self.smoothed = mfi if self.smoothed is None else self.smoothed + self.ema_a * (mfi - self.smoothed)
            out.append(dict(t=self.n / cfg.fs, mfi=mfi, mfi_smooth=self.smoothed, reps=self.reps.reps))
        return out

    @property
    def calibrating(self):
        return self.scale is None


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    cfg = Config(**ck["cfg"])
    model = FatigueRegressor(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg


def bar(v, width=30):
    k = int(round(v * width))
    return "[" + "#" * k + "-" * (width - k) + "]"


def show(r):
    state = "FRESH" if r["mfi_smooth"] < 0.4 else "MODERATE" if r["mfi_smooth"] < 0.75 else "FATIGUED"
    sys.stdout.write(f"\rt={r['t']:6.2f}s  MFI {r['mfi_smooth']:.2f} {bar(r['mfi_smooth'])} "
                     f"{state:<9} reps {r['reps']:3d}   ")
    sys.stdout.flush()


def run_replay(est, path, speed, chunk=50):
    arr = np.load(path)
    emg, imu = arr[1], arr[2]
    fs = est.cfg.fs
    t0 = time.perf_counter()
    results = []
    for s in range(0, len(emg), chunk):
        if speed > 0:
            wait = t0 + s / (fs * speed) - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        for r in est.push(emg[s:s + chunk], imu[s:s + chunk]):
            results.append(r)
            show(r)
    print()
    return results


def run_serial(est, port, baud):
    import serial  # pip install pyserial
    ser = serial.Serial(port, baud, timeout=1)
    print(f"reading {port} @ {baud}; calibrating for {est.cfg.calib_sec:.0f} s - keep the muscle fresh")
    emg, imu = [], []
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                vals = [float(v) for v in line.replace(" ", ",").split(",") if v]
            except ValueError:
                continue
            if len(vals) < 2:
                continue
            emg.append(vals[-2])
            imu.append(vals[-1])
            if len(emg) >= 25:
                for r in est.push(np.array(emg), np.array(imu)):
                    show(r)
                emg, imu = [], []
    except KeyboardInterrupt:
        print()
    finally:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", help="path to a *_raw.npy recording")
    g.add_argument("--serial", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0 = as fast as possible)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg = load_model(args.model, args.device)
    est = LiveFatigueEstimator(model, cfg, args.device)
    if args.replay:
        res = run_replay(est, args.replay, args.speed)
        if res:
            print(f"{len(res)} predictions, final MFI {res[-1]['mfi_smooth']:.2f}, reps {res[-1]['reps']}")
    else:
        run_serial(est, args.serial, args.baud)


if __name__ == "__main__":
    main()
