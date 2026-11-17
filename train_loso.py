"""Leave-One-Subject-Out training / evaluation of ResNet18-1D + BiLSTM MFI regressor.

Every recording in `recordings/` is one subject. For each fold:
    test  = 1 subject
    val   = 1 of the remaining subjects (early stopping only)
    train = the rest
Loss = RMSE, optimiser = Adam.

Usage:
    python train_loso.py                       # full LOSO
    python train_loso.py --epochs 3 --quick    # smoke test
    python train_loso.py --final               # LOSO, then train on ALL subjects for live use
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import FatigueSequenceDataset
from fatigue_model import FatigueRegressor, RMSELoss
from preprocessing import list_subjects, prepare_subject


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan")
    return dict(rmse=rmse, mae=mae, r2=r2, pearson_r=r)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, ws = [], [], []
    for x, y, _, w in loader:
        ps.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        ws.append(w.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)


def train_one(cfg, train_subj, val_subj, device, epochs, log_prefix=""):
    """Train a fresh model. If val_subj is None, train for exactly `epochs`."""
    model = FatigueRegressor(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = RMSELoss()

    tr_loader = DataLoader(FatigueSequenceDataset(train_subj, cfg.seq_len, cfg.augment),
                           batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                           num_workers=cfg.num_workers)
    va_loader = None
    if val_subj:
        va_loader = DataLoader(FatigueSequenceDataset(val_subj, cfg.seq_len, False),
                               batch_size=cfg.batch_size * 2, shuffle=False)

    best_rmse, best_state, best_epoch, bad = float("inf"), None, epochs, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y, _, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        tr_rmse = tot / max(n, 1)

        if va_loader is None:
            print(f"{log_prefix} ep {ep:3d}  train RMSE {tr_rmse:.4f}")
            continue
        yv, pv, _ = predict(model, va_loader, device)
        va_rmse = metrics(yv, pv)["rmse"]
        sched.step(va_rmse)
        flag = ""
        if va_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, bad = va_rmse, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        print(f"{log_prefix} ep {ep:3d}  train RMSE {tr_rmse:.4f}  val RMSE {va_rmse:.4f}{flag}")
        if bad >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-imu", action="store_true", help="feed IMU angle as 2nd input channel")
    ap.add_argument("--quick", action="store_true", help="only run the first 2 folds")
    ap.add_argument("--final", action="store_true", help="after LOSO, train on all subjects")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.use_imu:
        cfg.use_imu_input = True
    out_dir = args.out or os.path.join(cfg.out_dir, time.strftime("loso_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(args.device)

    here = os.path.dirname(os.path.abspath(__file__))
    rec_dir = cfg.recordings_dir if os.path.isabs(cfg.recordings_dir) else os.path.join(here, cfg.recordings_dir)
    subjects = [prepare_subject(b, cfg) for b in list_subjects(rec_dir)]
    print(f"device={device}  window={cfg.win_len} samples  hop={cfg.hop}  seq_len={cfg.seq_len}")
    for s in subjects:
        print(f"  {s['name']}: {s['duration_s']:.1f}s, {len(s['windows'])} windows, "
              f"{s['total_reps']} reps, calib RMS {s['calib_scale']:.1f}")

    folds = range(len(subjects)) if not args.quick else range(min(2, len(subjects)))
    fold_results, best_epochs = [], []
    pred_rows = []
    for k in folds:
        set_seed(cfg.seed + k)
        test = subjects[k]
        others = [s for i, s in enumerate(subjects) if i != k]
        val = [others[k % len(others)]]
        train = [s for s in others if s is not val[0]]
        prefix = f"[fold {k + 1}/{len(subjects)} test={test['name']}]"
        print(f"\n{prefix} train={[s['name'] for s in train]} val={val[0]['name']}")

        model, best_ep = train_one(cfg, train, val, device, cfg.epochs, prefix)
        best_epochs.append(best_ep)

        te_loader = DataLoader(FatigueSequenceDataset([test], cfg.seq_len, False),
                               batch_size=cfg.batch_size * 2, shuffle=False)
        yt, pt, wt = predict(model, te_loader, device)
        m = metrics(yt, pt)
        m.update(subject=test["name"], best_epoch=best_ep, n=len(yt))
        fold_results.append(m)
        print(f"{prefix} TEST  RMSE {m['rmse']:.4f}  MAE {m['mae']:.4f}  R2 {m['r2']:.3f}  r {m['pearson_r']:.3f}")

        for y, p, w in zip(yt, pt, wt):
            pred_rows.append((test["name"], int(w), float(test["t_end"][w]), int(test["reps"][w]), float(y), float(p)))
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.to_dict()},
                   os.path.join(out_dir, f"fold{k + 1}_{test['name']}.pt"))

    # ---- summary ----
    keys = ["rmse", "mae", "r2", "pearson_r"]
    summary = {k: dict(mean=float(np.nanmean([r[k] for r in fold_results])),
                       std=float(np.nanstd([r[k] for r in fold_results]))) for k in keys}
    print("\n=========== LOSO summary ===========")
    for r in fold_results:
        print(f"  {r['subject']:<28} RMSE {r['rmse']:.4f}  MAE {r['mae']:.4f}  R2 {r['r2']:.3f}  r {r['pearson_r']:.3f}")
    for k in keys:
        print(f"  {k:<10} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    with open(os.path.join(out_dir, "loso_results.json"), "w") as f:
        json.dump(dict(folds=fold_results, summary=summary, cfg=cfg.to_dict()), f, indent=2)
    with open(os.path.join(out_dir, "loso_predictions.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["subject", "window", "t_end_s", "reps", "mfi_true", "mfi_pred"])
        wr.writerows(pred_rows)
    plot_predictions(pred_rows, out_dir)

    # ---- final model on all subjects (for live inference) ----
    if args.final:
        n_ep = int(np.median(best_epochs)) if best_epochs else cfg.epochs
        print(f"\nTraining final model on all {len(subjects)} subjects for {n_ep} epochs")
        set_seed(cfg.seed)
        model, _ = train_one(cfg, subjects, None, device, n_ep, "[final]")
        path = os.path.join(out_dir, "final_model.pt")
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.to_dict()}, path)
        print(f"saved {path}")
    print(f"\nresults in {out_dir}")


def plot_predictions(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        rr = [r for r in rows if r[0] == name]
        t = [r[2] for r in rr]
        ax.plot(t, [r[4] for r in rr], label="true MFI", lw=2)
        ax.plot(t, [r[5] for r in rr], label="predicted", lw=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("MFI")
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loso_predictions.png"), dpi=120)


if __name__ == "__main__":
    main()
