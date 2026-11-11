"""Lightweight 1D-CNN models on the (filtered) raw EMG / IMU time series.

  EMG stream  (8 x T)  : 3 conv blocks (32-64-64 ch, k=7/5/5, 2x max-pool) -> [global avg ‖ global max] = 128
  IMU stream  (6 x T') : 2 conv blocks (16-32 ch, k=3)                       -> [avg ‖ max]            = 64
  Fusion               : concat -> Linear 64 -> ReLU -> Dropout
  Heads                : pose  -> n_pose logits
                         effort -> one Low/High logit PER ACTIVE POSE (pose-conditioned). Training uses only the
                                   logit of the true pose (Rest windows excluded); at run time the logit of the
                                   predicted pose is used (= hierarchical Pose -> Effort).

No hand-crafted features: the network sees the filtered sample sequence. Per-channel scaling statistics are
computed from the training windows only.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def _block(cin, cout, k, pool):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(2, ceil_mode=True))
    return nn.Sequential(*layers)


class EmgEncoder(nn.Module):
    out_dim = 128

    def __init__(self, n_ch=8, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(_block(n_ch, 32, 7, True), nn.Dropout(drop / 2), _block(32, 64, 5, True),
                                 nn.Dropout(drop / 2), _block(64, 64, 5, False))

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class ImuEncoder(nn.Module):
    out_dim = 64

    def __init__(self, n_ch=6, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(_block(n_ch, 16, 3, False), _block(16, 32, 3, False))

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class StreamNet(nn.Module):
    def __init__(self, streams: tuple, n_pose: int, n_effort: int, drop=0.3):
        super().__init__()
        self.streams = tuple(streams)
        self.emg = EmgEncoder() if "emg" in streams else None
        self.imu = ImuEncoder() if "imu" in streams else None
        d = (EmgEncoder.out_dim if self.emg else 0) + (ImuEncoder.out_dim if self.imu else 0)
        self.fuse = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.pose_head = nn.Linear(64, n_pose) if n_pose else None
        self.effort_head = nn.Linear(64, n_effort) if n_effort else None

    def forward(self, emg=None, imu=None):
        z = []
        if self.emg is not None:
            z.append(self.emg(emg))
        if self.imu is not None:
            z.append(self.imu(imu))
        h = self.fuse(torch.cat(z, 1))
        return (self.pose_head(h) if self.pose_head is not None else None,
                self.effort_head(h) if self.effort_head is not None else None)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ============================================================================================ normalization
def fit_norm(emg: np.ndarray, imu: np.ndarray) -> dict:
    """Per-channel statistics from TRAINING windows only. EMG is already high-passed (zero mean),
    so it is only scaled; IMU is z-scored."""
    return {"emg_scale": (np.sqrt((emg.astype(np.float64) ** 2).mean((0, 2))) + 1e-6).tolist(),
            "imu_mean": imu.mean((0, 2)).tolist(), "imu_std": (imu.std((0, 2)) + 1e-6).tolist()}


def apply_norm(norm: dict, emg: np.ndarray | None, imu: np.ndarray | None):
    e = i = None
    if emg is not None:
        e = (emg / np.asarray(norm["emg_scale"], np.float32)[None, :, None]).astype(np.float32)
    if imu is not None:
        i = ((imu - np.asarray(norm["imu_mean"], np.float32)[None, :, None]) /
             np.asarray(norm["imu_std"], np.float32)[None, :, None]).astype(np.float32)
    return e, i


# ============================================================================================ training
class TrainCfg:
    max_epochs = 100
    patience = 12
    batch = 256
    lr = 1e-3
    weight_decay = 1e-2
    effort_weight = 1.0
    gain_jitter = 0.10       # per-channel EMG gain augmentation (electrode contact variation)
    noise = 0.05


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_net(streams, emg, imu, pose_y, effort_pose, effort_y, n_pose, n_effort, val_mask, seed=C.SEED,
              cfg: type = TrainCfg, log=None):
    """emg/imu: normalized [N, ch, T]; pose_y: int [N] or None; effort_pose: int [N] (-1 = no effort label);
    effort_y: float [N] (1 = High). val_mask: windows of the inner validation trials (early stopping)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    net = StreamNet(streams, n_pose if pose_y is not None else 0, n_effort if effort_y is not None else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    def T(a, dt=torch.float32):
        return None if a is None else torch.as_tensor(a, dtype=dt, device=dev)

    tr, va = ~val_mask, val_mask
    E, I = T(emg), T(imu)
    PY = T(pose_y, torch.long) if pose_y is not None else None
    EP = T(effort_pose, torch.long) if effort_y is not None else None
    EY = T(effort_y) if effort_y is not None else None
    if PY is not None:
        counts = np.bincount(pose_y[tr], minlength=n_pose).astype(float)
        w_pose = T(np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_pose, 0.0))
    tr_idx = np.where(tr)[0]
    if EP is not None and PY is None:      # effort-only network: train only on windows with an effort label
        tr_idx = tr_idx[effort_pose[tr_idx] >= 0]
    va_idx = np.where(va)[0]
    if EP is not None and PY is None:
        va_idx = va_idx[effort_pose[va_idx] >= 0]

    def loss_on(idx, train):
        e = E[idx] if E is not None else None
        i = I[idx] if I is not None else None
        if train:
            if e is not None:
                e = e * (1 + cfg.gain_jitter * torch.randn(e.shape[0], e.shape[1], 1, device=dev)) \
                    + cfg.noise * torch.randn_like(e)
            if i is not None:
                i = i + cfg.noise * torch.randn_like(i)
        pl, el = net(e, i)
        loss = torch.zeros((), device=dev)
        if PY is not None:
            loss = loss + F.cross_entropy(pl, PY[idx], weight=w_pose)
        if EP is not None:
            ep = EP[idx]
            m = ep >= 0
            if m.any():
                logit = el[m].gather(1, ep[m][:, None])[:, 0]
                loss = loss + cfg.effort_weight * F.binary_cross_entropy_with_logits(logit, EY[idx][m])
        return loss

    best, best_state, bad, ep_done = np.inf, None, 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for b in range(0, len(perm), cfg.batch):
            idx = torch.as_tensor(perm[b:b + cfg.batch], device=dev)
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_on(idx, True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(torch.as_tensor(va_idx[b:b + 2048], device=dev), False).item()
                                for b in range(0, len(va_idx), 2048)])) if len(va_idx) else 0.0
        sched.step(vl)
        ep_done = epoch + 1
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    if log:
        log(f"    {'+'.join(streams)} {'pose' if PY is not None else ''}{'+' if PY is not None and EP is not None else ''}"
            f"{'effort' if EP is not None else ''}: {ep_done} epochs, best val loss {best:.3f}, {n_params(net)} params")
    return net.cpu(), {"epochs": ep_done, "best_val_loss": best, "params": n_params(net)}


@torch.no_grad()
def predict_net(net, emg, imu, batch=4096):
    net.eval()
    P, E = [], []
    n = len(emg) if emg is not None else len(imu)
    for b in range(0, n, batch):
        e = torch.as_tensor(emg[b:b + batch]) if emg is not None and net.emg is not None else None
        i = torch.as_tensor(imu[b:b + batch]) if imu is not None and net.imu is not None else None
        pl, el = net(e, i)
        if pl is not None:
            P.append(torch.softmax(pl, 1).numpy())
        if el is not None:
            E.append(torch.sigmoid(el).numpy())
    return (np.concatenate(P) if P else None), (np.concatenate(E) if E else None)


def inner_val_mask(meta, train_idx, frac=0.2, seed=C.SEED):
    """Early-stopping validation = ~20 % of the TRAINING trials (whole trials, >= 1 per label)."""
    rng = np.random.default_rng(seed)
    m = meta.iloc[train_idx]
    keys = []
    for _, g in m.groupby("group_label"):
        u = g.trial_key.unique()
        rng.shuffle(u)
        keys += list(u[:max(1, int(round(len(u) * frac)))])
    return m.trial_key.isin(keys).to_numpy()


def cpu_latency_ms(fn, n=200):
    for _ in range(5):
        fn(0)
    ts = []
    for k in range(n):
        t = time.perf_counter()
        fn(k)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))
