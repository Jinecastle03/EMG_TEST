"""Labeled segments -> sliding windows, and leakage-free splits.

Splits are decided per TRIAL GROUP before any window is created: every window carries the key of the trial
it came from (inter-trial rest windows belong to the preceding trial), and a trial key is in exactly one of
dev/test and in exactly one CV fold. Overlapping windows of one trial therefore never appear on both sides.

Labeled segments of a session
  * hold of a completed trial   [hold_start + onset_margin, hold_end]  -> trial pose / effort
  * inter-trial rest (new protocol only, meta.protocol.inter_trial_rest)  [hold_end + rest_settle, rest_end] -> Rest
  * prep / release transitions are never labeled (excluded from training).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .dsp import IMU_FEATURES, WindowConfig, extract_window, window_features
from .session import SessionData

REST_SETTLE_S = 1.0


@dataclass
class WindowSet:
    emg: np.ndarray          # [N, 8, n_emg] float32 filtered EMG (uV)
    imu: np.ndarray          # [N, 6, n_imu] float32
    feats: np.ndarray        # [N, 45] legacy features
    meta: pd.DataFrame       # one row per window
    cfg: WindowConfig
    dropped: dict

    def subset(self, idx):
        idx = np.asarray(idx)
        return WindowSet(self.emg[idx], self.imu[idx], self.feats[idx], self.meta.iloc[idx].reset_index(drop=True),
                         self.cfg, self.dropped)


def label_of(pose, effort):
    return pose if pose == C.REST or not effort else f"{pose}|{effort}"


def segments(s: SessionData) -> list[dict]:
    prot = s.meta.get("protocol", {})
    onset = float(prot.get("onset_margin_s", 0.5)) * s.fs
    gap_rest = bool(prot.get("inter_trial_rest", False))
    t = s.trials[s.trials.status == "completed"].sort_values("hold_start_host")
    out = []
    for order, (_, r) in enumerate(t.iterrows()):
        key = f"{s.session_id}#{int(r.trial_id)}"
        h0, h1, r1 = s.idx_to_clock([r.hold_start_idx, r.hold_end_idx, r.rest_end_idx])
        effort = "" if r.pose == C.REST else str(r.effort)
        base = {"session_id": s.session_id, "participant": s.participant, "trial_key": key, "trial_id": int(r.trial_id),
                "trial_order": order, "start_host": float(r.hold_start_host), "trial_pose": r.pose,
                "trial_effort": effort, "group_label": label_of(r.pose, effort)}
        out.append(base | {"seg": "hold", "c0": int(h0 + onset), "c1": int(h1),
                           "pose": r.pose, "effort": effort})
        if gap_rest and r1 > h1:
            out.append(base | {"seg": "gap_rest", "c0": int(h1 + REST_SETTLE_S * s.fs), "c1": int(r1), "pose": C.REST, "effort": ""})
    return out


def build_windows(sessions: list[SessionData], cfg: WindowConfig, progress=None) -> WindowSet:
    E, I, F, M = [], [], [], []
    dropped = {"invalid": 0, "reasons": {}}
    for s in sessions:
        if abs(s.fs - cfg.fs) > 1e-6:
            raise ValueError(f"{s.session_id}: fs {s.fs} != window config fs {cfg.fs}")
        for seg in segments(s):
            ends = np.arange(seg["c0"] + cfg.n_emg - 1, seg["c1"] + 1, cfg.stride)
            for e in ends:
                w = extract_window(s.clock, s.emg_filt, s.imu_clock, s.imu, int(e), cfg)
                if not w.valid:
                    dropped["invalid"] += 1
                    key = w.reason.split(" ")[0]
                    dropped["reasons"][key] = dropped["reasons"].get(key, 0) + 1
                    continue
                E.append(w.emg_grid.astype(np.float32))
                I.append(w.imu_grid.astype(np.float32))
                F.append(window_features(w))
                M.append({k: seg[k] for k in ("session_id", "participant", "trial_key", "trial_id", "trial_order", "start_host",
                                              "group_label", "seg", "pose", "effort")} |
                         {"end_clock": int(e), "t_in_seg": (e - seg["c0"]) / s.fs})
        if progress:
            progress(f"windows: {s.session_id} done ({len(M)} total)")
    meta = pd.DataFrame(M)
    feats = np.array(F) if F else np.zeros((0, 45))
    if len(meta):
        meta["label"] = [label_of(p, e) for p, e in zip(meta.pose, meta.effort)]
        meta["gyro_mag_mean"] = feats[:, 32 + IMU_FEATURES.index("gyro_mag_mean")]
    return WindowSet(np.stack(E) if E else np.zeros((0, 8, cfg.n_emg), np.float32),
                     np.stack(I) if I else np.zeros((0, 6, cfg.n_imu), np.float32), feats, meta, cfg, dropped)


# ======================================================================================== splits
def split_availability(meta: pd.DataFrame) -> dict:
    ns, npart = meta.session_id.nunique(), meta.participant.nunique()
    return {
        "trial": {"available": True, "reason": "test = last trials of every label (temporal hold-out) within each session"},
        "session": {"available": ns >= 2, "reason": f"{ns} session(s)" + ("" if ns >= 2 else " - needs >= 2 sessions; NOT evaluated")},
        "participant": {"available": npart >= 2,
                        "reason": f"{npart} participant(s)" + ("" if npart >= 2 else " - needs >= 2 participants; NOT evaluated")},
    }


def assign_splits(meta: pd.DataFrame, mode: str = "trial", test_reps: int = 2, n_folds: int = 4, seed: int = C.SEED) -> pd.DataFrame:
    """Returns one row per trial_key: split (dev/test) and fold (dev only, -1 for test)."""
    g = (meta.groupby("trial_key").agg(session_id=("session_id", "first"), participant=("participant", "first"),
                                       label=("group_label", "first"), start=("start_host", "first"))
         .reset_index().sort_values(["session_id", "start"]))
    if mode == "trial":
        g["rank_from_end"] = g.groupby(["session_id", "label"]).cumcount(ascending=False)
        g["split"] = np.where(g.rank_from_end < test_reps, "test", "dev")
    elif mode == "session":
        last = g.groupby("participant").session_id.agg(lambda s: sorted(s.unique())[-1])
        if (g.groupby("participant").session_id.nunique() < 2).all():
            raise ValueError("session-independent split needs >= 2 sessions of one participant")
        g["split"] = np.where(g.session_id == g.participant.map(last), "test", "dev")
    elif mode == "participant":
        parts = sorted(g.participant.unique())
        if len(parts) < 2:
            raise ValueError("participant-independent split needs >= 2 participants")
        g["split"] = np.where(g.participant == parts[-1], "test", "dev")
    else:
        raise ValueError(mode)
    g["fold"] = -1
    rng = np.random.default_rng(seed)
    dev = g[g.split == "dev"]
    group_col = {"session": "session_id", "participant": "participant"}.get(mode)
    if group_col and dev[group_col].nunique() >= 2:
        units = sorted(dev[group_col].unique())
        k = min(n_folds, len(units))
        fmap = {u: i % k for i, u in enumerate(units)}
        g.loc[dev.index, "fold"] = dev[group_col].map(fmap)
    else:
        for _, gg in dev.groupby("label"):       # label-stratified, trial-grouped folds
            idx = gg.index.to_numpy().copy()
            rng.shuffle(idx)
            g.loc[idx, "fold"] = np.arange(len(idx)) % n_folds
    return g[["trial_key", "session_id", "participant", "label", "split", "fold"]].reset_index(drop=True)


def attach_splits(ws: WindowSet, splits: pd.DataFrame):
    m = ws.meta.drop(columns=[c for c in ("split", "fold") if c in ws.meta]).merge(
        splits[["trial_key", "split", "fold"]], on="trial_key", how="left")
    assert len(m) == len(ws.meta)
    ws.meta = m
    # leakage guard: a trial key must be on one side only
    sides = ws.meta.groupby("trial_key").split.nunique()
    assert (sides == 1).all(), "trial appears in both dev and test"
    return ws
