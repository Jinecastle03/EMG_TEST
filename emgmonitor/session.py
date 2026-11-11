"""Session storage: raw EMG/IMU CSV writer (dedicated thread), trial table, metadata, loader.

Folder: data/sessions/<participant>/<session_id>/
  emg.csv       sample_index, package_num, device_ts, host_ts, trial_id, emg_ch1_uv..emg_ch8_uv   (RAW, unfiltered, 500 Hz)
  imu.csv       imu_index, sample_index, package_num, device_ts, host_ts, trial_id, acc_x_g..gyro_z_dps (RAW, ~50 Hz)
  trials.csv    one row per trial with the cue phases as sample indices (prep / hold start / hold end / rest end)
  events.csv    host_unix, event, detail
  predictions.csv  (only when live inference ran during the recording)
  metadata.json participant, session, kind, device layout, sampling rates, protocol, calibration snapshot

Raw files are written once and never modified afterwards. Windows / features are only computed at training time.
"""
from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import config as C
from .dsp import CausalFilter, SampleClock

TRIAL_COLUMNS = ["trial_id", "pose", "effort", "repetition", "condition", "status", "prep_start_idx", "hold_start_idx",
                 "hold_end_idx", "rest_end_idx", "hold_start_host", "hold_end_host", "quality", "note"]


def new_session_id(prefix="Session") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S")


class SessionWriter:
    def __init__(self, root: Path, participant: str, session_id: str, metadata: dict):
        self.dir = Path(root) / participant / session_id
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"{self.dir} already exists - refusing to overwrite")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.participant, self.session_id = participant, session_id
        self.meta = dict(metadata)
        self.meta.update({"format": "emgmonitor-session-1", "app_version": __version__, "participant_id": participant,
                          "session_id": session_id, "status": "recording",
                          "start_local": time.strftime("%Y-%m-%d %H:%M:%S"), "start_host_unix": time.time()})
        self.trials: list[dict] = []
        self._q: queue.Queue = queue.Queue()
        self._emg = open(self.dir / "emg.csv", "x", encoding="utf-8", newline="\n")
        self._imu = open(self.dir / "imu.csv", "x", encoding="utf-8", newline="\n")
        self._events = open(self.dir / "events.csv", "x", encoding="utf-8", newline="\n")
        self._pred = None
        self._emg.write("sample_index,package_num,device_ts,host_ts,trial_id," + ",".join(C.EMG_COLS) + "\n")
        self._imu.write("imu_index,sample_index,package_num,device_ts,host_ts,trial_id," + ",".join(C.IMU_COLS) + "\n")
        self._events.write("host_unix,event,detail\n")
        self.emg_rows = 0
        self.imu_rows = 0
        self._imu_index = 0
        self.error = ""
        self.write_metadata()
        self.write_trials()
        self._thread = threading.Thread(target=self._loop, name="SessionWriter", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API (any thread)
    def write_chunk(self, chunk, trial_id: int):
        self._q.put(("chunk", chunk, trial_id))

    def event(self, name: str, detail: str = ""):
        self._q.put(("event", time.time(), name, detail))

    def prediction(self, row: dict):
        self._q.put(("pred", row))

    def set_trials(self, trials: list[dict]):
        self.trials = [dict(t) for t in trials]
        self._q.put(("trials",))

    def close(self, status="completed", extra_meta: dict | None = None):
        self._q.put(("close", status, extra_meta or {}))
        self._thread.join(timeout=120)

    # ------------------------------------------------------------------ writer thread
    def _loop(self):
        try:
            last_flush = time.time()
            while True:
                item = self._q.get()
                kind = item[0]
                if kind == "chunk":
                    self._write_chunk(item[1], item[2])
                elif kind == "event":
                    self._events.write(f"{item[1]:.6f},{_csv(item[2])},{_csv(item[3])}\n")
                elif kind == "pred":
                    self._write_pred(item[1])
                elif kind == "trials":
                    self.write_trials()
                elif kind == "close":
                    self._finish(item[1], item[2])
                    return
                if time.time() - last_flush > 1.0:
                    for f in (self._emg, self._imu, self._events, self._pred):
                        if f:
                            f.flush()
                    last_flush = time.time()
        except Exception as e:  # never kill the acquisition because of disk errors; surface it
            self.error = f"{type(e).__name__}: {e}"

    def _write_chunk(self, ch, trial_id):
        n = len(ch.pkg)
        if n == 0:
            return
        idx = ch.sidx
        a = np.column_stack([idx, ch.pkg, ch.device_ts, np.full(n, ch.host_ts), np.full(n, trial_id), ch.emg])
        buf = io.StringIO()
        np.savetxt(buf, a, fmt=["%d", "%.0f", "%.6f", "%.6f", "%d"] + ["%.4f"] * C.N_EMG, delimiter=",")
        self._emg.write(buf.getvalue())
        self.emg_rows += n
        m = len(ch.imu_rows)
        if m:
            r = ch.imu_rows
            b = np.column_stack([self._imu_index + np.arange(m), idx[r], ch.pkg[r], ch.device_ts[r], np.full(m, ch.host_ts),
                                 np.full(m, trial_id), ch.imu])
            buf = io.StringIO()
            np.savetxt(buf, b, fmt=["%d", "%d", "%.0f", "%.6f", "%.6f", "%d"] + ["%.6f"] * C.N_IMU, delimiter=",")
            self._imu.write(buf.getvalue())
            self._imu_index += m
            self.imu_rows += m

    def _write_pred(self, row: dict):
        if self._pred is None:
            self._pred = open(self.dir / "predictions.csv", "x", encoding="utf-8", newline="\n")
            self._pred_cols = list(row.keys())
            self._pred.write(",".join(self._pred_cols) + "\n")
        self._pred.write(",".join(_csv(row.get(k, "")) for k in self._pred_cols) + "\n")

    def write_trials(self):
        df = pd.DataFrame(self.trials, columns=TRIAL_COLUMNS)
        tmp = self.dir / "trials.csv.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.dir / "trials.csv")

    def write_metadata(self):
        C.save_json(self.meta, self.dir / "metadata.json")

    def _finish(self, status, extra):
        for f in (self._emg, self._imu, self._events, self._pred):
            if f:
                f.close()
        self.write_trials()
        self.meta.update(extra)
        self.meta.update({"status": status, "stop_local": time.strftime("%Y-%m-%d %H:%M:%S"), "stop_host_unix": time.time(),
                          "emg_rows": self.emg_rows, "imu_rows": self.imu_rows,
                          "trials_completed": sum(t["status"] == "completed" for t in self.trials),
                          "writer_error": self.error})
        self.write_metadata()


def _csv(v) -> str:
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    s = str(v)
    return '"' + s.replace('"', '""') + '"' if ("," in s or '"' in s or "\n" in s) else s


# ====================================================================================== loading
@dataclass
class SessionData:
    path: Path
    meta: dict
    emg_raw: np.ndarray          # [N, 8]
    sample_index: np.ndarray     # [N]
    clock: np.ndarray            # [N] unwrapped sample clock (package_num based)
    emg_filt: np.ndarray         # [N, 8] causal filter over the whole stream (same as live)
    imu: np.ndarray              # [M, 6]
    imu_clock: np.ndarray        # [M]
    trials: pd.DataFrame
    trial_col: np.ndarray        # [N] trial_id per EMG row as recorded
    stats: dict = field(default_factory=dict)

    @property
    def session_id(self):
        return self.meta["session_id"]

    @property
    def participant(self):
        return self.meta.get("participant_id", "?")

    @property
    def fs(self):
        return float(self.meta.get("fs", C.FS_EMG_DEFAULT))

    def idx_to_clock(self, idx) -> np.ndarray:
        """Map stored sample_index values to the sample clock."""
        idx = np.asarray(idx, dtype=float)
        pos = np.clip(np.searchsorted(self.sample_index, idx), 0, len(self.sample_index) - 1)
        return self.clock[pos]


def load_session(path: Path, filter_cfg: dict | None = None) -> SessionData:
    path = Path(path)
    meta = C.load_json(path / "metadata.json")
    emg = pd.read_csv(path / "emg.csv", engine="c")
    imu = pd.read_csv(path / "imu.csv", engine="c")
    trials = pd.read_csv(path / "trials.csv") if (path / "trials.csv").exists() else pd.DataFrame(columns=TRIAL_COLUMNS)
    for c in ("effort", "quality", "note", "condition"):
        if c in trials:
            trials[c] = trials[c].fillna("").astype(str)
    fs = float(meta.get("fs", C.FS_EMG_DEFAULT))
    sidx = emg.sample_index.to_numpy(np.int64)
    clk = SampleClock()
    clock = clk.process(emg.package_num.to_numpy(float))
    raw = emg[C.EMG_COLS].to_numpy(float)
    filt = CausalFilter(fs, C.N_EMG, filter_cfg or meta.get("filter") or C.FILTER).process(raw)
    pos = np.clip(np.searchsorted(sidx, imu.sample_index.to_numpy()), 0, len(sidx) - 1)
    imu_clock = clock[pos]
    order = np.argsort(imu_clock, kind="stable")
    stats = {"emg_rows": len(emg), "imu_rows": len(imu), "lost_samples": clk.lost, "clock_discontinuities": clk.discontinuities,
             "duration_s": float((clock[-1] - clock[0]) / fs) if len(clock) else 0.0}
    stats["imu_rate_hz"] = float(len(imu) / stats["duration_s"]) if stats["duration_s"] else 0.0
    return SessionData(path, meta, raw, sidx, clock, filt, imu[C.IMU_COLS].to_numpy(float)[order], imu_clock[order],
                       trials, emg.trial_id.to_numpy(), stats)


def list_sessions(root: Path = C.SESSIONS_DIR) -> list[dict]:
    out = []
    for mp in sorted(Path(root).glob("*/*/metadata.json")):
        try:
            m = C.load_json(mp)
        except Exception:
            continue
        d = mp.parent
        info = {"path": str(d), "session_id": m.get("session_id", d.name), "participant": m.get("participant_id", "?"),
                "kind": m.get("kind", "?"), "data_source": m.get("data_source", "?"), "status": m.get("status", "?"),
                "start": m.get("start_local", ""), "labels": {}, "n_completed": 0}
        tp = d / "trials.csv"
        if tp.exists():
            try:
                t = pd.read_csv(tp)
                t = t[t.status == "completed"]
                info["n_completed"] = int(len(t))
                info["labels"] = {f"{p}|{e}" if isinstance(e, str) and e else p: int(n)
                                  for (p, e), n in t.fillna({"effort": ""}).groupby(["pose", "effort"]).size().items()}
            except Exception:
                pass
        out.append(info)
    return out
