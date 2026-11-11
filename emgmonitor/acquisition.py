"""Data sources. Each source runs in its own thread and puts `Chunk`s into `source.out`.

  MindRoveSource(board="wifi")       real MindRove Armband 2 (BoardIds.MINDROVE_WIFI_BOARD) via the MindRove
                                      Python SDK (pip `mindrove` 5.3.0 - same SDK version and row layout as the
                                      legacy Unity C# app: 39 rows, EMG 0-7, accel 20-22, gyro 23-25, pkg 26, ts 27)
  MindRoveSource(board="synthetic")  SDK SYNTHETIC_BOARD (no hardware; 250 Hz, 16 ch -> first 8 used). Only for
                                      testing the SDK code path; never used as training data.
  ReplaySource(session_dir)           plays a recorded session in real time (REPLAY MODE)

Channel rows are read from the SDK at connect time (never hard-coded). get_board_data() removes the returned
samples from the SDK ring buffer, so no sample is delivered twice.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .dsp import ImuDetector


@dataclass
class Chunk:
    emg: np.ndarray            # [n, 8] raw uV
    pkg: np.ndarray            # [n] package_num
    device_ts: np.ndarray      # [n]
    sidx: np.ndarray           # [n] running sample index
    imu_rows: np.ndarray       # [m] row positions (in this chunk) that carry a new IMU sample
    imu: np.ndarray            # [m, 6] acc xyz (g), gyro xyz (dps)
    host_ts: float             # time.time() when read from the SDK / file
    t_read: float              # time.perf_counter() when read
    battery: float = float("nan")


class Source(threading.Thread):
    kind = "?"

    def __init__(self):
        super().__init__(daemon=True, name=self.__class__.__name__)
        self.out: queue.Queue = queue.Queue(maxsize=2000)
        self.state = "idle"
        self.error = ""
        self.messages: list[str] = []
        self.fs = C.FS_EMG_DEFAULT
        self.layout: dict = {}
        self._stop = threading.Event()

    def log(self, msg):
        self.messages.append(time.strftime("%H:%M:%S ") + msg)
        del self.messages[:-200]

    def stop(self):
        self._stop.set()

    def _put(self, ch: Chunk):
        try:
            self.out.put(ch, timeout=1.0)
        except queue.Full:
            self.log("processing queue full - chunk dropped")


class MindRoveSource(Source):
    def __init__(self, board="wifi", ip="", port=0, timeout=0, poll_ms=10, stall_s=2.0):
        super().__init__()
        self.board_name = board
        self.kind = "DEVICE" if board == "wifi" else "SYNTHETIC"
        self.ip, self.port, self.timeout, self.poll_ms, self.stall_s = ip, port, timeout, poll_ms, stall_s
        self.sdk_version = ""
        self.battery = float("nan")

    def run(self):
        board = None
        streaming = False
        try:
            from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
            bid = (BoardIds.MINDROVE_WIFI_BOARD if self.board_name == "wifi" else BoardIds.SYNTHETIC_BOARD).value
            C.LOG_DIR.mkdir(parents=True, exist_ok=True)
            try:
                BoardShim.set_log_file(str(C.LOG_DIR / "mindrove_sdk.log"))
                BoardShim.enable_dev_board_logger()
            except Exception:
                pass
            self.state = "connecting"
            L = {"board_id": bid, "device_name": BoardShim.get_device_name(bid), "sampling_rate": BoardShim.get_sampling_rate(bid),
                 "num_rows": BoardShim.get_num_rows(bid), "emg_rows": list(BoardShim.get_emg_channels(bid))[:C.N_EMG],
                 "accel_rows": list(BoardShim.get_accel_channels(bid)), "gyro_rows": list(BoardShim.get_gyro_channels(bid)),
                 "package_num_row": BoardShim.get_package_num_channel(bid), "timestamp_row": BoardShim.get_timestamp_channel(bid)}
            try:
                L["battery_row"] = BoardShim.get_battery_channel(bid)
            except Exception:
                L["battery_row"] = -1
            try:
                self.sdk_version = str(BoardShim.get_version())
            except Exception:
                self.sdk_version = "?"
            self.layout, self.fs = L, float(L["sampling_rate"])
            self.log(f"SDK layout: {L}")
            if len(L["emg_rows"]) != C.N_EMG:
                self.log(f"WARNING: SDK reports {len(L['emg_rows'])} EMG channels")
            p = MindRoveInputParams()
            if self.ip:
                p.ip_address = self.ip
            if self.port:
                p.ip_port = int(self.port)
            if self.timeout:
                p.timeout = int(self.timeout)
            board = BoardShim(bid, p)
            board.prepare_session()
            board.start_stream(450000)
            streaming = True
            self.state = "streaming"
            self.log(f"stream started ({L['device_name']}, {self.fs:.0f} Hz)")
            det = ImuDetector()
            sidx0 = 0
            last_data = time.perf_counter()
            imu_cols = L["accel_rows"] + L["gyro_rows"]
            while not self._stop.is_set():
                n = board.get_board_data_count()
                if n > 0:
                    d = board.get_board_data(n)
                    t_read, host = time.perf_counter(), time.time()
                    emg = d[L["emg_rows"]].T.astype(float)
                    imu_all = d[imu_cols].T.astype(float) if len(imu_cols) == 6 else np.zeros((d.shape[1], 6))
                    mask = det.detect(imu_all)
                    rows = np.where(mask)[0]
                    if L["battery_row"] >= 0:
                        self.battery = float(d[L["battery_row"], -1])
                    pkg = d[L["package_num_row"]].astype(float) if L["package_num_row"] >= 0 else np.arange(sidx0, sidx0 + d.shape[1], dtype=float)
                    ts = d[L["timestamp_row"]].astype(float) if L["timestamp_row"] >= 0 else np.full(d.shape[1], host)
                    self._put(Chunk(emg, pkg, ts, np.arange(sidx0, sidx0 + d.shape[1]), rows, imu_all[rows], host, t_read, self.battery))
                    sidx0 += d.shape[1]
                    last_data = t_read
                    if self.state == "stalled":
                        self.state = "streaming"
                        self.log("data resumed")
                elif time.perf_counter() - last_data > self.stall_s and self.state == "streaming":
                    self.state = "stalled"
                    self.log(f"no data for {self.stall_s:.0f} s - is the PC on the armband Wi-Fi? firewall?")
                time.sleep(self.poll_ms / 1000)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.log("ERROR " + self.error + "  (armband on? PC joined the armband Wi-Fi? firewall allows python.exe?"
                     " previous session open -> restart program)")
            self.state = "error"
        finally:
            try:
                if board is not None:
                    if streaming:
                        board.stop_stream()
                    if board.is_prepared():
                        board.release_session()
                    self.log("session released")
            except Exception as e:
                self.log(f"release: {e}")
            if self.state != "error":
                self.state = "stopped"


class ReplaySource(Source):
    """Real-time playback of a recorded session (raw EMG + IMU exactly as stored)."""
    kind = "REPLAY"

    def __init__(self, session_dir, speed=1.0, chunk=10, loop=False, start_s=0.0):
        super().__init__()
        from .session import TRIAL_COLUMNS
        self.dir = session_dir
        self.meta = C.load_json(session_dir / "metadata.json")
        self.fs = float(self.meta.get("fs", C.FS_EMG_DEFAULT))
        self.layout = {"replay_of": str(session_dir)}
        e = pd.read_csv(session_dir / "emg.csv")
        i = pd.read_csv(session_dir / "imu.csv")
        self.emg = e[C.EMG_COLS].to_numpy(float)
        self.pkg = e.package_num.to_numpy(float)
        self.ts = e.device_ts.to_numpy(float)
        self.sidx = e.sample_index.to_numpy(np.int64)
        self.imu_pos = np.clip(np.searchsorted(self.sidx, i.sample_index.to_numpy()), 0, len(self.sidx) - 1)
        self.imu = i[C.IMU_COLS].to_numpy(float)
        self.trials = pd.read_csv(session_dir / "trials.csv") if (session_dir / "trials.csv").exists() else pd.DataFrame(columns=TRIAL_COLUMNS)
        self.trials = self.trials[self.trials.status == "completed"]
        self.trials["effort"] = self.trials["effort"].fillna("").astype(str)
        self.speed, self.chunk, self.loop = speed, chunk, loop
        self.start_pos = int(start_s * self.fs)
        self.pos = self.start_pos
        self.n = len(self.emg)
        self.labels_in_data = sorted({p if p == C.REST or not e else f"{p}|{e}" for p, e in zip(self.trials.pose, self.trials.effort)})
        self.poses_in_data = sorted(set(self.trials.pose))

    def truth_at(self, sidx: int):
        """Ground truth (pose, effort, phase) from the trial table: 'hold' inside a trial hold, 'rest' in
        the cued inter-trial rest, '' otherwise (unlabeled / transition)."""
        t = self.trials
        if len(t) == 0:
            return None
        h = t[(t.hold_start_idx <= sidx) & (sidx <= t.hold_end_idx)]
        if len(h):
            r = h.iloc[0]
            return r.pose, (r.effort or ""), "hold"
        if self.meta.get("protocol", {}).get("inter_trial_rest"):
            g = t[(t.hold_end_idx < sidx) & (sidx <= t.rest_end_idx)]
            if len(g):
                return C.REST, "", "rest"
        return None

    def progress(self):
        return (self.pos - self.start_pos) / max(1, self.n - self.start_pos), self.pos / self.fs

    def run(self):
        self.state = "streaming"
        self.log(f"replay {self.dir.name}: {self.n / self.fs:.0f} s at x{self.speed}")
        t0 = time.perf_counter()
        sent0 = self.pos
        ip = 0
        while not self._stop.is_set():
            if self.pos >= self.n:
                if not self.loop:
                    break
                self.pos, ip = self.start_pos, 0
                t0, sent0 = time.perf_counter(), self.pos
                self.log("replay loop restart")
            due = sent0 + int((time.perf_counter() - t0) * self.fs * self.speed)
            if due < self.pos + self.chunk:
                time.sleep(0.004)
                continue
            a, b = self.pos, min(self.n, max(self.pos + self.chunk, due))
            b = min(b, a + 200)
            while ip < len(self.imu_pos) and self.imu_pos[ip] < a:
                ip += 1
            jp = ip
            while jp < len(self.imu_pos) and self.imu_pos[jp] < b:
                jp += 1
            self._put(Chunk(self.emg[a:b], self.pkg[a:b], self.ts[a:b], self.sidx[a:b], self.imu_pos[ip:jp] - a,
                            self.imu[ip:jp], time.time(), time.perf_counter()))
            ip = jp
            self.pos = b
        self.state = "finished" if self.pos >= self.n else "stopped"
        self.log("replay " + self.state)
