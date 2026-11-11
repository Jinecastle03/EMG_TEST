"""Real-time processing engine (one worker thread, independent of the GUI refresh).

  source thread -> Chunk queue -> engine thread:
      sample clock (package_num) -> causal filter (same as training) -> ring buffers
      -> raw recording (SessionWriter, optional)
      -> every stride (50 ms): activity frame (RMS per channel, IMU means) for calibration / collection / display
      -> every stride, if a model is active: extract_window() (same function as training) -> predictor
         -> causal probability smoothing -> Prediction (or UNKNOWN with a reason)
  GUI timer (~30 Hz) only reads snapshots under a lock.

If inference is slower than the stride, stale windows are skipped (only the newest due window is predicted).
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .dsp import CausalFilter, SampleClock, WindowConfig, extract_window


class Ring:
    def __init__(self, cap, width, dtype=float):
        self.a = np.zeros((cap, width), dtype)
        self.cap, self.n, self.w = cap, 0, 0

    def append(self, x):
        x = np.asarray(x)
        k = len(x)
        if k == 0:
            return
        if k >= self.cap:
            x, k = x[-self.cap:], self.cap
        e = self.w + k
        if e <= self.cap:
            self.a[self.w:e] = x
        else:
            f = self.cap - self.w
            self.a[self.w:] = x[:f]
            self.a[:k - f] = x[f:]
        self.w = e % self.cap
        self.n = min(self.cap, self.n + k)

    def last(self, k):
        k = min(k, self.n)
        if k == 0:
            return self.a[:0].copy()
        s = (self.w - k) % self.cap
        if s + k <= self.cap:
            return self.a[s:s + k].copy()
        return np.concatenate([self.a[s:], self.a[:self.w]])


@dataclass
class ActivityFrame:
    clock: int
    sidx: int
    rms: np.ndarray            # [8] filtered RMS over the last 200 ms (uV)
    gyro_mean: np.ndarray      # [3]
    acc_mean: np.ndarray       # [3]
    gyro_mag: float
    host_ts: float


@dataclass
class Prediction:
    t: float
    end_clock: int
    sidx: int
    pose: str                  # class name or "UNKNOWN"
    pose_conf: float
    effort: str | None         # "Low"/"High"/None (Rest, no effort model, unknown)
    effort_conf: float
    classes: list
    proba: np.ndarray          # smoothed
    proba_raw: np.ndarray
    p_high: dict
    inference_ms: float
    latency_ms: float          # SDK read of the newest sample -> prediction ready (processing path only)
    reason: str = ""
    truth: tuple | None = None


@dataclass
class EngineStatus:
    source_kind: str = "NONE"
    source_state: str = "idle"
    emg_rate: float = 0.0
    imu_rate: float = 0.0
    fs: float = 0.0
    lost_samples: int = 0
    clock_discontinuities: int = 0
    last_data_age: float = float("inf")
    battery: float = float("nan")
    flat_channels: list = field(default_factory=list)
    noisy_channels: list = field(default_factory=list)
    recording: str = ""
    queue: int = 0
    dropped_windows: int = 0
    infer_rate: float = 0.0


class Engine:
    BUF_S = 12.0

    def __init__(self):
        self.lock = threading.RLock()
        self.source = None
        self._thread = None
        self._stop = threading.Event()
        self.status = EngineStatus()
        self.predictor = None
        self.predictor_error = ""
        self.inference_on = False
        self.smooth_k = 5
        self.min_conf = 0.45
        self.calibration = None
        self.writer = None
        self.trial_id = -1
        self.listeners: list = []
        self.prediction: Prediction | None = None
        self.activity: ActivityFrame | None = None
        self.activation = float("nan")
        self.last_sidx = -1
        self.pred_hist = collections.deque(maxlen=400)
        self._reset_stream(C.FS_EMG_DEFAULT)

    # ------------------------------------------------------------------ control (GUI thread)
    def start(self, source):
        self.stop()
        self.source = source
        self._reset_stream(source.fs)
        self.status = EngineStatus(source_kind=source.kind, source_state="connecting")
        self._stop.clear()
        source.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Engine")
        self._thread.start()

    def stop(self):
        if self.source is not None:
            self.source.stop()
            self.source.join(timeout=5)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self.stop_recording()
        if self.source is not None:
            self.status.source_state = self.source.state
        self.source = None

    def set_predictor(self, pred):
        with self.lock:
            self.predictor = pred
            self.predictor_error = ""
            self._ph.clear()
            self._next_end = None
            self.prediction = None

    def add_listener(self, fn):
        with self.lock:
            self.listeners.append(fn)

    def remove_listener(self, fn):
        with self.lock:
            if fn in self.listeners:
                self.listeners.remove(fn)

    def start_recording(self, writer):
        with self.lock:
            self.writer = writer
            self.trial_id = -1
        writer.event("recording_start", self.status.source_kind)

    def stop_recording(self, status="completed", extra=None):
        with self.lock:
            w, self.writer = self.writer, None
        if w is not None:
            w.event("recording_stop", status)
            w.close(status, extra)
        return w

    # ------------------------------------------------------------------ snapshots (GUI thread)
    def snapshot(self, seconds=3.0, filtered=True):
        with self.lock:
            n = int(seconds * self.fs)
            y = (self.filt if filtered else self.raw).last(n)
            t = self.clk.last(n)[:, 0]
            it = self.iclk.last(int(seconds * 300))[:, 0]
            iv = self.imu.last(int(seconds * 300))
        t0 = t[-1] if len(t) else 0
        return (t - t0) / self.fs, y, (it - t0) / self.fs, iv

    def recent_raw_std(self, seconds=1.0):
        with self.lock:
            return self.raw.last(int(seconds * self.fs)).std(0) if self.raw.n else np.zeros(C.N_EMG)

    # ------------------------------------------------------------------ worker
    def _reset_stream(self, fs):
        self.fs = float(fs)
        cap = int(self.BUF_S * self.fs)
        self.raw, self.filt = Ring(cap, C.N_EMG), Ring(cap, C.N_EMG)
        self.clk = Ring(cap, 1, np.int64)
        self.sid = Ring(cap, 1, np.int64)
        self.imu, self.iclk = Ring(int(self.BUF_S * 100), C.N_IMU), Ring(int(self.BUF_S * 100), 1, np.int64)
        self.filter = CausalFilter(self.fs, C.N_EMG)
        self.clock = SampleClock()
        self.act_cfg = WindowConfig(C.WINDOW_MS, C.STRIDE_MS, self.fs)
        self._next_act = None
        self._next_end = None
        self._ph = collections.deque(maxlen=50)
        self._rate = collections.deque()       # (perf, n_emg, n_imu)
        self._last_chunk_t = None
        self._inf_times = collections.deque(maxlen=40)

    def _loop(self):
        src = self.source
        last_q = time.perf_counter()
        while not self._stop.is_set():
            try:
                ch = src.out.get(timeout=0.1)
            except queue.Empty:
                self._update_status(src)
                continue
            try:
                self._process(ch, src)
            except Exception as e:     # keep streaming; show the problem
                src.log(f"engine error: {type(e).__name__}: {e}")
            if time.perf_counter() - last_q > 0.2:
                self._update_status(src)
                last_q = time.perf_counter()
        # source already stopped: process what is still queued so recordings are complete
        while True:
            try:
                ch = src.out.get_nowait()
            except queue.Empty:
                break
            try:
                self._process(ch, src)
            except Exception as e:
                src.log(f"engine error: {type(e).__name__}: {e}")
        self._update_status(src)

    def _update_status(self, src):
        st = self.status
        st.source_state = src.state
        st.queue = src.out.qsize()
        st.fs = self.fs
        st.last_data_age = time.perf_counter() - self._last_chunk_t if self._last_chunk_t else float("inf")
        st.battery = getattr(src, "battery", float("nan"))
        now = time.perf_counter()
        while self._rate and now - self._rate[0][0] > 3.0:
            self._rate.popleft()
        if len(self._rate) >= 2:
            dt = self._rate[-1][0] - self._rate[0][0]
            if dt > 0.5:
                st.emg_rate = sum(r[1] for r in list(self._rate)[1:]) / dt
                st.imu_rate = sum(r[2] for r in list(self._rate)[1:]) / dt
        if self._inf_times and len(self._inf_times) > 2:
            d = self._inf_times[-1] - self._inf_times[0]
            st.infer_rate = (len(self._inf_times) - 1) / d if d > 0 else 0.0
        st.lost_samples, st.clock_discontinuities = self.clock.lost, self.clock.discontinuities
        st.recording = str(self.writer.dir) if self.writer else ""
        if self.raw.n > self.fs * 0.5:
            sd = self.recent_raw_std(1.0)
            st.flat_channels = [c + 1 for c in range(C.N_EMG) if sd[c] < 1e-3]
            if self.activity is not None:
                st.noisy_channels = [c + 1 for c in range(C.N_EMG) if self.activity.rms[c] > 2000]

    def _process(self, ch, src):
        if abs(src.fs - self.fs) > 1e-9:
            with self.lock:
                self._reset_stream(src.fs)
        clk = self.clock.process(ch.pkg)
        f = self.filter.process(ch.emg)
        with self.lock:
            self.raw.append(ch.emg)
            self.filt.append(f)
            self.clk.append(clk[:, None])
            self.sid.append(ch.sidx[:, None])
            if len(ch.imu_rows):
                self.imu.append(ch.imu)
                self.iclk.append(clk[ch.imu_rows][:, None])
            self.last_sidx = int(ch.sidx[-1])
            w = self.writer
            tid = self.trial_id
        if w is not None:
            w.write_chunk(ch, tid)
        self._last_chunk_t = ch.t_read
        self._rate.append((time.perf_counter(), len(ch.pkg), len(ch.imu_rows)))
        end_now = int(clk[-1])
        # ---- activity frames every stride
        if self._next_act is None:
            self._next_act = end_now
        if end_now >= self._next_act:
            self._activity(end_now, ch)
            self._next_act += ((end_now - self._next_act) // self.act_cfg.stride + 1) * self.act_cfg.stride
        # ---- inference
        if self.inference_on and self.predictor is not None:
            p = self.predictor
            if self._next_end is None:
                self._next_end = end_now
            if end_now >= self._next_end:
                skipped = (end_now - self._next_end) // p.cfg.stride
                self.status.dropped_windows += int(max(0, skipped))
                if src.out.qsize() > 3:
                    # backlog: never let a slow model delay / drop raw data; skip this window (counted)
                    self.status.dropped_windows += 1
                else:
                    self._infer(p, end_now, ch, src)   # window ends at the newest sample (lowest latency)
                self._next_end += (int(max(0, skipped)) + 1) * p.cfg.stride   # fixed 50 ms cadence, no drift

    def _buffers(self, seconds):
        with self.lock:
            n = int(seconds * self.fs)
            return (self.clk.last(n)[:, 0], self.filt.last(n), self.iclk.last(int(seconds * 100))[:, 0],
                    self.imu.last(int(seconds * 100)))

    def _activity(self, end, ch):
        clk, filt, ic, iv = self._buffers(0.6)
        w = extract_window(clk, filt, ic, iv, end, self.act_cfg, need_imu=False)
        rms = np.sqrt((w.emg_actual ** 2).mean(0)) if len(w.emg_actual) else np.zeros(C.N_EMG)
        ia = w.imu_actual
        gm = ia[:, 3:].mean(0) if len(ia) else np.full(3, np.nan)
        am = ia[:, :3].mean(0) if len(ia) else np.full(3, np.nan)
        mag = float(np.linalg.norm(ia[:, 3:], axis=1).mean()) if len(ia) else float("nan")
        fr = ActivityFrame(end, int(ch.sidx[-1]), rms, gm, am, mag, ch.host_ts)
        cal = self.calibration
        act = cal.activation(rms) if cal is not None else float("nan")
        with self.lock:
            self.activity, self.activation = fr, act
            ls = list(self.listeners)
        for fn in ls:
            try:
                fn(fr)
            except Exception as e:
                self.source.log(f"listener error: {e}")

    def _unknown(self, reason, end, ch, p, inf_ms=0.0, raw=None):
        cls = list(p.pose_classes) if p else []
        return Prediction(time.time(), end, int(ch.sidx[-1]), "UNKNOWN", 0.0, None, 0.0, cls,
                          np.zeros(len(cls)), raw if raw is not None else np.zeros(len(cls)), {}, inf_ms,
                          (time.perf_counter() - ch.t_read) * 1000, reason)

    def _infer(self, p, end, ch, src):
        st = self.status
        if abs(p.fs - self.fs) > 1e-9:
            pr = self._unknown(f"model fs {p.fs:.0f} Hz != stream {self.fs:.0f} Hz", end, ch, p)
        elif st.flat_channels:
            pr = self._unknown(f"flat EMG channel(s) {st.flat_channels}", end, ch, p)
        else:
            clk, filt, ic, iv = self._buffers(p.cfg.window_ms / 1000 + 1.0)
            w = extract_window(clk, filt, ic, iv, end, p.cfg, need_imu=p.needs_imu)
            if not w.valid:
                pr = self._unknown("invalid window: " + w.reason, end, ch, p)
            else:
                t0 = time.perf_counter()
                try:
                    pp, ph = p.predict(w)
                except Exception as e:
                    self.predictor_error = f"{type(e).__name__}: {e}"
                    pr = self._unknown("model error: " + self.predictor_error, end, ch, p)
                    pp = None
                if pp is not None:
                    inf_ms = (time.perf_counter() - t0) * 1000
                    self._ph.append((pp, ph))
                    k = max(1, self.smooth_k)
                    hist = list(self._ph)[-k:]
                    P = np.mean([h[0] for h in hist], 0)
                    H = np.nanmean(np.array([h[1] for h in hist], float), 0) if len(p.active_poses) else np.zeros(0)
                    j = int(P.argmax())
                    pose, conf = p.pose_classes[j], float(P[j])
                    eff, econf = None, 0.0
                    if pose in p.active_poses:
                        v = H[p.active_poses.index(pose)]
                        if np.isfinite(v):
                            eff, econf = ("High" if v >= 0.5 else "Low"), float(max(v, 1 - v))
                    reason = ""
                    if conf < self.min_conf:
                        reason = f"low confidence {conf:.2f} (best: {pose})"
                        pose, eff = "UNKNOWN", None
                    pr = Prediction(time.time(), end, int(ch.sidx[-1]), pose, conf, eff, econf, list(p.pose_classes), P, pp,
                                    {a: float(H[i]) for i, a in enumerate(p.active_poses)}, inf_ms,
                                    (time.perf_counter() - ch.t_read) * 1000, reason)
        if hasattr(src, "truth_at"):
            pr.truth = src.truth_at(pr.sidx)
        self._inf_times.append(time.perf_counter())
        with self.lock:
            self.prediction = pr
            self.pred_hist.append(pr)
            w = self.writer
        if w is not None:
            row = {"host_ts": pr.t, "sample_index": pr.sidx, "end_clock": pr.end_clock, "model": p.name, "pose": pr.pose,
                   "pose_conf": pr.pose_conf, "effort": pr.effort or "", "effort_conf": pr.effort_conf,
                   "inference_ms": pr.inference_ms, "latency_ms": pr.latency_ms, "reason": pr.reason}
            for c, v in zip(pr.classes, pr.proba_raw):
                row[f"p_{c}"] = float(v)
            for a in p.active_poses:
                row[f"p_high_{a}"] = pr.p_high.get(a, float("nan"))
            w.prediction(row)
