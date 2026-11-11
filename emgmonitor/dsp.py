"""Signal conditioning, sample clock, timestamp-aligned windows and hand-crafted features.

All functions here are used by BOTH offline training (whole recorded session) and the real-time engine
(ring buffers), so the model input is computed by the same code in both cases.

Time base
---------
`device_ts` from the SDK is a per-packet PC timestamp (many identical values) and cannot time single
samples. `package_num` increments by 1 per EMG sample (+3 when 2 samples were lost, observed in the
legacy session), so the unwrapped package number is used as the sample clock: t = clock / fs.
IMU samples are stored in the same SDK row stream; each IMU sample gets the clock of the EMG row in
which it appeared, so EMG and IMU share one time axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from . import config as C


# --------------------------------------------------------------------------------------------- filter
def _rbj_highpass(fs, f0, q):
    w0 = 2 * np.pi * f0 / fs
    cos, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    b = np.array([(1 + cos) / 2, -(1 + cos), (1 + cos) / 2])
    a = np.array([1 + alpha, -2 * cos, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])


def _rbj_notch(fs, f0, q):
    w0 = 2 * np.pi * f0 / fs
    cos, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    b = np.array([1.0, -2 * cos, 1.0])
    a = np.array([1 + alpha, -2 * cos, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])


def filter_sos(fs: float, cfg: dict | None = None) -> np.ndarray:
    """Second-order sections identical to the legacy Unity EmgFilterBank / ML preprocessing.py."""
    cfg = cfg or C.FILTER
    secs = []
    if cfg["highpass_order"] >= 4:
        secs += [_rbj_highpass(fs, cfg["highpass_hz"], 0.54119610), _rbj_highpass(fs, cfg["highpass_hz"], 1.30656296)]
    else:
        secs += [_rbj_highpass(fs, cfg["highpass_hz"], 0.70710678)]
    f = cfg["notch_hz"]
    while f < fs / 2 * 0.95:
        secs.append(_rbj_notch(fs, f, cfg["notch_q"]))
        if not cfg["notch_harmonics"]:
            break
        f += cfg["notch_hz"]
    return np.array(secs)


class CausalFilter:
    """Streaming causal IIR filter (per channel). Filtering a stream in chunks gives exactly the same
    output as filtering it at once. State is primed to the first sample (removes the DC step of the
    raw MindRove signal, which has offsets of 10-40 mV)."""

    def __init__(self, fs: float, n_ch: int = C.N_EMG, cfg: dict | None = None):
        self.sos = filter_sos(fs, cfg)
        self.n_ch = n_ch
        self._zi_unit = sosfilt_zi(self.sos)
        self.zi = None

    def reset(self):
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: [n, ch] raw -> [n, ch] filtered."""
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.empty((0, self.n_ch))
        if self.zi is None:
            self.zi = np.stack([self._zi_unit * x[0, c] for c in range(self.n_ch)], axis=-1)  # [sec, 2, ch]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


# --------------------------------------------------------------------------------------------- clock
class SampleClock:
    """Unwraps package_num into a monotonic sample clock. Normal step: +1; lost samples: +k (k <= max_gap).
    Wrap-around, reset or garbage -> +1 (assume no loss) and count it as a discontinuity."""

    def __init__(self, max_gap: int = 50):
        self.max_gap = max_gap
        self.last_pkg = None
        self.clock = -1
        self.discontinuities = 0
        self.lost = 0

    def process(self, pkg: np.ndarray) -> np.ndarray:
        pkg = np.asarray(pkg, dtype=float)
        out = np.empty(len(pkg), dtype=np.int64)
        if len(pkg) == 0:
            return out
        prev = self.last_pkg
        d = np.empty(len(pkg))
        d[0] = 1 if prev is None else pkg[0] - prev
        d[1:] = np.diff(pkg)
        bad = ~np.isfinite(d) | (d < 1) | (d > self.max_gap)
        if prev is None:
            bad[0] = False
        self.discontinuities += int(bad.sum())
        d[bad] = 1
        self.lost += int((d - 1).sum())
        steps = d.astype(np.int64)
        if prev is None:
            steps[0] = 0
            out[:] = np.cumsum(steps)
        else:
            out[:] = self.clock + np.cumsum(steps)
        self.clock = int(out[-1])
        self.last_pkg = float(pkg[-1]) if np.isfinite(pkg[-1]) else self.last_pkg
        return out


# --------------------------------------------------------------------------------------------- IMU detection
class ImuDetector:
    """New IMU sample = row whose 6 values differ from the previous IMU sample and are not all zero
    (port of the legacy Unity ImuSampleDetector, ChangeDetection mode)."""

    def __init__(self):
        self.last = None
        self.rows = 0
        self.held = 0
        self.empty = 0

    def detect(self, v: np.ndarray) -> np.ndarray:
        """v: [n, 6] -> boolean mask of rows with a new IMU sample."""
        n = len(v)
        self.rows += n
        mask = np.zeros(n, bool)
        if n == 0:
            return mask
        finite = np.isfinite(v).all(1)
        zero = (v == 0).all(1)
        prev = np.vstack([self.last if self.last is not None else np.full(6, np.nan), v[:-1]])
        # change detection is relative to the last ACCEPTED sample; with held rows in between, the previous
        # row equals the last accepted one, so comparing to the previous valid row is equivalent except after
        # empty rows - handle that sequentially only when needed.
        if (zero | ~finite).any():
            last = self.last
            for i in range(n):
                if not finite[i] or zero[i]:
                    self.empty += int(zero[i])
                    continue
                if last is not None and np.array_equal(v[i], last):
                    self.held += 1
                    continue
                mask[i] = True
                last = v[i].copy()
            self.last = last
            return mask
        same = (v == prev).all(1)
        mask = ~same
        self.held += int(same.sum())
        self.last = v[-1].copy()
        return mask


# --------------------------------------------------------------------------------------------- windows
@dataclass
class WindowConfig:
    window_ms: float = C.WINDOW_MS
    stride_ms: float = C.STRIDE_MS
    fs: float = C.FS_EMG_DEFAULT
    imu_hz: float = C.IMU_HZ_NOMINAL
    min_emg_coverage: float = C.MIN_EMG_COVERAGE
    min_imu_fraction: float = C.MIN_IMU_FRACTION

    @property
    def n_emg(self) -> int:
        return int(round(self.fs * self.window_ms / 1000))

    @property
    def stride(self) -> int:
        return int(round(self.fs * self.stride_ms / 1000))

    @property
    def n_imu(self) -> int:
        return int(round(self.imu_hz * self.window_ms / 1000))

    @property
    def imu_step(self) -> float:
        return self.fs / self.imu_hz        # IMU period in sample-clock units (10 at 500/50 Hz)

    @property
    def min_imu_samples(self) -> int:
        return max(2, int(np.ceil(self.n_imu * self.min_imu_fraction)))

    def to_dict(self):
        return {"window_ms": self.window_ms, "stride_ms": self.stride_ms, "fs": self.fs, "imu_hz": self.imu_hz,
                "min_emg_coverage": self.min_emg_coverage, "min_imu_fraction": self.min_imu_fraction,
                "n_emg": self.n_emg, "n_imu": self.n_imu, "stride_samples": self.stride}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("window_ms", "stride_ms", "fs", "imu_hz", "min_emg_coverage", "min_imu_fraction") if k in d})


@dataclass
class Window:
    end_clock: int
    emg_grid: np.ndarray        # [8, n_emg]  filtered EMG on the regular sample grid (end-n+1 .. end)
    imu_grid: np.ndarray        # [6, n_imu]  IMU on a regular 50 Hz grid ending at end (causal sample-hold)
    emg_actual: np.ndarray      # [k, 8] filtered EMG samples really received inside the window
    imu_actual: np.ndarray      # [j, 6] IMU samples really received inside the window
    coverage: float
    valid: bool
    reason: str = ""


def extract_window(emg_clock: np.ndarray, emg_filt: np.ndarray, imu_clock: np.ndarray, imu_vals: np.ndarray,
                   end_clock: int, cfg: WindowConfig, need_imu: bool = True) -> Window:
    """Window covering sample-clock interval (end - n_emg, end]. EMG and IMU are selected by their own
    clock values (never by array position), so both streams cover the same time span.
    Missing EMG samples (packet loss) are linearly interpolated on the grid; the window is invalid if
    coverage < min_emg_coverage. IMU grid uses the latest IMU sample at or before each grid time (causal)."""
    n = cfg.n_emg
    lo = np.searchsorted(emg_clock, end_clock - n + 1)
    hi = np.searchsorted(emg_clock, end_clock, side="right")
    actual = emg_filt[lo:hi]
    k = hi - lo
    coverage = k / n
    reason = ""
    if k < 2:
        return Window(end_clock, np.zeros((C.N_EMG, n)), np.zeros((C.N_IMU, cfg.n_imu)), actual,
                      np.zeros((0, C.N_IMU)), coverage, False, "no EMG")
    if k == n:
        grid = actual.T.copy()
    else:
        t = np.arange(end_clock - n + 1, end_clock + 1)
        tc = emg_clock[lo:hi]
        grid = np.stack([np.interp(t, tc, actual[:, c]) for c in range(actual.shape[1])])
    valid = coverage >= cfg.min_emg_coverage
    if not valid:
        reason = f"EMG coverage {coverage:.2f}"

    ilo = np.searchsorted(imu_clock, end_clock - n + 1)
    ihi = np.searchsorted(imu_clock, end_clock, side="right")
    imu_actual = imu_vals[ilo:ihi]
    m = cfg.n_imu
    gt = end_clock - (m - 1 - np.arange(m)) * cfg.imu_step
    idx = np.searchsorted(imu_clock, gt, side="right") - 1
    if len(imu_clock) == 0 or idx[0] < 0:
        imu_grid = np.zeros((C.N_IMU, m))
        imu_ok = False
    else:
        imu_grid = imu_vals[idx].T.copy()
        stale = (gt - imu_clock[idx]).max()
        imu_ok = stale <= 3 * cfg.imu_step and len(imu_actual) >= cfg.min_imu_samples
    if need_imu and not imu_ok:
        valid = False
        reason = (reason + "; " if reason else "") + f"IMU samples {len(imu_actual)}"
    return Window(end_clock, grid, imu_grid, actual, imu_actual, coverage, valid, reason)


# --------------------------------------------------------------------------------------------- features (legacy)
EMG_FEATURES = [f"emg_{k}_ch{c}" for k in ("rms", "mav", "wl", "std") for c in range(1, 9)]
IMU_FEATURES = ([f"gyro_{a}_mean" for a in "xyz"] + [f"gyro_{a}_std" for a in "xyz"] + ["gyro_mag_mean"]
                + [f"acc_{a}_mean" for a in "xyz"] + [f"acc_{a}_std" for a in "xyz"])
ALL_FEATURES = EMG_FEATURES + IMU_FEATURES


def emg_features(x: np.ndarray) -> np.ndarray:
    """x: [n, 8] filtered EMG -> RMS, MAV, WL, STD per channel (legacy features.py definitions)."""
    rms = np.sqrt((x ** 2).mean(0))
    mav = np.abs(x).mean(0)
    wl = np.abs(np.diff(x, axis=0)).sum(0)
    std = x.std(0)
    return np.concatenate([rms, mav, wl, std])


def imu_features(v: np.ndarray) -> np.ndarray:
    """v: [j, 6] (acc xyz, gyro xyz) -> 13 legacy IMU statistics."""
    if len(v) < 2:
        return np.full(13, np.nan)
    acc, gyro = v[:, :3], v[:, 3:]
    mag = np.sqrt((gyro ** 2).sum(1))
    return np.concatenate([gyro.mean(0), gyro.std(0), [mag.mean()], acc.mean(0), acc.std(0)])


def window_features(w: Window) -> np.ndarray:
    return np.concatenate([emg_features(w.emg_actual), imu_features(w.imu_actual)])


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(x) ** 2).mean(0)) if len(x) else np.zeros(C.N_EMG)
