"""Parsers for FrameGenCapture session artifacts (InputLog_*.bin,
FrameLog_*.csv).  No window-building logic."""

import numpy as np

INPUTLOG_MAGIC = 0x4E494746  # 'FGIN'
INPUTLOG_VERSION = 1

EVENT_DTYPE = np.dtype(
    [("t", "<f8"), ("type", "<u2"), ("code", "<u2"), ("value", "<i4")]
)

# evdev constants (linux/input-event-codes.h)
EV_SYN, EV_KEY, EV_REL = 0, 1, 2
REL_X, REL_Y = 0, 1
KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE = 17, 30, 31, 32, 57

EXPECTED_COLUMNS = [
    "FrameIndex", "GFrameCounter", "TickTimestamp", "TimeSeconds",
    "DeltaTime", "Key_W", "Key_A", "Key_S", "Key_D", "Key_Jump",
    "MouseAngX", "MouseAngY", "MouseAngRateX", "MouseAngRateY",
    "VelX", "VelY", "VelZ", "CamAngVelPitch", "CamAngVelYaw",
    "CamLocX", "CamLocY", "CamLocZ", "CamRotPitch", "CamRotYaw",
    "CamRotRoll", "Teleport", "InBurst",
]

# Schema v2 appends columns at the end so v1 indices never move.
# Both names are accepted for the pump anchor; capture-side used
# BeginFrameTimestamp.
OPTIONAL_COLUMNS = ["PumpTimestamp", "BeginFrameTimestamp",
                    "InputSamplingTimestamp",
                    "CharLocX", "CharLocY", "CharLocZ",
                    "CharVelX", "CharVelY", "CharVelZ",
                    "DevZToWorldZ_X", "DevZToWorldZ_Y",
                    "DevZToWorldZ_Z", "DevZToWorldZ_W",
                    "ProjNearClip"]


def anchor_times(log):
    return log["TickTimestamp"]


def read_inputlog(path):
    with open(path, "rb") as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError(f"{path}: truncated header ({len(header)} bytes)")
    magic, version = np.frombuffer(header[:8], dtype="<u4")
    if magic != INPUTLOG_MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08X} (expected FGIN)")
    if version != INPUTLOG_VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    return np.fromfile(path, dtype=EVENT_DTYPE, offset=16)


def mouse_reports(events):
    out_t, out_dx, out_dy = [], [], []
    acc_dx = acc_dy = 0
    acc_t = None
    for ev in events:
        et = ev["type"]
        if et == EV_REL:
            if ev["code"] == REL_X:
                acc_dx += int(ev["value"])
                acc_t = ev["t"]
            elif ev["code"] == REL_Y:
                acc_dy += int(ev["value"])
                acc_t = ev["t"]
        elif et == EV_SYN:
            if acc_t is not None:
                out_t.append(acc_t)
                out_dx.append(acc_dx)
                out_dy.append(acc_dy)
                acc_dx = acc_dy = 0
                acc_t = None
    rep = np.zeros(len(out_t), dtype=[("t", "<f8"), ("dx", "<i8"), ("dy", "<i8")])
    rep["t"], rep["dx"], rep["dy"] = out_t, out_dx, out_dy
    return rep


def key_events(events):
    return events[events["type"] == EV_KEY]


def read_framelog(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    got = list(data.dtype.names)
    missing = [c for c in EXPECTED_COLUMNS if c not in got]
    unknown = [c for c in got
               if c not in EXPECTED_COLUMNS + OPTIONAL_COLUMNS]
    if missing or unknown:
        raise ValueError(
            f"{path}: column mismatch. missing={missing} unknown={unknown}"
        )
    # Reorder: expected columns first (canonical order), then any optional
    # columns that were in the CSV (in the order they appeared).
    optional_present = [c for c in got if c in OPTIONAL_COLUMNS]
    data = data[EXPECTED_COLUMNS + optional_present]
    return data


def window_slices(tick_ts, sample_ts):
    n = len(tick_ts)
    start = np.zeros(n, dtype=np.int64)
    end = np.zeros(n, dtype=np.int64)
    start[1:] = np.searchsorted(sample_ts, tick_ts[:-1], side="right")
    end[1:] = np.searchsorted(sample_ts, tick_ts[1:], side="right")
    return start, end


def window_sum(tick_ts, sample_ts, values, shift=0.0):
    start, end = window_slices(tick_ts + shift, sample_ts)
    c = np.concatenate([[0], np.cumsum(values)])
    return c[end] - c[start]


def per_frame_sums(tick_ts, reports, shift=0.0):
    return (window_sum(tick_ts, reports["t"], reports["dx"], shift),
            window_sum(tick_ts, reports["t"], reports["dy"], shift))


def _lag1_fit(ang, wsum, row_ok=None):
    a = ang[2:]
    w = wsum[1:-1]
    mask = np.abs(w) > 0
    if row_ok is not None:
        mask &= row_ok[1:-1] & row_ok[2:]
    if mask.sum() < 30 or np.std(a[mask]) < 1e-9:
        return 0.0, -np.inf
    k = float(np.sum(a[mask] * w[mask]) / np.sum(w[mask] ** 2))
    resid = a[mask] - k * w[mask]
    ss_tot = np.sum((a[mask] - a[mask].mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / max(ss_tot, 1e-12)
    return k, float(r2)


def fit_boundary_shift(tick_ts, ang, reports, comp="dx", row_ok=None,
                       shifts_ms=None):
    # Trim reports to the tick range so pre-world events don't
    # contaminate the first block's boundary shift fit.
    lo_t = tick_ts[0]
    hi_t = tick_ts[-1]
    mask = (reports["t"] >= lo_t) & (reports["t"] <= hi_t)
    if not np.any(mask):
        return 0.0, 0.0, -np.inf
    reports = reports[mask]

    if shifts_ms is None:
        shifts_ms = np.arange(-4.0, 20.001, 0.25)
    vals = reports[comp]
    rt = reports["t"]
    best = (0.0, 0.0, -np.inf)

    # Coarse pass: bracket the peak (R^2 is flat between report crossings).
    for s in shifts_ms:
        w = window_sum(tick_ts, rt, vals, shift=s * 1e-3)
        k, r2 = _lag1_fit(ang, w, row_ok)
        if r2 > best[2]:
            best = (s * 1e-3, k, r2)

    # Exact pass: evaluate at every shift in the winning cell where a window
    # boundary lands exactly on a report timestamp.
    step_s = (float(shifts_ms[1] - shifts_ms[0]) / 2.0) * 1e-3
    for s in _breakpoint_shifts(tick_ts, rt, best[0] - step_s, best[0] + step_s):
        w = window_sum(tick_ts, rt, vals, shift=s)
        k, r2 = _lag1_fit(ang, w, row_ok)
        if r2 > best[2]:
            best = (s, k, r2)
    return best


def _breakpoint_shifts(tick_ts, rt, lo, hi):
    parts = []
    for t in tick_ts[1:]:                       # window boundaries
        ia = np.searchsorted(rt, t + lo)
        ib = np.searchsorted(rt, t + hi, side="right")
        if ia < ib:
            parts.append(rt[ia:ib] - t)
    if not parts:
        return np.array([])
    return np.unique(np.concatenate(parts))


def drift_shifted(tick_ts, s0, alpha, t0):
    return tick_ts + s0 + alpha * (tick_ts - t0)


def fit_drift(tick_ts, ang, reports, comp="dx", row_ok=None, n_windows=5):
    n = len(tick_ts)
    n_windows = max(1, min(n_windows, n // 8))
    mid_t = []
    shift = []
    kept = 0
    for i in range(n_windows):
        lo = int(n * i / n_windows)
        hi = int(n * (i + 1) / n_windows)
        row_ok_blk = row_ok[lo:hi] if row_ok is not None else None
        s, k, r2 = fit_boundary_shift(tick_ts[lo:hi], ang[lo:hi], reports,
                                      comp, row_ok_blk)
        # Exclude blocks where the boundary-shift fit is unreliable
        # (low R² means the raw input doesn't explain the control rotation
        # well, e.g. during startup transients).  A threshold of 0.95
        # is conservative: blocks with good data achieve R² > 0.99.
        if r2 >= 0.95:
            mid_t.append(0.5 * (tick_ts[lo] + tick_ts[hi - 1]))
            shift.append(s)
            kept += 1
    if kept < 2:
        # Not enough reliable blocks — return zero drift.
        return 0.0, 0.0, tick_ts[0], 0.0
    mid_t = np.array(mid_t)
    shift = np.array(shift)
    t0 = tick_ts[0]
    x = mid_t - t0
    A = np.vstack([np.ones(len(mid_t)), x]).T
    (s0, alpha), *_ = np.linalg.lstsq(A, shift, rcond=None)
    resid = shift - (s0 + alpha * x)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((shift - shift.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(s0), float(alpha), float(t0), float(r2)


def refine_teleports(log, reports, shift=0.0):
    n = len(log)
    out = np.zeros(n, dtype=bool)
    sdx = window_sum(anchor_times(log), reports["t"], reports["dx"])
    sdy = window_sum(anchor_times(log), reports["t"], reports["dy"])
    pitch = ((log["CamRotPitch"] + 180.0) % 360.0) - 180.0
    clamp = np.abs(pitch) > 87.0
    k_ref = 0.1747
    cnt_thresh = 30
    err_thresh = 5.0
    for r in range(1, n - 1):
        if clamp[r] or clamp[r + 1]:
            continue
        # Check TOTAL rotation magnitude vs TOTAL counts.  The per-axis check
        # is too aggressive: a fast flick can have large yaw counts (724) and
        # small pitch counts (21), with a 7 deg pitch error that is a genuine
        # prediction error, not a teleport.  Exclude only when the total
        # observed rotation is large but the total counts are small.
        total_counts = np.sqrt(sdx[r]**2 + sdy[r]**2)
        total_rot = np.sqrt(log["MouseAngX"][r + 1]**2 + log["MouseAngY"][r + 1]**2)
        total_pred = k_ref * total_counts
        total_err = abs(total_rot - total_pred)
        if total_err > err_thresh and total_counts < cnt_thresh:
            out[r] = True
    return out