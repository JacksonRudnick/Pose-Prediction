"""Rotation experiment: fit input-window delay and sensitivity from mouse
counts, and score input-based rotation prediction vs history baselines.

The delay is a pair (n, delta): frame i's window is (t[i-n-1] - delta,
t[i-n] - delta].  Per-tick window + shift by n rows.

Window clock: InputSamplingTimestamp (default) or TickTimestamp/BeginFrameTimestamp.
State columns are shifted back one row so row i = frame i's pose (n=0 expected).

History baselines see only real frames (--hist_stride default 2); const-vel,
mt-2, mt-5.  Split: random SPLIT_BLOCK-frame blocks (default) or last 20%.
"""

import argparse
import os
import platform
import time
from bisect import bisect_right

import numpy as np
import fgcap

# the logged state columns (everything except the frame counter and clocks)
# are moved back one row so each row holds its own frame's pose
SHIFT_STATE = True
WINDOW_CLOCK = "InputSamplingTimestamp"
CLOCKS = ("InputSamplingTimestamp", "TickTimestamp", "BeginFrameTimestamp")

N_TICKS = (0, 1, 2)
DELTA_RANGE_MS = 2.0
FINE_STEP_MS = 0.005
EDGE_N_TICKS = (0, 1, 2, 3)
EDGE_DELTA_RANGE_MS = 8.0
EDGE_STEP_MS = 0.5
EDGE_WARN_MS = 0.1

CLAMP_DEG = 87.0
SKIP_START_FRAMES = 5
TRAIN_FRAC = 0.8
HITCH_TOL_MS = 2.0
HIST_TAPS = 5
K_REF = 0.1747
DROP_MARGIN_MS = 2.0
OUTLIER_DEG = 5.0

assert HIST_TAPS >= max(N_TICKS) + 1


def hist_span(taps=HIST_TAPS):
    return taps * HIST_STRIDE

HIST_STRIDE = 2
SPLIT_MODE = "blocks"
SPLIT_BLOCK = 60
SPLIT_SEED = 0
VAL_FRAC = 1.0 - TRAIN_FRAC

N_BOOT = 10000
SEED = 0
JERK_BUCKETS = {"steady": (0.0, 1.0), "moderate": (1.0, 5.0),
                "reversal": (5.0, np.inf)}


def read_frame_log(cap_dir, state_cols, shift=SHIFT_STATE):
    name = [n for n in os.listdir(cap_dir)
            if n.startswith("FrameLog_") and n.endswith(".csv")][0]
    data = np.genfromtxt(os.path.join(cap_dir, name), delimiter=",", names=True)
    clocks = [c for c in CLOCKS if c in data.dtype.names]
    data = data[['GFrameCounter', *clocks, *state_cols]].copy()
    if shift:
        nxt_ok = np.diff(data['GFrameCounter']) == 1
        out = data[:-1].copy()
        for c in state_cols:
            v = np.asarray(data[c][1:], dtype=np.float64).copy()
            v[~nxt_ok] = np.nan
            out[c] = v
        data = out
    return data


def get_frame_log_data(cap_dir, shift=SHIFT_STATE):
    return read_frame_log(cap_dir, ['CamRotYaw', 'CamRotPitch', 'MouseAngX', 'MouseAngY'],
                          shift)


def window_times(log, clock=WINDOW_CLOCK):
    if clock not in log.dtype.names:
        raise ValueError(f"frame log has no {clock} column (has: "
                         f"{', '.join(c for c in CLOCKS if c in log.dtype.names)})")
    t = np.asarray(log[clock], dtype=np.float64)
    bad = ~np.isfinite(t) | (t <= 0)
    if bad.any():
        raise ValueError(f"{clock} has {int(bad.sum())} missing or zero values")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{clock} is not strictly increasing")
    return t


def clock_offsets(log, clock):
    t = np.asarray(log[clock], dtype=np.float64)
    out = {}
    for c in CLOCKS:
        if c != clock and c in log.dtype.names:
            out[c] = (np.asarray(log[c], dtype=np.float64) - t) * 1e3
    return out


def get_input_reports(cap_dir):
    """Returns (reports sorted by time, timestamps of SYN_DROPPED events)."""
    name = [n for n in os.listdir(cap_dir)
            if n.startswith("InputLog_") and n.endswith(".bin")][0]
    events = fgcap.read_inputlog(os.path.join(cap_dir, name))
    dropped = (events["type"] == fgcap.EV_SYN) & (events["code"] == 3)
    drop_t = np.sort(events["t"][dropped].astype(np.float64))
    if len(drop_t):
        print(f"WARNING: {len(drop_t)} SYN_DROPPED events -- mouse counts were lost")
    rep = fgcap.mouse_reports(events)
    return rep[np.argsort(rep["t"], kind="stable")], drop_t


def improvement(base_err, meth_err):
    mb = np.mean(base_err)
    return 100.0 * (1.0 - np.mean(meth_err) / mb) if mb > 0 else 0.0


def split_rows(rows_all, n_frames, mode=None, seed=None):
    mode = SPLIT_MODE if mode is None else mode
    seed = SPLIT_SEED if seed is None else seed
    if mode == "last":
        cut = int(len(rows_all) * TRAIN_FRAC)
        return rows_all[:cut], rows_all[cut:]
    if mode != "blocks":
        raise ValueError(f"unknown split mode {mode!r}")
    n_blocks = int(np.ceil(n_frames / SPLIT_BLOCK))
    perm = np.random.default_rng(seed).permutation(n_blocks)
    val_block = np.zeros(n_blocks, dtype=bool)
    val_block[perm[:int(round(VAL_FRAC * n_blocks))]] = True
    is_val = val_block[rows_all // SPLIT_BLOCK]
    return rows_all[~is_val], rows_all[is_val]


def block_of(rows):
    return np.asarray(rows) // SPLIT_BLOCK


def _resampled_sums(values, groups, rng, n_boot):
    _, g = np.unique(groups, return_inverse=True)
    G = int(g.max()) + 1
    per = np.zeros((G, values.shape[1]))
    np.add.at(per, g, values)
    out = np.empty((n_boot, values.shape[1]))
    chunk = 1000
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        draws = rng.integers(0, G, (m, G))
        cnt = np.zeros((m, G))
        np.add.at(cnt, (np.repeat(np.arange(m), G), draws.ravel()), 1.0)
        out[s:s + m] = cnt @ per
    return out


def block_bootstrap(base_err, meth_err, groups, rng, n_boot=N_BOOT):
    sums = _resampled_sums(np.stack([base_err, meth_err], 1), groups, rng, n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (1.0 - sums[:, 1] / sums[:, 0])


def block_bootstrap_diff(base_err, meth_err, groups, rng, n_boot=N_BOOT):
    v = np.stack([meth_err - base_err, np.ones(len(base_err))], 1)
    sums = _resampled_sums(v, groups, rng, n_boot)
    return sums[:, 0] / sums[:, 1]


def history_pred(d, taps, stride=None):
    R = HIST_STRIDE if stride is None else int(stride)
    d = np.asarray(d, dtype=np.float64)
    one_d = d.ndim == 1
    d2 = np.nan_to_num(d.reshape(len(d), -1))
    n = len(d2)
    cs = np.vstack([np.zeros((1, d2.shape[1])), np.cumsum(d2, axis=0)])
    tau = -R / 2.0 - R * np.arange(taps)          # centres, relative to the end of i-1
    if taps == 1:
        w = np.ones(1)
    else:
        tc = tau - tau.mean()
        w = 1.0 / taps + tc * (0.5 - tau.mean()) / np.sum(tc ** 2)
    out = np.zeros_like(d2)
    i = np.arange(taps * R, n)
    for j in range(taps):
        out[i] += w[j] * (cs[i - j * R] - cs[i - (j + 1) * R]) / R
    return out[:, 0] if one_d else out


def real_velocity(d, j=0, stride=None):
    R = HIST_STRIDE if stride is None else int(stride)
    d = np.asarray(d, dtype=np.float64)
    one_d = d.ndim == 1
    d2 = np.nan_to_num(d.reshape(len(d), -1))
    n = len(d2)
    cs = np.vstack([np.zeros((1, d2.shape[1])), np.cumsum(d2, axis=0)])
    out = np.zeros_like(d2)
    i = np.arange((j + 1) * R, n)
    out[i] = (cs[i - j * R] - cs[i - (j + 1) * R]) / R
    return out[:, 0] if one_d else out


def mt2_coef(stride=None):
    R = HIST_STRIDE if stride is None else int(stride)
    return (R + 1) / (2.0 * R)


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def per_frame_delta(angle):
    angle = np.asarray(angle, dtype=np.float64)
    d = np.zeros(len(angle))
    d[1:] = wrap_deg(angle[1:] - angle[:-1])
    return d


def window_counts(frame_t, report_t, cum_counts, D_s):
    edges = np.searchsorted(report_t, frame_t + D_s, side="right")
    c = np.zeros(len(frame_t))
    c[1:] = cum_counts[edges[1:]] - cum_counts[edges[:-1]]
    return c


def shift_rows(c, n):
    c = np.asarray(c)
    if n == 0:
        return c.copy()
    out = np.zeros_like(c)
    if n > 0:
        out[n:] = c[:-n]
    else:
        out[:n] = c[-n:]
    return out


def window_counts_nd(frame_t, report_t, cum_counts, n, delta_ms):
    return shift_rows(window_counts(frame_t, report_t, cum_counts, -delta_ms * 1e-3), n)


def delta_grid(range_ms, step_ms):
    k = int(round(range_ms / step_ms))
    return np.round(np.arange(-k, k + 1) * step_ms, 9)


def candidate_grid(n_ticks, deltas_ms):
    cand_n = np.repeat(np.asarray(n_ticks, dtype=int), len(deltas_ms))
    cand_d = np.tile(np.asarray(deltas_ms, dtype=np.float64), len(n_ticks))
    return cand_n, cand_d


def candidate_counts(frame_t, report_t, cum_counts, n_ticks, deltas_ms):
    base = np.stack([window_counts(frame_t, report_t, cum_counts, -d * 1e-3)
                     for d in deltas_ms], 1)
    return np.concatenate([shift_rows(base, n) for n in n_ticks], axis=1)


# ── delay fit ──────────────────────────────────────────────────────────────

def get_delay(frame_t, target, rows, report_t, report_counts, n_ticks, deltas_ms,
              sse_tol=1e-3):
    frame_t = np.asarray(frame_t, dtype=np.float64)
    C = np.concatenate([[0.0], np.cumsum(np.asarray(report_counts, np.float64))])
    y = target[rows]
    yy = float(y @ y)
    n_ticks = tuple(int(x) for x in n_ticks)
    deltas_ms = np.asarray(deltas_ms, dtype=np.float64)
    assert rows.min() - max(n_ticks) - 1 >= 0

    cand_n, cand_d = candidate_grid(n_ticks, deltas_ms)
    nd = len(deltas_ms)
    yc = np.zeros(len(cand_n))
    cc = np.zeros(len(cand_n))
    for di, d in enumerate(deltas_ms):
        base = window_counts(frame_t, report_t, C, -d * 1e-3)
        for ni, nt in enumerate(n_ticks):
            c = base[rows - nt]                 # shifted window, on the fit rows
            yc[ni * nd + di] = float(y @ c)
            cc[ni * nd + di] = float(c @ c)
    scores = np.where(cc > 0, yc ** 2 / np.where(cc > 0, cc, 1.0), 0.0)   # = yy - min SSE

    best = int(np.argmax(scores))
    n_best, d_best = int(cand_n[best]), float(cand_d[best])
    k = yc[best] / cc[best]
    sse = yy - scores

    # how clearly the best n wins: best R2 reachable at each n
    r2_by_n = {nt: 1.0 - (yy - scores[cand_n == nt].max()) / yy for nt in n_ticks}
    # plateau over the delta values of the best n only
    sel = cand_n == n_best
    on_plateau = sel & ((sse - sse[best]) <= sse_tol * abs(sse[best]) + 1e-12 * yy)
    return {
        "n": n_best,
        "delta_ms": d_best,
        "k": float(k),
        "r2": 1.0 - sse[best] / yy,
        "r2_by_n": r2_by_n,
        "plateau_ms": (float(cand_d[on_plateau].min()), float(cand_d[on_plateau].max())),
        "n_rows": int(len(rows)),
        "cand_n": cand_n,
        "cand_d": cand_d,
        "sse": sse,
        "yy": yy,
    }


def describe_fit(f, deltas_ms):
    """One-line summary of a get_delay result, with the edge warning."""
    others = sorted(((r2, nt) for nt, r2 in f["r2_by_n"].items() if nt != f["n"]),
                    reverse=True)
    margin = (f"; next best n={others[0][1]} R2 {others[0][0]:.4f}" if others else "")
    edge = ""
    lim = float(np.max(np.abs(deltas_ms)))
    if abs(f["delta_ms"]) > lim - EDGE_WARN_MS:
        edge = f"   WARNING: best delta is at the edge of the +/-{lim:g} ms grid"
    return (f"n = {f['n']}, delta = {f['delta_ms']:+.3f} ms, k = {f['k']:+.5f}, "
            f"R2 = {f['r2']:.5f}{margin}{edge}")


def resolve_seam(target, counts, k):
    pred = k * counts
    cands = np.stack([target, target + 360.0, target - 360.0])
    pick = np.argmin(np.abs(cands - pred), axis=0)
    fixed = cands[pick, np.arange(len(target))]
    return fixed, int(np.sum(pick != 0))


def history_ok(delta_ok, taps):
    n = len(delta_ok)
    cs = np.concatenate([[0], np.cumsum(delta_ok.astype(np.int64))])
    out = np.zeros(n, dtype=bool)
    i = np.arange(taps, n)
    out[i] = (cs[i] - cs[i - taps]) == taps
    return out


TIMING_WARMUP = 200


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def time_per_call(fn, arg_list, warmup=TIMING_WARMUP):
    clock = time.perf_counter_ns
    ts = []
    for j, args in enumerate(arg_list):
        t0 = clock()
        fn(*args)
        dt = clock() - t0
        if j >= warmup:
            ts.append(dt)
    ts = np.asarray(ts, dtype=np.float64) / 1e3
    return float(np.median(ts)), float(np.percentile(ts, 99)), float(ts.max())


def timing_report(t, rows, report_t, Cx, Cy, nx, dx_s, ny, dy_s, kx, ky, dyaw, dpitch):
    rt = report_t.tolist()
    cx_cum, cy_cum = Cx.tolist(), Cy.tolist()

    def input_step(tx0, tx1, ty0, ty1):
        # counts in (t[i-n-1] - delta, t[i-n] - delta] on each axis, times k
        lo = bisect_right(rt, tx0 - dx_s)
        hi = bisect_right(rt, tx1 - dx_s)
        yaw = kx * (cx_cum[hi] - cx_cum[lo])
        lo = bisect_right(rt, ty0 - dy_s)
        hi = bisect_right(rt, ty1 - dy_s)
        pitch = ky * (cy_cum[hi] - cy_cum[lo])
        return yaw, pitch

    R = HIST_STRIDE
    c2 = mt2_coef()

    # inputs: the yaw / pitch of the last real frames (unwrapped)
    def mt2_step(y0, y1, y2, p0, p1, p2):
        uy0, uy1 = (y0 - y1) / R, (y1 - y2) / R
        up0, up1 = (p0 - p1) / R, (p1 - p2) / R
        return uy0 + c2 * (uy0 - uy1), up0 + c2 * (up0 - up1)

    def cv_step(y0, y1, p0, p1):
        return (y0 - y1) / R, (p0 - p1) / R

    tl = t.tolist()
    # unwrapped angles: running sum of the per-frame deltas
    dy = np.concatenate([[0.0], np.cumsum(np.nan_to_num(dyaw))]).tolist()
    dp = np.concatenate([[0.0], np.cumsum(np.nan_to_num(dpitch))]).tolist()
    args_in = [(tl[i - nx - 1], tl[i - nx], tl[i - ny - 1], tl[i - ny]) for i in rows]

    ref = np.stack([kx * window_counts_nd(t, report_t, Cx, nx, dx_s * 1e3)[rows],
                    ky * window_counts_nd(t, report_t, Cy, ny, dy_s * 1e3)[rows]], 1)
    got = np.array([input_step(*a) for a in args_in])
    max_diff = float(np.max(np.abs(got - ref)))
    # angle at the end of frame f = dy[f + 1]
    args_m2 = [(dy[i], dy[i - R], dy[i - 2 * R], dp[i], dp[i - R], dp[i - 2 * R])
               for i in rows]
    args_cv = [(dy[i], dy[i - R], dp[i], dp[i - R]) for i in rows]
    frame_us = 1e6 * float(np.median(np.diff(t)))

    print(f"\nper-frame cost (Python, one call per frame, {len(rows) - TIMING_WARMUP} frames "
          f"after {TIMING_WARMUP} warm-up; {cpu_name()})")
    print(f"  {'predictor':<12}{'median us':>11}{'p99 us':>10}{'max us':>10}"
          f"{'% of frame (p99)':>18}")
    for name, fn, args in (("input", input_step, args_in),
                           ("const-vel", cv_step, args_cv),
                           ("mt-2", mt2_step, args_m2)):
        med, p99, mx = time_per_call(fn, args)
        print(f"  {name:<12}{med:>11.2f}{p99:>10.2f}{mx:>10.2f}{100 * p99 / frame_us:>17.3f}%")
    print(f"  frame interval {frame_us:.0f} us.  'input' = two window lookups (binary "
          f"search over the report times) and two multiplies.")
    print(f"  per-frame input step matches the vectorized prediction to {max_diff:.2e} deg")


def add_common_args(ap):
    ap.add_argument("--hist_stride", type=int, default=HIST_STRIDE,
                    help=f"history baselines use every this-many-th frame as real "
                         f"(default {HIST_STRIDE}; 1 = every frame, the old baselines)")
    ap.add_argument("--split", choices=("blocks", "last"), default=SPLIT_MODE,
                    help=f"train/val split: random {SPLIT_BLOCK}-frame blocks (default) "
                         f"or the last 20%% of usable rows (old)")
    ap.add_argument("--split_seed", type=int, default=SPLIT_SEED,
                    help=f"seed for the random block split (default {SPLIT_SEED})")


def apply_common_args(a):
    global HIST_STRIDE, SPLIT_MODE, SPLIT_SEED
    if a.hist_stride < 1:
        raise SystemExit("--hist_stride must be >= 1")
    HIST_STRIDE, SPLIT_MODE, SPLIT_SEED = a.hist_stride, a.split, a.split_seed


def run(cap_dir, quiet=False, timing=False, hitch_filter=True, clock=WINDOW_CLOCK,
        shift=SHIFT_STATE):
    log = get_frame_log_data(cap_dir, shift)
    reports, drop_t = get_input_reports(cap_dir)

    t = window_times(log, clock)
    fid = np.asarray(log['GFrameCounter'], dtype=np.int64)
    n = len(log)
    dt_med = float(np.median(np.diff(t)))
    rt = reports["t"].astype(np.float64)
    deltas = delta_grid(DELTA_RANGE_MS, FINE_STEP_MS)
    edge_deltas = delta_grid(EDGE_DELTA_RANGE_MS, EDGE_STEP_MS)
    n_max = max(N_TICKS)

    # ---- per-frame targets: change since previous frame, wrapped ----
    dyaw = per_frame_delta(log['CamRotYaw'])
    dpitch = per_frame_delta(log['CamRotPitch'])
    Cx = np.concatenate([[0.0], np.cumsum(reports["dx"].astype(np.float64))])
    Cy = np.concatenate([[0.0], np.cumsum(reports["dy"].astype(np.float64))])

    # ---- fixed row selection (decided once, never depends on the delay) ----
    ok = np.isfinite(t) & np.isfinite(log['CamRotYaw']) & np.isfinite(log['CamRotPitch'])
    prev_ok = np.zeros(n, dtype=bool)
    prev_ok[1:] = (fid[1:] - fid[:-1] == 1) & ok[:-1]          # real previous frame

    clamp = np.abs(wrap_deg(log['CamRotPitch'])) > CLAMP_DEG
    no_clamp = np.zeros(n, dtype=bool)
    no_clamp[1:] = ~clamp[1:] & ~clamp[:-1]                    # both ends free

    # hitch: the interval ending at i and the one ending at i-1 must be normal
    long_int = np.zeros(n, dtype=bool)
    long_int[1:] = np.abs(np.diff(t) - dt_med) > HITCH_TOL_MS * 1e-3
    no_hitch = ~long_int
    no_hitch[1:] &= ~long_int[:-1]
    if not hitch_filter:
        no_hitch = np.ones(n, dtype=bool)

    # teleport / spawn: big rotation with almost no input, checked once at a
    # fixed reference window, independent of the search: the frame's own
    # input read (n = 0) when the state is shifted, one row back (n = 1) when
    # the rows are as logged
    n_ref = 0 if shift else 1
    c0x = window_counts_nd(t, rt, Cx, n_ref, 0.0)
    c0y = window_counts_nd(t, rt, Cy, n_ref, 0.0)
    rot = np.hypot(dyaw, dpitch)
    cnt = np.hypot(c0x, c0y)
    teleport = (np.abs(rot - K_REF * cnt) > 5.0) & (cnt < 30)

    # lost input: a SYN_DROPPED event inside any candidate's window (or within
    # DROP_MARGIN_MS of it) means that candidate's counts are incomplete.  The
    # span covers every candidate: from t[i - n_max - 1] - DELTA_RANGE to
    # t[i] + DELTA_RANGE, so no candidate is scored on lost input.
    near_drop = np.zeros(n, dtype=bool)
    if len(drop_t):
        m = DROP_MARGIN_MS * 1e-3 + DELTA_RANGE_MS * 1e-3
        i = np.arange(n)
        t_lo = t[np.clip(i - n_max - 1, 0, n - 1)] - m
        t_hi = t + m
        lo = np.searchsorted(drop_t, t_lo, side="right")
        hi = np.searchsorted(drop_t, t_hi, side="right")
        near_drop = hi > lo
        near_drop[:n_max + 1] = True

    delta_ok = ok & prev_ok & no_clamp & ~teleport             # genuine rotation delta
    usable = (delta_ok & no_hitch & ~near_drop
              & history_ok(delta_ok, hist_span()))
    usable[:max(SKIP_START_FRAMES, n_max + 1)] = False

    rows_all = np.flatnonzero(usable)
    fit_rows, val_rows = split_rows(rows_all, n)
    print(f"state columns {'moved back one row (row i = frame i)' if shift else 'as logged'}")
    print(f"history baselines on real frames every {HIST_STRIDE} frame(s); split "
          + (f"random {SPLIT_BLOCK}-frame blocks, {VAL_FRAC:.0%} to validation, seed {SPLIT_SEED}"
             if SPLIT_MODE == "blocks" else "last 20% of usable rows"))
    print(f"window clock: {clock} (tick interval median {dt_med * 1e3:.3f} ms, "
          f"p1-p99 {np.percentile(np.diff(t), 1) * 1e3:.3f}-"
          f"{np.percentile(np.diff(t), 99) * 1e3:.3f} ms)")
    for c, off in clock_offsets(log, clock).items():
        print(f"  {c} - {clock}: median {np.median(off):+.3f} ms, "
              f"p1-p99 {np.percentile(off, 1):+.3f} .. {np.percentile(off, 99):+.3f} ms")
    print(f"frames {n}, usable {len(rows_all)} (fit {len(fit_rows)}, val {len(val_rows)})")
    print(f"  excluded: gaps {int(np.sum(~prev_ok[1:]))}, clamp {int(np.sum(~no_clamp[1:]))}, "
          f"hitch {int(np.sum(~no_hitch))}{'' if hitch_filter else ' (filter off)'}, "
          f"teleport {int(np.sum(teleport & ~clamp))}, "
          f"near SYN_DROPPED {int(np.sum(near_drop[n_max + 1:]))}")

    # ---- yaw seam fix (pitch can't move >180 deg in one tick, so no fix) ----
    # uses the wide, coarse grid: it only needs a rough k
    coarse_yaw = get_delay(t, dyaw, fit_rows, rt, reports["dx"], EDGE_N_TICKS, edge_deltas)
    cx_coarse = window_counts_nd(t, rt, Cx, coarse_yaw["n"], coarse_yaw["delta_ms"])
    dyaw_fixed, _ = resolve_seam(dyaw, cx_coarse, coarse_yaw["k"])
    changed = dyaw_fixed != dyaw
    print(f"  yaw seam fixes: fit {int(np.sum(changed[fit_rows]))}, "
          f"val {int(np.sum(changed[val_rows]))}")

    targets = {"Yaw": dyaw_fixed, "Pitch": dpitch}
    counts_cum = {"Yaw": Cx, "Pitch": Cy}
    count_field = {"Yaw": "dx", "Pitch": "dy"}
    fits = {}

    for axis in ("Yaw", "Pitch"):
        y = targets[axis]
        rc = reports[count_field[axis]]
        a = np.abs(y[fit_rows])
        print(f"\n[{axis}] |delta| on fit rows: mean {a.mean():.3f}, median "
              f"{np.median(a):.3f}, p99 {np.percentile(a, 99):.2f}, max {a.max():.1f} deg")

        edge = get_delay(t, y, fit_rows, rt, rc, EDGE_N_TICKS, edge_deltas)
        fine = get_delay(t, y, fit_rows, rt, rc, N_TICKS, deltas)
        fits[axis] = fine

        print(f"  wide check (n {EDGE_N_TICKS}, delta +/-{EDGE_DELTA_RANGE_MS:g} ms "
              f"step {EDGE_STEP_MS}): n = {edge['n']}, delta = {edge['delta_ms']:+.1f} ms, "
              f"R2 = {edge['r2']:.4f}")
        if edge["n"] != fine["n"] or abs(edge["delta_ms"]) > DELTA_RANGE_MS:
            print(f"  WARNING: the wide check's best is outside the main grid")
        print(f"  fit:   {describe_fit(fine, deltas)}")
        print(f"         plateau delta [{fine['plateau_ms'][0]:+.3f}, "
              f"{fine['plateau_ms'][1]:+.3f}] ms (SSE within 0.1% of the minimum)")
        print("         best R2 at each n: " +
              ", ".join(f"n={nt}: {r2:.5f}" for nt, r2 in fine["r2_by_n"].items()))

        # the 10 largest deltas on fit rows, with the counts that drove them
        c_best = window_counts_nd(t, rt, counts_cum[axis], fine["n"], fine["delta_ms"])
        top = fit_rows[np.argsort(np.abs(y[fit_rows]))[-10:][::-1]]
        print("  largest deltas (row: delta deg | k*counts deg):")
        for r in top:
            print(f"    {r:6d}: {y[r]:+9.3f} | {fine['k'] * c_best[r]:+9.3f}")

    # ---- one delay for both axes (used by the position script) ----
    # Each axis's error is divided by its own total, so yaw (larger angles)
    # does not drown out pitch.  Equivalent to minimising (1-R2_yaw)+(1-R2_pitch).
    joint = fits["Yaw"]["sse"] / fits["Yaw"]["yy"] + fits["Pitch"]["sse"] / fits["Pitch"]["yy"]
    jb = int(np.argmin(joint))
    n_joint = int(fits["Yaw"]["cand_n"][jb])
    d_joint = float(fits["Yaw"]["cand_d"][jb])
    print(f"\njoint delay (yaw + pitch): n = {n_joint}, delta = {d_joint:+.3f} ms")
    result = {"n_joint": n_joint, "delta_joint_ms": d_joint,
              "n_yaw": fits["Yaw"]["n"], "delta_yaw_ms": fits["Yaw"]["delta_ms"],
              "n_pitch": fits["Pitch"]["n"], "delta_pitch_ms": fits["Pitch"]["delta_ms"],
              "k_yaw": fits["Yaw"]["k"], "k_pitch": fits["Pitch"]["k"],
              "fit_rows": fit_rows, "val_rows": val_rows, "clock": clock, "shift": shift}
    if quiet:
        return result

    nx, dxm = fits["Yaw"]["n"], fits["Yaw"]["delta_ms"]
    ny, dym = fits["Pitch"]["n"], fits["Pitch"]["delta_ms"]
    cx = window_counts_nd(t, rt, Cx, nx, dxm)
    cy = window_counts_nd(t, rt, Cy, ny, dym)
    kx, ky = fits["Yaw"]["k"], fits["Pitch"]["k"]

    # ---- verification: rows the input does not explain ----
    # Checked over every genuine delta (not only fit/val rows), plus every row
    # near a SYN_DROPPED event.  MouseAng is printed at i-1, i, i+1 so you can
    # see which tick the engine's own controller rotation lines up with.
    #   camera delta ~= MouseAng(i)  and  k*counts differs -> input side (lost counts?)
    #   camera delta != MouseAng     and  MouseAng ~= k*counts -> camera got extra rotation
    res_yaw = dyaw_fixed - kx * cx
    res_pitch = dpitch - ky * cy
    check = (delta_ok & no_hitch) & ((np.abs(res_yaw) > OUTLIER_DEG)
                                     | (np.abs(res_pitch) > OUTLIER_DEG))
    if len(drop_t):
        check |= near_drop
    check[:max(2, n_max + 1)] = False
    check[-1] = False
    rows_chk = np.flatnonzero(check)
    mx = np.asarray(log['MouseAngX'], dtype=np.float64)
    my = np.asarray(log['MouseAngY'], dtype=np.float64)
    in_fit = np.isin(np.arange(n), fit_rows)
    in_val = np.isin(np.arange(n), val_rows)
    print(f"\nrows not explained by input (|residual| > {OUTLIER_DEG} deg) "
          f"or near SYN_DROPPED: {len(rows_chk)}")
    for r in rows_chk:
        where = "fit" if in_fit[r] else ("val" if in_val[r] else "excluded")
        flag = " SYN_DROPPED" if near_drop[r] else ""
        print(f"  row {r} ({where}{flag}), t = {t[r]:.6f}")
        print(f"    yaw:   camera {dyaw_fixed[r]:+9.3f} | k*counts {kx * cx[r]:+9.3f} | "
              f"MouseAngX[i-1,i,i+1] {mx[r-1]:+9.3f} {mx[r]:+9.3f} {mx[r+1]:+9.3f}")
        print(f"    pitch: camera {dpitch[r]:+9.3f} | k*counts {ky * cy[r]:+9.3f} | "
              f"MouseAngY[i-1,i,i+1] {my[r-1]:+9.3f} {my[r]:+9.3f} {my[r+1]:+9.3f}")
    if len(drop_t):
        print("  SYN_DROPPED times: " + ", ".join(f"{x:.6f}" for x in drop_t))

    # ---- validation: 2D error, input vs history baselines ----
    v = val_rows
    ty, tp = dyaw_fixed[v], dpitch[v]

    pred = {
        "input": (kx * cx[v], ky * cy[v]),
        "const-vel": (history_pred(dyaw_fixed, 1)[v], history_pred(dpitch, 1)[v]),
        "mt-2": (history_pred(dyaw_fixed, 2)[v], history_pred(dpitch, 2)[v]),
    }
    err = {m: np.hypot(py - ty, pp - tp) for m, (py, pp) in pred.items()}

    print(f"\nvalidation, 2D error (deg), {len(v)} rows")
    print(f"  {'method':<10}{'MAE':>10}{'median':>10}{'p99':>10}")
    for m, e in err.items():
        print(f"  {m:<10}{e.mean():>10.4f}{np.median(e):>10.4f}"
              f"{np.percentile(e, 99):>10.4f}")
    for base in ("const-vel", "mt-2"):
        imp = 1.0 - err["input"].mean() / err[base].mean()
        print(f"  input vs {base}: {imp:.1%} lower error")

    # ---- bootstrap CIs per motion bucket ----
    # bucket by the change in the 2D rotation delta entering the target frame
    jerk = np.hypot(dyaw_fixed[v] - dyaw_fixed[v - 1], dpitch[v] - dpitch[v - 1])
    moving = np.hypot(ty, tp) > 0
    groups = {name: (jerk >= lo) & (jerk < hi) for name, (lo, hi) in JERK_BUCKETS.items()}
    groups["moving"] = moving
    groups["all"] = np.ones(len(v), dtype=bool)

    rng = np.random.default_rng(SEED)
    gid = block_of(v)
    print(f"\nbootstrap: input vs stronger baseline per bucket "
          f"(resampling {SPLIT_BLOCK}-frame blocks, {N_BOOT} resamples, seed {SEED})")
    print(f"  {'bucket':<10}{'n':>6}{'const-vel':>11}{'mt-2':>9}{'input':>9}"
          f"{'stronger':>10}{'improvement [95% CI]':>28}")
    for name, sel in groups.items():
        ns = int(sel.sum())
        if ns < 20:
            print(f"  {name:<10}{ns:>6}  (too few rows)")
            continue
        e_cv, e_m2, e_in = err["const-vel"][sel], err["mt-2"][sel], err["input"][sel]
        if e_cv.mean() <= e_m2.mean():
            base_name, e_base = "const-vel", e_cv
        else:
            base_name, e_base = "mt-2", e_m2
        pt = improvement(e_base, e_in)
        st = block_bootstrap(e_base, e_in, gid[sel], rng)
        lo_ci, hi_ci = np.percentile(st, [2.5, 97.5])
        print(f"  {name:<10}{ns:>6}{e_cv.mean():>11.4f}{e_m2.mean():>9.4f}"
              f"{e_in.mean():>9.4f}{base_name:>10}"
              f"{f'{pt:.1f}% [{lo_ci:.1f}, {hi_ci:.1f}]':>28}")
    print("  errors are 2D angular error in degrees; buckets by |delta_i - delta_(i-1)|")

    if timing:
        timing_report(t, v, rt, Cx, Cy, nx, dxm * 1e-3, ny, dym * 1e-3,
                      kx, ky, dyaw_fixed, dpitch)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir")
    ap.add_argument("--timing", action="store_true",
                    help="also measure the per-frame cost of each predictor")
    ap.add_argument("--no_hitch_filter", action="store_true",
                    help="keep rows next to irregular tick intervals")
    ap.add_argument("--clock", choices=CLOCKS, default=WINDOW_CLOCK,
                    help=f"frame-log clock the input windows follow (default {WINDOW_CLOCK})")
    ap.add_argument("--no_state_shift", action="store_true",
                    help="keep the logged rows as they are (pose is one frame behind its row)")
    add_common_args(ap)
    a = ap.parse_args()
    apply_common_args(a)
    run(a.cap_dir, timing=a.timing, hitch_filter=not a.no_hitch_filter, clock=a.clock,
        shift=not a.no_state_shift)