"""Position experiment: predict dpos[i] = CharLoc[i] - CharLoc[i-1]

Input model (keys + camera yaw, XY only): three-parameter OLS per tick
(move: keep old velocity + turn toward direction + acceleration push;
brake: friction that stops, no reversal).  Z and airborne use mt-2.

Ablations: no yaw, no keys, keys -1/+1.  --train_caps for leave-one-map-out.
"""

import argparse
import contextlib
import io
import math
import os

import numpy as np
import fgcap
import e_rot as check_delay

# evdev key codes: W, A, S, D, Space
KEY_CODES = (17, 30, 31, 32, 57)
KEY_NAMES = ("W", "A", "S", "D", "Jump")
ENGINE_KEY_COLS = ('Key_W', 'Key_A', 'Key_S', 'Key_D', 'Key_Jump')

SKIP_START_FRAMES = 5
HIST_TAPS = 5
TELEPORT_CM = 30.0          # a per-tick move larger than this is a teleport/spawn
AIR_CM = 3.0                # |dz| above this in a recent step -> airborne (--air_cm)
JUMP_WINDOW = 30            # ticks after a Jump press treated as airborne (--jump_window)
NEAR_EDGE = 3               # "near a key change" = within this many frames (--near)
CAP_PCT = 99.0              # speed cap percentile (move rows, train)
POS_BUCKETS = {"steady": (0.0, 0.37), "moderate": (0.37, 1.15),
               "transient": (1.15, np.inf)}

N_BOOT = 10000
SEED = 0


def get_frame_log_data(cap_dir, shift=check_delay.SHIFT_STATE):
    """Same row convention as e_rot (shift=True: row i holds frame i's state)."""
    return check_delay.read_frame_log(
        cap_dir, ['CharLocX', 'CharLocY', 'CharLocZ', 'CamRotYaw', *ENGINE_KEY_COLS], shift)


def get_key_events(cap_dir):
    name = [n for n in os.listdir(cap_dir)
            if n.startswith("InputLog_") and n.endswith(".bin")][0]
    ev = fgcap.read_inputlog(os.path.join(cap_dir, name))
    keep = ((ev["type"] == fgcap.EV_KEY) & np.isin(ev["code"], KEY_CODES)
            & (ev["value"] != 2))
    ev = ev[keep]
    return ev[np.argsort(ev["t"], kind="stable")]


def key_state_at(key_ev, times):
    out = np.zeros((len(times), len(KEY_CODES)))
    for j, code in enumerate(KEY_CODES):
        e = key_ev[key_ev["code"] == code]
        if len(e) == 0:
            continue
        idx = np.searchsorted(e["t"], times, side="right") - 1
        has = idx >= 0
        out[has, j] = (e["value"][idx[has]] == 1).astype(float)
    return out


def key_times(t, n_ticks, delta_s):
    idx = np.clip(np.arange(len(t)) - n_ticks, 0, len(t) - 1)
    return t[idx] - delta_s


def key_edges(key_ev, key_t, codes=KEY_CODES):
    n = len(key_t)
    press = np.zeros(n, dtype=bool)
    release = np.zeros(n, dtype=bool)
    sel = np.isin(key_ev["code"], codes)
    for val, out in ((1, press), (0, release)):
        et = key_ev["t"][sel & (key_ev["value"] == val)]
        c = np.searchsorted(et, key_t, side="right")
        out[1:] = c[1:] > c[:-1]
    return press, release


def frames_since(edge):
    n = np.full(len(edge), 1e6)
    last = -1
    for i, e in enumerate(edge):
        if e:
            last = i
        if last >= 0:
            n[i] = i - last
    return n


def edge_offset_histogram(engine_edge, evdev_edge, max_off=5):
    ev_idx = np.flatnonzero(evdev_edge)
    offs = []
    unmatched = 0
    for i in np.flatnonzero(engine_edge):
        if len(ev_idx) == 0:
            unmatched += 1
            continue
        k = np.searchsorted(ev_idx, i)
        cands = ev_idx[max(k - 1, 0):k + 1] - i
        o = cands[np.argmin(np.abs(cands))]
        if abs(o) <= max_off:
            offs.append(int(o))
        else:
            unmatched += 1
    return offs, unmatched


def history_ok(delta_ok, taps):
    n = len(delta_ok)
    cs = np.concatenate([[0], np.cumsum(delta_ok.astype(np.int64))])
    out = np.zeros(n, dtype=bool)
    i = np.arange(taps, n)
    out[i] = (cs[i] - cs[i - taps]) == taps
    return out


def err3(pred, y):
    return np.linalg.norm(pred - y, axis=1)


def improvement(base_err, meth_err):
    mb = np.mean(base_err)
    return 100.0 * (1.0 - np.mean(meth_err) / mb) if mb > 0 else 0.0


def ci_row(name, base_err, meth_err, groups):
    rng = np.random.default_rng(SEED)
    ns = len(base_err)
    if ns < 20:
        return f"  {name:<24}{ns:>6}  (too few rows)"
    pt = improvement(base_err, meth_err)
    lo, hi = np.percentile(check_delay.block_bootstrap(base_err, meth_err, groups, rng,
                                                       N_BOOT), [2.5, 97.5])
    return (f"  {name:<24}{ns:>6}{base_err.mean():>10.4f}{meth_err.mean():>10.4f}"
            f"{f'{pt:+.1f}% [{lo:+.1f}, {hi:+.1f}]':>26}")


CI_HDR = (f"  {'group':<24}{'n':>6}{'mt-2':>10}{'method':>10}"
          f"{'method vs mt-2 [95% CI]':>26}")


def prepare(cap_dir, n_ticks=None, delta_ms=None, verbose=True,
            clock=check_delay.WINDOW_CLOCK, shift=check_delay.SHIFT_STATE):
    log = get_frame_log_data(cap_dir, shift)
    key_ev = get_key_events(cap_dir)
    if (n_ticks is None) != (delta_ms is None):
        raise ValueError("give both n_ticks and delta_ms, or neither")
    if n_ticks is None:
        with contextlib.redirect_stdout(io.StringIO()):
            mouse = check_delay.run(cap_dir, quiet=True, clock=clock, shift=shift)
        n_ticks, delta_ms = mouse["n_joint"], mouse["delta_joint_ms"]
        if verbose:
            print(f"input delay from mouse counts (e_rot, yaw+pitch): n = {n_ticks}, "
                  f"delta = {delta_ms:+.3f} ms (yaw n={mouse['n_yaw']} "
                  f"{mouse['delta_yaw_ms']:+.3f}, pitch n={mouse['n_pitch']} "
                  f"{mouse['delta_pitch_ms']:+.3f})")
    n_ticks = int(n_ticks)
    delta_s = float(delta_ms) * 1e-3

    t = check_delay.window_times(log, clock)
    fid = np.asarray(log['GFrameCounter'], dtype=np.int64)
    pos = np.stack([log['CharLocX'], log['CharLocY'], log['CharLocZ']], 1).astype(np.float64)
    yaw = np.asarray(log['CamRotYaw'], dtype=np.float64)
    n = len(log)

    dpos = np.zeros_like(pos)
    dpos[1:] = pos[1:] - pos[:-1]

    ok = np.isfinite(t) & np.all(np.isfinite(pos), axis=1) & np.isfinite(yaw)
    prev_ok = np.zeros(n, dtype=bool)
    prev_ok[1:] = (fid[1:] - fid[:-1] == 1) & ok[:-1]
    teleport = np.linalg.norm(dpos, axis=1) > TELEPORT_CM
    delta_ok = ok & prev_ok & ~teleport
    span = check_delay.hist_span(HIST_TAPS)
    usable = delta_ok & history_ok(delta_ok, span)
    # keys come from n ticks back, and the "keys -1" ablation reads n + 1;
    # history_ok guarantees consecutive ticks back to i - span - 1
    assert n_ticks + 2 <= span + 1, "n_ticks too large for the history span"
    usable[:max(SKIP_START_FRAMES, n_ticks + 2)] = False

    rows_all = np.flatnonzero(usable)
    tr, v = check_delay.split_rows(rows_all, n)
    dt_med = float(np.median(np.diff(t)))

    if verbose:
        print(f"frames {n}, usable {len(rows_all)} (fit {len(tr)}, val {len(v)})")
        print(f"  excluded: gaps {int(np.sum(~prev_ok[1:]))}, teleport {int(np.sum(teleport))}")
        per_key = ", ".join(f"{k}:{int(np.sum(key_ev['code'] == c))}"
                            for k, c in zip(KEY_NAMES, KEY_CODES))
        print(f"  key delay n = {n_ticks}, delta = {delta_s * 1e3:+.3f} ms on {clock}, "
              f"key events {len(key_ev)} ({per_key})")

    mt2 = check_delay.history_pred(dpos, 2)

    return dict(name=os.path.basename(os.path.normpath(cap_dir)), log=log, key_ev=key_ev,
                n_ticks=n_ticks, delta_s=delta_s, key_t=key_times(t, n_ticks, delta_s),
                t=t, n=n, dpos=dpos, yaw=yaw,
                rows_all=rows_all, tr=tr, v=v, dt_med=dt_med, mt2=mt2)


def key_direction(keys, yaw_deg, use_yaw=True):
    """World-space intended direction (XY, unit length) and whether a movement
    key is held.  Unreal axes: X forward, Y right, yaw about Z."""
    fwd = keys[:, 0] - keys[:, 2]            # W - S
    right = keys[:, 3] - keys[:, 1]          # D - A
    if use_yaw:
        a = np.radians(yaw_deg)
    else:
        a = np.zeros_like(yaw_deg)
    fx, fy = np.cos(a), np.sin(a)
    rx, ry = -np.sin(a), np.cos(a)
    d = np.stack([fwd * fx + right * rx, fwd * fy + right * ry], 1)
    norm = np.linalg.norm(d, axis=1)
    held = norm > 1e-9
    d[held] /= norm[held, None]
    d[~held] = 0.0
    return d, held


def model_inputs(data, variant, yaw_lag, key_shift=0):
    dpos, t, n = data["dpos"], data["t"], data["n"]
    # key_shift -1 = one tick earlier = one more tick back
    key_t = key_times(t, data["n_ticks"] - key_shift, data["delta_s"])
    keys = key_state_at(data["key_ev"], key_t)
    yaw = data["yaw"]
    if yaw_lag:
        yaw = np.concatenate([[yaw[0]], yaw[:-1]])

    # previous velocity from the last two real frames (per frame)
    u0 = check_delay.real_velocity(dpos, 0)
    u1 = check_delay.real_velocity(dpos, 1)
    prev = u0[:, :2].copy()
    s = np.linalg.norm(prev, axis=1)

    if variant == "no keys":
        held = s > 0.01
        d = np.zeros((n, 2))
        d[held] = prev[held] / s[held, None]
    else:
        d, held = key_direction(keys, yaw, use_yaw=(variant != "no yaw"))

    dz1 = np.abs(u0[:, 2])
    dz2 = np.abs(u1[:, 2])
    # jump: from the press until JUMP_WINDOW ticks later (the whole arc)
    jp, _ = key_edges(data["key_ev"], key_t, codes=(KEY_CODES[4],))
    in_jump = frames_since(jp) <= JUMP_WINDOW
    air = (dz1 > AIR_CM) | (dz2 > AIR_CM) | in_jump
    return dict(d=d, held=held, air=air, prev=prev, s=s)


def move_features(m, rows, form):
    v, s, d = m["prev"][rows], m["s"][rows], m["d"][rows]
    if form == "input":
        # dp = a*v + b*s*d + c*d   ->  per axis: [v_ax, s*d_ax, d_ax]
        X = np.concatenate([np.stack([v[:, 0], s * d[:, 0], d[:, 0]], 1),
                            np.stack([v[:, 1], s * d[:, 1], d[:, 1]], 1)])
    else:
        # dp = (a*s + b*(v.d) + c) * d   ->  fit the step length along d
        X = np.stack([s, np.sum(v * d, axis=1), np.ones(len(rows))], 1)
    return X


def fit_model(parts, form):
    Xm, ym, Xb, yb, caps = [], [], [], [], []
    for data, m, rows in parts:
        dxy = data["dpos"][:, :2]
        mv = rows[m["held"][rows] & ~m["air"][rows]]
        br = rows[~m["held"][rows] & ~m["air"][rows] & (m["s"][rows] > 0.01)]
        Xm.append(move_features(m, mv, form))
        if form == "input":
            ym.append(np.concatenate([dxy[mv, 0], dxy[mv, 1]]))
        else:
            ym.append(np.sum(dxy[mv] * m["d"][mv], axis=1))
        v, s = m["prev"][br], m["s"][br]
        Xb.append(np.concatenate([np.stack([v[:, 0], v[:, 0] / s], 1),
                                  np.stack([v[:, 1], v[:, 1] / s], 1)]))
        yb.append(np.concatenate([dxy[br, 0], dxy[br, 1]]))
        caps.append(np.linalg.norm(dxy[mv], axis=1))
    w_move, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
    Xb, yb = np.concatenate(Xb), np.concatenate(yb)
    if len(yb) >= 2:
        w_brake, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    else:                       # e.g. "no keys": every moving row counts as held
        w_brake = np.array([1.0, 0.0])
    vmax = float(np.percentile(np.concatenate(caps), CAP_PCT))
    return dict(form=form, move=w_move, brake=w_brake, vmax=vmax)


def predict_model(data, m, W):
    n = data["n"]
    pred = data["mt2"].copy()
    rows = np.arange(2, n)
    ground = ~m["air"][rows]
    mv = rows[ground & m["held"][rows]]
    br = rows[ground & ~m["held"][rows]]

    if len(mv):
        X = move_features(m, mv, W["form"])
        if W["form"] == "input":
            out = X @ W["move"]
            xy = np.stack([out[:len(mv)], out[len(mv):]], 1)
        else:
            xy = (X @ W["move"])[:, None] * m["d"][mv]
        sp = np.linalg.norm(xy, axis=1)
        over = sp > W["vmax"]
        xy[over] *= (W["vmax"] / sp[over])[:, None]
        pred[mv, :2] = xy

    if len(br):
        v, s = m["prev"][br], m["s"][br]
        e, f = W["brake"]
        safe = np.maximum(s, 1e-9)
        new_s = np.maximum(e * s + f, 0.0)          # speed after braking, no reversal
        new_s[s <= 0.01] = 0.0
        pred[br, :2] = v / safe[:, None] * new_s[:, None]
    return pred


def print_weights(W, dt):
    a, b, c = W["move"]
    e, f = W["brake"]
    if W["form"] == "input":
        print(f"    move:  a {a:+.4f} (old velocity kept)   b {b:+.4f} (turned toward d)   "
              f"c {c:+.4f} cm/tick (push)")
    else:
        print(f"    move:  a {a:+.4f} (|v|)   b {b:+.4f} (v.d)   c {c:+.4f} cm/tick")
    print(f"    brake: e {e:+.4f} (speed kept)   f {f:+.4f} cm/tick")
    print(f"    cap:   {W['vmax']:.3f} cm/tick = {W['vmax'] / dt:.1f} cm/s")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def direction_diagnostic(data):
    SETTLE = 5
    dpos, n, rows = data["dpos"], data["n"], data["rows_all"]
    keys = key_state_at(data["key_ev"], data["key_t"])
    combo = (keys[:, :4] > 0.5) @ np.array([1, 2, 4, 8])      # W=1 A=2 S=4 D=8
    same = np.zeros(n, dtype=int)
    for i in range(1, n):
        same[i] = same[i - 1] + 1 if combo[i] == combo[i - 1] else 0

    dz = np.abs(dpos[:, 2])
    step = np.linalg.norm(dpos[:, :2], axis=1)
    sel = np.zeros(n, dtype=bool)
    sel[rows] = True
    sel &= (same >= SETTLE) & (step > 1.0) & (dz < AIR_CM) & (combo > 0)

    move_ang = np.degrees(np.arctan2(dpos[:, 1], dpos[:, 0]))
    names = {1: "W", 2: "A", 4: "S", 8: "D", 1 | 8: "W+D", 1 | 2: "W+A",
             4 | 8: "S+D", 4 | 2: "S+A"}
    expect = {1: 0, 2: -90, 4: 180, 8: 90, 9: 45, 3: -45, 12: 135, 6: -135}

    yaw = data["yaw"]
    yaw_prev = np.concatenate([[yaw[0]], yaw[:-1]])
    print(f"\ndirection check: step direction relative to camera forward, by keys held "
          f"(settled >= {SETTLE} frames, step > 1 cm, |dz| < {AIR_CM} cm)")
    print(f"  {'keys':<6}{'n':>6}{'expect':>8}  {'median offset':>14}  "
          f"{'median |error|':>15}  {'within 20 deg':>14}")
    for ref_name, ref in (("yaw[i]", yaw), ("yaw[i-1]", yaw_prev)):
        print(f"  using {ref_name}")
        rel = wrap180(move_ang - ref)
        for c, nm in names.items():
            s = sel & (combo == c)
            if s.sum() < 10:
                print(f"  {nm:<6}{int(s.sum()):>6}  (too few)")
                continue
            r = rel[s]
            off = wrap180(r - expect[c])
            err = np.abs(off)
            print(f"  {nm:<6}{int(s.sum()):>6}{expect[c]:>+8d}  {np.median(off):>+14.1f}  "
                  f"{np.median(err):>15.1f}  {100 * np.mean(err < 20):>13.0f}%")

    # does the error grow with how fast the camera is turning?
    rel = wrap180(move_ang - yaw)
    exp_arr = np.array([expect.get(int(c), np.nan) for c in combo])
    err = np.abs(wrap180(rel - exp_arr))
    yaw_rate = np.abs(wrap180(np.diff(yaw, prepend=yaw[0])))
    print("  direction error (yaw[i]) by camera turn rate this tick:")
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 1e9)):
        s = sel & (yaw_rate >= lo) & (yaw_rate < hi) & np.isfinite(exp_arr)
        if s.sum() >= 10:
            print(f"    {lo:>4}-{hi if hi < 1e9 else 'inf':<4} deg/tick: n {int(s.sum()):>5}, "
                  f"median |error| {np.median(err[s]):5.1f} deg")


def timing_report(data, W, yaw_lag, rows):
    dpos, n = data["dpos"], data["n"]
    keys = key_state_at(data["key_ev"], data["key_t"])
    yaw = data["yaw"]
    if yaw_lag:
        yaw = np.concatenate([[yaw[0]], yaw[:-1]])
    m = model_inputs(data, "full", yaw_lag)
    a, b, c = (float(x) for x in W["move"])
    e, f = (float(x) for x in W["brake"])
    vmax = W["vmax"]
    mt2 = data["mt2"]

    def input_step(vx, vy, w, a_, s_, d_, yaw_deg, air, m2x, m2y, m2z, vz):
        # m2*: the mt-2 prediction, used on airborne frames and for Z
        if air:
            return m2x, m2y, m2z
        s = math.hypot(vx, vy)
        fwd, right = w - s_, d_ - a_
        if fwd or right:
            r = math.radians(yaw_deg)
            cy, sy = math.cos(r), math.sin(r)
            dx, dy = fwd * cy - right * sy, fwd * sy + right * cy
            dn = math.hypot(dx, dy)
            dx, dy = dx / dn, dy / dn
            px = a * vx + b * s * dx + c * dx
            py = a * vy + b * s * dy + c * dy
            sp = math.hypot(px, py)
            if sp > vmax:
                px, py = px * vmax / sp, py * vmax / sp
        elif s > 0.01:
            ns = max(e * s + f, 0.0)
            px, py = vx / s * ns, vy / s * ns
        else:
            px = py = 0.0
        return px, py, m2z

    c2 = check_delay.mt2_coef()

    def mt2_step(x0, y0, z0, x1, y1, z1):
        # u_0, u_1: per-frame velocities of the last two real steps
        return (x0 + c2 * (x0 - x1), y0 + c2 * (y0 - y1), z0 + c2 * (z0 - z1))

    dp = check_delay.real_velocity(dpos, 0).tolist()
    dp1 = check_delay.real_velocity(dpos, 1).tolist()
    k = keys.tolist()
    yl = yaw.tolist()
    air = m["air"].tolist()
    m2 = mt2.tolist()
    args_in = [(dp[i][0], dp[i][1], k[i][0], k[i][1], k[i][2], k[i][3], yl[i],
                air[i], m2[i][0], m2[i][1], m2[i][2], dp[i][2]) for i in rows]
    args_m2 = [(*dp[i], *dp1[i]) for i in rows]

    # the per-frame function must give the same answer as the vectorized one
    ref = predict_model(data, m, W)[rows]
    got = np.array([input_step(*a_) for a_ in args_in])
    max_diff = float(np.max(np.abs(got - ref)))

    frame_us = 1e6 * data["dt_med"]
    print(f"\nper-frame cost (Python, one call per frame, {len(rows) - check_delay.TIMING_WARMUP} "
          f"frames after warm-up; {check_delay.cpu_name()})")
    print(f"  {'predictor':<12}{'median us':>11}{'p99 us':>10}{'max us':>10}"
          f"{'% of frame (p99)':>18}")
    for name, fn, args in (("input", input_step, args_in), ("mt-2", mt2_step, args_m2)):
        med, p99, mx = check_delay.time_per_call(fn, args)
        print(f"  {name:<12}{med:>11.2f}{p99:>10.2f}{mx:>10.2f}{100 * p99 / frame_us:>17.3f}%")
    print(f"  frame interval {frame_us:.0f} us.  per-frame input step matches the "
          f"vectorized prediction to {max_diff:.2e} cm")


def run(cap_dir, n_ticks=None, delta_ms=None, train_caps=None, timing=False,
        clock=check_delay.WINDOW_CLOCK, shift=check_delay.SHIFT_STATE):
    data = prepare(cap_dir, n_ticks, delta_ms, clock=clock, shift=shift)
    t, n, dpos, tr, v = data["t"], data["n"], data["dpos"], data["tr"], data["v"]
    key_t, key_ev, log = data["key_t"], data["key_ev"], data["log"]

    # ---- check 1: evdev key edges vs the engine's own key columns ----
    eng_keys = np.stack([log[c] for c in ENGINE_KEY_COLS], 1) > 0.5
    eng_edge = np.zeros(n, dtype=bool)
    eng_edge[1:] = np.any(eng_keys[1:] != eng_keys[:-1], axis=1)
    p0, r0 = key_edges(key_ev, key_t)
    offs, unmatched = edge_offset_histogram(eng_edge, p0 | r0)
    print(f"\nkey timing check: engine key-column changes vs evdev changes "
          f"(n = {data['n_ticks']}, delta = {data['delta_s'] * 1e3:+.3f} ms)")
    print(f"  engine changes {int(eng_edge.sum())}, evdev changes {int((p0 | r0).sum())}, "
          f"unmatched within 5 frames {unmatched}")
    if offs:
        vals, cnts = np.unique(offs, return_counts=True)
        print("  offset (evdev frame - engine frame): " +
              ", ".join(f"{v_:+d}: {c}" for v_, c in zip(vals, cnts)))

    # ---- check 2: when does movement respond to a key change? (model-free) ----
    wasd = KEY_CODES[:4]
    pw, rw = key_edges(key_ev, key_t, codes=wasd)
    step = np.zeros(n)
    step[1:] = np.linalg.norm(dpos[1:] - dpos[:-1], axis=1)
    offsets = range(-2, 5)
    print(f"\nmovement response to W/A/S/D changes (train rows): "
          f"mean |dpos[e+j] - dpos[e+j-1]| (cm)")
    print(f"  {'offset j':<10}" + "".join(f"{j:>+8d}" for j in offsets))
    resp = {}
    for name, ed in (("press", pw), ("release", rw)):
        e_idx = np.flatnonzero(ed)
        e_idx = e_idx[(e_idx >= 3) & (e_idx + max(offsets) < n) & np.isin(e_idx, tr)]
        means = [step[e_idx + j].mean() for j in offsets]
        resp[name] = dict(zip(offsets, means))
        print(f"  {name:<10}" + "".join(f"{m_:>8.3f}" for m_ in means) + f"   ({len(e_idx)} edges)")
    both = {j: resp["press"][j] + resp["release"][j] for j in offsets if j >= 0}
    print(f"  -> movement reacts at j = {max(both, key=both.get):+d}")

    direction_diagnostic(data)

    # ---- training sources ----
    if train_caps:
        others = [prepare(c, None, None, verbose=False, clock=clock, shift=shift)
                  for c in train_caps]
        print(f"\nleave-one-map-out: weights fitted on all usable rows of "
              f"{', '.join(o['name'] for o in others)}; evaluated on {data['name']} val rows")
        sources = [(o, o["rows_all"]) for o in others]
    else:
        print(f"\nweights fitted on this capture's train rows")
        sources = [(data, tr)]

    # yaw for tick i or i-1?  Decided on the training sources.
    print("\nyaw used for the key direction (training rows, input model, MAE cm):")
    lag_err = {}
    for lag in (0, 1):
        parts = [(d_, model_inputs(d_, "full", lag), r_) for d_, r_ in sources]
        W = fit_model(parts, "input")
        e = np.mean(np.concatenate([err3(predict_model(d_, m_, W)[r_], d_["dpos"][r_])
                                    for d_, m_, r_ in parts]))
        lag_err[lag] = e
        print(f"  yaw[i{'-1' if lag else ''}]{'':<4} train MAE {e:.4f}")
    yaw_lag = min(lag_err, key=lag_err.get)
    print(f"  -> using yaw[i{'-1' if yaw_lag else ''}]")

    # ---- fit every variant ----
    variants = {
        "input":        ("full", "input", 0),
        "input-proj":   ("full", "proj", 0),
        "abl: no yaw":  ("no yaw", "input", 0),
        "abl: no keys": ("no keys", "input", 0),
        "abl: keys -1": ("full", "input", -1),
        "abl: keys +1": ("full", "input", +1),
    }
    pred = {
        "const-vel": check_delay.history_pred(dpos, 1),
        "mt-2": data["mt2"],
        "mt-5": check_delay.history_pred(dpos, HIST_TAPS),
    }
    weights = {}
    m_main = None
    for name, (var, form, shift) in variants.items():
        parts = [(d_, model_inputs(d_, var, yaw_lag, shift), r_) for d_, r_ in sources]
        W = fit_model(parts, form)
        weights[name] = W
        m_eval = model_inputs(data, var, yaw_lag, shift)
        pred[name] = predict_model(data, m_eval, W)
        if name == "input":
            m_main = m_eval

    # input model only near a W/A/S/D change, mt-2 otherwise
    wasd_edge = pw | rw
    near_all = frames_since(wasd_edge) <= NEAR_EDGE
    pred["input-near"] = np.where(near_all[:, None], pred["input"], data["mt2"])
    # control: the same number of changes, placed at random frames
    rng_r = np.random.default_rng(SEED + 1)
    rand_edge = np.zeros(n, dtype=bool)
    rand_edge[rng_r.choice(np.arange(2, n), size=int(wasd_edge.sum()), replace=False)] = True
    near_rand = frames_since(rand_edge) <= NEAR_EDGE
    pred["ctl: near, random"] = np.where(near_rand[:, None], pred["input"], data["mt2"])

    print("\nfitted weights (per tick)")
    for name in ("input", "input-proj"):
        print(f"  {name}")
        print_weights(weights[name], data["dt_med"])

    cov = m_main
    vv = v
    n_air = int(cov["air"][vv].sum())
    n_mv = int((cov["held"][vv] & ~cov["air"][vv]).sum())
    print(f"\nval rows: {len(vv)}  (move {n_mv}, brake {len(vv) - n_mv - n_air}, "
          f"airborne -> mt-2 {n_air})")

    # ---- validation table ----
    y = dpos
    err = {m_: err3(p[v], y[v]) for m_, p in pred.items()}
    print(f"\nvalidation, 3D position error (cm), {len(v)} rows")
    print(f"  {'method':<16}{'MAE':>10}{'median':>10}{'p99':>10}{'vs mt-2':>10}")
    for m_, e in err.items():
        print(f"  {m_:<16}{e.mean():>10.4f}{np.median(e):>10.4f}"
              f"{np.percentile(e, 99):>10.4f}{improvement(err['mt-2'], e):>+9.1f}%")

    # ---- bootstrap by motion bucket ----
    jerk = np.linalg.norm(dpos[v] - dpos[v - 1], axis=1)
    gid = check_delay.block_of(v)
    print(f"\nbootstrap: input vs mt-2 by motion bucket "
          f"(resampling {check_delay.SPLIT_BLOCK}-frame blocks, {N_BOOT} resamples, seed {SEED})")
    print(CI_HDR)
    for name in ("input", "input-near", "input-proj"):
        print(f"  [{name}]")
        for b, (lo, hi) in POS_BUCKETS.items():
            s = (jerk >= lo) & (jerk < hi)
            print(ci_row(b, err["mt-2"][s], err[name][s], gid[s]))
        print(ci_row("all", err["mt-2"], err[name], gid))

    # ---- ablations ----
    print("\nablations and controls: each vs mt-2 on all val rows")
    print(CI_HDR)
    for name in ("input", "abl: no yaw", "abl: no keys", "abl: keys -1", "abl: keys +1",
                 "input-near", "ctl: near, random"):
        print(ci_row(name, err["mt-2"], err[name], gid))

    # ---- where the gain is ----
    edge = p0 | r0
    n_since = frames_since(edge)
    n_press, n_release = frames_since(p0), frames_since(r0)
    near = n_since[v] <= NEAR_EDGE
    last_is_press = n_press[v] <= n_release[v]
    groups = {
        f"<= {NEAR_EDGE} after any change": near,
        f"<= {NEAR_EDGE} after a press": near & last_is_press,
        f"<= {NEAR_EDGE} after a release": near & ~last_is_press,
        "no change nearby": ~near,
        "move rows": cov["held"][v] & ~cov["air"][v],
        "brake rows": ~cov["held"][v] & ~cov["air"][v],
    }
    print("\nwhere the gain is (vs mt-2)")
    print(CI_HDR)
    for m_ in ("input", "input-near"):
        print(f"  [{m_}]")
        for name, s in groups.items():
            print(ci_row(name, err["mt-2"][s], err[m_][s], gid[s]))

    # ---- signed error along the previous motion direction ----
    prev_dir = check_delay.real_velocity(dpos, 0)[v]
    speed = np.linalg.norm(prev_dir, axis=1)
    has_dir = speed > 0.01
    u = np.zeros_like(prev_dir)
    u[has_dir] = prev_dir[has_dir] / speed[has_dir, None]
    sgn = ("mt-2", "const-vel", "input", "input-near")
    signed = {m_: np.sum((pred[m_][v] - y[v]) * u, axis=1) for m_ in sgn}
    print("\nsigned error along previous motion direction (cm; + = overshoot)")
    print(f"  {'group':<24}{'n':>6}" + "".join(f"{m_:>12}" for m_ in sgn))
    for name, s in groups.items():
        s = s & has_dir
        if s.sum() == 0:
            continue
        print(f"  {name:<24}{int(s.sum()):>6}" +
              "".join(f"{signed[m_][s].mean():>+12.3f}" for m_ in sgn))

    if timing:
        timing_report(data, weights["input"], yaw_lag, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="override the whole-tick part of the delay (needs --delta_ms); "
                         "default: fit both from the mouse")
    ap.add_argument("--delta_ms", type=float, default=None,
                    help="override the fine offset of the delay (ms; needs --n)")
    ap.add_argument("--train_caps", nargs="+", default=None,
                    help="fit on these captures instead (leave-one-map-out)")
    ap.add_argument("--air_cm", type=float, default=AIR_CM,
                    help=f"|dz| per tick above this counts as airborne (default {AIR_CM})")
    ap.add_argument("--jump_window", type=int, default=JUMP_WINDOW,
                    help=f"ticks after a Jump press counted as airborne (default {JUMP_WINDOW})")
    ap.add_argument("--near", type=int, default=NEAR_EDGE,
                    help=f"frames after a key change that count as near (default {NEAR_EDGE})")
    ap.add_argument("--timing", action="store_true",
                    help="also measure the per-frame cost of the input model and mt-2")
    ap.add_argument("--clock", choices=check_delay.CLOCKS, default=check_delay.WINDOW_CLOCK,
                    help=f"frame-log clock the key windows follow "
                         f"(default {check_delay.WINDOW_CLOCK})")
    ap.add_argument("--no_state_shift", action="store_true",
                    help="keep the logged rows as they are (state is one frame behind its row)")
    check_delay.add_common_args(ap)
    a = ap.parse_args()
    check_delay.apply_common_args(a)
    AIR_CM, JUMP_WINDOW, NEAR_EDGE = a.air_cm, a.jump_window, a.near
    print(f"settings: air_cm {AIR_CM}, jump_window {JUMP_WINDOW}, near {NEAR_EDGE}, "
          f"history every {check_delay.HIST_STRIDE} frame(s), split {check_delay.SPLIT_MODE}"
          + (f" (seed {check_delay.SPLIT_SEED})" if check_delay.SPLIT_MODE == "blocks" else ""))
    run(a.cap_dir, a.n, a.delta_ms, a.train_caps, a.timing, clock=a.clock,
        shift=not a.no_state_shift)