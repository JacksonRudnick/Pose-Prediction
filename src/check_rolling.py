"""Rolling (real-time) delay/sensitivity estimator: causal, per-frame
updates from real frames only, with optional injected change stress test.

Estimator: exponential running sum per candidate, best by
sum_axes (A^2/B) / Y.  Learning on real frames every --hist_stride.
Injection: delay +/-ms, +/-ticks, or sensitivity scale at --inject_at.
"""

import argparse
import collections
import contextlib
import io
import time

import numpy as np

import e_rot as cd
import e_pos as cp

N_MAX = 2
DELTA_RANGE_MS = 2.0
DELTA_STEP_MS = 0.05
DEFAULT_HALFLIVES = ("30", "120", "600", "inf")
DELTA_TOL_MS = 0.2
K_TOL = 0.005
STAY_FRAMES = 60
ERR_WINDOW = 120
ERR_RATIO = 1.5
ERR_ABS = 0.005
ERR_STAY = 60
FALLBACK_N, FALLBACK_DELTA_MS = 1, 0.0



def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def edges_varying(ev_t, ev_val, t, n_fr, delta_fr_s):
    nf = len(t)
    i = np.arange(nf)
    lo_i = i - n_fr - 1
    hi_i = i - n_fr
    valid = (lo_i >= 0) & (hi_i < nf)
    lo_t = t[np.clip(lo_i, 0, nf - 1)] - delta_fr_s
    hi_t = t[np.clip(hi_i, 0, nf - 1)] - delta_fr_s
    press = np.zeros(nf, dtype=bool)
    release = np.zeros(nf, dtype=bool)
    for val, out in ((1, press), (0, release)):
        et = ev_t[ev_val == val]
        hi = np.searchsorted(et, hi_t, side="right")
        lo = np.searchsorted(et, lo_t, side="right")
        out[:] = valid & (hi > lo)
    return press, release


def shift_ticks_earlier(ev_t, sel, t, delta_s, ticks):
    ev_t = ev_t.copy()
    keep = np.ones(len(ev_t), dtype=bool)
    edges = t - delta_s
    idx = np.flatnonzero(sel)
    for _ in range(ticks):
        e = ev_t[idx]
        j = np.searchsorted(edges, e, side="left")        # edges[j-1] < e <= edges[j]
        ok = (j >= 2) & (j < len(t))
        jj = np.clip(j, 2, len(t) - 1)
        r = (e - edges[jj - 1]) / (edges[jj] - edges[jj - 1])
        new = edges[jj - 2] + r * (edges[jj - 1] - edges[jj - 2])
        ev_t[idx] = np.where(ok, new, e)
        keep[idx[~ok]] = False
    return ev_t, keep


def recovered_by_error(err_roll, err_oracle, rows, start):
    r = rows[rows >= start]
    if len(r) < ERR_WINDOW + ERR_STAY:
        return None
    cr = np.concatenate([[0.0], np.cumsum(err_roll[r])])
    co = np.concatenate([[0.0], np.cumsum(err_oracle[r])])
    m = len(r) - ERR_WINDOW + 1
    mean_r = (cr[ERR_WINDOW:ERR_WINDOW + m] - cr[:m]) / ERR_WINDOW
    mean_o = (co[ERR_WINDOW:ERR_WINDOW + m] - co[:m]) / ERR_WINDOW
    ok = mean_r <= ERR_RATIO * mean_o + ERR_ABS
    run = 0
    for p, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= ERR_STAY:
            return int(r[p - ERR_STAY + 1])
    return None


def converged_at(n_r, d_r, kx, ky, n_ref, d_ref, kx_ref, ky_ref, moving):
    with np.errstate(invalid="ignore"):
        ok = ((n_r == n_ref) & (np.abs(d_r - d_ref) <= DELTA_TOL_MS)
              & (np.abs(kx - kx_ref) <= K_TOL * abs(kx_ref))
              & (np.abs(ky - ky_ref) <= K_TOL * abs(ky_ref)))
    idx = np.flatnonzero(moving)
    run = 0
    for p, i in enumerate(idx):
        run = run + 1 if ok[i] else 0
        if run >= STAY_FRAMES:
            return int(idx[p - STAY_FRAMES + 1])
    return None


def stride_sum(a, R):
    # NaN steps (missing frames) count as 0; learning frames never include them
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    if R == 1:
        return a.copy()
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    out = np.zeros_like(a)
    out[R - 1:] = cs[R:] - cs[:-R]
    return out


def learn_mask(usable, motion, R):
    n = len(usable)
    all_ok = np.zeros(n, dtype=bool)
    ok_sum = stride_sum(usable.astype(np.float64), R)
    all_ok[R - 1:] = ok_sum[R - 1:] == R
    real = (np.arange(n) % R) == 0
    return real & all_ok & motion


def n_switches(n_r, start):
    a, b = n_r[start:-1], n_r[start + 1:]
    return int(np.sum((a >= 0) & (b >= 0) & (a != b)))


# ── the rolling estimator ──────────────────────────────────────────────────

def rolling_estimate(cx, cy, y_yaw, y_pitch, update, cand_n, cand_d, halflife):
    n, m = cx.shape
    lam = 1.0 if halflife == np.inf else 0.5 ** (1.0 / halflife)
    Ax = np.zeros(m); Bx = np.zeros(m); Ay = np.zeros(m); By = np.zeros(m)
    Yx = Yy = 0.0
    n_r = np.full(n, -1); d_r = np.full(n, np.nan); jbest = np.full(n, -1)
    kx = np.full(n, np.nan); ky = np.full(n, np.nan)
    for i in range(n):
        if Yx > 0 and Yy > 0 and np.all(Bx > 0) and np.all(By > 0):
            score = (Ax ** 2 / Bx) / Yx + (Ay ** 2 / By) / Yy
            j = int(np.argmax(score))
            n_r[i], d_r[i], jbest[i] = cand_n[j], cand_d[j], j
            kx[i], ky[i] = Ax[j] / Bx[j], Ay[j] / By[j]
        if update[i]:
            Ax = lam * Ax + y_yaw[i] * cx[i]
            Bx = lam * Bx + cx[i] ** 2
            Ay = lam * Ay + y_pitch[i] * cy[i]
            By = lam * By + cy[i] ** 2
            Yx = lam * Yx + y_yaw[i] ** 2
            Yy = lam * Yy + y_pitch[i] ** 2
    return n_r, d_r, jbest, kx, ky


# ── main ───────────────────────────────────────────────────────────────────

def run(cap_dir, halflives, inject_delay_ms=0.0, inject_k_scale=1.0,
        inject_at=0.5, inject_delay_ticks=0, n_max=N_MAX,
        delta_range_ms=DELTA_RANGE_MS, timing=False, hitch_filter=True,
        clock=cd.WINDOW_CLOCK, shift=cd.SHIFT_STATE):
    if n_max + 1 > cd.HIST_TAPS:
        raise ValueError(f"n_max {n_max} needs HIST_TAPS >= {n_max + 1} in e_rot.py "
                         f"(usable rows need consecutive ticks n_max + 1 back)")

    # offline reference (fit on the unchanged data, train frames only)
    off = quiet(cd.run, cap_dir, quiet=True, hitch_filter=hitch_filter, clock=clock,
                shift=shift)
    fit_rows, val_rows = off["fit_rows"], off["val_rows"]
    n_off, d_off = off["n_joint"], off["delta_joint_ms"]
    n_off_x, d_off_x = off["n_yaw"], off["delta_yaw_ms"]
    n_off_y, d_off_y = off["n_pitch"], off["delta_pitch_ms"]

    # ---- data ----
    log = cd.get_frame_log_data(cap_dir, shift)
    reports, _ = quiet(cd.get_input_reports, cap_dir)
    plog = cp.get_frame_log_data(cap_dir, shift)
    key_ev = cp.get_key_events(cap_dir)
    t = cd.window_times(log, clock)
    n = len(t)
    dt_ms = float(np.median(np.diff(t))) * 1e3

    # targets (seam-fixed with a fixed reference, independent of any fit)
    dyaw = cd.per_frame_delta(log["CamRotYaw"])
    dpitch = cd.per_frame_delta(log["CamRotPitch"])
    Cx0 = np.concatenate([[0.0], np.cumsum(reports["dx"].astype(np.float64))])
    c1 = cd.window_counts_nd(t, reports["t"].astype(np.float64), Cx0, 0 if shift else 1, 0.0)
    dyaw, _ = cd.resolve_seam(dyaw, c1, cd.K_REF)

    # ---- optional injected change ----
    t_change = t[0] + inject_at * (t[-1] - t[0])
    rt = reports["t"].astype(np.float64).copy()
    dx = reports["dx"].astype(np.float64).copy()
    dy = reports["dy"].astype(np.float64).copy()
    kt = key_ev["t"].astype(np.float64).copy()
    kv = key_ev["value"].copy()
    changed = bool(inject_delay_ms) or bool(inject_delay_ticks) or inject_k_scale != 1.0
    if changed:
        r_after = rt > t_change
        k_after = kt > t_change
        if inject_delay_ticks:
            # placed with the offline delta, so the truth after the change is
            # exactly n + ticks with the same delta
            rt, keep = shift_ticks_earlier(rt, r_after, t, d_off * 1e-3, inject_delay_ticks)
            rt, dx, dy, r_after = rt[keep], dx[keep], dy[keep], r_after[keep]
            kt, keep = shift_ticks_earlier(kt, k_after, t, d_off * 1e-3, inject_delay_ticks)
            kt, kv, k_after = kt[keep], kv[keep], k_after[keep]
        rt[r_after] -= inject_delay_ms * 1e-3      # longer delay = input lands earlier vs ticks
        dx[r_after] *= inject_k_scale
        dy[r_after] *= inject_k_scale
        o = np.argsort(rt, kind="stable"); rt, dx, dy = rt[o], dx[o], dy[o]
        kt[k_after] -= inject_delay_ms * 1e-3
        o = np.argsort(kt, kind="stable"); kt, kv = kt[o], kv[o]
    Cx = np.concatenate([[0.0], np.cumsum(dx)])
    Cy = np.concatenate([[0.0], np.cumsum(dy)])

    # counts per frame per candidate (n-major: index = n * len(deltas) + d)
    n_ticks = tuple(range(n_max + 1))
    deltas = cd.delta_grid(delta_range_ms, DELTA_STEP_MS)
    cand_n, cand_d = cd.candidate_grid(n_ticks, deltas)
    cx = cd.candidate_counts(t, rt, Cx, n_ticks, deltas)
    cy = cd.candidate_counts(t, rt, Cy, n_ticks, deltas)

    # usable rotation frames with mouse motion in any candidate's window
    usable = np.zeros(n, dtype=bool)
    usable[fit_rows] = True
    usable[val_rows] = True
    usable[:n_max + 1] = False
    moving = usable & ((np.abs(cx).sum(1) + np.abs(cy).sum(1)) > 0)

    # learning pairs: real frame r and the real frame R before it
    R = cd.HIST_STRIDE
    cx_l, cy_l = stride_sum(cx, R), stride_sum(cy, R)
    yaw_l, pitch_l = stride_sum(dyaw, R), stride_sum(dpitch, R)
    motion_l = (np.abs(cx_l).sum(1) + np.abs(cy_l).sum(1)) > 0
    update = learn_mask(usable, motion_l, R)

    n_ref = n_off + (inject_delay_ticks if changed else 0)
    d_ref = d_off + (inject_delay_ms if changed else 0.0)
    ksc = inject_k_scale if changed else 1.0
    kx_ref, ky_ref = off["k_yaw"] / ksc, off["k_pitch"] / ksc

    print(f"session {cap_dir}  (windows on {clock})")
    print(f"  offline fit (train frames, unchanged data): joint n = {n_off}, "
          f"delta = {d_off:+.3f} ms (yaw n={n_off_x} {d_off_x:+.3f}, "
          f"pitch n={n_off_y} {d_off_y:+.3f}), k {off['k_yaw']:+.5f} / {off['k_pitch']:+.5f}")
    print(f"  candidates: n {n_ticks[0]}..{n_ticks[-1]} x delta {deltas[0]:+.2f} .. "
          f"{deltas[-1]:+.2f} ms step {DELTA_STEP_MS} ms = {len(cand_n)}; "
          f"learning updates {int(update.sum())} (real-to-real pairs, every {R} frame(s)), "
          f"moving frames {int(moving.sum())} of {n}"
          f"{'' if hitch_filter else ' (hitch filter off)'}")
    if changed:
        print(f"  INJECTED at t = {t_change - t[0]:.1f} s: delay +{inject_delay_ticks} tick(s) "
              f"{inject_delay_ms:+.3f} ms, count scale x{inject_k_scale:g}  "
              f"(expected new n = {n_ref}, delta = {d_ref:+.3f} ms, "
              f"k {kx_ref:+.5f} / {ky_ref:+.5f})")
        if not n_ref in n_ticks or abs(d_ref) > delta_range_ms:
            print("  WARNING: the expected new delay is outside the candidate grid")

    # offline predictions (per-axis n, delta and k, as in e_rot)
    def counts_fixed(C, n_, d_):
        return cd.window_counts_nd(t, rt, C, n_, d_)
    off_px = off["k_yaw"] * counts_fixed(Cx, n_off_x, d_off_x)
    off_py = off["k_pitch"] * counts_fixed(Cy, n_off_y, d_off_y)

    # with a random block split, validation frames are spread over the whole
    # session; after an injected change only those after the change point
    # test the change (they include the catch-up period)
    start_v = int(np.searchsorted(t, t_change)) if changed else 0
    # the estimator has no estimate until every candidate has seen counts on
    # a learning frame (the same frame for every half-life); validation frames
    # before that are cold start, reported separately, not normal play
    first_est = n
    seen = np.zeros(cx_l.shape[1], dtype=bool)
    seen_y = np.zeros(cy_l.shape[1], dtype=bool)
    yx_seen = yy_seen = False
    for i in np.flatnonzero(update):
        seen |= cx_l[i] != 0
        seen_y |= cy_l[i] != 0
        yx_seen |= yaw_l[i] != 0
        yy_seen |= pitch_l[i] != 0
        if seen.all() and seen_y.all() and yx_seen and yy_seen:
            first_est = i + 1
            break
    n_early = int(np.sum((val_rows >= start_v) & (val_rows < first_est)))
    v = val_rows[(val_rows >= start_v) & (val_rows >= first_est)]
    gid = cd.block_of(v)
    err_off = np.hypot(off_px[v] - dyaw[v], off_py[v] - dpitch[v])
    mt2y = cd.history_pred(dyaw, 2)[v]          # real frames only
    mt2p = cd.history_pred(dpitch, 2)[v]
    err_mt2 = np.hypot(mt2y - dyaw[v], mt2p - dpitch[v])

    # ---- key timing: the engine's own logged key state is the reference ----
    # The engine log is never changed by the injection, so after an injected
    # delay change the offline delay places key changes on the wrong tick and
    # the rolling delay has to put them back.
    eng_keys = np.stack([plog[c] for c in cp.ENGINE_KEY_COLS], 1) > 0.5
    eng_edge = np.zeros(n, dtype=bool)
    eng_edge[1:] = np.any(eng_keys[1:] != eng_keys[:-1], axis=1)

    def key_match(n_fr, d_fr_ms, from_frame):
        """Share of engine key changes (at or after from_frame) that the evdev
        key changes, placed with a per-frame (n, delta), put on the same tick."""
        none = n_fr < 0
        n_use = np.where(none, FALLBACK_N, n_fr)
        d_use = np.where(none, FALLBACK_DELTA_MS, d_fr_ms)
        pr, rl = edges_varying(kt, kv, t, n_use, d_use * 1e-3)
        ref = eng_edge.copy()
        ref[:from_frame] = False
        offs, unmatched = cp.edge_offset_histogram(ref, pr | rl)
        total = int(ref.sum())
        same = int(np.sum(np.asarray(offs) == 0)) if offs else 0
        early = int(np.sum(np.asarray(offs) < 0)) if offs else 0
        late = int(np.sum(np.asarray(offs) > 0)) if offs else 0
        return total, same, early, late, unmatched

    rng = np.random.default_rng(cd.SEED)

    def ci(base, meth):
        pt = cd.improvement(base, meth)
        lo, hi = np.percentile(cd.block_bootstrap(base, meth, gid, rng), [2.5, 97.5])
        return f"{pt:+6.1f}% [{lo:+.1f}, {hi:+.1f}]"

    def diff_ci(base, meth):
        """mean(meth) - mean(base) with a block-bootstrap 95% CI (+ = rolling worse)."""
        st = cd.block_bootstrap_diff(base, meth, gid, rng)
        lo, hi = np.percentile(st, [2.5, 97.5])
        return f"{(meth - base).mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"

    # ---- rotation table ----
    print(f"\nrotation, validation frames ({len(v)}"
          + (", after the change point" if changed else "")
          + (f"; {n_early} before the first estimate left out" if n_early else "")
          + "), 2D error in deg")
    print(f"  {'estimate':<18}{'MAE':>9}{'vs mt-2':>26}{'minus offline':>28}"
          f"   delay on val (share of frames at each n; delta mean [min, max] ms)")
    print(f"  {'offline (fixed)':<18}{err_off.mean():>9.4f}{ci(err_mt2, err_off):>26}"
          f"{'--':>28}   yaw n={n_off_x} {d_off_x:+.3f}, pitch n={n_off_y} {d_off_y:+.3f}")

    start = int(np.searchsorted(t, t_change)) if changed else 0

    # oracle: the offline fit, corrected by the known injected change after
    # the change point.  It is what a perfectly calibrated system would do.
    post = np.arange(n) >= start if changed else np.zeros(n, dtype=bool)
    tk = inject_delay_ticks if changed else 0
    dly = inject_delay_ms if changed else 0.0
    orc_x = np.where(post,
                     off["k_yaw"] / ksc * counts_fixed(Cx, n_off_x + tk, d_off_x + dly),
                     off["k_yaw"] * counts_fixed(Cx, n_off_x, d_off_x))
    orc_y = np.where(post,
                     off["k_pitch"] / ksc * counts_fixed(Cy, n_off_y + tk, d_off_y + dly),
                     off["k_pitch"] * counts_fixed(Cy, n_off_y, d_off_y))
    err_oracle_all = np.hypot(orc_x - dyaw, orc_y - dpitch)
    err_off_all = np.hypot(off_px - dyaw, off_py - dpitch)    # offline, every frame
    mrows = np.flatnonzero(moving)          # usable frames with mouse motion
    fps = 1000.0 / dt_ms
    recovery = {}
    rolled = {}
    switches = {}
    per_frame = {}                          # returned for plotting (e_rt_plot.py)

    def time_txt(at_frame):
        """'X s since change / Y s of mouse motion (frames, updates)'.  Mouse
        motion counts usable frames with motion; updates counts the learning
        updates (real-to-real pairs) between the change and the recovery."""
        if at_frame is None:
            return "never"
        mov = int(moving[start:start + at_frame].sum())
        upd = int(update[start:start + at_frame].sum())
        return (f"{at_frame / fps:.1f} s / {mov / fps:.1f} s motion "
                f"({at_frame} fr, {upd} upd)")

    for hl in halflives:
        n_r, d_r, j_r, kx_r, ky_r = rolling_estimate(cx_l, cy_l, yaw_l, pitch_l, update,
                                                     cand_n, cand_d, hl)
        rolled[hl] = (n_r, d_r)
        jj = np.where(j_r >= 0, j_r, 0)
        px = np.nan_to_num(kx_r) * cx[np.arange(n), jj]
        py = np.nan_to_num(ky_r) * cy[np.arange(n), jj]
        err_r = np.hypot(px[v] - dyaw[v], py[v] - dpitch[v])
        nv, dv = n_r[v], d_r[v]
        share = ", ".join(f"{nt}:{100 * np.mean(nv == nt):.0f}%" for nt in n_ticks)
        d_txt = (f"{np.nanmean(dv):+.3f} [{np.nanmin(dv):+.2f}, {np.nanmax(dv):+.2f}]"
                 if np.any(np.isfinite(dv)) else "--")
        name = f"rolling hl={'inf' if hl == np.inf else int(hl)}"
        print(f"  {name:<18}{err_r.mean():>9.4f}{ci(err_mt2, err_r):>26}"
              f"{diff_ci(err_off, err_r):>28}   n {share}; delta {d_txt}")

        # recovery, from the start (cold start) or from the injected change
        c_at = converged_at(n_r[start:], d_r[start:], kx_r[start:], ky_r[start:],
                            n_ref, d_ref, kx_ref, ky_ref, moving[start:])
        err_all = np.hypot(px - dyaw, py - dpitch)
        e_at = recovered_by_error(err_all, err_oracle_all, mrows, start)
        recovery[name] = (time_txt(c_at), time_txt(None if e_at is None else e_at - start))
        switches[name] = n_switches(n_r, start)
        per_frame[name] = {
            "err": err_all, "n": n_r, "delta_ms": d_r, "kx": kx_r, "ky": ky_r,
            # absolute frame indices (None = did not recover)
            "rec_param": None if c_at is None else start + c_at,
            "rec_err": e_at,
        }
    print("  'minus offline' = rolling MAE - offline MAE with 95% CI (+ = rolling is worse)")

    what = "the injected change" if changed else "a cold start"
    print(f"\nrecovery after {what}: seconds since {'the change' if changed else 'the start'} "
          f"/ seconds of mouse motion (frames, learning updates)")
    print(f"  {'estimate':<18}{'parameter-based':>44}{'error-based':>44}{'n switches':>12}")
    for name, (p_txt, e_txt) in recovery.items():
        print(f"  {name:<18}{p_txt:>44}{e_txt:>44}{switches[name]:>12}")
    print(f"  parameter-based: n right, delta within {DELTA_TOL_MS} ms and yaw k and pitch k "
          f"within {K_TOL:.1%} of the "
          + ("post-change truth" if changed else "offline fit")
          + f", staying there for {STAY_FRAMES} moving frames")
    print(f"  error-based: mean error over {ERR_WINDOW} moving frames <= {ERR_RATIO} x a perfectly "
          f"calibrated predictor's + {ERR_ABS} deg, for {ERR_STAY} consecutive positions")
    print("  n switches: frames where the chosen n changes; each one repeats or skips one "
          "tick of input on that frame")

    # ---- key timing table ----
    where = "after the change point" if changed else "whole session"
    print(f"\nkey timing: engine key changes ({where}) matched on the same tick by evdev "
          f"key changes placed with each delay")
    print(f"  {'delay':<18}{'engine changes':>15}{'same tick':>16}{'early':>8}{'late':>8}"
          f"{'unmatched':>11}")
    rows_kt = [("offline (fixed)", np.full(n, n_off), np.full(n, d_off))]
    rows_kt += [(f"rolling hl={'inf' if hl == np.inf else int(hl)}", *rolled[hl])
                for hl in halflives]
    for name, n_series, d_series in rows_kt:
        total, same, early, late, unm = key_match(n_series, d_series, start)
        pct = 100.0 * same / total if total else 0.0
        print(f"  {name:<18}{total:>15}{f'{same} ({pct:.1f}%)':>16}{early:>8}{late:>8}{unm:>11}")
    print("  early / late = matched within 5 ticks but on an earlier / later tick")

    if timing:
        timing_report(t, rt, dx, dy, dyaw, dpitch, update, dt_ms, deltas,
                      n_sets=((1,), n_ticks))

    # per-frame results for plotting; printing above is unchanged
    return {
        "t": t - t[0], "fps": fps, "start": start, "changed": changed,
        "moving": moving, "err_offline": err_off_all, "err_oracle": err_oracle_all,
        "rolling": per_frame,
    }


class StreamingEstimator:
    """Streaming estimator: one call per frame; ring buffer for n shifts."""
    def __init__(self, rt, Cx, Cy, n_ticks, deltas_ms, halflife, stride=1):
        self.rt, self.Cx, self.Cy = rt, Cx, Cy
        self.stride = int(stride)
        # the last `stride` frames' candidate counts and rotations, summed on
        # a learning frame (a real frame) to give the real-to-real pair
        self.recent = collections.deque(maxlen=self.stride)
        self.n_ticks = tuple(n_ticks)
        self.delta_s = np.asarray(deltas_ms) * 1e-3
        nd = len(deltas_ms)
        m = nd * len(self.n_ticks)
        self.lam = 1.0 if halflife == np.inf else 0.5 ** (1.0 / halflife)
        self.Ax = np.zeros(m); self.Bx = np.zeros(m)
        self.Ay = np.zeros(m); self.By = np.zeros(m)
        self.Yx = 0.0; self.Yy = 0.0
        self.j = -1; self.kx = 0.0; self.ky = 0.0
        self.lo = None
        z = np.zeros(nd)
        # buf[n] = (counts x, counts y) of the tick n ticks ago
        self.buf = collections.deque([(z, z)] * (max(self.n_ticks) + 1),
                                     maxlen=max(self.n_ticks) + 1)

    def start(self, t0):
        self.lo = np.searchsorted(self.rt, t0 - self.delta_s, side="right")

    def step(self, t_now, y_yaw, y_pitch, learn):
        hi = np.searchsorted(self.rt, t_now - self.delta_s, side="right")
        bx = self.Cx[hi] - self.Cx[self.lo]
        by = self.Cy[hi] - self.Cy[self.lo]
        self.lo = hi
        self.buf.appendleft((bx, by))
        cx = np.concatenate([self.buf[nt][0] for nt in self.n_ticks])
        cy = np.concatenate([self.buf[nt][1] for nt in self.n_ticks])
        if self.j >= 0:
            pred = (self.kx * cx[self.j], self.ky * cy[self.j])
        else:
            pred = (0.0, 0.0)
        self.recent.append((cx, cy, y_yaw, y_pitch))
        if learn:
            if self.stride > 1:
                cx = sum(r[0] for r in self.recent)
                cy = sum(r[1] for r in self.recent)
                y_yaw = sum(r[2] for r in self.recent)
                y_pitch = sum(r[3] for r in self.recent)
            lam = self.lam
            self.Ax = lam * self.Ax + y_yaw * cx
            self.Bx = lam * self.Bx + cx * cx
            self.Ay = lam * self.Ay + y_pitch * cy
            self.By = lam * self.By + cy * cy
            self.Yx = lam * self.Yx + y_yaw * y_yaw
            self.Yy = lam * self.Yy + y_pitch * y_pitch
            if self.Yx > 0 and self.Yy > 0 and self.Bx.min() > 0 and self.By.min() > 0:
                score = (self.Ax * self.Ax / self.Bx) / self.Yx \
                    + (self.Ay * self.Ay / self.By) / self.Yy
                j = int(np.argmax(score))
                self.j = j
                self.kx = self.Ax[j] / self.Bx[j]
                self.ky = self.Ay[j] / self.By[j]
        return pred


def timing_report(t, rt, dx, dy, dyaw, dpitch, update, dt_ms, deltas, n_sets,
                  halflife=120.0, warmup=cd.TIMING_WARMUP):
    Cx = np.concatenate([[0.0], np.cumsum(dx)])
    Cy = np.concatenate([[0.0], np.cumsum(dy)])
    n = len(t)
    clock = time.perf_counter_ns
    frame_us = dt_ms * 1e3
    print(f"\nper-frame cost of the estimator (NumPy, one call per frame, half-life "
          f"{halflife:g}, {n - warmup} frames after {warmup} warm-up; {cd.cpu_name()})")
    print(f"  {'candidates':<16}{'frame type':<16}{'n':>7}{'median us':>11}{'p99 us':>10}"
          f"{'max us':>10}{'% of frame (p99)':>18}")
    for n_ticks in n_sets:
        cand_n, cand_d = cd.candidate_grid(n_ticks, deltas)
        est = StreamingEstimator(rt, Cx, Cy, n_ticks, deltas, halflife, cd.HIST_STRIDE)
        est.start(t[0])
        ts = np.zeros(n)
        chosen = np.full(n, -1)
        for i in range(1, n):
            chosen[i] = est.j
            t0 = clock()
            est.step(t[i], dyaw[i], dpitch[i], update[i])
            ts[i] = (clock() - t0) / 1e3
        # same choice as the vectorized estimator?  (both use frames < i)
        R = cd.HIST_STRIDE
        cx_all = stride_sum(cd.candidate_counts(t, rt, Cx, n_ticks, deltas), R)
        cy_all = stride_sum(cd.candidate_counts(t, rt, Cy, n_ticks, deltas), R)
        _, _, j_ref, _, _ = rolling_estimate(cx_all, cy_all, stride_sum(dyaw, R),
                                             stride_sum(dpitch, R), update,
                                             cand_n, cand_d, halflife)
        both = (j_ref[1:] >= 0) & (chosen[1:] >= 0)
        agree = 100.0 * np.mean(j_ref[1:][both] == chosen[1:][both]) if both.any() else 0.0

        label_c = (f"{len(cand_n)} (n {n_ticks[0]} only)" if len(n_ticks) == 1
                   else f"{len(cand_n)} (n {n_ticks[0]}..{n_ticks[-1]})")
        idx = np.arange(n) >= warmup
        for label, sel in (("learning", idx & update), ("predict only", idx & ~update)):
            # 'learning' = real frames that update the sums
            x = ts[sel]
            if len(x) == 0:
                continue
            p99 = np.percentile(x, 99)
            print(f"  {label_c:<16}{label:<16}{len(x):>7}{np.median(x):>11.2f}{p99:>10.2f}"
                  f"{x.max():>10.2f}{100 * p99 / frame_us:>17.3f}%")
        print(f"  {'':<16}matches the vectorized estimator's choice on {agree:.2f}% of frames")
    print(f"  frame interval {frame_us:.0f} us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir")
    ap.add_argument("--halflives", nargs="+", default=list(DEFAULT_HALFLIVES),
                    help="estimator half-lives in learning updates; 'inf' = never forget")
    ap.add_argument("--inject_delay_ms", type=float, default=0.0,
                    help="make the input delay this much longer after the change point")
    ap.add_argument("--inject_delay_ticks", type=int, default=0,
                    help="make the input delay this many whole ticks longer after the "
                         "change point (exact, even with tick jitter)")
    ap.add_argument("--inject_k_scale", type=float, default=1.0,
                    help="multiply mouse counts after the change point")
    ap.add_argument("--inject_at", type=float, default=0.5,
                    help="change point as a fraction of the session (default 0.5)")
    ap.add_argument("--n_max", type=int, default=N_MAX,
                    help=f"candidates cover n = 0 .. this whole ticks back (default {N_MAX})")
    ap.add_argument("--delta_range_ms", type=float, default=DELTA_RANGE_MS,
                    help=f"fine offset from -this to +this ms (default {DELTA_RANGE_MS})")
    ap.add_argument("--no_hitch_filter", action="store_true",
                    help="keep rows next to irregular tick intervals (passed to e_rot)")
    ap.add_argument("--clock", choices=cd.CLOCKS, default=cd.WINDOW_CLOCK,
                    help=f"frame-log clock the input windows follow (default {cd.WINDOW_CLOCK})")
    ap.add_argument("--timing", action="store_true",
                    help="also measure the per-frame cost of the estimator "
                         "(n = 1 only, and n = 0 .. n_max)")
    ap.add_argument("--no_state_shift", action="store_true",
                    help="keep the logged rows as they are (state is one frame behind its row)")
    cd.add_common_args(ap)
    a = ap.parse_args()
    cd.apply_common_args(a)
    hls = [np.inf if h == "inf" else float(h) for h in a.halflives]
    run(a.cap_dir, hls, a.inject_delay_ms, a.inject_k_scale, a.inject_at,
        a.inject_delay_ticks, a.n_max, a.delta_range_ms, a.timing,
        hitch_filter=not a.no_hitch_filter, clock=a.clock, shift=not a.no_state_shift)