"""Diagnostic: for each pair of neighbouring disturbed frames, move the
shared window boundary to minimise joint 2D residual.
"""

import argparse
import contextlib
import io
import os

import numpy as np

import e_rot as cd

RESID_DEG = 5.0          # a frame counts as disturbed above this 2D residual
SHIFT_MS = np.arange(-16.0, 16.0001, 0.05)
NEAR_MS = 20.0


def run(cap_dir, resid_deg=RESID_DEG):
    with contextlib.redirect_stdout(io.StringIO()):
        fit = cd.run(cap_dir, quiet=True)
        log = cd.get_frame_log_data(cap_dir)
        reports, drop_t = cd.get_input_reports(cap_dir)

    t = np.asarray(log["TickTimestamp"], np.float64)
    n = len(t)
    rt = reports["t"].astype(np.float64)
    Cx = np.concatenate([[0.0], np.cumsum(reports["dx"].astype(np.float64))])
    Cy = np.concatenate([[0.0], np.cumsum(reports["dy"].astype(np.float64))])
    Dx, Dy = fit["yaw_delay_ms"] * 1e-3, fit["pitch_delay_ms"] * 1e-3
    kx, ky = fit["k_yaw"], fit["k_pitch"]

    cx = cd.window_counts(t, rt, Cx, Dx)
    cy = cd.window_counts(t, rt, Cy, Dy)
    dyaw, _ = cd.resolve_seam(cd.per_frame_delta(log["CamRotYaw"]), cx, kx)
    dpitch = cd.per_frame_delta(log["CamRotPitch"])
    ry, rp = dyaw - kx * cx, dpitch - ky * cy
    r2 = np.hypot(ry, rp)

    # frames we can trust: real previous frame, pitch away from the clamp
    fid = np.asarray(log["GFrameCounter"], np.int64)
    ok = np.zeros(n, dtype=bool)
    ok[1:] = fid[1:] - fid[:-1] == 1
    clamp = np.abs(cd.wrap_deg(log["CamRotPitch"])) > cd.CLAMP_DEG
    ok[1:] &= ~clamp[1:] & ~clamp[:-1]
    ok[0] = False

    # time from the input pump to the tick timestamp (both logged by the plugin)
    name = [f for f in os.listdir(cap_dir)
            if f.startswith("FrameLog_") and f.endswith(".csv")][0]
    extra = np.genfromtxt(os.path.join(cap_dir, name), delimiter=",", names=True,
                          usecols=("TickTimestamp", "InputSamplingTimestamp",
                                   "BeginFrameTimestamp"))
    pump_ms = (extra["TickTimestamp"] - extra["InputSamplingTimestamp"]) * 1e3
    begin_ms = (extra["TickTimestamp"] - extra["BeginFrameTimestamp"]) * 1e3
    pump_ok = np.isfinite(pump_ms) & (extra["InputSamplingTimestamp"] > 0)

    mx = np.asarray(log["MouseAngX"], np.float64)
    my = np.asarray(log["MouseAngY"], np.float64)
    dt_ms = np.zeros(n)
    dt_ms[1:] = np.diff(t) * 1e3
    gaps = np.diff(rt) * 1e3
    gap_p999 = float(np.percentile(gaps, 99.9)) if len(gaps) else np.nan

    bad = ok & (r2 > resid_deg)
    pairs, singles = [], []
    i = 1
    while i < n - 1:
        if bad[i]:
            if bad[i + 1] and ok[i + 1]:
                pairs.append(i)
                i += 2
                continue
            singles.append(i)
        i += 1

    print(f"session {cap_dir}")
    print(f"  tau yaw {-fit['yaw_delay_ms']:.3f} ms, pitch {-fit['pitch_delay_ms']:.3f} ms; "
          f"median tick {np.median(dt_ms[1:]):.3f} ms; report gap p99.9 {gap_p999:.2f} ms")
    if pump_ok.any():
        pv = pump_ms[pump_ok]
        print(f"  pump->tick (TickTimestamp - InputSamplingTimestamp): median "
              f"{np.median(pv):.2f} ms, p99 {np.percentile(pv, 99):.2f} ms, "
              f"max {pv.max():.2f} ms; frames above median + 3 ms: "
              f"{int(np.sum(pv > np.median(pv) + 3.0))}")
    print(f"  frames with 2D residual > {resid_deg} deg: {int(bad.sum())} "
          f"({len(pairs)} neighbouring pairs, {len(singles)} single)")

    def split_residual(i, s_ms):
        """residuals of frames i and i+1 with their shared boundary moved"""
        s = s_ms * 1e-3
        b_x = np.searchsorted(rt, t[i] + Dx + s, side="right")
        b_y = np.searchsorted(rt, t[i] + Dy + s, side="right")
        a_x = np.searchsorted(rt, t[i - 1] + Dx, side="right")
        a_y = np.searchsorted(rt, t[i - 1] + Dy, side="right")
        c_x = np.searchsorted(rt, t[i + 1] + Dx, side="right")
        c_y = np.searchsorted(rt, t[i + 1] + Dy, side="right")
        r_i = np.hypot(dyaw[i] - kx * (Cx[b_x] - Cx[a_x]),
                       dpitch[i] - ky * (Cy[b_y] - Cy[a_y]))
        r_j = np.hypot(dyaw[i + 1] - kx * (Cx[c_x] - Cx[b_x]),
                       dpitch[i + 1] - ky * (Cy[c_y] - Cy[b_y]))
        return r_i, r_j

    rows = []
    if pairs:
        print(f"\n  {'time s':>8} {'intervals ms':>14} {'before deg':>16} {'shift ms':>9} "
              f"{'after deg':>14} {'cam-engine deg':>15} {'gap ms':>7} {'pump->tick ms':>15}")
    for i in pairs:
        ri, rj = split_residual(i, SHIFT_MS)
        cost = ri + rj
        best = int(np.argmin(cost))
        s_best = float(SHIFT_MS[best])
        # if several shifts are equally good, report the one closest to 0
        tied = np.flatnonzero(cost <= cost[best] + 1e-9)
        s_best = float(SHIFT_MS[tied[np.argmin(np.abs(SHIFT_MS[tied]))]])
        ai, aj = split_residual(i, s_best)
        w = lambda x: abs(float(cd.wrap_deg(x)))     # yaw is compared modulo 360
        cam_eng = max(w(dyaw[i] - mx[i]) if np.isfinite(mx[i]) else np.nan,
                      w(dyaw[i + 1] - mx[i + 1]) if np.isfinite(mx[i + 1]) else np.nan,
                      abs(dpitch[i] - my[i]) if np.isfinite(my[i]) else np.nan,
                      abs(dpitch[i + 1] - my[i + 1]) if np.isfinite(my[i + 1]) else np.nan)
        b = t[i] + Dx
        near = (rt > b - NEAR_MS * 1e-3) & (rt < b + NEAR_MS * 1e-3)
        g = np.diff(rt[near]) * 1e3
        gmax = float(g.max()) if len(g) else np.nan
        rows.append((t[i] - t[0], s_best, gmax, dt_ms[i], dt_ms[i + 1], cam_eng,
                     pump_ms[i] if pump_ok[i] else np.nan,
                     pump_ms[i + 1] if pump_ok[i + 1] else np.nan))
        print(f"  {t[i] - t[0]:>8.2f} {dt_ms[i]:>6.2f} {dt_ms[i + 1]:>6.2f}   "
              f"{r2[i]:>6.2f} {r2[i + 1]:>6.2f}   {s_best:>+8.2f}  {float(ai):>5.2f} {float(aj):>5.2f}"
              f"   {cam_eng:>13.3f}   {gmax:>6.2f}   {rows[-1][6]:>6.2f} {rows[-1][7]:>6.2f}")

    if singles:
        print(f"\n  single disturbed frames (no disturbed neighbour): "
              + ", ".join(f"{t[i] - t[0]:.2f}s ({r2[i]:.1f} deg)" for i in singles[:20])
              + (" ..." if len(singles) > 20 else ""))

    if not rows:
        return
    a = np.array(rows)
    times, shifts, gm, d1, d2, ce, p1, p2 = a.T
    print("\n  summary")
    if len(times) > 1:
        sp = np.diff(times)
        print(f"    spacing between pairs: median {np.median(sp):.2f} s, "
              f"p10 {np.percentile(sp, 10):.2f} s, p90 {np.percentile(sp, 90):.2f} s")
        # some disturbances are too small to cross the threshold, so a regular
        # cause shows up as spacings that are whole multiples of one period
        per = float(np.percentile(sp, 10))
        if per > 0:
            ratio = sp / per
            on_grid = np.abs(ratio - np.round(ratio)) * per <= 0.1
            print(f"    shortest common spacing {per:.2f} s; spacings that are a whole "
                  f"multiple of it (within 0.1 s): {int(on_grid.sum())} of {len(sp)}")
        print("    spacings (s): " + ", ".join(f"{x:.2f}" for x in sp[:40])
              + (" ..." if len(sp) > 40 else ""))
    print(f"    best shift: median {np.median(shifts):+.2f} ms, |shift| median "
          f"{np.median(np.abs(shifts)):.2f} ms, range [{shifts.min():+.2f}, {shifts.max():+.2f}] ms")
    print(f"    pairs fixed by moving the boundary <= 1.5 ms (about one report): "
          f"{int(np.sum(np.abs(shifts) <= 1.5))} of {len(shifts)}")
    med_tick = np.median(dt_ms[1:])
    odd_tick = np.sum((np.abs(d1 - med_tick) > 0.5) | (np.abs(d2 - med_tick) > 0.5))
    print(f"    pairs with a tick interval more than 0.5 ms from the median: {int(odd_tick)} "
          f"of {len(shifts)}")
    print(f"    camera vs engine's own per-tick input (MouseAng): largest difference "
          f"{np.nanmax(ce):.3f} deg (near 0 = engine applied the rotation as logged)")
    print(f"    largest report gap near the boundary: median {np.nanmedian(gm):.2f} ms "
          f"(session p99.9 {gap_p999:.2f} ms)")
    if pump_ok.any():
        med = np.median(pump_ms[pump_ok])
        big = np.abs(shifts) > 1.5
        for label, sel in (("large shift (> 1.5 ms)", big), ("small shift (<= 1.5 ms)", ~big)):
            if sel.any():
                extra_ms = np.nanmax(np.stack([p1[sel], p2[sel]]), axis=0) - med
                print(f"    pump->tick on {label} pairs: extra over the session median, "
                      f"median {np.nanmedian(extra_ms):+.2f} ms, range "
                      f"[{np.nanmin(extra_ms):+.2f}, {np.nanmax(extra_ms):+.2f}] ms")
        print("    if the large-shift pairs show extra pump->tick time close to |shift|, the")
        print("    engine stalled between reading input and stamping the tick")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir", required=True)
    ap.add_argument("--resid_deg", type=float, default=RESID_DEG)
    a = ap.parse_args()
    run(a.cap_dir, a.resid_deg)