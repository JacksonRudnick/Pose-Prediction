#!/usr/bin/env python3
"""Capture-session validation: input log, frame log, timestamps, mouse rate,
input-to-rotation cross-check, burst pattern.
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np

import fgcap
import helper

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results = []


def report(status, name, detail):
    _results.append(status)
    print(f"[{status}] {name}: {detail}")


def check_inputlog(events):
    n = len(events)
    if n == 0:
        return report(FAIL, "inputlog", "zero events")
    span = events["t"][-1] - events["t"][0]
    report(PASS, "inputlog", f"{n} events over {span:.1f}s")
    inversions = int(np.sum(np.diff(events["t"]) < 0))
    if inversions == 0:
        report(PASS, "monotonic", "timestamps non-decreasing")
    elif inversions < n * 0.001:
        report(WARN, "monotonic", f"{inversions} inversions")
    else:
        report(FAIL, "monotonic", f"{inversions} inversions")


def check_mouse_rate(reports):
    if len(reports) < 100:
        report(WARN, "mouse-rate", f"only {len(reports)} reports")
        return
    dt = np.diff(reports["t"])
    dup = float(np.mean(dt == 0.0))
    med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else float("inf")
    rate = 1.0 / med if med > 0 else 0.0
    hist_note = ", ".join(f"{lo * 1e3:.0f}-{hi * 1e3:.0f}ms:{int(c)}" for c, lo, hi in _top_bins(dt, 4))
    status = PASS if rate > 100 else WARN
    report(status, "mouse-rate", f"median interval {med * 1e3:.3f}ms (~{rate:.0f}Hz); bins: {hist_note}")
    if dup > 0.05:
        report(FAIL, "timestamps", f"{dup:.1%} duplicate stamps")
    else:
        report(PASS, "timestamps", f"{dup:.1%} duplicate stamps")


def _top_bins(dt, k):
    counts, edges = np.histogram(dt, bins=50)
    order = np.argsort(counts)[::-1][:k]
    return [(counts[i], edges[i], edges[i + 1]) for i in sorted(order)]


def check_framelog(log, reports=None):
    n = len(log)
    d = np.diff(log["TickTimestamp"])
    med = float(np.median(d))
    dev = (d - med) * 1e6
    tight = d[np.abs(dev) < 1000]
    tight_std = float(np.std(tight))
    p95_dev = float(np.percentile(dev, 95))
    hitch_count = int(np.sum(d > 0.02))
    hitch_max = float(np.max(d)) if hitch_count > 0 else 0.0
    status = PASS if tight_std < med * 0.01 else WARN
    report(status, "fixed-dt",
           f"{n} rows, median dt {med * 1e3:.3f}ms, "
           f"tight-core std {tight_std * 1e6:.1f}us "
           f"(~{tight_std / med * 100:.1f}% of interval), "
           f"p95 deviation {p95_dev:.0f}us, "
           f"{hitch_count} hitches >20ms (max {hitch_max * 1e3:.0f}ms) "
           f"({'holding' if status == PASS else 'varies — not holding'})")
    tp = int(np.sum(log["Teleport"] > 0))
    if reports is not None:
        teleport = fgcap.refine_teleports(log, reports)
        n_tele = int(np.sum(teleport))
        report(PASS, "teleport", f"engine: {tp} rows; refine_teleports: {n_tele}")
    else:
        report(WARN, "teleport", f"{tp} rows flagged (use refine_teleports)")
    if np.any(np.diff(log["TickTimestamp"]) <= 0):
        report(FAIL, "tick-anchor", "not strictly increasing")
    else:
        report(PASS, "tick-anchor", "strictly increasing")


def check_domain_overlap(log, events):
    t0, t1 = log["TickTimestamp"][0], log["TickTimestamp"][-1]
    e0, e1 = events["t"][0], events["t"][-1]
    inter = min(t1, e1) - max(t0, e0)
    frac = inter / (t1 - t0) if t1 > t0 else 0
    if frac > 0.9:
        report(PASS, "domain", f"overlap {frac:.0%}")
    elif inter > 0:
        report(WARN, "domain", f"overlap {frac:.0%}")
    else:
        report(FAIL, "domain", f"NO overlap")


def check_boundary_shift(cap_dir, log, reports, row_ok):
    """Report the anchored window shift d exactly as the frames table has it
    (the table is built/saved if absent), so E0 and the experiments never
    disagree on a session's delay."""
    import frames_table as ft
    path = cap_dir.rstrip("/") + ".frames.npz"
    if not os.path.exists(path):
        frames, reports_t, info = ft.build(cap_dir)
        ft.save(path, frames, reports_t, info)
    else:
        _, _, info = ft.load(path)
    d = info["d"]; tau = info["tau"]
    status = PASS if abs(d) < 5e-3 else WARN
    report(status, "boundary-shift",
           f"d = {d * 1e3:+.4f} ms (tau = {tau * 1e3:.4f} ms): "
           f"k {info['k']:+.5f}, R^2 {info['r2']:.4f} "
           f"(the exact value the frames table uses)")


def check_anchor_gap(log):
    names = log.dtype.names or ()
    pump_cols = [c for c in ("InputSamplingTimestamp", "BeginFrameTimestamp") if c in names]
    if not pump_cols:
        return report(PASS, "anchor", "TickTimestamp only")
    col = pump_cols[0]
    gap = (log["TickTimestamp"] - log[col]) * 1e3
    if np.any(gap < -0.05):
        report(WARN, "anchor-gap", f"{int(np.sum(gap < -0.05))} rows pump after Tick")
    report(PASS, "anchor-gap", f"tick - {col}: median {np.median(gap):.2f} ms, std {np.std(gap):.3f} ms")


def check_drift(log, reports, row_ok):
    tick_ts = fgcap.anchor_times(log)
    sdx_full = fgcap.window_sum(tick_ts, reports["t"], reports["dx"])
    dyaw_sf, _ = helper.resolve_yaw_delta(
        np.asarray(log["MouseAngX"], dtype=np.float64), sdx_full)
    s0, alpha, t0, r2 = fgcap.fit_drift(
        tick_ts, dyaw_sf, reports, "dx", row_ok)
    span = log["TickTimestamp"][-1] - log["TickTimestamp"][0]
    alpha_ppm = alpha * 1e6
    drift_ms = alpha * span * 1e3
    status = PASS if abs(alpha_ppm) < 5.0 else WARN
    report(status, "drift",
           f"shift(t) = {s0*1e3:+.2f} ms + {alpha_ppm:+.1f} ppm * (t-t0); "
           f"drift over session {drift_ms:+.2f} ms (line R^2 {r2:.3f}; "
           f"{'clock drift, apply linear term' if status == WARN else 'constant offset'})")


def check_crosscheck(log, reports):
    if len(reports) < 100:
        report(WARN, "cross-check", "too few mouse reports to fit")
        return
    sdx, sdy = fgcap.per_frame_sums(fgcap.anchor_times(log), reports)
    pitch = ((log["CamRotPitch"] + 180.0) % 360.0) - 180.0
    clamp_ok = np.abs(pitch) <= 87.0
    # Compute seam-fixed yaw (matches the E1 pipeline)
    dyaw_sf, _ = helper.resolve_yaw_delta(
        np.asarray(log["MouseAngX"], dtype=np.float64), sdx)

    # Use refine_teleports instead of the engine's Teleport flag
    teleport_ok = ~fgcap.refine_teleports(log, reports)
    row_ok = clamp_ok & teleport_ok
    n_clamped = int(np.sum(~clamp_ok))
    if n_clamped:
        report(PASS, "clamp-mask",
               f"{n_clamped} clamp-adjacent frames excluded from fits")
    for axis, ang, raw in (("yaw", dyaw_sf, sdx),
                           ("pitch", log["MouseAngY"], sdy)):
        if np.std(ang) < 1e-9:
            report(WARN, f"cross-{axis}",
                   "no rotation on this axis in session — cannot fit "
                   "(rotate on both axes during capture)")
            continue
        best = None
        for lag in (0, 1):
            a = ang[1 + lag:]
            r = raw[1:len(raw) - lag] if lag else raw[1:]
            n = len(log)
            cm = (row_ok[lag:n - 1] & row_ok[1 + lag:n]) if lag \
                else (row_ok[0:n - 1] & row_ok[1:n])
            mask = (np.abs(r) > 0) & cm
            if mask.sum() < 30:
                continue
            k = float(np.sum(a[mask] * r[mask]) / np.sum(r[mask] ** 2))
            resid = a[mask] - k * r[mask]
            ss = 1.0 - np.sum(resid ** 2) / max(np.sum((a[mask] - a[mask].mean()) ** 2), 1e-12)
            if best is None or ss > best[2]:
                best = (lag, k, ss, resid, mask.sum())
        if best is None:
            report(WARN, f"cross-{axis}", "no frames with motion — wiggle the "
                   "mouse during capture")
            continue
        lag, k, r2, resid, m = best
        out = int(np.sum(np.abs(resid) > 5 * np.std(resid))) if len(resid) else 0
        status = PASS if r2 > 0.95 else (WARN if r2 > 0.7 else FAIL)
        report(status, f"cross-{axis}",
               f"best at lag {lag}: sensitivity {k:.5f} deg/count, "
               f"R^2 {r2:.4f} over {m} frames, {out} outliers "
               f"({'raw input explains control rotation' if status == PASS else 'unexplained rotation — smoothing/recoil/aim-assist or join misalignment'})")
        if lag != 1:
            report(WARN, f"lag-{axis}",
                   "best alignment is lag 0; with the OnWorldTickStart anchor "
                   "convention lag 1 was expected — verify the anchor")


def check_frames_dir(frames_dir, log, burst_len, burst_period):
    names = [f for f in os.listdir(frames_dir) if f.startswith("frame_")]
    nums = sorted(int(re.search(r"(\d+)", f).group(1)) for f in names)
    if not nums:
        return report(FAIL, "frames", "no frame files")
    runs = []
    s = p = nums[0]
    for x in nums[1:]:
        if x == p + 1:
            p = x
        else:
            runs.append((s, p))
            s = p = x
    runs.append((s, p))
    lens = np.array([b - a + 1 for a, b in runs])
    full = int(np.sum(lens >= burst_len))
    report(PASS if full == len(runs) else WARN, "bursts",
           f"{len(nums)} frames in {len(runs)} runs; lengths min/med/max "
           f"{lens.min()}/{int(np.median(lens))}/{lens.max()}; {full}/{len(runs)} complete")
    in_burst = set(log["GFrameCounter"][log["InBurst"] > 0].astype(np.int64))
    have = set(nums)
    missing = len(in_burst - have)
    if in_burst:
        report(PASS if missing / len(in_burst) < 0.05 else WARN, "inburst-join",
               f"{missing}/{len(in_burst)} missing")


# ── self-test: synthesize a session and validate it ─────────────────────────

def selftest():
    rng = np.random.default_rng(7)
    fps, dur, hz, sens = 60, 10.0, 1000, 0.022
    tick = np.arange(int(dur * fps)) / fps + 100.0
    rt = np.arange(int(dur * hz)) / hz + 100.0 + rng.normal(0, 2e-5, int(dur * hz))
    dx = np.round(3 * np.sin(rt) + 40 * np.exp(-((rt - 105) ** 2) / 0.01)).astype(int)
    dy = np.round(1.5 * np.cos(rt * 0.7)).astype(int)
    evs = []
    for t, x, y in zip(rt, dx, dy):
        if x: evs.append((t, fgcap.EV_REL, fgcap.REL_X, x))
        if y: evs.append((t, fgcap.EV_REL, fgcap.REL_Y, y))
        evs.append((t, fgcap.EV_SYN, 0, 0))
    arr = np.array(evs, dtype=fgcap.EVENT_DTYPE)
    tmp = tempfile.mkdtemp()
    binpath = os.path.join(tmp, "InputLog_t.bin")
    with open(binpath, "wb") as f:
        f.write(np.array([fgcap.INPUTLOG_MAGIC, fgcap.INPUTLOG_VERSION], dtype="<u4").tobytes() + b"\0" * 8)
        arr.tofile(f)
    cdx = np.concatenate([[0], np.cumsum(dx)])
    cdy = np.concatenate([[0], np.cumsum(dy)])
    idx = np.searchsorted(rt, tick, side="right")
    wsx = np.diff(cdx[idx], prepend=0)
    wsy = np.diff(cdy[idx], prepend=0)
    ang = np.zeros(len(tick)); ang[1:] = sens * wsx[:-1]
    angy = np.zeros(len(tick)); angy[1:] = sens * wsy[:-1]
    rows = []
    for i, t in enumerate(tick):
        rows.append([i, 1000 + i, t, t - 100.0, 1.0 / fps, 0, 0, 0, 0, 0,
                     ang[i], angy[i], ang[i] * fps, angy[i] * fps,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1 if (1000 + i) % 16 < 4 else 0])
    csvpath = os.path.join(tmp, "FrameLog_t.csv")
    np.savetxt(csvpath, np.array(rows), delimiter=",", header=",".join(fgcap.EXPECTED_COLUMNS), comments="")
    fdir = os.path.join(tmp, "Frames"); os.mkdir(fdir)
    for i in range(len(tick)):
        g = 1000 + i
        if g % 16 < 4 and not (g == 1041):
            open(os.path.join(fdir, f"frame_{g:06d}.bin"), "wb").close()
    print(f"selftest session in {tmp}\n")
    run(tmp, 4, 16)
    print("\nselftest expectation: all PASS except one WARN from the deliberate burst hole.")


def run(cap_dir, burst_len, burst_period):
    frames_dir, csv, binpath = helper.resolve_capture_paths(cap_dir)
    events = fgcap.read_inputlog(binpath)
    log = fgcap.read_framelog(csv)
    reports = fgcap.mouse_reports(events)
    check_inputlog(events)
    check_mouse_rate(reports)
    check_framelog(log, reports)
    check_domain_overlap(log, events)
    check_anchor_gap(log)
    pitch = ((log["CamRotPitch"] + 180.0) % 360.0) - 180.0
    clamp_ok = np.abs(pitch) <= 87.0
    teleport_ok = ~fgcap.refine_teleports(log, reports)
    row_ok = clamp_ok & teleport_ok
    check_drift(log, reports, row_ok)
    check_crosscheck(log, reports)
    check_boundary_shift(cap_dir, log, reports, row_ok)
    check_frames_dir(frames_dir, log, burst_len, burst_period)
    n_fail = _results.count(FAIL)
    n_warn = _results.count(WARN)
    print(f"\n== {len(_results)} checks: "
          f"{_results.count(PASS)} pass, {n_warn} warn, {n_fail} fail ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir")
    ap.add_argument("--burst-len", type=int, default=4)
    ap.add_argument("--burst-period", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cap_dir:
        sys.exit(run(a.cap_dir, a.burst_len, a.burst_period))
    else:
        ap.print_help()