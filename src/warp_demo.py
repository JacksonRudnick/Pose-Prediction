"""warp_demo — export three warped images for one high-motion frame.

Writes, at full size, into --out_dir:
  history_<baseline>.png  previous frame rotated by the history baseline's prediction
  input.png               previous frame rotated by the raw-input prediction
  true_rotation.png       previous frame rotated by the true rotation

Rotation only: the camera position is held fixed, so depth is not used.
Delay, sensitivity, split and history predictor come from e_rot.py.
Only validation rows in the "reversal" bucket are candidates.

Usage:
  python3 warp_demo.py --cap_dir CAP_DIR [--out_dir warp_out]
                       [--baseline mt-2|const-vel] [--pick max|median]
                       [--row I] [--flip] [--fov 90]
"""

import argparse
import contextlib
import io
import os

import numpy as np

import e_rot as cd

W, H = 1920, 1080
FOV_DEG = 90.0
FOCAL = (W / 2.0) / np.tan(np.radians(FOV_DEG) / 2.0)
HOLE = np.array([26, 26, 32], np.uint8)

IMAGE_OFFSET = 1
MIN_JERK_DEG = cd.JERK_BUCKETS["reversal"][0]
MAX_PITCH_DEG = 45.0
MAX_HOLE_FRAC = 0.10
MEASURE_SPLAT = 3
MAX_HOLE_CHECKS = 200


def set_fov(fov_deg):
    global FOCAL
    FOCAL = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ── IO ───────────────────────────────────────────────────────────────────────

def read_color(path, flip=False):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != W * H * 4:
        raise ValueError(f"{path}: {raw.size} bytes, expected {W*H*4}")
    img = raw.reshape(H, W, 4)[..., [2, 1, 0]]
    return img[::-1] if flip else img


def find_frames_dir(cap_dir):
    """First folder under cap_dir holding frame_*.bin."""
    for root, _, files in sorted(os.walk(cap_dir)):
        if any(f.startswith("frame_") and f.endswith(".bin") for f in files):
            return root
    raise SystemExit("no frame_*.bin color captures found (pass --frames_dir)")


def file_ids(d, prefix):
    out = set()
    for f in os.listdir(d):
        if f.startswith(prefix) and f.endswith(".bin"):
            try:
                out.add(int(f[len(prefix):-4]))
            except ValueError:
                pass
    return out


def cam_basis(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp_, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    fwd = np.array([cp_ * cy, cp_ * sy, sp])
    right = np.array([cy * sp * sr - sy * cr,
                      sy * sp * sr + cy * cr,
                      cp_ * sr])
    up = np.cross(fwd, right)
    up /= (np.linalg.norm(up) + 1e-12)
    return np.stack([fwd, right, up], axis=1)


def pixel_rays():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    px = (xx - W / 2.0) / FOCAL
    py = -(yy - H / 2.0) / FOCAL
    return np.stack([np.ones_like(px), px, py], axis=-1)


def warp_forward(src, depth, cam_t, R_t, cam_p, R_p, splat=7):
    rays = pixel_rays() @ R_t.T
    fwd_t = R_t[:, 0]
    denom = rays @ fwd_t
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    P = cam_t[None, None, :] + rays * (depth / denom)[..., None]

    rel = P - cam_p[None, None, :]
    Pc = rel @ R_p
    fc = Pc[..., 0]
    front = fc > 1e-3
    u = np.where(front, Pc[..., 1] / np.where(front, fc, 1), -1e9) * FOCAL + W / 2
    v = -np.where(front, Pc[..., 2] / np.where(front, fc, 1), -1e9) * FOCAL + H / 2

    out = np.zeros((H, W, 3), np.uint8)
    zb = np.full(H * W, np.inf)
    di = np.round(v).astype(np.int64)
    dj = np.round(u).astype(np.int64)
    offs = range(-(splat // 2), splat // 2 + 1)
    for oy in offs:
        for ox in offs:
            ii, jj = di + oy, dj + ox
            ok = front & (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
            fi, fj = ii[ok], jj[ok]
            zc = fc[ok]
            cols = src[np.nonzero(ok)[0], np.nonzero(ok)[1]]
            lin = fi * W + fj
            order = np.argsort(-zc)
            lo, zo, co = lin[order], zc[order], cols[order]
            flat = out.reshape(-1, 3)
            keep = zo < zb[lo]
            zb[lo[keep]] = zo[keep]
            flat[lo[keep]] = co[keep]
    holes = ~np.isfinite(zb.reshape(H, W))
    out[holes] = HOLE
    return out, holes


def run(cap_dir, out_dir, baseline="mt-2", pick="max", row=None, splat=7,
        flip=False, fov=FOV_DEG, frames_dir=None, image_offset=IMAGE_OFFSET):
    from PIL import Image
    set_fov(fov)
    frames_dir = frames_dir or find_frames_dir(cap_dir)
    fit = quiet(cd.run, cap_dir, quiet=True)
    nx, dxm, kx = fit["n_yaw"], fit["delta_yaw_ms"], fit["k_yaw"]
    ny, dym, ky = fit["n_pitch"], fit["delta_pitch_ms"], fit["k_pitch"]
    val_rows = fit["val_rows"]

    log = cd.read_frame_log(cap_dir, ["CamRotYaw", "CamRotPitch", "CamRotRoll"])
    t = cd.window_times(log)
    n = len(log)

    reports, _ = quiet(cd.get_input_reports, cap_dir)
    rt = reports["t"].astype(np.float64)
    Cx = np.concatenate([[0.0], np.cumsum(reports["dx"].astype(np.float64))])
    Cy = np.concatenate([[0.0], np.cumsum(reports["dy"].astype(np.float64))])
    cx = cd.window_counts_nd(t, rt, Cx, nx, dxm)
    cy = cd.window_counts_nd(t, rt, Cy, ny, dym)
    dyaw, _ = cd.resolve_seam(cd.per_frame_delta(log["CamRotYaw"]), cx, kx)
    dpitch = cd.per_frame_delta(log["CamRotPitch"])

    rot_pred = {
        "input": (kx * cx, ky * cy),
        "mt-2": (cd.history_pred(dyaw, 2), cd.history_pred(dpitch, 2)),
        "const-vel": (cd.history_pred(dyaw, 1), cd.history_pred(dpitch, 1)),
    }

    def rot_err(m):
        py, pp = rot_pred[m]
        return np.hypot(py - dyaw, pp - dpitch)

    err_base = rot_err(baseline)

    yaw = np.asarray(log["CamRotYaw"], np.float64)
    pitch = np.asarray(log["CamRotPitch"], np.float64)
    roll = np.asarray(log["CamRotRoll"], np.float64)
    pitch_w = cd.wrap_deg(pitch)

    jerk = np.zeros(n)
    jerk[1:] = np.hypot(np.diff(dyaw), np.diff(dpitch))

    v = val_rows[val_rows >= 1]
    rev = v[jerk[v] >= MIN_JERK_DEG]

    # ---- frame selection ----
    gfc = log["GFrameCounter"].astype(np.int64)
    have = file_ids(frames_dir, "frame_")

    def img_id(i):
        return int(gfc[i]) + image_offset

    def img_path(i):
        return os.path.join(frames_dir, f"frame_{img_id(i):06d}.bin")

    def eligible(i):
        s = i - 1
        return (s >= 0 and jerk[i] >= MIN_JERK_DEG
                and abs(pitch_w[s]) <= MAX_PITCH_DEG and abs(pitch_w[i]) <= MAX_PITCH_DEG
                and img_id(s) in have and img_id(i) in have)

    zero = np.zeros(3)

    def rotate(src, R_from, R_to, sp):
        # pure rotation: constant depth is exact, camera position stays at 0
        depth = np.full((H, W), 1000.0, np.float32)
        return warp_forward(src, depth, zero, R_from, zero, R_to, sp)

    def true_warp(i, sp):
        s = i - 1
        src = read_color(img_path(s), flip)
        R_s = cam_basis(yaw[s], pitch[s], roll[s])
        R_i = cam_basis(yaw[i], pitch[i], roll[i])
        w, h = rotate(src, R_s, R_i, sp)
        return src, R_s, w, h

    if row is not None:
        i = int(row)
        if not (1 <= i < n):
            raise SystemExit(f"--row {i} is outside 1..{n - 1}")
        if not eligible(i):
            print(f"WARNING: row {i} fails the selection rules")
        if i not in set(val_rows.tolist()):
            print(f"WARNING: row {i} is not a validation row")
    else:
        cands = [int(i) for i in rev if eligible(int(i))]
        if not cands:
            raise SystemExit("no eligible reversal frame with images")
        if pick == "max":
            cands.sort(key=lambda i: -err_base[i])
        else:
            med = float(np.median(err_base[cands]))
            cands.sort(key=lambda i: abs(err_base[i] - med))
        i = None
        for c in cands[:MAX_HOLE_CHECKS]:
            *_, h = true_warp(c, MEASURE_SPLAT)
            if h.mean() < MAX_HOLE_FRAC:
                i = c
                break
        if i is None:
            raise SystemExit(f"no candidate with under {MAX_HOLE_FRAC:.0%} holes "
                             f"in the true-rotation warp")

    s = i - 1
    src, R_s, true_img, _ = true_warp(i, splat)

    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    for m, name in ((baseline, f"history_{baseline}"), ("input", "input")):
        py, pp = float(rot_pred[m][0][i]), float(rot_pred[m][1][i])
        R_p = cam_basis(yaw[s] + py, pitch[s] + pp, roll[s])
        outputs[name] = rotate(src, R_s, R_p, splat)[0]
        e = float(np.hypot(py - dyaw[i], pp - dpitch[i]))
        print(f"{name}: error {e:.4f} deg")
    outputs["true_rotation"] = true_img

    print(f"row {i} (images {img_id(s)} -> {img_id(i)}), "
          f"true turn {float(np.hypot(dyaw[i], dpitch[i])):.2f} deg")
    for name, img in outputs.items():
        p = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(img).save(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir", required=True)
    ap.add_argument("--out_dir", default="warp_out")
    ap.add_argument("--baseline", choices=("mt-2", "const-vel"), default="mt-2")
    ap.add_argument("--pick", choices=("max", "median"), default="max",
                    help="max: frame where the history baseline is worst; "
                         "median: a typical reversal frame")
    ap.add_argument("--row", type=int, default=None,
                    help="use this (shifted) row instead of picking one")
    ap.add_argument("--splat", type=int, default=7)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--fov", type=float, default=FOV_DEG)
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--image_offset", type=int, default=IMAGE_OFFSET)
    cd.add_common_args(ap)
    a = ap.parse_args()
    cd.apply_common_args(a)
    run(a.cap_dir, a.out_dir, a.baseline, a.pick, a.row, a.splat, a.flip,
        a.fov, a.frames_dir, a.image_offset)