import os

import numpy as np

import fgcap


def wrap_angle_deg(deg):
    return ((deg + 180.0) % 360.0) - 180.0


def resolve_yaw_delta(mouse_ang, count_sum, k_ref=0.1747):
    """Correct yaw deltas that wrapped crossing the 0/360 seam.

    Uses lag-1 pairing: MouseAng[r] should be ~ k_ref * counts[r-1].
    """
    resolved = np.asarray(mouse_ang, dtype=np.float64).copy()
    cnt = np.asarray(count_sum, dtype=np.float64)
    predicted = k_ref * cnt
    # Lag-1 alignment: MouseAng[r] ~ k * counts[r-1].
    # Compare resolved[1:] (r >= 1) with predicted[:-1] (r-1 >= 0).
    diff = resolved[1:] - predicted[:-1]
    near_wrap = np.abs(diff) > 180.0
    opp_sign = np.sign(resolved[1:]) != np.sign(cnt[:-1])
    fix_sub = near_wrap & opp_sign & (np.abs(cnt[:-1]) > 0)
    resolved[1:][fix_sub] += 360.0 * np.sign(cnt[:-1][fix_sub])
    fix_mask = np.zeros(len(mouse_ang), dtype=bool)
    fix_mask[1:][fix_sub] = True
    return resolved, fix_mask


def true_delta_2d(cam_yaw, cam_pitch):
    n = len(cam_yaw)
    dyaw = np.zeros(n)
    dpitch = np.zeros(n)
    dyaw[:-1] = wrap_angle_deg(cam_yaw[1:] - cam_yaw[:-1])
    dpitch[:-1] = wrap_angle_deg(cam_pitch[1:] - cam_pitch[:-1])
    return dyaw, dpitch


def vec_mag(yaw, pitch):
    return np.sqrt(yaw * yaw + pitch * pitch)


# Pixel-space conversion for rotation error.
# At 90 deg horizontal FOV across 1920 px the focal length is 960 px, so
# near-centre error of theta deg = FOCAL * tan(theta) ~ 960 * pi/180 * theta.
PX_PER_DEG = 960.0 * np.pi / 180.0


def resolve_capture_paths(cap_dir):
    frames_dir = os.path.join(cap_dir, "Frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"{frames_dir}: missing Frames/ directory")
    frame_logs = [n for n in os.listdir(cap_dir)
                  if n.startswith("FrameLog_") and n.endswith(".csv")]
    input_logs = [n for n in os.listdir(cap_dir)
                  if n.startswith("InputLog_") and n.endswith(".bin")]
    if len(frame_logs) != 1:
        raise FileNotFoundError(f"{cap_dir}: expected one FrameLog_*.csv, found {len(frame_logs)}")
    if len(input_logs) != 1:
        raise FileNotFoundError(f"{cap_dir}: expected one InputLog_*.bin, found {len(input_logs)}")
    return frames_dir, os.path.join(cap_dir, frame_logs[0]), os.path.join(cap_dir, input_logs[0])


def get_data(frames_dir, frame_log_path, input_log_path):
    return fgcap.read_inputlog(input_log_path), fgcap.read_framelog(frame_log_path)