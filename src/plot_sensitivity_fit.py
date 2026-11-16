"""Plot the through-origin sensitivity fits from e_rot: training points,
validation points, and the fitted line for yaw and pitch.
"""

import argparse
import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np

import e_rot


def _fit_data(cap_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        result = e_rot.run(cap_dir, quiet=True)

    log = e_rot.get_frame_log_data(cap_dir, result["shift"])
    reports, _ = e_rot.get_input_reports(cap_dir)
    t = e_rot.window_times(log, result["clock"])
    report_t = reports["t"].astype(np.float64)
    counts_cum = {
        "Yaw": np.concatenate([[0.0], np.cumsum(reports["dx"].astype(np.float64))]),
        "Pitch": np.concatenate([[0.0], np.cumsum(reports["dy"].astype(np.float64))]),
    }
    angles = {
        "Yaw": e_rot.per_frame_delta(log["CamRotYaw"]),
        "Pitch": e_rot.per_frame_delta(log["CamRotPitch"]),
    }
    settings = {
        "Yaw": (result["n_yaw"], result["delta_yaw_ms"], result["k_yaw"]),
        "Pitch": (result["n_pitch"], result["delta_pitch_ms"], result["k_pitch"]),
    }

    data = {}
    for axis in ("Yaw", "Pitch"):
        n_ticks, delta_ms, sensitivity = settings[axis]
        counts = e_rot.window_counts_nd(
            t, report_t, counts_cum[axis], n_ticks, delta_ms
        )
        target = angles[axis]
        if axis == "Yaw":
            target, _ = e_rot.resolve_seam(target, counts, sensitivity)
        data[axis] = {
            "counts": counts,
            "target": target,
            "sensitivity": sensitivity,
            "n_ticks": n_ticks,
            "delta_ms": delta_ms,
            "fit_rows": result["fit_rows"],
            "val_rows": result["val_rows"],
        }
    return data


def _plot_axis(ax, axis, values, max_points):
    counts = values["counts"]
    target = values["target"]
    fit_rows = values["fit_rows"]
    val_rows = values["val_rows"]
    sensitivity = values["sensitivity"]

    rng = np.random.default_rng(0)
    if max_points and len(fit_rows) > max_points:
        fit_plot = np.sort(rng.choice(fit_rows, max_points, replace=False))
    else:
        fit_plot = fit_rows
    if max_points and len(val_rows) > max_points:
        val_plot = np.sort(rng.choice(val_rows, max_points, replace=False))
    else:
        val_plot = val_rows

    ax.scatter(
        counts[fit_plot],
        target[fit_plot],
        s=5,
        alpha=0.25,
        label=f"fit ({len(fit_rows):,})",
    )
    ax.scatter(
        counts[val_plot],
        target[val_plot],
        s=12,
        alpha=0.7,
        label=f"validation ({len(val_rows):,})",
    )

    finite = np.isfinite(counts) & np.isfinite(target)
    x_limit = np.max(np.abs(counts[finite])) if finite.any() else 1.0
    x = np.array([-x_limit, x_limit])
    ax.plot(x, sensitivity * x, color="black", linewidth=1.5,
            label=f"fit: {sensitivity:+.5f} deg/count")

    rows = np.concatenate([fit_rows, val_rows])
    mask = np.isfinite(counts[rows]) & np.isfinite(target[rows])
    residual = target[rows][mask] - sensitivity * counts[rows][mask]
    ss_tot = np.sum((target[rows][mask] - np.mean(target[rows][mask])) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / max(ss_tot, 1e-12)

    ax.axhline(0.0, color="0.7", linewidth=0.7)
    ax.axvline(0.0, color="0.7", linewidth=0.7)
    ax.set_title(
        f"{axis}: n={values['n_ticks']}, delay={values['delta_ms']:+.3f} ms, "
        f"R²={r2:.5f}"
    )
    ax.set_xlabel("Mouse counts in selected input window")
    ax.set_ylabel("Camera rotation delta (degrees)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize="small")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cap_dir")
    parser.add_argument("--out", default=None, help="output path")
    parser.add_argument("--max-points", type=int, default=10000)
    args = parser.parse_args()

    if args.max_points < 0:
        parser.error("--max-points must be non-negative")
    if not os.path.isdir(args.cap_dir):
        parser.error(f"capture directory does not exist: {args.cap_dir}")

    data = _fit_data(args.cap_dir)
    output = args.out or os.path.join(args.cap_dir, "sensitivity_fit.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, axis in zip(axes, ("Yaw", "Pitch")):
        _plot_axis(ax, axis, data[axis], args.max_points)
    fig.suptitle("Input counts to camera rotation sensitivity fit")
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
