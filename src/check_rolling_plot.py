"""Plot rotation error over time after injected changes.  Rows = sessions,
columns = delay +1 ms, sensitivity x2, delay +1 tick.  Trailing mean over
mouse-motion time, restarted at the change point.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (column title, run() keyword arguments)
CHANGES = (
    ("delay +1 ms", {"inject_delay_ms": 1.0}),
    ("sensitivity ×2", {"inject_k_scale": 2.0}),
    ("delay +1 tick", {"inject_delay_ticks": 1}),
)

# half-life lines take categorical slots in fixed order; the two references are neutral
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
OFFLINE_STYLE = {"color": "#8a8985", "lw": 1.2, "ls": "-"}
ORACLE_STYLE = {"color": "#0b0b0b", "lw": 1.0, "ls": (0, (4, 2))}


def load_script(path):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("check_rolling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trailing_mean(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - w, 0)
    return (cs[i] - cs[lo]) / (i - lo)


def curves(res, window):
    mrows = np.flatnonzero(res["moving"])
    k0 = int(np.searchsorted(mrows, res["start"]))      # first moving frame after the change
    x = (np.arange(len(mrows)) - k0) / res["fps"]

    def smooth(err):
        e = err[mrows]
        return np.concatenate([trailing_mean(e[:k0], window), trailing_mean(e[k0:], window)])

    lines = {"offline": smooth(res["err_offline"]), "oracle": smooth(res["err_oracle"])}
    rec_x = {}
    for name, r in res["rolling"].items():
        lines[name] = smooth(r["err"])
        if r["rec_err"] is not None:
            rec_x[name] = (int(np.searchsorted(mrows, r["rec_err"])) - k0) / res["fps"]
    return x, k0, lines, rec_x


def short(name):
    hl = name.split("=")[-1]
    return "never forget" if hl == "inf" else f"hl {hl}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cap_dirs", nargs="+", help="capture dirs, one row each")
    ap.add_argument("--script", required=True,
                    help="path to the real-time estimator script (with run())")
    ap.add_argument("--names", nargs="+",
                    help="row titles (default: the cap dir folder names)")
    ap.add_argument("--halflives", nargs="+", default=None,
                    help="half-lives in learning updates, 'inf' = never forget "
                         "(default: the script's DEFAULT_HALFLIVES)")
    ap.add_argument("--inject_at", type=float, default=0.5,
                    help="change point as a fraction of the session (default 0.5)")
    ap.add_argument("--window", type=int, default=None,
                    help="trailing mean over this many moving frames (default: ERR_WINDOW)")
    ap.add_argument("--pre", type=float, default=10.0,
                    help="seconds of mouse motion to show before the change (default 10)")
    ap.add_argument("--post", type=float, default=None,
                    help="seconds of mouse motion to show after it (default: to the end)")
    ap.add_argument("--linear", action="store_true", help="linear y-axis instead of log")
    ap.add_argument("--out", default="rt_after_change.pdf",
                    help="output figure; format from the extension (default rt_after_change.pdf)")
    # also used by e_rt's own CLI; passed through to e_rot so the settings match
    # (e.g. hist stride).  Loaded before parsing, so --script must come first
    # on the command line only if e_rot needs it -- it doesn't: we pre-parse.
    pre, _ = ap.parse_known_args()
    mod = load_script(pre.script)
    cd = mod.cd
    cd.add_common_args(ap)
    a = ap.parse_args()
    cd.apply_common_args(a)

    names = a.names or [os.path.basename(os.path.normpath(d)) for d in a.cap_dirs]
    if len(names) != len(a.cap_dirs):
        ap.error("--names needs one name per cap dir")
    hls = [np.inf if h == "inf" else float(h) for h in (a.halflives or mod.DEFAULT_HALFLIVES)]
    window = a.window or mod.ERR_WINDOW

    nr, nc = len(a.cap_dirs), len(CHANGES)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.4 * nr + 0.6),
                             sharex=False, sharey="row", squeeze=False)
    handles = {}
    for r, (cap, sess) in enumerate(zip(a.cap_dirs, names)):
        for c, (title, kw) in enumerate(CHANGES):
            print(f"\n===== {sess}: {title} =====", flush=True)
            res = mod.run(cap, hls, inject_at=a.inject_at, **kw)
            x, k0, lines, rec_x = curves(res, window)
            ax = axes[r, c]
            keep = x >= -a.pre
            if a.post is not None:
                keep &= x <= a.post
            pre_m, post_m = keep & (x < 0), keep & (x >= 0)

            def draw(y, label, **style):
                # pre and post drawn as separate segments: the window restarts at the change
                h, = ax.plot(x[pre_m], y[pre_m], **style)
                ax.plot(x[post_m], y[post_m], **style)
                handles.setdefault(label, h)

            draw(lines["offline"], "offline (fixed)", **OFFLINE_STYLE)
            for i, name in enumerate(res["rolling"]):
                col = SERIES_COLORS[i % len(SERIES_COLORS)]
                draw(lines[name], short(name), color=col, lw=1.2)
                if name in rec_x and (a.post is None or rec_x[name] <= a.post):
                    xr = rec_x[name]
                    j = int(np.searchsorted(x, xr))
                    if j < len(x):
                        ax.plot([xr], [lines[name][j]], "o", ms=4.5, color=col,
                                mec="white", mew=0.8, zorder=5)
            draw(lines["oracle"], "oracle (calibrated to the change)", **ORACLE_STYLE)

            ax.axvline(0.0, color="#52514e", lw=0.8, zorder=0)
            if not a.linear:
                ax.set_yscale("log")
            ax.grid(True, which="major", color="#e4e3df", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{sess}\nerror (deg)", fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("seconds of mouse motion since the change", fontsize=9)
            ax.tick_params(labelsize=8)

    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center",
               ncol=min(len(handles), 6), fontsize=8, frameon=False)
    fig.text(0.5, 0.005, f"Trailing mean over {window} moving frames, restarted at the change. "
             "Dot = error-based recovery.", ha="center", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.4 / (2.4 * nr + 0.6)))
    fig.savefig(a.out, dpi=200)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()