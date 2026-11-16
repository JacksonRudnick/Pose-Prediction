# InPose: Raw Input as a Pose Signal for Frame Extrapolation

Code for my CS master's thesis at the University of Central Arkansas (December 2026).

Frame extrapolation raises frame rate without the extra latency of interpolation, but it has to guess the camera pose of the generated frame. Current G-buffer-free methods guess it from past motion, so they are wrong exactly when the motion changes. This project shows that the raw mouse and keyboard input the player has already sent can stand in for that guess.

This is **not** a new frame-generation pipeline. It is a better pose signal that any extrapolation pipeline can use. No images are generated; only the pose is evaluated.

## Main results

Measured in Unreal Engine 5.7 on three recorded sessions (3 maps, 60 FPS, simulated 2× extrapolation).

**Rotation (mouse)**
- One delay and one sensitivity (k ≈ 0.175 °/count) explain at least 99.77% of squared per-frame rotation.
- Error is 95.0–99.1% lower than the best history baseline (mt-2), and 94.4–99.97% lower on large-motion frames. Significant in every bucket on every session.
- Mean error of 0.02–0.12°, which is about 0.3–2 px on a 1080p, 90° FOV screen (history baselines: 21–63 px).
- Cost: under 1 µs per frame.

**Real-time estimator**
- Finds the delay and sensitivity while the app runs, with no engine-side calibration.
- With a half-life of 30 updates, it recovers from an injected delay or sensitivity change within 0.7–8.4 s of mouse motion.
- Cost: about 20 µs on the frames where it learns (243 candidates).

**Translation (keyboard)**
- Keys land on the frame the engine reads them for 99.6–99.7% of key changes.
- A movement model driven by keys and camera yaw cuts position error by 21.6–40.5% on large-motion frames.
- Used only within 3 frames of a key change: 14.3–18.2% better than mt-2 on every session. On every frame, it is not reliably better.
- Main takeaway: for translation, input is mostly a *timing* signal that says when history-based extrapolation is about to fail.

## How it works

1. **Input window.** Each frame's rotation comes from the mouse counts read at the start of that same frame. The window is tied to the engine's input-read times, plus a small offset δ (about 0.12–0.18 ms).
2. **Offline fit.** For each candidate delay, the best sensitivity has a closed form (k = A/B from two running sums). The delay with the highest score wins. Yaw and pitch scores are normalized and added together for a joint fit.
3. **Real-time estimator.** The same sums are updated every frame with exponential decay (half-life h). This covers 243 candidates: n ∈ {0, 1, 2} whole frames and δ from −2 to +2 ms in 0.05 ms steps.
4. **Translation model.** The next step is a mix of the previous velocity and the key direction (rotated by camera yaw), capped at a top speed. A separate braking model is used when no key is held. Weights are fitted with least squares.
5. **Baselines.** Constant velocity, second-order extrapolation (mt-2), and a five-point line fit (mt-5). Each result is compared against the stronger baseline in that group.
6. **Stats.** 60-frame block bootstrap (10,000 resamples) with 95% confidence intervals.

## Citation

```bibtex
@mastersthesis{rudnick2026inpose,
  author = {Jackson Rudnick},
  title  = {InPose: Raw Input as a Pose Signal for Frame Extrapolation},
  school = {University of Central Arkansas},
  address = {Conway, Arkansas},
  year   = {2026},
  month  = {December}
}
```

## Acknowledgements

Advisor: Jonathan Baarsch. Committee: Dr. Ademola Adesokan and Dr. Jobair Hossain Faruk.