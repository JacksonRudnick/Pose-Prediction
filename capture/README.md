# Capture: Unreal Engine data recorder

Unreal Engine 5.7 project module (`FrameGen`) that records what the pose-prediction experiments in `../src` need: raw mouse and keyboard events and per-frame engine state, on one shared clock. Optionally it also saves color and depth frames.

Raw input capture is **Linux only** (it reads evdev). On other platforms the module still logs the per-frame CSV, but there is no `InputLog` data.

## Read this before running

**The input thread records every key from every keyboard on the machine, not just W/A/S/D/Space.** It opens each `/dev/input/event*` mouse and keyboard device and logs every `EV_KEY`, `EV_REL` and `EV_SYN` event, including keys typed into other windows. Don't type passwords or anything private during a capture, and don't publish raw `InputLog_*.bin` files from a machine you also use for other things.

## Requirements

- Unreal Engine 5.7
- Linux, with your user in the `input` group so `/dev/input/event*` can be opened:
  ```bash
  sudo usermod -a -G input $USER
  ```
  Log out and back in afterwards. If no device can be opened, the thread writes `Saved/CaptureData/diag_input.txt` explaining why and capture continues without raw input.
- `Content/` (a map and a playable character) is not included in this repo.

## Build

Open `FrameGen.uproject` with UE 5.7 (right-click the file and choose "Switch Unreal Engine version" if it isn't associated), or generate project files and build the `FrameGen` game target.

## What it records

Everything goes to `<Project>/Saved/CaptureData/`, tagged with a session id `YYYYMMDD_HHMMSS`:

| Output | Contents |
|---|---|
| `FrameLog_<id>.csv` | One row per game tick: timestamps, key state, camera and character pose |
| `InputLog_<id>.bin` | Raw evdev events at their native rate (mouse around 1 kHz) |
| `Frames/frame_NNNNNN.bin` | Back-buffer color, burst frames only |
| `Depth/depth_NNNNNN.bin` | Scene depth, burst frames only |
| `diag_input.txt` | Only written if no input device could be opened |

Pass `-nocapture` on the command line to disable the whole subsystem.

### Timing

The capture subsystem forces a fixed timestep of `TargetFramerate x 2` (default 30 -> 60 fps), turns VSync off and sets `t.MaxFPS` to match.

All timestamps are `CLOCK_MONOTONIC` seconds. The evdev devices are switched to that clock with `EVIOCSCLOCKID`, so input events and frame timestamps are directly comparable with no calibration. If that call fails on a device, its events are stamped with the time the read returned instead (less accurate).

### `InputLog_*.bin`

A 16-byte header followed by 16-byte records, all little-endian.

```
header:  uint32 magic = 0x4E494746 ("FGIN"), uint32 version = 1, uint64 reserved = 0
record:  double t, uint16 type, uint16 code, int32 value      (evdev type/code/value)
```

Only `EV_KEY` (keys), `EV_REL` (mouse motion: `REL_X`, `REL_Y`, raw counts before any sensitivity or acceleration) and `EV_SYN` (report boundaries) are kept. `../src/fgcap.py` parses this format.

### `FrameLog_*.csv`

| Column(s) | Meaning |
|---|---|
| `FrameIndex` | Capture tick counter (increments every tick) |
| `GFrameCounter` | Engine game-thread frame counter |
| `TickTimestamp` | Time at the start of the world tick (`OnWorldTickStart`) |
| `BeginFrameTimestamp` | Time of `OnBeginFrame` |
| `InputSamplingTimestamp` | Time of `OnSamplingInput`, just before the engine polls OS input. The experiments use this as the input-window clock. |
| `TimeSeconds`, `DeltaTime` | World time and tick delta |
| `Key_W/A/S/D/Jump` | `IsInputKeyDown` for W, A, S, D, Space as the engine sees them |
| `MouseAngX/Y` | Control-rotation change this tick (yaw, pitch), in degrees. This is after sensitivity, not raw counts. |
| `MouseAngRateX/Y` | The same, divided by `DeltaTime` |
| `VelX/Y/Z` | Camera velocity from the change in camera location |
| `CamAngVelPitch/Yaw` | Camera angular velocity from the change in view rotation |
| `CamLocX/Y/Z`, `CamRotPitch/Yaw/Roll` | Player view point |
| `CharLocX/Y/Z`, `CharVelX/Y/Z` | Pawn location and character-movement velocity |
| `Teleport` | 1 if camera speed exceeds about 50,000 cm/s or angular speed exceeds 5000 °/s |
| `InBurst` | 1 if this frame is inside a color/depth burst window |
| `DevZToWorldZ_X/Y/Z/W`, `ProjNearClip` | Projection constants for linearizing depth (reverse-Z, infinite far plane) |

State columns describe the pose as read at the tick. The analysis code shifts them back one row so that row *i* holds frame *i*'s own pose (`SHIFT_STATE` in `../src/e_rot.py`).

### Color and depth frames

These are raw dumps with **no header**: the width and height are not stored in the files.

- `Frames/frame_NNNNNN.bin`: BGRA8 (`FColor`), the size of the game window's back buffer.
- `Depth/depth_NNNNNN.bin`: little-endian `float32`, row-major, from a `SceneCapture2D` with `SCS_SceneDepth` at `DepthCaptureResolutionX × DepthCaptureResolutionY` (default 1920 × 1080).

Frames are only saved in bursts: `BurstLength` consecutive frames out of every `BurstPeriod` (default 8 of 16, chosen by frame counter). The file number is the render-thread frame counter; `GFrameCounter` in the CSV is the game-thread counter, so check the offset between them before joining frames to rows. 

## Configuration

Settings live on `UCaptureOrchestrator` and are read from the `Game` config, section `[/Script/FrameGen.CaptureOrchestrator]` in `DefaultGame.ini`:

| Setting | Default | |
|---|---|---|
| `TargetFramerate` | 30 | Base rate; the tick runs at 2x this |
| `bCaptureColor` | true | Save color frames |
| `bCaptureDepth` | true | Save depth frames |
| `bCaptureRawInput` | true | Run the evdev thread |
| `DepthCaptureResolutionX/Y` | 1920 / 1080 | |
| `BurstLength` / `BurstPeriod` | 8 / 16 | Clamped so length <= period |

## How it works

- **Game thread** (`CaptureOrchestrator`): hooks the world tick, builds the CSV row, and starts GPU readbacks from the back buffer and the depth target.
- **Input thread** (`InputCaptureThread`): polls the evdev devices at above-normal priority and pushes events to a queue. It is independent of the game's frame rate.
- **Writer thread** (`FrameDataWriter`): lowest priority. It drains the queues and writes to disk, so nothing on the render or game thread waits on I/O.

Readback uses a ring of slots. If the writer can't keep up, frames are dropped rather than stalling the game, and the counts (armed, enqueued, ring-exhausted, cap-dropped) are logged at shutdown. Check them before trusting a capture.

## Layout

```
FrameGen.uproject
Source/
  FrameGen.Target.cs
  FrameGen/
    FrameGen.Build.cs
    Public/    CaptureOrchestrator.h  FrameDataWriter.h  InputCaptureThread.h
    Private/   CaptureOrchestrator.cpp  FrameDataWriter.cpp  InputCaptureThread.cpp
```