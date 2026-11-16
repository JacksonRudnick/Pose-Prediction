// Copyright Epic Games, Inc. All Rights Reserved.

#include "InputCaptureThread.h"

#include "CaptureOrchestrator.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Misc/Paths.h"

#if PLATFORM_LINUX
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

// Older kernel headers expose only the `time` timeval member; newer ones
// provide these accessor macros. Define them if absent so both compile.
#ifndef input_event_sec
#define input_event_sec time.tv_sec
#define input_event_usec time.tv_usec
#endif
#endif

#if PLATFORM_LINUX

/** True if the device at fd supports the given event type. */
static bool EvdevHasType(int fd, int type) {
  unsigned long bits[EV_MAX / (8 * sizeof(unsigned long)) + 1] = {};
  if (ioctl(fd, EVIOCGBIT(0, sizeof(bits)), bits) < 0)
    return false;
  return (bits[type / (8 * sizeof(unsigned long))] >>
          (type % (8 * sizeof(unsigned long)))) &
         1;
}

/** True if the device at fd supports the given event type + code. */
static bool EvdevHasCode(int fd, int type, int code) {
  unsigned long bits[KEY_MAX / (8 * sizeof(unsigned long)) + 1] = {};
  if (ioctl(fd, EVIOCGBIT(type, sizeof(bits)), bits) < 0)
    return false;
  return (bits[code / (8 * sizeof(unsigned long))] >>
          (code % (8 * sizeof(unsigned long)))) &
         1;
}

#endif // PLATFORM_LINUX

FInputCaptureThread::FInputCaptureThread(
    TQueue<FRawInputEvent, EQueueMode::Spsc> *InQueue)
    : EventQueue(InQueue) {}

FInputCaptureThread::~FInputCaptureThread() { CloseInputDevices(); }

bool FInputCaptureThread::Init() {
#if PLATFORM_LINUX
  if (!OpenInputDevices()) {
    UE_LOG(LogTemp, Warning,
           TEXT("[InputCapture] No evdev devices found — "
                "raw input unavailable, falling back to per-frame CSV input"));

    // Write diag file to help user debug evdev permissions
    {
      IPlatformFile &PF = FPlatformFileManager::Get().GetPlatformFile();
      IFileHandle *Diag =
          PF.OpenWrite(*(FPaths::ProjectSavedDir() / TEXT("CaptureData") /
                         TEXT("diag_input.txt")),
                       false, false);
      if (Diag) {
        const char *Msg =
            "InputCapture: No evdev devices could be opened.\n"
            "This usually means the running user is not in the 'input' group.\n"
            "Fix: sudo usermod -a -G input $USER\n"
            "Then log out and back in (or reboot) for the change to take "
            "effect.\n"
            "Alternative: run the game as root (not recommended).\n"
            "\n"
            "Without evdev, capture falls back to per-frame IsInputKeyDown +\n"
            "control-rotation delta in the CSV. This works but loses the raw\n"
            "mouse deltas (pre-sensitivity, pre-acceleration) that the "
            "training\n"
            "pipeline needs for the Light variant.\n";
        Diag->Write((const uint8 *)Msg, strlen(Msg));
        delete Diag;
      }
    }

    bAvailable = false;
    return true;
  }

  UE_LOG(LogTemp, Log,
         TEXT("[InputCapture] Opened %d evdev device(s) for raw input"),
         DeviceFds.Num());
  bAvailable = true;
  return true;
#else
  UE_LOG(LogTemp, Log,
         TEXT("[InputCapture] Evdev raw input not supported on this platform"));
  bAvailable = false;
  return true;
#endif
}

uint32 FInputCaptureThread::Run() {
#if PLATFORM_LINUX
  if (!bAvailable || DeviceFds.Num() == 0) {
    return 0;
  }

  // Switch evdev devices from CLOCK_REALTIME to CLOCK_MONOTONIC
  // so kernel timestamps match FCaptureClock::Now() — no calibration needed
  TArray<bool> UseKernelTimestamps;
  UseKernelTimestamps.Reserve(DeviceFds.Num());
  for (int32 fd : DeviceFds) {
    int ClockId = CLOCK_MONOTONIC;
    const bool bOk = (ioctl(fd, EVIOCSCLOCKID, &ClockId) == 0);
    if (!bOk) {
      UE_LOG(LogTemp, Warning,
             TEXT("[InputCapture] EVIOCSCLOCKID failed on fd %d — "
                  "falling back to receipt-time stamps for this device"),
             fd);
    }
    UseKernelTimestamps.Add(bOk);
  }

  TArray<pollfd> PollFds;
  for (int32 fd : DeviceFds) {
    PollFds.Add({fd, POLLIN, 0});
  }

  struct input_event EvBuf[64];

  while (!bStopRequested) {
    int ret = poll(PollFds.GetData(), PollFds.Num(), 100); // 100ms timeout
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      UE_LOG(LogTemp, Error, TEXT("[InputCapture] poll() failed: %d"), errno);
      break;
    }
    if (ret == 0)
      continue;

    for (int32 i = 0; i < PollFds.Num(); ++i) {
      // Device error / unplugged: close it and stop polling it, or a
      // hung-up fd makes poll() return immediately forever (busy spin).
      if (PollFds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
        UE_LOG(LogTemp, Warning,
               TEXT("[InputCapture] Device fd %d hung up — dropping it"),
               PollFds[i].fd);
        close(PollFds[i].fd);
        DeviceFds[i] = -1;  // guard against double-close in shutdown
        PollFds[i].fd = -1; // negative fds are ignored by poll()
        PollFds[i].revents = 0;
        continue;
      }

      if (!(PollFds[i].revents & POLLIN))
        continue;

      // Non-blocking drain: read until EAGAIN (fds keep O_NONBLOCK).
      for (;;) {
        ssize_t n = read(PollFds[i].fd, EvBuf, sizeof(EvBuf));
        if (n < 0) {
          if (errno == EINTR)
            continue;
          break; // EAGAIN: device drained
        }
        if (n == 0)
          break;

        const double ReceiptTime = FCaptureClock::Now();

        int Count = static_cast<int>(n / sizeof(struct input_event));
        for (int j = 0; j < Count; ++j) {
          const struct input_event &KernelEv = EvBuf[j];

          // Keep key, relative-motion, and sync events; SYN_REPORT
          // marks report boundaries so the offline parser can group
          // simultaneous REL_X/REL_Y into one mouse report. EV_MSC
          // scan codes and everything else are noise for our purposes.
          if (KernelEv.type != EV_KEY && KernelEv.type != EV_REL &&
              KernelEv.type != EV_SYN)
            continue;

          FRawInputEvent Event;
          if (UseKernelTimestamps[i]) {
            // Kernel timestamp is already CLOCK_MONOTONIC (set via
            // EVIOCSCLOCKID), same domain as FCaptureClock::Now() —
            // no conversion needed.
            Event.Timestamp = (double)KernelEv.input_event_sec +
                              (double)KernelEv.input_event_usec * 1e-6;
          } else {
            Event.Timestamp = ReceiptTime;
          }
          Event.Type = KernelEv.type;
          Event.Code = KernelEv.code;
          Event.Value = KernelEv.value;

          EventQueue->Enqueue(Event);
        }
      }
    }
  }
#else
  FPlatformProcess::Sleep(0.01f);
#endif

  return 0;
}

void FInputCaptureThread::Stop() {
  bStopRequested = true;
}

#if PLATFORM_LINUX
bool FInputCaptureThread::OpenInputDevices() {
  for (int i = 0; i < 64; ++i) {
    FString Path = FString::Printf(TEXT("/dev/input/event%d"), i);
    FTCHARToUTF8 Utf8(*Path);

    int fd = open(Utf8.Get(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
      continue;

    bool bIsMouse = EvdevHasType(fd, EV_REL) &&
                    EvdevHasCode(fd, EV_REL, REL_X) &&
                    EvdevHasCode(fd, EV_REL, REL_Y);

    bool bIsKeyboard =
        EvdevHasType(fd, EV_KEY) &&
        (EvdevHasCode(fd, EV_KEY, KEY_A) || EvdevHasCode(fd, EV_KEY, KEY_W) ||
         EvdevHasCode(fd, EV_KEY, KEY_ENTER));

    if (bIsMouse || bIsKeyboard) {
      // fds stay O_NONBLOCK so Run()'s drain loop gets EAGAIN when empty
      DeviceFds.Add(fd);

      char devname[256] = {};
      ioctl(fd, EVIOCGNAME(sizeof(devname)), devname);
      UE_LOG(LogTemp, Log,
             TEXT("[InputCapture] Opened /dev/input/event%d (%s) — %s"), i,
             UTF8_TO_TCHAR(devname),
             bIsMouse ? TEXT("mouse") : TEXT("keyboard"));
    } else {
      close(fd);
    }
  }

  return DeviceFds.Num() > 0;
}
#endif // PLATFORM_LINUX

void FInputCaptureThread::CloseInputDevices() {
#if PLATFORM_LINUX
  for (int32 fd : DeviceFds) {
    if (fd >= 0)
      close(fd);
  }
#endif
  DeviceFds.Empty();
}