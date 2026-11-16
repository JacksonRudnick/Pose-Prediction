// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Containers/Queue.h"
#include "CoreMinimal.h"
#include "HAL/Runnable.h"

/** One evdev input event (CLOCK_MONOTONIC timestamp, matching FCaptureClock). */
struct FRawInputEvent {
  double Timestamp;
  uint16 Type;
  uint16 Code;
  int32  Value;
};

/** Reads evdev devices (/dev/input/event*) via poll()/read() at above-normal priority. */
class FInputCaptureThread : public FRunnable {
public:
  FInputCaptureThread(TQueue<FRawInputEvent, EQueueMode::Spsc> *InQueue);
  virtual ~FInputCaptureThread();

  virtual bool Init() override;
  virtual uint32 Run() override;
  virtual void Stop() override;

    bool IsAvailable() const { return bAvailable; }

private:
  bool OpenInputDevices();
  void CloseInputDevices();

  TQueue<FRawInputEvent, EQueueMode::Spsc> *EventQueue;

  FThreadSafeBool bStopRequested = false;
  FThreadSafeBool bAvailable = false;

  TArray<int32> DeviceFds;
};