// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Containers/Queue.h"
#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/PlatformFileManager.h"

#include "InputCaptureThread.h"

struct FColorFrameWrite {
  uint64 FrameIndex = ~0ull;
  FString Path;
  TArray<FColor> Pixels;
};

struct FDepthFrameWrite {
  uint64 FrameIndex = ~0ull;
  FString Path;
  TArray<float> DepthData;
  FIntPoint Size = FIntPoint::ZeroValue;
};

/** Low-priority writer thread draining SPSC queues to disk (input, CSV, color, depth). */
class FFrameDataWriter : public FRunnable {
public:
  FFrameDataWriter(
      TQueue<FRawInputEvent, EQueueMode::Spsc> *InInputQueue,
      TQueue<FString, EQueueMode::Spsc> *InCsvLineQueue,
      TQueue<FColorFrameWrite, EQueueMode::Spsc> *InColorQueue,
      FThreadSafeCounter *InPendingColorWrites,
      TQueue<FDepthFrameWrite, EQueueMode::Spsc> *InDepthQueue,
      FThreadSafeCounter *InPendingDepthWrites,
      const FString &InInputLogPath,
      const FString &InCsvPath,
      const FString &InCsvHeader);

  virtual ~FFrameDataWriter();

  virtual bool Init() override;
  virtual uint32 Run() override;
  virtual void Stop() override;
  virtual void Exit() override;

private:
  void FlushInputBuffer();

  TQueue<FRawInputEvent, EQueueMode::Spsc> *InputQueue;
  TQueue<FString, EQueueMode::Spsc> *CsvLineQueue;
  TQueue<FColorFrameWrite, EQueueMode::Spsc> *ColorQueue;
  TQueue<FDepthFrameWrite, EQueueMode::Spsc> *DepthQueue;
  FThreadSafeCounter *PendingColorWrites;
  FThreadSafeCounter *PendingDepthWrites;

  FString InputLogPath;
  FString CsvPath;
  FString CsvHeader;

  IFileHandle *InputLogFile = nullptr;
  IFileHandle *CsvFile = nullptr;

  TArray<uint8> InputBuffer;
  FString CsvBuffer;

  double LastFlushTime = 0.0;
  int32 InputEventsSinceFlush = 0;

  FThreadSafeBool bStopRequested = false;

  static constexpr int32 MaxInputEventsBeforeFlush = 2048;
  static constexpr double FlushIntervalSec = 0.5;
};