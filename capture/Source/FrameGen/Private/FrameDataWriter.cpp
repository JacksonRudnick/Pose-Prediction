// Copyright Epic Games, Inc. All Rights Reserved.

#include "FrameDataWriter.h"

#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/ScopeLock.h"

// Binary input log: [16-byte header: uint32 Magic(FGIN), uint32 Version, uint64 Reserved]
// then 16-byte records: double Timestamp, uint16 Type, uint16 Code, int32 Value
static constexpr uint32 kInputLogMagic = 0x4E494746;

struct FInputLogHeader {
  uint32 Magic = kInputLogMagic;
  uint32 Version = 1;
  uint64 Reserved = 0;
};

FFrameDataWriter::FFrameDataWriter(
    TQueue<FRawInputEvent, EQueueMode::Spsc> *InInputQueue,
    TQueue<FString, EQueueMode::Spsc> *InCsvLineQueue,
    TQueue<FColorFrameWrite, EQueueMode::Spsc> *InColorQueue,
    FThreadSafeCounter *InPendingColorWrites,
    TQueue<FDepthFrameWrite, EQueueMode::Spsc> *InDepthQueue,
    FThreadSafeCounter *InPendingDepthWrites,
    const FString &InInputLogPath,
    const FString &InCsvPath,
    const FString &InCsvHeader)
    : InputQueue(InInputQueue)
    , CsvLineQueue(InCsvLineQueue)
    , ColorQueue(InColorQueue)
    , DepthQueue(InDepthQueue)
    , PendingColorWrites(InPendingColorWrites)
    , PendingDepthWrites(InPendingDepthWrites)
    , InputLogPath(InInputLogPath)
    , CsvPath(InCsvPath)
    , CsvHeader(InCsvHeader) {}

FFrameDataWriter::~FFrameDataWriter() {
  if (InputLogFile) delete InputLogFile;
  if (CsvFile) delete CsvFile;
}

bool FFrameDataWriter::Init() {
  IPlatformFile &PF = FPlatformFileManager::Get().GetPlatformFile();

  InputLogFile = PF.OpenWrite(*InputLogPath, false, false);
  if (!InputLogFile) {
    UE_LOG(LogTemp, Error,
           TEXT("[FrameDataWriter] Failed to open input log: %s"),
           *InputLogPath);
    return false;
  }

  FInputLogHeader Hdr;
  InputLogFile->Write(reinterpret_cast<const uint8 *>(&Hdr), sizeof(Hdr));

  CsvFile = PF.OpenWrite(*CsvPath, false, false);
  if (!CsvFile) {
    UE_LOG(LogTemp, Error,
           TEXT("[FrameDataWriter] Failed to open CSV: %s"), *CsvPath);
    return false;
  }

  FTCHARToUTF8 HdrUtf8(*CsvHeader);
  CsvFile->Write(reinterpret_cast<const uint8 *>(HdrUtf8.Get()), HdrUtf8.Length());

  LastFlushTime = FPlatformTime::Seconds();
  UE_LOG(LogTemp, Log,
         TEXT("[FrameDataWriter] Started — %s, %s"),
         *InputLogPath, *CsvPath);
  return true;
}

uint32 FFrameDataWriter::Run() {
  while (!bStopRequested) {
    bool bDidWork = false;

    FRawInputEvent Ev;
    while (InputQueue->Dequeue(Ev)) {
      InputBuffer.Append(reinterpret_cast<const uint8 *>(&Ev),
                          sizeof(FRawInputEvent));
      ++InputEventsSinceFlush;
      bDidWork = true;
    }

    FString Line;
    while (CsvLineQueue->Dequeue(Line)) {
      CsvBuffer += Line;
      bDidWork = true;
    }

    FColorFrameWrite ColorJob;
    while (ColorQueue->Dequeue(ColorJob)) {
      TArrayView64<const uint8> Raw(
          reinterpret_cast<const uint8 *>(ColorJob.Pixels.GetData()),
          ColorJob.Pixels.Num() * sizeof(FColor));
      FFileHelper::SaveArrayToFile(Raw, *ColorJob.Path);
      if (PendingColorWrites) {
        PendingColorWrites->Decrement();
      }
      bDidWork = true;
    }

    FDepthFrameWrite DepthJob;
    while (DepthQueue->Dequeue(DepthJob)) {
      TArrayView64<const uint8> Raw(
          reinterpret_cast<const uint8 *>(DepthJob.DepthData.GetData()),
          DepthJob.DepthData.Num() * sizeof(float));
      FFileHelper::SaveArrayToFile(Raw, *DepthJob.Path);
      if (PendingDepthWrites) {
        PendingDepthWrites->Decrement();
      }
      bDidWork = true;
    }

    const double Now = FPlatformTime::Seconds();
    if (InputEventsSinceFlush >= MaxInputEventsBeforeFlush ||
        (Now - LastFlushTime) >= FlushIntervalSec) {
      FlushInputBuffer();
      LastFlushTime = Now;
    }

    if (!bDidWork) {
      FPlatformProcess::Sleep(0.002f);
    }
  }

  FRawInputEvent Ev;
  while (InputQueue->Dequeue(Ev)) {
    InputBuffer.Append(reinterpret_cast<const uint8 *>(&Ev),
                        sizeof(FRawInputEvent));
  }
  FString Line;
  while (CsvLineQueue->Dequeue(Line)) {
    CsvBuffer += Line;
  }
  FColorFrameWrite ColorJob;
  while (ColorQueue->Dequeue(ColorJob)) {
    TArrayView64<const uint8> Raw(
        reinterpret_cast<const uint8 *>(ColorJob.Pixels.GetData()),
        ColorJob.Pixels.Num() * sizeof(FColor));
    FFileHelper::SaveArrayToFile(Raw, *ColorJob.Path);
    if (PendingColorWrites) PendingColorWrites->Decrement();
  }
  FDepthFrameWrite DepthJob;
  while (DepthQueue->Dequeue(DepthJob)) {
    TArrayView64<const uint8> Raw(
        reinterpret_cast<const uint8 *>(DepthJob.DepthData.GetData()),
        DepthJob.DepthData.Num() * sizeof(float));
    FFileHelper::SaveArrayToFile(Raw, *DepthJob.Path);
    if (PendingDepthWrites) PendingDepthWrites->Decrement();
  }
  FlushInputBuffer();

  return 0;
}

void FFrameDataWriter::Stop() {
  bStopRequested = true;
}

void FFrameDataWriter::Exit() {
  FlushInputBuffer();
}

void FFrameDataWriter::FlushInputBuffer() {
  if (InputBuffer.Num() > 0 && InputLogFile) {
    InputLogFile->Write(InputBuffer.GetData(), InputBuffer.Num());
    InputLogFile->Flush();
    InputBuffer.Empty();
    InputEventsSinceFlush = 0;
  }

  if (CsvBuffer.Len() > 0 && CsvFile) {
    FTCHARToUTF8 Utf8(*CsvBuffer);
    CsvFile->Write(reinterpret_cast<const uint8 *>(Utf8.Get()), Utf8.Length());
    CsvFile->Flush();
    CsvBuffer.Empty();
  }
}