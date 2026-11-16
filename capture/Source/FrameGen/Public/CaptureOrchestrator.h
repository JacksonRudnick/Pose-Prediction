#pragma once

#include "Containers/Queue.h"
#include "CoreMinimal.h"
#include "HAL/RunnableThread.h"
#include "Subsystems/WorldSubsystem.h"

#include "FrameDataWriter.h"
#include "InputCaptureThread.h"
#include "RHIGPUReadback.h"
#include "CaptureOrchestrator.generated.h"

class FFrameDataWriter;
class SWindow;

/** Single source of truth for all capture timestamps (CLOCK_MONOTONIC on Linux, matching evdev). */
struct FCaptureClock
{
	static double Now()
	{
#if PLATFORM_LINUX
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1.0e9;
#else
		return FPlatformTime::Seconds();
#endif
	}
};

DECLARE_MULTICAST_DELEGATE_OneParam(FFrameGenCaptureDelegate, uint64);

/**
 * Central tick backbone for the capture pipeline. Runs at 2x target framerate
 * with fixed timestep. Outputs: FrameLog_*.csv, InputLog_*.bin,
 * Frames/frame_*.bin (BGRA8), Depth/depth_*.bin (float32).
 */
UCLASS(Config = Game)
class FRAMEGEN_API UCaptureOrchestrator : public UWorldSubsystem {
  GENERATED_BODY()

public:
  virtual void Initialize(FSubsystemCollectionBase &Collection) override;
  virtual void Deinitialize() override;

  virtual bool DoesSupportWorldType(EWorldType::Type WorldType) const override;

  uint64 GetFrameIndex() const { return FrameIndex; }
  int32 GetCaptureFramerate() const { return TargetFramerate * 2; }

  FFrameGenCaptureDelegate OnCaptureFrame;

  UPROPERTY(Config, EditAnywhere, Category = "Capture",
            meta = (ClampMin = "15", ClampMax = "240"))
  int32 TargetFramerate = 30;

  UPROPERTY(Config, EditAnywhere, Category = "Capture")
  bool bCaptureColor = true;

  UPROPERTY(Config, EditAnywhere, Category = "Capture")
  bool bCaptureDepth = true;

  UPROPERTY(Config, EditAnywhere, Category = "Capture")
  bool bCaptureRawInput = true;

  UPROPERTY(Config, EditAnywhere, Category = "Capture",
            meta = (ClampMin = "32"))
  int32 DepthCaptureResolutionX = 1920;

  UPROPERTY(Config, EditAnywhere, Category = "Capture",
            meta = (ClampMin = "32"))
  int32 DepthCaptureResolutionY = 1080;

  /** Burst: capture BurstLength ticks out of every BurstPeriod. */
  UPROPERTY(Config, EditAnywhere, Category = "Capture", meta = (ClampMin = "1"))
  int32 BurstLength = 8;

  UPROPERTY(Config, EditAnywhere, Category = "Capture", meta = (ClampMin = "1"))
  int32 BurstPeriod = 16;

private:
  uint64 FrameIndex = 0;
  FDelegateHandle TickStartHandle;
  FDelegateHandle BeginFrameHandle;
  FDelegateHandle SamplingInputHandle;
  FDelegateHandle BackBufferReadyHandle;

  TQueue<FRawInputEvent, EQueueMode::Spsc> InputEventQueue;
  TQueue<FString, EQueueMode::Spsc> CsvLineQueue;

  FInputCaptureThread *InputCapture = nullptr;
  FRunnableThread *InputCaptureThread = nullptr;

  FFrameDataWriter *DataWriter = nullptr;
  FRunnableThread *DataWriterThread = nullptr;

  double BeginFrameTimestamp = 0.0;
  double InputSamplingTimestamp = 0.0;
  FRotator PreviousControlRotation = FRotator::ZeroRotator;
  FVector PreviousCameraLocation = FVector::ZeroVector;
  FRotator PreviousCameraRotation = FRotator::ZeroRotator;
  bool bHasPreviousFrame = false;

  TQueue<FColorFrameWrite, EQueueMode::Spsc> ColorWriteQueue;
  FThreadSafeCounter PendingColorWrites;
  int32 MaxPendingColorWrites = 16;

  struct FSurfaceReadback {
    TUniquePtr<FRHIGPUTextureReadback> Readback;
    uint64 FrameIndex = ~0ull;
    FIntPoint Size = FIntPoint::ZeroValue;
    int32 bInFlight = 0; // 0=free, 1=in_flight, 2=draining
  };

  TArray<FSurfaceReadback> ReadbackRing;
  int32 RingExhaustedCount = 0;
  int32 ArmedCount = 0;
  int32 EnqueuedCount = 0;
  int32 CapDroppedCount = 0;
  FString FrameCaptureDir;

  TQueue<FDepthFrameWrite, EQueueMode::Spsc> DepthWriteQueue;
  FThreadSafeCounter PendingDepthWrites;
  int32 MaxPendingDepthWrites = 16;

  struct FDepthReadbackSlot {
    TUniquePtr<FRHIGPUTextureReadback> Readback;
    uint64 FrameIndex = ~0ull;
    FIntPoint Size = FIntPoint::ZeroValue;
    int32 bInFlight = 0; // 0=free, 1=in_flight, 2=draining
  };

  TArray<FDepthReadbackSlot> DepthReadbackRing;
  int32 DepthRingExhaustedCount = 0;
  int32 DepthArmedCount = 0;
  int32 DepthEnqueuedCount = 0;
  int32 DepthCapDroppedCount = 0;
  FString DepthCaptureDir;

  UPROPERTY()
  TObjectPtr<class USceneCaptureComponent2D> DepthCaptureComponent;

  UPROPERTY()
  TObjectPtr<AActor> DepthCaptureActor;

  UPROPERTY()
  TObjectPtr<class UTextureRenderTarget2D> DepthRenderTarget;

  bool Tick(float DeltaTime);
  void ApplyFixedTimestepSettings();
  void LogTickToCsv(uint64 FrameIdx, double TickTimestamp, double BeginFrameTs,
                    double InputSamplingTs, double TimeSec, double DeltaSec);
  void OnBackBufferReady(SWindow &Window, const FTextureRHIRef &BackBuffer);
  void PollCompletedReadbacks();
  bool ShouldCapture() const;

  void UpdateDepthCapturePose();

};