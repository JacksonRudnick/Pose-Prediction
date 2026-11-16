// Copyright Epic Games, Inc. All Rights Reserved.

#include "CaptureOrchestrator.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Framework/Application/SlateApplication.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "HAL/PlatformFileManager.h"

#include "FrameDataWriter.h"
#include "InputCaptureThread.h"
#include "InputCoreTypes.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RHI.h"
#include "RHIGPUReadback.h"
#include "RenderingThread.h"
#include "Widgets/SWindow.h"

void UCaptureOrchestrator::Initialize(FSubsystemCollectionBase &Collection) {
  Super::Initialize(Collection);
  if (FParse::Param(FCommandLine::Get(), TEXT("nocapture")))
    return;

  UWorld *World = GetWorld();
  if (!World)
    return;

  ApplyFixedTimestepSettings();

  const FString BaseDir = FPaths::ProjectSavedDir() / TEXT("CaptureData");
  IPlatformFile &PF = FPlatformFileManager::Get().GetPlatformFile();
  PF.CreateDirectoryTree(*BaseDir);

  const FString SessionId = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
  const FString CsvPath =
      BaseDir / FString::Printf(TEXT("FrameLog_%s.csv"), *SessionId);
  const FString InputLogPath =
      BaseDir / FString::Printf(TEXT("InputLog_%s.bin"), *SessionId);

  FrameCaptureDir = BaseDir / TEXT("Frames");
  PF.CreateDirectoryTree(*FrameCaptureDir);

  DepthCaptureDir = BaseDir / TEXT("Depth");
  PF.CreateDirectoryTree(*DepthCaptureDir);

  const FString CsvHeader =
      TEXT("FrameIndex,GFrameCounter,TickTimestamp,BeginFrameTimestamp,"
           "InputSamplingTimestamp,"
           "TimeSeconds,DeltaTime"
           ",Key_W,Key_A,Key_S,Key_D,Key_Jump"
           ",MouseAngX,MouseAngY,MouseAngRateX,MouseAngRateY"
           ",VelX,VelY,VelZ,CamAngVelPitch,CamAngVelYaw"
           ",CamLocX,CamLocY,CamLocZ,CamRotPitch,CamRotYaw,CamRotRoll"
           ",CharLocX,CharLocY,CharLocZ,CharVelX,CharVelY,CharVelZ"
           ",Teleport,InBurst"
           ",DevZToWorldZ_X,DevZToWorldZ_Y,DevZToWorldZ_Z,DevZToWorldZ_W"
           ",ProjNearClip\n");

  BurstLength = FMath::Min(BurstLength, BurstPeriod);

  // 2*BurstLength slots so a stalled drain never exhausts the ring mid-burst
  ReadbackRing.SetNum(2 * BurstLength);
  MaxPendingColorWrites = 4 * BurstLength;

  DepthReadbackRing.SetNum(2 * BurstLength);
  MaxPendingDepthWrites = 4 * BurstLength;

  DataWriter = new FFrameDataWriter(
      &InputEventQueue, &CsvLineQueue, &ColorWriteQueue, &PendingColorWrites,
      &DepthWriteQueue, &PendingDepthWrites, InputLogPath, CsvPath, CsvHeader);
  DataWriterThread = FRunnableThread::Create(
      DataWriter, TEXT("FrameDataWriter"), 0, TPri_Lowest);

  if (bCaptureRawInput) {
    InputCapture = new FInputCaptureThread(&InputEventQueue);
    InputCaptureThread = FRunnableThread::Create(
        InputCapture, TEXT("InputCapture"), 0, TPri_AboveNormal);
  }

  if (bCaptureDepth) {
    FActorSpawnParameters SpawnParams;
    SpawnParams.SpawnCollisionHandlingOverride =
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    DepthCaptureActor = World->SpawnActor<AActor>(
        FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
    if (DepthCaptureActor) {
      // Ensure root component exists so SetActorLocationAndRotation works.
      if (!DepthCaptureActor->GetRootComponent()) {
        USceneComponent *Root = NewObject<USceneComponent>(DepthCaptureActor);
        DepthCaptureActor->SetRootComponent(Root);
        Root->RegisterComponent();
      }

      DepthCaptureComponent =
          NewObject<USceneCaptureComponent2D>(DepthCaptureActor);
      DepthCaptureComponent->AttachToComponent(
          DepthCaptureActor->GetRootComponent(),
          FAttachmentTransformRules::SnapToTargetNotIncludingScale);
      DepthCaptureComponent->RegisterComponent();

      // Create the float render target for scene depth.
      DepthRenderTarget = NewObject<UTextureRenderTarget2D>(DepthCaptureActor);
      DepthRenderTarget->RenderTargetFormat = RTF_R32f;
      DepthRenderTarget->ClearColor = FLinearColor::Black;
      DepthRenderTarget->InitAutoFormat(DepthCaptureResolutionX,
                                        DepthCaptureResolutionY);
      DepthRenderTarget->UpdateResourceImmediate(true);

      DepthCaptureComponent->TextureTarget = DepthRenderTarget;
      DepthCaptureComponent->CaptureSource = SCS_SceneDepth;
      DepthCaptureComponent->bCaptureEveryFrame = true;
      DepthCaptureComponent->bCaptureOnMovement = false;
      DepthCaptureComponent->ShowFlags.SetPostProcessing(false);
      DepthCaptureComponent->bAlwaysPersistRenderingState = true;
      DepthCaptureComponent->PrimitiveRenderMode =
          ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;

      UE_LOG(LogTemp, Log,
             TEXT("[CaptureOrchestrator] Depth capture: %dx%d R32f"),
             DepthCaptureResolutionX, DepthCaptureResolutionY);
    }
  }

  BeginFrameHandle = FCoreDelegates::OnBeginFrame.AddLambda(
      [this]() { BeginFrameTimestamp = FCaptureClock::Now(); });

  // Fires before PollGameDeviceState — captures input-sampling timestamp
  // closer to the real OS read than OnBeginFrame
  SamplingInputHandle = FCoreDelegates::OnSamplingInput.AddLambda(
      [this]() { InputSamplingTimestamp = FCaptureClock::Now(); });

  TickStartHandle = FWorldDelegates::OnWorldTickStart.AddLambda(
      [this](UWorld *W, ELevelTick, float DeltaTime) {
        if (W == this->GetWorld())
          this->Tick(DeltaTime);
      });

  if ((bCaptureColor || bCaptureDepth) && FSlateApplication::IsInitialized()) {
    BackBufferReadyHandle =
        FSlateApplication::Get()
            .GetRenderer()
            ->OnBackBufferReadyToPresent()
            .AddUObject(this, &UCaptureOrchestrator::OnBackBufferReady);
  }

  UE_LOG(LogTemp, Log,
         TEXT("[CaptureOrchestrator] Initialized — %d fps -> capture @ %d fps"),
         TargetFramerate, GetCaptureFramerate());
  UE_LOG(LogTemp, Log,
         TEXT("[CaptureOrchestrator] Color=%d  Depth=%d  RawInput=%d"),
         (int)bCaptureColor, (int)bCaptureDepth, (int)bCaptureRawInput);
  UE_LOG(LogTemp, Log, TEXT("[CaptureOrchestrator] Burst: %d on / %d period"),
         BurstLength, BurstPeriod);
  UE_LOG(LogTemp, Log, TEXT("[CaptureOrchestrator] CSV: %s"), *CsvPath);
  UE_LOG(LogTemp, Log, TEXT("[CaptureOrchestrator] Input: %s"), *InputLogPath);
}

void UCaptureOrchestrator::Deinitialize() {
  if (BeginFrameHandle.IsValid()) {
    FCoreDelegates::OnBeginFrame.Remove(BeginFrameHandle);
    BeginFrameHandle.Reset();
  }
  if (SamplingInputHandle.IsValid()) {
    FCoreDelegates::OnSamplingInput.Remove(SamplingInputHandle);
    SamplingInputHandle.Reset();
  }
  if (TickStartHandle.IsValid()) {
    FWorldDelegates::OnWorldTickStart.Remove(TickStartHandle);
    TickStartHandle.Reset();
  }
  if (BackBufferReadyHandle.IsValid() && FSlateApplication::IsInitialized()) {
    FSlateApplication::Get().GetRenderer()->OnBackBufferReadyToPresent().Remove(
        BackBufferReadyHandle);
    BackBufferReadyHandle.Reset();
  }
  FlushRenderingCommands();

  UE_LOG(LogTemp, Log,
         TEXT("[CaptureOrchestrator] Color accounting: armed=%d  enqueued=%d  "
              "ring-exhausted=%d  cap-dropped=%d"),
         ArmedCount, EnqueuedCount, RingExhaustedCount, CapDroppedCount);
  UE_LOG(LogTemp, Log,
         TEXT("[CaptureOrchestrator] Depth accounting: armed=%d  enqueued=%d  "
              "ring-exhausted=%d  cap-dropped=%d"),
         DepthArmedCount, DepthEnqueuedCount, DepthRingExhaustedCount,
         DepthCapDroppedCount);

  if (DepthCaptureActor) {
    DepthCaptureActor->Destroy();
    DepthCaptureActor = nullptr;
    DepthCaptureComponent = nullptr;
    DepthRenderTarget = nullptr;
  }

  if (InputCaptureThread) {
    InputCapture->Stop();
    InputCaptureThread->WaitForCompletion();
    delete InputCaptureThread;
    InputCaptureThread = nullptr;
  }
  if (InputCapture) {
    delete InputCapture;
    InputCapture = nullptr;
  }

  if (DataWriterThread) {
    DataWriter->Stop();
    DataWriterThread->WaitForCompletion();
    delete DataWriterThread;
    DataWriterThread = nullptr;
  }
  if (DataWriter) {
    delete DataWriter;
    DataWriter = nullptr;
  }

  UE_LOG(LogTemp, Log, TEXT("[CaptureOrchestrator] Deinitialized"));
  Super::Deinitialize();
}

bool UCaptureOrchestrator::Tick(float DeltaTime) {
  UWorld *World = GetWorld();
  if (!World || !ShouldCapture()) {
    ++FrameIndex;
    return true;
  }

  const double TickTimestamp = FCaptureClock::Now();

  if (bCaptureDepth) {
    UpdateDepthCapturePose();
  }

  LogTickToCsv(FrameIndex, TickTimestamp, BeginFrameTimestamp,
               InputSamplingTimestamp, World->GetTimeSeconds(), DeltaTime);

  PollCompletedReadbacks();

  OnCaptureFrame.Broadcast(FrameIndex);

  ++FrameIndex;
  return true;
}

bool UCaptureOrchestrator::DoesSupportWorldType(
    EWorldType::Type WorldType) const {
  return WorldType == EWorldType::Game || WorldType == EWorldType::GameRPC ||
         WorldType == EWorldType::PIE;
}

bool UCaptureOrchestrator::ShouldCapture() const {
#if WITH_EDITOR
  UWorld *W = GetWorld();
  return W && (W->WorldType == EWorldType::Game ||
               W->WorldType == EWorldType::GameRPC ||
               W->WorldType == EWorldType::PIE);
#else
  return true;
#endif
}

void UCaptureOrchestrator::ApplyFixedTimestepSettings() {
  if (GEngine) {
    GEngine->bUseFixedFrameRate = true;
    GEngine->FixedFrameRate = (float)GetCaptureFramerate();
  }
  IConsoleVariable *Cvar;
  Cvar = IConsoleManager::Get().FindConsoleVariable(TEXT("t.MaxFPS"));
  if (Cvar)
    Cvar->Set(GetCaptureFramerate(), ECVF_SetByCode);
  Cvar = IConsoleManager::Get().FindConsoleVariable(TEXT("r.VSync"));
  if (Cvar)
    Cvar->Set(0, ECVF_SetByCode);
  UE_LOG(LogTemp, Log,
         TEXT("[CaptureOrchestrator] Fixed timestep: %d fps, VSync off"),
         GetCaptureFramerate());
}

void UCaptureOrchestrator::UpdateDepthCapturePose() {
  if (!DepthCaptureActor || !DepthCaptureComponent)
    return;

  UWorld *World = GetWorld();
  APlayerController *PC = World ? World->GetFirstPlayerController() : nullptr;
  if (!PC)
    return;

  FVector CamLoc;
  FRotator CamRot;
  PC->GetPlayerViewPoint(CamLoc, CamRot);

  DepthCaptureActor->SetActorLocationAndRotation(
      CamLoc, CamRot, false, nullptr, ETeleportType::TeleportPhysics);
}

void UCaptureOrchestrator::LogTickToCsv(uint64 FrameIdx, double TickTimestamp,
                                        double BeginFrameTs,
                                        double InputSamplingTs, double TimeSec,
                                        double DeltaSec) {
  uint64 GFrameCnt = GFrameCounter;

  const int32 InBurst =
      ((GFrameCnt % (uint64)BurstPeriod) < (uint64)BurstLength) ? 1 : 0;

  double MouseAngX = 0, MouseAngY = 0, MouseAngRateX = 0, MouseAngRateY = 0;
  double VelX = 0, VelY = 0, VelZ = 0;
  double CamAngVelP = 0, CamAngVelY = 0;
  double CamLocX = 0, CamLocY = 0, CamLocZ = 0;
  double CamRotP = 0, CamRotY = 0, CamRotR = 0;
  double CharLocX = 0, CharLocY = 0, CharLocZ = 0;
  double CharVelX = 0, CharVelY = 0, CharVelZ = 0;
  int32 bKeyW = 0, bKeyA = 0, bKeyS = 0, bKeyD = 0, bKeyJump = 0;
  int32 Teleport = 0;

  double DevZToWorldZ_X = 0.0, DevZToWorldZ_Y = 0.0;
  double DevZToWorldZ_Z = 0.0, DevZToWorldZ_W = 0.0;
  double ProjNearClip = 0.0;

  UWorld *World = GetWorld();
  APlayerController *PC = World ? World->GetFirstPlayerController() : nullptr;
  if (PC) {
    bKeyW = PC->IsInputKeyDown(EKeys::W) ? 1 : 0;
    bKeyA = PC->IsInputKeyDown(EKeys::A) ? 1 : 0;
    bKeyS = PC->IsInputKeyDown(EKeys::S) ? 1 : 0;
    bKeyD = PC->IsInputKeyDown(EKeys::D) ? 1 : 0;
    bKeyJump = PC->IsInputKeyDown(EKeys::SpaceBar) ? 1 : 0;

    FRotator CtrlRot = PC->GetControlRotation();
    if (bHasPreviousFrame) {
      MouseAngY = FRotator::NormalizeAxis(CtrlRot.Pitch -
                                          PreviousControlRotation.Pitch);
      MouseAngX =
          FRotator::NormalizeAxis(CtrlRot.Yaw - PreviousControlRotation.Yaw);
      if (DeltaSec > SMALL_NUMBER) {
        MouseAngRateX = MouseAngX / DeltaSec;
        MouseAngRateY = MouseAngY / DeltaSec;
      }
    }
    PreviousControlRotation = CtrlRot;

    FVector CurLoc;
    FRotator CurRot;
    PC->GetPlayerViewPoint(CurLoc, CurRot);
    CamLocX = CurLoc.X;
    CamLocY = CurLoc.Y;
    CamLocZ = CurLoc.Z;
    CamRotP = CurRot.Pitch;
    CamRotY = CurRot.Yaw;
    CamRotR = CurRot.Roll;

    // Reverse-Z infinite-far projection: LinearZ = 1/(DeviceZ/NearClip - 0)
    if (APlayerCameraManager *CamMgr = PC->PlayerCameraManager) {
      FMinimalViewInfo ViewInfo = CamMgr->GetCameraCacheView();
      ProjNearClip = ViewInfo.GetFinalPerspectiveNearClipPlane();
      if (ProjNearClip > 0.0) {
        DevZToWorldZ_Z = 1.0 / ProjNearClip;
      }
    }

    if (APawn *Pawn = PC->GetPawn()) {
      FVector CharLoc = Pawn->GetActorLocation();
      CharLocX = CharLoc.X;
      CharLocY = CharLoc.Y;
      CharLocZ = CharLoc.Z;
      if (ACharacter *Char = Cast<ACharacter>(Pawn)) {
        if (UCharacterMovementComponent *Move = Char->GetCharacterMovement()) {
          const FVector V = Move->Velocity;
          CharVelX = V.X;
          CharVelY = V.Y;
          CharVelZ = V.Z;
        }
      }
    }

    if (bHasPreviousFrame) {
      CamAngVelP =
          FRotator::NormalizeAxis(CurRot.Pitch - PreviousCameraRotation.Pitch);
      CamAngVelY =
          FRotator::NormalizeAxis(CurRot.Yaw - PreviousCameraRotation.Yaw);
      CamAngVelP = DeltaSec > SMALL_NUMBER ? CamAngVelP / DeltaSec : 0;
      CamAngVelY = DeltaSec > SMALL_NUMBER ? CamAngVelY / DeltaSec : 0;

      FVector LocDelta = CurLoc - PreviousCameraLocation;
      VelX = DeltaSec > SMALL_NUMBER ? LocDelta.X / DeltaSec : 0;
      VelY = DeltaSec > SMALL_NUMBER ? LocDelta.Y / DeltaSec : 0;
      VelZ = DeltaSec > SMALL_NUMBER ? LocDelta.Z / DeltaSec : 0;

      double SpeedSq = VelX * VelX + VelY * VelY + VelZ * VelZ;
      if (SpeedSq > 2.5e9 || FMath::Abs(CamAngVelP) > 5000.0 ||
          FMath::Abs(CamAngVelY) > 5000.0)
        Teleport = 1;
    }
    PreviousCameraLocation = CurLoc;
    PreviousCameraRotation = CurRot;
    bHasPreviousFrame = true;
  }

  FString Line = FString::Printf(
      TEXT("%llu,%llu,%.9f,%.9f,%.9f,%.6f,%.6f"
           ",%d,%d,%d,%d,%d"
           ",%.6f,%.6f,%.6f,%.6f"
           ",%.6f,%.6f,%.6f,%.6f,%.6f"
           ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f"
           ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f"
           ",%d,%d"
           ",%.9f,%.9f,%.9f,%.9f"
           ",%.6f\n"),
      FrameIdx, GFrameCnt, TickTimestamp, BeginFrameTs, InputSamplingTs,
      TimeSec, DeltaSec, bKeyW, bKeyA, bKeyS, bKeyD, bKeyJump, MouseAngX,
      MouseAngY, MouseAngRateX, MouseAngRateY, VelX, VelY, VelZ, CamAngVelP,
      CamAngVelY, CamLocX, CamLocY, CamLocZ, CamRotP, CamRotY, CamRotR,
      CharLocX, CharLocY, CharLocZ, CharVelX, CharVelY, CharVelZ, Teleport,
      InBurst, DevZToWorldZ_X, DevZToWorldZ_Y, DevZToWorldZ_Z, DevZToWorldZ_W,
      ProjNearClip);

  CsvLineQueue.Enqueue(Line);
}

void UCaptureOrchestrator::OnBackBufferReady(SWindow &Window,
                                             const FTextureRHIRef &BackBuffer) {
  if (!GEngine || !GEngine->GameViewport)
    return;
  TSharedPtr<SWindow> GameWin = GEngine->GameViewport->GetWindow();
  if (!GameWin.IsValid() || &Window != GameWin.Get())
    return;

  if (!BackBuffer.IsValid())
    return;

  const uint64 RTFrame = GFrameCounterRenderThread;
  const bool bInBurstWindow =
      (RTFrame % (uint64)BurstPeriod) < (uint64)BurstLength;
  if (!bInBurstWindow)
    return;

  if (bCaptureColor) {
    FSurfaceReadback *S = nullptr;
    for (FSurfaceReadback &Slot : ReadbackRing) {
      if (FPlatformAtomics::InterlockedCompareExchange(&Slot.bInFlight, 1, 0) ==
          0) {
        S = &Slot;
        break;
      }
    }
    if (S) {
      ++ArmedCount;
      FRHICommandListImmediate &RHICmdList =
          FRHICommandListExecutor::GetImmediateCommandList();
      const FIntPoint Size = BackBuffer->GetSizeXY();
      if (Size.X > 0 && Size.Y > 0) {
        if (!S->Readback.IsValid())
          S->Readback =
              MakeUnique<FRHIGPUTextureReadback>(TEXT("FrameGenColorReadback"));
        S->Readback->EnqueueCopy(RHICmdList, BackBuffer, FIntVector::ZeroValue,
                                 0, FIntVector(Size.X, Size.Y, 1));
        S->FrameIndex = RTFrame;
        S->Size = Size;
      }
    } else {
      ++RingExhaustedCount;
    }
  }

  if (bCaptureDepth && DepthRenderTarget && DepthRenderTarget->GetResource()) {
    FTextureRHIRef DepthTex =
        DepthRenderTarget->GetResource()->GetTexture2DRHI();
    if (DepthTex.IsValid()) {
      FDepthReadbackSlot *DS = nullptr;
      for (FDepthReadbackSlot &Slot : DepthReadbackRing) {
        if (FPlatformAtomics::InterlockedCompareExchange(&Slot.bInFlight, 1,
                                                         0) == 0) {
          DS = &Slot;
          break;
        }
      }
      if (DS) {
        ++DepthArmedCount;
        FRHICommandListImmediate &RHICmdList =
            FRHICommandListExecutor::GetImmediateCommandList();
        const FIntPoint Size = DepthTex->GetSizeXY();
        if (Size.X > 0 && Size.Y > 0) {
          if (!DS->Readback.IsValid())
            DS->Readback = MakeUnique<FRHIGPUTextureReadback>(
                TEXT("FrameGenDepthReadback"));
          DS->Readback->EnqueueCopy(RHICmdList, DepthTex, FIntVector::ZeroValue,
                                    0, FIntVector(Size.X, Size.Y, 1));
          DS->FrameIndex = RTFrame;
          DS->Size = Size;
        }
      } else {
        ++DepthRingExhaustedCount;
      }
    }
  }
}

void UCaptureOrchestrator::PollCompletedReadbacks() {
  TArray<FSurfaceReadback> *ColorRing = &ReadbackRing;
  TQueue<FColorFrameWrite, EQueueMode::Spsc> *ColorQueue = &ColorWriteQueue;
  FThreadSafeCounter *ColorPending = &PendingColorWrites;
  FString *ColorDir = &FrameCaptureDir;
  const int32 ColorMaxPending = MaxPendingColorWrites;
  int32 *ColorEnq = &EnqueuedCount;
  int32 *ColorDrop = &CapDroppedCount;

  TArray<FDepthReadbackSlot> *DepthRing = &DepthReadbackRing;
  TQueue<FDepthFrameWrite, EQueueMode::Spsc> *DepthQueue = &DepthWriteQueue;
  FThreadSafeCounter *DepthPending = &PendingDepthWrites;
  FString *DepthDir = &DepthCaptureDir;
  const int32 DepthMaxPending = MaxPendingDepthWrites;
  int32 *DepthEnq = &DepthEnqueuedCount;
  int32 *DepthDrop = &DepthCapDroppedCount;

  ENQUEUE_RENDER_COMMAND(
      FrameGenDrainReadbacks)([ColorRing, ColorQueue, ColorPending, ColorDir,
                               ColorMaxPending, ColorEnq, ColorDrop, DepthRing,
                               DepthQueue, DepthPending, DepthDir,
                               DepthMaxPending, DepthEnq,
                               DepthDrop](FRHICommandListImmediate &) {
    for (auto &S : *ColorRing) {
      if (FPlatformAtomics::InterlockedCompareExchange(&S.bInFlight, 2, 1) != 1)
        continue;
      if (!S.Readback.IsValid() || !S.Readback->IsReady()) {
        FPlatformAtomics::InterlockedExchange(&S.bInFlight, 1);
        continue;
      }
      int32 RowPitch = 0, BufH = 0;
      void *Mapped = S.Readback->Lock(RowPitch, &BufH);
      if (Mapped && S.Size.X > 0 && S.Size.Y > 0) {
        const int32 W = S.Size.X, H = S.Size.Y;
        TArray<FColor> Pixels;
        Pixels.SetNumUninitialized(W * H);
        const FColor *Src = reinterpret_cast<const FColor *>(Mapped);
        for (int32 y = 0; y < H; ++y)
          FMemory::Memcpy(Pixels.GetData() + y * W, Src + y * RowPitch,
                          W * sizeof(FColor));
        S.Readback->Unlock();
        const uint64 Idx = S.FrameIndex;
        FString Path =
            *ColorDir / FString::Printf(TEXT("frame_%06llu.bin"), Idx);
        if (ColorPending->GetValue() < ColorMaxPending) {
          ColorPending->Increment();
          ColorQueue->Enqueue(FColorFrameWrite{Idx, Path, MoveTemp(Pixels)});
          ++*ColorEnq;
        } else {
          ++*ColorDrop;
        }
      } else if (Mapped) {
        S.Readback->Unlock();
      }
      FPlatformAtomics::InterlockedExchange(&S.bInFlight, 0);
    }

    for (auto &S : *DepthRing) {
      if (FPlatformAtomics::InterlockedCompareExchange(&S.bInFlight, 2, 1) != 1)
        continue;
      if (!S.Readback.IsValid() || !S.Readback->IsReady()) {
        FPlatformAtomics::InterlockedExchange(&S.bInFlight, 1);
        continue;
      }
      int32 RowPitch = 0, BufH = 0;
      void *Mapped = S.Readback->Lock(RowPitch, &BufH);
      if (Mapped && S.Size.X > 0 && S.Size.Y > 0) {
        const int32 W = S.Size.X, H = S.Size.Y;
        const int32 FloatPitch = RowPitch / sizeof(float);
        TArray<float> DepthData;
        DepthData.SetNumUninitialized(W * H);
        const float *Src = reinterpret_cast<const float *>(Mapped);
        for (int32 y = 0; y < H; ++y)
          FMemory::Memcpy(DepthData.GetData() + y * W, Src + y * FloatPitch,
                          W * sizeof(float));
        S.Readback->Unlock();
        const uint64 Idx = S.FrameIndex;
        FString Path =
            *DepthDir / FString::Printf(TEXT("depth_%06llu.bin"), Idx);
        if (DepthPending->GetValue() < DepthMaxPending) {
          DepthPending->Increment();
          DepthQueue->Enqueue(
              FDepthFrameWrite{Idx, Path, MoveTemp(DepthData), S.Size});
          ++*DepthEnq;
        } else {
          ++*DepthDrop;
        }
      } else if (Mapped) {
        S.Readback->Unlock();
      }
      FPlatformAtomics::InterlockedExchange(&S.bInFlight, 0);
    }
  });
}