// Copyright (c) 2023 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Generation/PoissonDiscSampling.h"

#include "CarlaMeshGeneration.h"

#include "PCGContext.h"
#include "PCGComponent.h"
#include "PCGCustomVersion.h"
#include "PCGPin.h"
#include "Data/PCGPointData.h"
#include "Data/PCGSplineData.h"

#include <unordered_map>
#include <algorithm>
#include <optional>
#include <atomic>
#include <vector>
#include <random>
#include <array>
#include <span>
#include <bit>

using RealT = float;
using IntT = int32;
using V2 = UE::Math::TVector2<RealT>;
using I2 = FIntPoint;
using Edge = std::pair<V2, V2>;

UPCGPoissonDiscSamplingSettings::UPCGPoissonDiscSamplingSettings()
{
}

TArray<FPCGPinProperties> UPCGPoissonDiscSamplingSettings::InputPinProperties() const
{
  TArray<FPCGPinProperties> Properties;
  FPCGPinProperties& InputPinProperty = Properties.Emplace_GetRef(PCGPinConstants::DefaultInputLabel, EPCGDataType::Spline);
  InputPinProperty.SetRequiredPin();
  return Properties;
}

TArray<FPCGPinProperties> UPCGPoissonDiscSamplingSettings::OutputPinProperties() const
{
  return Super::DefaultPointOutputPinProperties();
}

FPCGElementPtr UPCGPoissonDiscSamplingSettings::CreateElement() const
{
  return MakeShared<FPCGPoissonDiscSampling>();
}

static RealT KahanDeterminant(
  V2 u, V2 v)
{
  RealT
    a = u.X, b = v.X,
    c = u.Y, d = v.Y;
  RealT w = b * c;
  return std::fma(-b, c, w) + std::fma(a, d, -w);
}

static bool IsInsideSpline(
  std::span<Edge> Edges,
  V2 point)
{
  const std::size_t BatchSize = 32;
  const double Threshold = 1e-4;

  auto ComputeDelta = [&](Edge e)
  {
    auto [v0, v1] = e;
    V2 u = v0 - point;
    V2 v = v1 - point;
    RealT det = KahanDeterminant(u, v);
    RealT dot = u.Dot(v);
    return std::atan2(det, dot);
  };

  double TotalTheta = 0.0;
  if (Edges.size() > BatchSize)
  {
    std::atomic<double> SharedTheta = 0.0;
    ParallelFor(Edges.size(), [&](int32 Index)
    {
      std::size_t Begin = (std::size_t)Index * BatchSize;
      std::size_t End = std::min(Begin + BatchSize, Edges.size());
      double local_theta = 0.0;
      for (std::size_t i = Begin; i != End; ++i)
        local_theta += ComputeDelta(Edges[i]);
      (void)SharedTheta.fetch_add(local_theta, std::memory_order_relaxed);
    });
    TotalTheta = SharedTheta.load(std::memory_order_acquire);
  }
  else
  {
    for (Edge& e : Edges)
      TotalTheta += ComputeDelta(e);
  }

  double WindingNumber = std::abs(TotalTheta) / (double)PI;

  return WindingNumber > Threshold;
}

static FBox ComputeSplineBoundingBox(
  std::span<V2> Points)
{
  V2 Min, Max;
  if (Points.empty())
    return FBox();
  Min = Points[0];
  Max = Points[0];
  for (std::size_t i = 1; i != Points.size(); ++i)
  {
    Min = V2::Min(Min, Points[i]);
    Max = V2::Max(Max, Points[i]);
  }
  return FBox(FVector(Min.X, Min.Y, 0.0), FVector(Max.X, Max.Y, 0.0));
}

static std::vector<V2> GeneratePoissonDiscPoints(
  FPCGContext* Context,
  FBox SplineBB,
  std::span<Edge> Edges,
  const UPCGPoissonDiscSamplingSettings& Settings)
{
  std::random_device RD;
  std::ranlux48 PRNG(RD());
  std::uniform_real_distribution<RealT> URD((RealT)0, (RealT)1);

  auto Sqrt2 = FMath::Sqrt((RealT)2);
  auto Pi = (RealT)PI;
  auto Tau = Pi * 2;

  auto Origin = V2(SplineBB.Min.X, SplineBB.Min.Y);
  auto Extent3D = SplineBB.GetExtent();
  auto Extent = V2(Extent3D.X, Extent3D.Y);
  auto R = (RealT)Settings.MinDistance;
  auto R2 = R * R;
  auto MaxRetries = Settings.MaxRetries;
  auto ZeroVec = V2(0, 0);

  auto CellSize = R / Sqrt2; // Do not use InvSqrt, we want precision.
  auto GridFloat = Extent / CellSize;
  FIntPoint GridSize(FMath::CeilToInt(GridFloat.X), FMath::CeilToInt(GridFloat.Y));
  auto CellCount = GridSize.X * GridSize.Y;
  bool UseBitMask = 4096 >= (CellCount / 8);

  std::unordered_map<IntT, V2> Grid;
  std::vector<bool> Occupancy;
  if (UseBitMask)
    Occupancy = std::vector<bool>(CellCount);
  
  auto GetRandomScalar01 = [&]
  {
    return URD(PRNG);
  };

  auto GetRandomScalar = [&](RealT Min, RealT Max)
  {
      return std::fma(GetRandomScalar01(), Max - Min, Min);
  };

  auto GetRandomPoint = [&](V2 Min, V2 Max)
  {
      return V2(
        GetRandomScalar(Min.X, Max.X),
        GetRandomScalar(Min.Y, Max.Y));
  };

  auto GridCoordToFlatIndex = [&](I2 Key)
  {
    return Key.X + Key.Y * GridSize.X;
  };

  auto GridQuery = [&](I2 Key) -> std::optional<V2>
  {
      auto Flat = GridCoordToFlatIndex(Key);
      if (UseBitMask)
        if (!Occupancy[Flat])
          return std::nullopt;
      auto it = Grid.find(Flat);
      if (it == Grid.cend())
        return std::nullopt;
      return it->second;
  };

  auto GridAdd = [&](I2 Key, V2 Value)
  {
      auto Flat = GridCoordToFlatIndex(Key);
      if (UseBitMask)
        Occupancy[Flat] = true;
      Grid.insert({ Flat, Value });
  };
  
  std::vector<V2> Results2D;
  std::vector<V2> Pending;
  Results2D.reserve(CellCount);
  Pending.reserve(CellCount);
  Grid.reserve(CellCount);

  V2 First = GetRandomPoint(Origin, Extent);
  Results2D.push_back(First);
  Pending.push_back(First);

  UE_LOG(
    LogCarlaMeshGeneration,
    Log,
    TEXT("Generating Poisson Disc Sampling points array.\n")
    TEXT(" R = %f.\n")
    TEXT(" Max Retries = %u.\n")
    TEXT(" Grid Size = %ix%i.\n")
    TEXT(" AABB = { Min: (%f, %f), Max: (%f, %f) }.\n")
    TEXT(" Cell Count = %i.\n")
    TEXT(" Cell Size = %f.\n"),
    R, (unsigned)MaxRetries,
    GridSize.X, GridSize.Y,
    SplineBB.Min.X, SplineBB.Min.Y, SplineBB.Max.X, SplineBB.Max.Y,
    CellCount,
    CellSize
  );

  while (!Pending.empty())
  {
    std::uniform_int_distribution<IntT> UID(0, (IntT)Pending.size() - 1);
    IntT Index = UID(PRNG);
    check(Index < (IntT)Pending.size());
    V2 Point = Pending[Index];
    bool Found = false;
    for (IntT i = 0; i != MaxRetries; ++i)
    {
      RealT Theta = GetRandomScalar(0, Tau);
      RealT Rho = GetRandomScalar(R, R * 2);
      V2 SinCos;
      FMath::SinCos(&SinCos.Y, &SinCos.X, Theta);
      V2 NewPoint = Point + Rho * SinCos;
      if (
        NewPoint.X < 0 || NewPoint.X >= Extent.X ||
        NewPoint.Y < 0 || NewPoint.Y >= Extent.Y)
        continue;
      V2 Tmp = NewPoint / CellSize;
      I2 GridCoord((IntT)Tmp.X, (IntT)Tmp.Y);
      I2 Begin(
        std::max(GridCoord.X - 2, 0),
        std::max(GridCoord.Y - 2, 0));
      I2 End(
        std::min(GridCoord.X + 2, GridSize.X),
        std::min(GridCoord.Y + 2, GridSize.Y));

      const I2 DefaultOffsets[] =
      {
        I2(-2, -2), I2(-1, -2), I2(0, -2), I2(1, -2), I2(2, -2),
        I2(-1, -2), I2(-1, -1), I2(0, -1), I2(1, -1), I2(2, -1),
        I2(-0, -2), I2(-1, -0), I2(0, -0), I2(1, -0), I2(2, -0),
        I2(1, -2), I2(-1, 1), I2(0, 1), I2(1, 1), I2(2, 1),
        I2(2, -2), I2(-1, 2), I2(0, 2), I2(1, 2), I2(2, 2)
      };
      std::vector<I2> TestPositions;
      TestPositions.reserve(sizeof(DefaultOffsets) / sizeof(DefaultOffsets[0]));
      for (I2 Offset : DefaultOffsets)
      {
        I2 TestPoint = GridCoord + Offset;
        if (
          TestPoint.X < 0 || TestPoint.Y < 0 ||
          TestPoint.X >= GridSize.X || TestPoint.Y >= GridSize.Y)
          continue;
        TestPositions.push_back(TestPoint);
      }

      auto TestNeighbor = [&](I2 TestPosition)
      {
        auto NeighborOpt = GridQuery(TestPosition);
        if (!NeighborOpt.has_value())
          return true;
        V2 Neighbor = NeighborOpt.value();
        return V2::DistSquared(NewPoint, Neighbor) >= R2;
      };

      bool OK = true;
      if (TestPositions.size() >= 16)
      {
        std::atomic_bool SharedOK = true;
        ParallelFor(TestPositions.size(), [&](int32 Index)
          {
            if (SharedOK.load(std::memory_order_acquire))
              if (!TestNeighbor(TestPositions[Index]))
                SharedOK.store(false, std::memory_order_release);
          });
        OK = SharedOK.load(std::memory_order_acquire);
      }
      else
      {
        for (I2 TestPosition : TestPositions)
        {
          OK = TestNeighbor(TestPosition);
          if (!OK)
            break;
        }
      }
      if (OK)
      {
        Results2D.push_back(NewPoint);
        Pending.push_back(NewPoint);
        GridAdd(GridCoord, NewPoint);
        Found = true;
        break;
      }
    }
    if (!Found)
      Pending.erase(Pending.begin() + Index);
  }

  return Results2D;
}

bool FPCGPoissonDiscSampling::ExecuteInternal(
  FPCGContext* Context) const
{ 
  auto MakePCGPoint = [&](V2 Point)
  {
      FPCGPoint P;
      P.Transform.SetLocation(FVector(Point.X, Point.Y, 0.0F));
      return P;
  };

  auto SettingsPtr = Context->GetInputSettings<UPCGPoissonDiscSamplingSettings>();
  check(SettingsPtr);
  auto& Settings = *SettingsPtr;

  auto Inputs = Context->InputData.GetInputsByPin(PCGPinConstants::DefaultInputLabel);
  auto Output = FPCGContext::NewObject_AnyThread<UPCGPointData>(Context);

  for (FPCGTaggedData& Input : Inputs)
  {
    const UPCGSplineData* InputData = Cast<UPCGSplineData>(Input.Data);
    check(InputData != nullptr);

    auto LocalToWorld = InputData->GetTransform();
    int32 SplineSampleCount = Settings.SplineSampleCount;
    
    std::vector<V2> Results2D;
    {
      std::vector<Edge> SplineEdges(SplineSampleCount);
      std::vector<V2> SplinePoints;
      SplinePoints.reserve(SplineSampleCount);
      for (int32 i = 0; i != SplineSampleCount; ++i)
      {
        float Alpha = (float)i / (float)SplineSampleCount;
        FVector P = InputData->GetLocationAtAlpha(Alpha);
        P = LocalToWorld.TransformPosition(P);
        SplinePoints.push_back(V2(P.X, P.Y));
      }
      for (std::size_t i = 0; i != SplinePoints.size(); ++i)
      {
        std::size_t Next = i + 1;
        Next = Next < SplinePoints.size() ? Next : 0;
        SplineEdges[i] = std::make_pair(SplinePoints[i], SplinePoints[Next]);
      }
      FBox SplineBoundingBox = ComputeSplineBoundingBox(SplinePoints);
      Results2D = GeneratePoissonDiscPoints(
        Context, SplineBoundingBox, SplineEdges, Settings);
    }

    Output->InitializeFromData(InputData);
    auto& OutputPoints = Output->GetMutablePoints();
    OutputPoints.Reserve(Results2D.size());
    for (auto& Point2D : Results2D)
      OutputPoints.Add(MakePCGPoint(Point2D));
    Context->OutputData.TaggedData.Add_GetRef(Input).Data = Output;
  }

  return true;
}
