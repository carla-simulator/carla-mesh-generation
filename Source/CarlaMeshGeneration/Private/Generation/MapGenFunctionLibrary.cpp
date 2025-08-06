// Copyright (c) 2023 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Generation/MapGenFunctionLibrary.h"

// Engine headers
#include "AssetRegistry/AssetRegistryModule.h"
#include "Materials/MaterialInstance.h"
#include "StaticMeshAttributes.h"
#include "RenderingThread.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "PhysicsEngine/BodySetup.h"
// Carla C++ headers

// Carla plugin headers
#include "CarlaMeshGeneration.h"
#include "Paths/GenerationPathsHelper.h"

#if WITH_EDITOR
#include "Editor/Transactor.h"
#endif

#if ENGINE_MAJOR_VERSION < 5
using V2 = FVector2D;
using V3 = FVector;
#else
using V2 = FVector2f;
using V3 = FVector3f;
#endif

DEFINE_LOG_CATEGORY(LogCarlaMapGenFunctionLibrary);
static const float OSMToCentimetersScaleFactor = 100.0f;

FMeshDescription UMapGenFunctionLibrary::BuildMeshDescriptionFromData(
  const FProceduralCustomMesh& Data,
  const TArray<FProcMeshTangent>& ParamTangents,
  UMaterialInstance* MaterialInstance  )
{

  int32 VertexCount = Data.Vertices.Num();
  int32 VertexInstanceCount = Data.Triangles.Num();
  int32 PolygonCount = Data.Vertices.Num()/3;

	FMeshDescription MeshDescription;
  FStaticMeshAttributes AttributeGetter(MeshDescription);
  AttributeGetter.Register();

  auto PolygonGroupNames = AttributeGetter.GetPolygonGroupMaterialSlotNames();
  auto VertexPositions = AttributeGetter.GetVertexPositions();
  auto Tangents = AttributeGetter.GetVertexInstanceTangents();
  auto BinormalSigns = AttributeGetter.GetVertexInstanceBinormalSigns();
  auto Normals = AttributeGetter.GetVertexInstanceNormals();
  auto Colors = AttributeGetter.GetVertexInstanceColors();
  auto UVs = AttributeGetter.GetVertexInstanceUVs();

  // Calculate the totals for each ProcMesh element type
  FPolygonGroupID PolygonGroupForSection;
  MeshDescription.ReserveNewVertices(VertexCount);
  MeshDescription.ReserveNewVertexInstances(VertexInstanceCount);
  MeshDescription.ReserveNewPolygons(PolygonCount);
  MeshDescription.ReserveNewEdges(PolygonCount * 2);
  UVs.SetNumIndices(4);

  // Create Materials
  TMap<UMaterialInterface*, FPolygonGroupID> UniqueMaterials;
	const int32 NumSections = 1;
	UniqueMaterials.Reserve(1);
  FPolygonGroupID NewPolygonGroup = MeshDescription.CreatePolygonGroup();

  if( MaterialInstance != nullptr ){
    UMaterialInterface *Material = MaterialInstance;
    UniqueMaterials.Add(Material, NewPolygonGroup);
    PolygonGroupNames[NewPolygonGroup] = Material->GetFName();
  }else{
    UE_LOG(LogCarlaMapGenFunctionLibrary, Error, TEXT("MaterialInstance is nullptr"));
  }
  PolygonGroupForSection = NewPolygonGroup;



  // Create the vertex
  int32 NumVertex = Data.Vertices.Num();
  TMap<int32, FVertexID> VertexIndexToVertexID;
  VertexIndexToVertexID.Reserve(NumVertex);
  for (int32 VertexIndex = 0; VertexIndex < NumVertex; ++VertexIndex)
  {
    auto& Vert = Data.Vertices[VertexIndex];
    const FVertexID VertexID = MeshDescription.CreateVertex();
    VertexPositions[VertexID] = V3(Vert);
    VertexIndexToVertexID.Add(VertexIndex, VertexID);
  }

  // Create the VertexInstance
  int32 NumIndices = Data.Triangles.Num();
  int32 NumTri = NumIndices / 3;
  TMap<int32, FVertexInstanceID> IndiceIndexToVertexInstanceID;
  IndiceIndexToVertexInstanceID.Reserve(NumVertex);
  for (int32 IndiceIndex = 0; IndiceIndex < NumIndices; IndiceIndex++)
  {
    const int32 VertexIndex = Data.Triangles[IndiceIndex];
    const FVertexID VertexID = VertexIndexToVertexID[VertexIndex];
    const FVertexInstanceID VertexInstanceID =
    MeshDescription.CreateVertexInstance(VertexID);
    IndiceIndexToVertexInstanceID.Add(IndiceIndex, VertexInstanceID);
    Normals[VertexInstanceID] = V3(Data.Normals[VertexIndex]);

    if(ParamTangents.Num() == Data.Vertices.Num())
    {
      Tangents[VertexInstanceID] = V3(ParamTangents[VertexIndex].TangentX);
      BinormalSigns[VertexInstanceID] =
        ParamTangents[VertexIndex].bFlipTangentY ? -1.f : 1.f;
    }else{

    }
    Colors[VertexInstanceID] = FLinearColor(0,0,0);
    if(Data.UV0.Num() == Data.Vertices.Num())
    {
      UVs.Set(VertexInstanceID, 0, V2(Data.UV0[VertexIndex]));
    }else{
      UVs.Set(VertexInstanceID, 0, V2(0,0));
    }
    UVs.Set(VertexInstanceID, 1, V2(0,0));
    UVs.Set(VertexInstanceID, 2, V2(0,0));
    UVs.Set(VertexInstanceID, 3, V2(0,0));
  }

  for (int32 TriIdx = 0; TriIdx < NumTri; TriIdx++)
  {
    FVertexID VertexIndexes[3];
    TArray<FVertexInstanceID> VertexInstanceIDs;
    VertexInstanceIDs.SetNum(3);

    for (int32 CornerIndex = 0; CornerIndex < 3; ++CornerIndex)
    {
      const int32 IndiceIndex = (TriIdx * 3) + CornerIndex;
      const int32 VertexIndex = Data.Triangles[IndiceIndex];
      VertexIndexes[CornerIndex] = VertexIndexToVertexID[VertexIndex];
      VertexInstanceIDs[CornerIndex] =
        IndiceIndexToVertexInstanceID[IndiceIndex];
    }

    // Insert a polygon into the mesh
    MeshDescription.CreatePolygon(NewPolygonGroup, VertexInstanceIDs);

  }

  return MeshDescription;
}

UStaticMesh* UMapGenFunctionLibrary::CreateMesh(
    const FProceduralCustomMesh& Data,
    const TArray<FProcMeshTangent>& ParamTangents,
    UMaterialInstance* MaterialInstance,
    FString MapName,
    FString FolderName,
    FName MeshName)
{
  IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

  UStaticMesh::FBuildMeshDescriptionsParams Params;
  Params.bBuildSimpleCollision = false;

  FString PackageName = UGenerationPathsHelper::GetMapContentDirectoryPath(MapName) + FolderName + "/" + MeshName.ToString();

  if (!PlatformFile.DirectoryExists(*PackageName))
  {
    //PlatformFile.CreateDirectory(*PackageName);
  }


  FMeshDescription Description = BuildMeshDescriptionFromData(Data,ParamTangents, MaterialInstance);

  if (Description.Polygons().Num() > 0)
  {
    UPackage* Package = CreatePackage(*PackageName);
    check(Package);
    UStaticMesh* Mesh = NewObject<UStaticMesh>( Package, MeshName, RF_Public | RF_Standalone);

    Mesh->InitResources();

    Mesh->SetLightingGuid(FGuid::NewGuid());
    Mesh->GetStaticMaterials().Add(FStaticMaterial(MaterialInstance));
    Mesh->NaniteSettings.bEnabled = true;
    Mesh->BuildFromMeshDescriptions({ &Description }, Params);
    // Ensure Mesh has a BodySetup
    Mesh->CreateBodySetup();
    UBodySetup* BodySetup = Mesh->GetBodySetup();
    if (BodySetup)
    {
        // Set to use complex as simple collision
        BodySetup->CollisionTraceFlag = CTF_UseComplexAsSimple;
        BodySetup->InvalidatePhysicsData();
        BodySetup->ClearPhysicsMeshes();
    }
    // Build mesh from source
    Mesh->NeverStream = false;
    TArray<UObject*> CreatedAssets;
    CreatedAssets.Add(Mesh);

    // Notify asset registry of new asset
    FAssetRegistryModule::AssetCreated(Mesh);
    //UPackage::SavePackage(Package, Mesh, EObjectFlags::RF_Public | EObjectFlags::RF_Standalone, *(MeshName.ToString()), GError, nullptr, true, true, SAVE_NoError);
#if ENGINE_MAJOR_VERSION > 4
    TArray<UStaticMesh*> MeshToEnableNanite;
    MeshToEnableNanite.Add(Mesh);
    UStaticMesh::BatchBuild(MeshToEnableNanite);
#endif
    // Finalize mesh
    Mesh->Build(false); // Rebuilds mesh data, optionally pass true to mark as async
    Mesh->PostEditChange();
    Package->MarkPackageDirty();
    Mesh->ComplexCollisionMesh = Mesh;
    return Mesh;
  }
  return nullptr;
}

// Transverse Mercator projection, see e.g. https://proj.org/en/stable/operations/projections/tmerc.html
FVector2D UMapGenFunctionLibrary::GetTransversemercProjection(float lat, float lon, float lat0, float lon0)
{
  // Earth radius in m
  const float R = 6373000.0f;
  float latt = FMath::DegreesToRadians(lat);
  float lonn  = FMath::DegreesToRadians(lon - lon0);
  float latt0 = FMath::DegreesToRadians(lat0);
  float eps = atan(tan(latt)/cos(lonn));
  float nab = asinh(sin(lonn)/sqrt(tan(latt)*tan(latt)+cos(lonn)*cos(lonn)));
  float x = R*nab;
  float y = R*eps;
  float eps0 = atan(tan(latt0)/cos(0));
  float nab0 = asinh(sin(0)/sqrt(tan(latt0)*tan(latt0)+cos(0)*cos(0)));
  float x0 = R*nab0;
  float y0 = R*eps0;
  FVector2D Result = FVector2D(x, -(y - y0)) * OSMToCentimetersScaleFactor;

  return Result;
}

FVector2D UMapGenFunctionLibrary::InverseTransverseMercatorProjection(float x, float y, float lat0, float lon0)
{
    const float R = 6373000.0f;

    x /= OSMToCentimetersScaleFactor;
    y = -y / OSMToCentimetersScaleFactor;

    float latt0 = FMath::DegreesToRadians(lat0);
    float eps0 = latt0; // atan(tan(latt0)/cos(0));
    float y0 = R * eps0;

    float eps = (y + y0) / R;
    float nab = x / R;

    float lat = FMath::RadiansToDegrees(atan(sin(eps)/sqrt(tan(nab)*tan(nab)+cos(eps)*cos(eps))));
    float lon = lon0 + FMath::RadiansToDegrees(atan(sinh(nab) / cos(eps)));

    return FVector2D(lat, lon);
}

void UMapGenFunctionLibrary::SetThreadToSleep(float seconds){
  //FGenericPlatformProcess::Sleep(seconds);
}

void UMapGenFunctionLibrary::FlushRenderingCommandsInBlueprint(){
#if ENGINE_MAJOR_VERSION < 5
  FlushRenderingCommands(true);
#else
    FlushRenderingCommands();
#endif
 	FlushPendingDeleteRHIResources_GameThread();
}

void UMapGenFunctionLibrary::CleanupGEngine(){
  GEngine->PerformGarbageCollectionAndCleanupActors();
#if WITH_EDITOR
  FText TransResetText(FText::FromString("Clean up after Move actors to sublevels"));
  if ( GEditor->Trans )
  {
    GEditor->Trans->Reset(TransResetText);
    GEditor->Cleanse(true, true, TransResetText);
  }
#endif
}


UInstancedStaticMeshComponent* UMapGenFunctionLibrary::AddInstancedStaticMeshComponentToActor(AActor* TargetActor){
  if ( !TargetActor )
  {
      UE_LOG(LogCarlaMapGenFunctionLibrary, Warning, TEXT("Invalid TargetActor in AddInstancedStaticMeshComponentToActor"));
      return nullptr;
  }

  if (!TargetActor->GetRootComponent())
  {
      USceneComponent* NewRoot = NewObject<USceneComponent>(TargetActor, TEXT("GeneratedRootComponent"));
      TargetActor->SetRootComponent(NewRoot);
      NewRoot->RegisterComponent();
  }

  // Crear el componente instanciado
  UInstancedStaticMeshComponent* ISMComponent = NewObject<UInstancedStaticMeshComponent>(TargetActor);
  if (!ISMComponent)
  {
      UE_LOG(LogCarlaMapGenFunctionLibrary, Error, TEXT("COuld not create UInstancedStaticMeshComponent"));
      return nullptr;
  }

  ISMComponent->SetupAttachment(TargetActor->GetRootComponent());
  ISMComponent->RegisterComponent();

  TargetActor->AddInstanceComponent(ISMComponent);

  return ISMComponent;
}

UStaticMeshComponent* UMapGenFunctionLibrary::AddStaticMeshComponentToActor(AActor* TargetActor){
  if ( !TargetActor )
  {
      UE_LOG(LogCarlaMapGenFunctionLibrary, Warning, TEXT("Invalid TargetActor in AddInstancedStaticMeshComponentToActor"));
      return nullptr;
  }

  if (!TargetActor->GetRootComponent())
  {
      USceneComponent* NewRoot = NewObject<USceneComponent>(TargetActor, TEXT("GeneratedRootComponent"));
      TargetActor->SetRootComponent(NewRoot);
      NewRoot->RegisterComponent();
  }

  // Crear el componente instanciado
  UStaticMeshComponent* SMComponent = NewObject<UStaticMeshComponent>(TargetActor);
  if (!SMComponent)
  {
      UE_LOG(LogCarlaMapGenFunctionLibrary, Error, TEXT("Could not create UStaticMeshComponent"));
      return nullptr;
  }

  SMComponent->SetupAttachment(TargetActor->GetRootComponent());
  SMComponent->RegisterComponent();

  TargetActor->AddInstanceComponent(SMComponent);

  return SMComponent;
}

USceneComponent* UMapGenFunctionLibrary::AddSceneComponentToActor(AActor* TargetActor)
{
    if (!TargetActor)
    {
        UE_LOG(LogCarlaMapGenFunctionLibrary, Warning, TEXT("Invalid TargetActor in AddSceneComponentToActor"));
        return nullptr;
    }

    if (!TargetActor->GetRootComponent())
    {
        USceneComponent* NewRoot = NewObject<USceneComponent>(TargetActor, TEXT("GeneratedRoot"));
        TargetActor->SetRootComponent(NewRoot);
        NewRoot->RegisterComponent();
    }

    USceneComponent* SceneComp = NewObject<USceneComponent>(TargetActor);
    if (!SceneComp)
    {
        UE_LOG(LogCarlaMapGenFunctionLibrary, Error, TEXT("Could not create USceneComponent"));
        return nullptr;
    }

    SceneComp->SetupAttachment(TargetActor->GetRootComponent());
    SceneComp->RegisterComponent();

    TargetActor->AddInstanceComponent(SceneComp);

    return SceneComp;
}


void UMapGenFunctionLibrary::SmoothVerticesDeep(
  TArray<FVector>& Vertices,
  const TArray<int32>& Indices,
  int Depth,                 // Number of neighbor levels
  int NumIterations,
  float SmoothingFactor   // Blend between original and averaged
)
{
  const int32 NumVertices = Vertices.Num();

  // Step 1: Build adjacency map
  TMap<int32, TSet<int32>> VertexNeighbors;
  const int32 NumTriangles = Indices.Num() / 3;

  for (int32 i = 0; i < NumTriangles; ++i)
  {
      int32 I0 = Indices[i * 3 + 0];
      int32 I1 = Indices[i * 3 + 1];
      int32 I2 = Indices[i * 3 + 2];

      VertexNeighbors.FindOrAdd(I0).Add(I1);
      VertexNeighbors.FindOrAdd(I0).Add(I2);

      VertexNeighbors.FindOrAdd(I1).Add(I0);
      VertexNeighbors.FindOrAdd(I1).Add(I2);

      VertexNeighbors.FindOrAdd(I2).Add(I0);
      VertexNeighbors.FindOrAdd(I2).Add(I1);
  }

  // Step 2: Iterative smoothing
  for (int32 Iter = 0; Iter < NumIterations; ++Iter)
  {
    TArray<FVector> NewVertices = Vertices;
    for (int32 i = 0; i < NumVertices; ++i)
    {
        TSet<int32> Visited;
        TQueue<TPair<int32, int32>> Queue; // Pair of (vertex index, current depth)
        Visited.Add(i);
        Queue.Enqueue(TPair<int32, int32>(i, 0));
        TArray<int32> CollectedNeighbors;

        while (!Queue.IsEmpty())
        {
          TPair<int32, int32> Current;
          Queue.Dequeue(Current);
          int32 CurrentIndex = Current.Key;
          int32 CurrentDepth = Current.Value;

          if (CurrentDepth >= Depth)
              continue;
          const TSet<int32>* Neighbors = VertexNeighbors.Find(CurrentIndex);
          if (!Neighbors) continue;

          for (int32 NeighborIndex : *Neighbors)
          {
            if (!Visited.Contains(NeighborIndex))
            {
              Visited.Add(NeighborIndex);
              CollectedNeighbors.Add(NeighborIndex);
              Queue.Enqueue(TPair<int32, int32>(NeighborIndex, CurrentDepth + 1));
            }
          }
        }

        if (CollectedNeighbors.Num() > 0)
        {
          FVector Average = FVector::ZeroVector;
          for (int32 NeighborIdx : CollectedNeighbors)
          {
            Average += Vertices[NeighborIdx];
          }
          Average /= CollectedNeighbors.Num();
          NewVertices[i] = FMath::Lerp(Vertices[i], Average, SmoothingFactor);
        }
    }
    Vertices = NewVertices;
  }
}

float UMapGenFunctionLibrary::BicubicSampleG16(const TArrayView64<const uint16>& Pixels, int Width, int Height, float X, float Y)
{
  int ix = FMath::FloorToInt(X);
  int iy = FMath::FloorToInt(Y);
  float fx = X - ix;
  float fy = Y - iy;

  float patch[4][4];

  // Fetch surrounding 4x4 pixel values
  for (int m = -1; m <= 2; ++m)
  {
      for (int n = -1; n <= 2; ++n)
      {
          uint16 val = GetPixelG16(Pixels, Width, Height, ix + n, iy + m);
          patch[m + 1][n + 1] = val / 65535.0f;  // Normalize to [0,1]
      }
  }

  float col[4];
  for (int i = 0; i < 4; ++i)
  {
      col[i] = CubicHermite(patch[i][0], patch[i][1], patch[i][2], patch[i][3], fx);
  }

  float result = CubicHermite(col[0], col[1], col[2], col[3], fy);
  return FMath::Clamp(result, 0.0f, 1.0f); // Final result in [0,1]
}
