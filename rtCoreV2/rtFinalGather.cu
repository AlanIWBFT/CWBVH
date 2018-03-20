#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "rtDebugFunc.h"
#include "rtHelperFunc.h"
#include "rtWarpAggregation.h"
#include "rtSurfelSorting.h"
#include "rtTrace.h"
#include "cub/cub.cuh"
#include "sh_lightsample.h"
#include "sh_warpreduction.h"
#include "progress_bar.h"

#define USE_ADAPTIVE_SAMPLING 0
#define USE_CORRELATED_SAMPLING 0
#define USE_JITTERED_SAMPLING 1

const int NumSampleTheta = 32;
const int NumSamplePhi = NumSampleTheta * 4;
const int TotalSamplePerTexel = NumSampleTheta * NumSamplePhi;

const int NumBucketTheta = 8;
const int NumBucketPhi = 32;
const int TotalBucketPerTexel = NumBucketTheta * NumBucketPhi;

const int NumSamplePerBucketTheta = NumSampleTheta / NumBucketTheta;
const int NumSamplePerBucketPhi = NumSamplePhi / NumBucketPhi;
const int TotalSamplePerBucket = NumSamplePerBucketTheta * NumSamplePerBucketPhi;

const int ImportanceImageFilteringFactor = 16;

const int MaxRayBufferSize = 64 * 64 * 4096;
const int ImageBlockSize = MaxRayBufferSize / TotalSamplePerTexel;

__device__ float4 BucketRadiance[ImageBlockSize * TotalBucketPerTexel];
__device__ float DownsampledBucketImportance[ImageBlockSize / ImportanceImageFilteringFactor * TotalBucketPerTexel];
__device__ int BucketRayStartOffsetInTexel[ImageBlockSize * TotalBucketPerTexel];
__device__ int TexelToRayIDMap[ImageBlockSize];

struct RayStartInfo
{
	float3 RayInWorldSpace;
	float TangentZ;
};

__global__ void BucketRayGenKernel(
    const float4* WorldPositionTexture,
	const float4* WorldNormalTexture,
	const float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
	const int Offset,
	MortonHash6* InHashes,
    const int NumMappedTexel,
    int* const OutRayCount,
    Ray* const OutRayBuffer,
    RayStartInfo* const OutRayStartInfoBuffer
)
{
	const int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId / TotalSamplePerTexel >= NumMappedTexel)
		return;
	int targetTexel = Offset + threadId / TotalSamplePerTexel;
	
	targetTexel = InHashes[targetTexel].OriginalPosition;

	const int targetRay = threadId % TotalSamplePerTexel;
	const int targetBucketInTexel = targetRay / TotalSamplePerBucket;
	const int targetBucket = threadId / TotalSamplePerBucket;
	const int targetRayInBucket = targetRay % TotalSamplePerBucket;
	const int targetBucketTheta = targetBucketInTexel / NumBucketPhi;
	const int targetBucketPhi = targetBucketInTexel % NumBucketPhi;
	const int targetRayThetaInBucket = targetRayInBucket / NumSamplePerBucketPhi;
	const int targetRayPhiInBucket = targetRayInBucket % NumSamplePerBucketPhi;
	const int sampleIndexTheta = targetRayThetaInBucket + targetBucketTheta * NumSamplePerBucketTheta;
	const int sampleIndexPhi = targetRayPhiInBucket + targetBucketPhi * NumSamplePerBucketPhi;

	if (targetTexel >= SizeX * SizeY)
		return;

	float3 worldPosition = make_float3(WorldPositionTexture[targetTexel]);
	float3 worldNormal = make_float3(WorldNormalTexture[targetTexel]);
	float texelRadius = TexelRadiusTexture[targetTexel];

	if (texelRadius == 0.0f)
		return;

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	curand_init(Offset * TotalSamplePerTexel + threadId, 0, 0, &randState);
#endif
#endif

	float RandA = 0.5f;
	float RandB = 0.5f;
	float RandC = 0.5f;
	float RandD = 0.5f;

#if USE_JITTERED_SAMPLING
	RandA = curand_uniform(&randState);
	RandB = curand_uniform(&randState);
	RandC = curand_uniform(&randState);
	RandD = curand_uniform(&randState);
#endif

	float U = 1.0f * (sampleIndexTheta + RandA) / NumSampleTheta;
	float Phi = 2.f * 3.1415926f * (sampleIndexPhi + RandB) / NumSamplePhi;
	float3 RayInLocalSpace = make_float3(sqrtf(1 - U*U) * cos(Phi), sqrtf(1 - U*U) * sin(Phi), U);
	float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);

	float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.5f;
	RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * 0.5f;
	RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * 0.5f;

	Ray ray;
	ray.RayOriginAndNearClip = make_float4(RayOrigin, 0.01f);
	ray.RayDirectionAndFarClip = make_float4(RayInWorldSpace, 1e20);

	int rayID = threadId;
	atomicAggInc(OutRayCount);

	if (targetRay == 0)
		TexelToRayIDMap[threadId / TotalSamplePerTexel] = rayID;

	if (targetRayInBucket == 0)
		BucketRayStartOffsetInTexel[targetBucket] = TotalSamplePerBucket * targetBucketInTexel;

	OutRayBuffer[rayID] = ray;
	OutRayStartInfoBuffer[rayID].RayInWorldSpace = RayInWorldSpace;
	OutRayStartInfoBuffer[rayID].TangentZ = texelRadius == 0.0f ? -1.0f: RayInLocalSpace.z;
}


__device__ float3 SampleRadiance(RayResult result, float3 RayInWorldSpace)
{
	if (result.HitTriangleIndex == -1)
	{
		return make_float3(1.0f);
	}
	else
	{
        return make_float3(0.0f);
	}
}

__global__ void BucketGatherKernel(
    const RayResult* const RayHitResultBuffer,
    const RayStartInfo* const RayStartInfoBuffer
)
{
	int targetBucket = 2 * blockIdx.x + threadIdx.y;
	if (targetBucket >= ImageBlockSize * TotalBucketPerTexel) return;

	int targetTexel = targetBucket / TotalBucketPerTexel;
	int targetBucketInTexel = targetBucket % TotalBucketPerTexel;

	int totalRayInThisBucket;
	if (targetBucketInTexel == TotalBucketPerTexel - 1)
		totalRayInThisBucket = TotalSamplePerTexel - BucketRayStartOffsetInTexel[targetBucket];
	else
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket + 1] - BucketRayStartOffsetInTexel[targetBucket];

	int rayStart = targetTexel * TotalSamplePerTexel + BucketRayStartOffsetInTexel[targetBucket];
	int rayEnd = rayStart + totalRayInThisBucket;

	float4 bucketRadiance = make_float4(0.0f);

	for (int offset = 0; offset < totalRayInThisBucket; offset += warpSize)
	{
		int rayID = rayStart + offset + threadIdx.x;
		if (rayID < rayEnd)
		{
			float3 radiance = RayStartInfoBuffer[rayID].TangentZ > 0.0f ? SampleRadiance(RayHitResultBuffer[rayID], RayStartInfoBuffer[rayID].RayInWorldSpace) * RayStartInfoBuffer[rayID].TangentZ: make_float3(0.0f);

			typedef cub::WarpReduce<float> WarpReduce;
			__shared__ typename WarpReduce::TempStorage temp_storage;

			radiance.x = WarpReduce(temp_storage).Sum(radiance.x);
			radiance.y = WarpReduce(temp_storage).Sum(radiance.y);
			radiance.z = WarpReduce(temp_storage).Sum(radiance.z);

			float luminance = getLuminance(radiance);
			float variance = 0.0f;

			if (lane_id() % NumSamplePerBucketPhi != NumSamplePerBucketPhi - 1)
			{
				float luminanceNeighbour = __shfl_down_sync(__activemask(), luminance, 1);
				variance += (luminance - luminanceNeighbour) * (luminance - luminanceNeighbour);
			}

			if (lane_id() / NumSamplePerBucketPhi != NumSamplePerBucketTheta - 1)
			{
				float luminanceNeighbour = __shfl_down_sync(__activemask(), luminance, NumSamplePerBucketPhi);
				variance += (luminance - luminanceNeighbour) * (luminance - luminanceNeighbour);
			}

			variance = WarpReduce(temp_storage).Sum(variance);

			bucketRadiance += make_float4(radiance, variance);
		}
	}

	if (threadIdx.x == 0)
	{
		BucketRadiance[targetBucket] += bucketRadiance;
	}
}

__global__ void DownsampleImportanceKernel()
{
	int targetTexel = blockIdx.x;
	if (targetTexel >= ImageBlockSize / ImportanceImageFilteringFactor) return;
	int targetBucket = targetTexel * TotalBucketPerTexel + threadIdx.x;
	if (targetBucket >= ImageBlockSize / ImportanceImageFilteringFactor * TotalBucketPerTexel) return;

	float importance = 0.00001f;

	for (int i = 0; i < ImportanceImageFilteringFactor; i++)
	{
		int sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + threadIdx.x;
		importance += BucketRadiance[sourceBucket].w;
	}

	typedef cub::BlockReduce<float, TotalBucketPerTexel> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	
	float totalImportanceOnThread0 = BlockReduce(temp_storage).Sum(importance);
	
	__shared__ float totalImportance;

	if (threadIdx.x == 0)
		totalImportance = totalImportanceOnThread0;

	__syncthreads();

	DownsampledBucketImportance[targetBucket] = importance / totalImportance;

}

__global__ void ScatterImportanceAndCalculateSampleNumKernel()
{
	int targetBucket = blockIdx.x * TotalBucketPerTexel + threadIdx.x;
	if (targetBucket >= ImageBlockSize * TotalBucketPerTexel) return;

	float importance = DownsampledBucketImportance[(blockIdx.x / ImportanceImageFilteringFactor) * TotalBucketPerTexel + threadIdx.x];

	int numSamplesThisBucket = importance * TotalSamplePerTexel;

	int numSamplesBeforeThisBucket = 0;

	typedef cub::BlockScan<int, TotalBucketPerTexel> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	BlockScan(temp_storage).ExclusiveSum(numSamplesThisBucket, numSamplesBeforeThisBucket);

	BucketRayStartOffsetInTexel[targetBucket] = numSamplesBeforeThisBucket;
}

__global__ void BucketAdaptiveRayGenKernel(
	const float4* WorldPositionTexture,
	const float4* WorldNormalTexture,
	const float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
	const int Offset,
	MortonHash6* InHashes,
    const int NumMappedTexel,
    int* const OutRayCount,
    Ray* const OutRayBuffer,
    RayStartInfo* const OutRayStartInfoBuffer
)
{
	int targetBucket = 2 * blockIdx.x + threadIdx.y;
	if (targetBucket >= ImageBlockSize * TotalBucketPerTexel) return;
	int targetBucketInTexel = targetBucket % TotalBucketPerTexel;
	int targetTexel = targetBucket / TotalBucketPerTexel;

	if (targetTexel >= NumMappedTexel)
		return;

	int targetTexelGlobal = InHashes[Offset + targetTexel].OriginalPosition;

	if (targetTexelGlobal >= SizeX * SizeY)
		return;

	float3 worldPosition = make_float3(WorldPositionTexture[targetTexelGlobal]);
	float3 worldNormal = make_float3(WorldNormalTexture[targetTexelGlobal]);
	float texelRadius = TexelRadiusTexture[targetTexelGlobal];

	if (texelRadius == 0.0f)
		return;

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	int threadId = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
	curand_init(ImageBlockSize + TotalSamplePerTexel + Offset * TotalSamplePerTexel + threadId, 0, 0, &randState);
#endif
#endif

	int totalRayInThisBucket;
	if (targetBucketInTexel == TotalBucketPerTexel - 1)
	{
		totalRayInThisBucket = TotalSamplePerTexel - BucketRayStartOffsetInTexel[targetBucket];
	}
	else {
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket + 1] - BucketRayStartOffsetInTexel[targetBucket];
	}
	int rayStart = targetTexel * TotalSamplePerTexel + BucketRayStartOffsetInTexel[targetBucket];
	int rayEnd = rayStart + totalRayInThisBucket;

	int numSampleTheta = sqrtf(totalRayInThisBucket / 4.0f);
	numSampleTheta = max(numSampleTheta, 1);
	int numSamplePhi = totalRayInThisBucket / numSampleTheta;
	numSamplePhi = max(numSamplePhi, 1);
	int numActualSamplesTaken = numSampleTheta * numSamplePhi;

	for (int offset = 0; offset < totalRayInThisBucket; offset += warpSize)
	{
		int RayIdInBucket = offset + threadIdx.x;
		int rayID = rayStart + RayIdInBucket;
		if (rayID < rayEnd)
		{
			const int targetRayInBucket = RayIdInBucket;
			const int targetBucketTheta = targetBucketInTexel / NumBucketPhi;
			const int targetBucketPhi = targetBucketInTexel % NumBucketPhi;
			const int targetRayThetaInBucket = targetRayInBucket / numSamplePhi;
			const int targetRayPhiInBucket = targetRayInBucket % numSamplePhi;

			float RandA = 0.5f;
			float RandB = 0.5f;
// 			float RandC = 0.5f;
// 			float RandD = 0.5f;

#if USE_JITTERED_SAMPLING
			RandA = curand_uniform(&randState);
			RandB = curand_uniform(&randState);
// 			RandC = curand_uniform(&randState);
// 			RandD = curand_uniform(&randState);
#endif
			
			float U = 1.0f * ((targetRayThetaInBucket + RandA) / numSampleTheta + targetBucketTheta) / NumBucketTheta;
			float Phi = 2.f * 3.1415926f * ((targetRayPhiInBucket + RandB) / numSamplePhi + targetBucketPhi) / NumBucketPhi;
			float3 RayInLocalSpace = make_float3(sqrtf(1 - U*U) * cos(Phi), sqrtf(1 - U*U) * sin(Phi), U);
			float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);

			float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.5f;
			//RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * 0.5f;
			//RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * 0.5f;

			Ray ray;
			ray.RayOriginAndNearClip = make_float4(RayOrigin, 0.01f);
			ray.RayDirectionAndFarClip = make_float4(RayInWorldSpace, 1e20);

			atomicAggInc(OutRayCount);

			if (targetBucketInTexel == 0 && targetRayInBucket == 0)
				TexelToRayIDMap[targetTexel] = rayID;

			OutRayBuffer[rayID] = ray;
			OutRayStartInfoBuffer[rayID].RayInWorldSpace = RayInWorldSpace;
			OutRayStartInfoBuffer[rayID].TangentZ = texelRadius == 0.0f ? -1.0f : RayInLocalSpace.z;

			if (targetRayInBucket >= numActualSamplesTaken)
				OutRayStartInfoBuffer[rayID].TangentZ = 0.0f;
		}
	}
}

__global__ void BucketShadingKernel(
    const float4* WorldPositionTexture,
	const float4* WorldNormalTexture,
	const float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
	const int Offset,
	MortonHash6* InHashes,
    const int NumMappedTexel,
	GatheredLightSample* OutLightmapData)
{
	int targetTexel = Offset + blockIdx.x;

	if (targetTexel >= NumMappedTexel)
		return;

	targetTexel = InHashes[targetTexel].OriginalPosition;

	float3 worldPosition = make_float3(WorldPositionTexture[targetTexel]);
	float3 worldNormal = make_float3(WorldNormalTexture[targetTexel]);
	float texelRadius = TexelRadiusTexture[targetTexel];
	bool isTwoSided = WorldPositionTexture[targetTexel].w == 1.0f;

	if (texelRadius == 0.0f)
		return;

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

	const int sampleIndexTheta = threadIdx.x;
	const int sampleIndexPhi = threadIdx.y;

	float U = 1.0f * (sampleIndexTheta + 0.5f) / NumBucketTheta;
	float Phi = 2.f * 3.1415926f * (sampleIndexPhi + 0.5f) / NumBucketPhi;

	float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
	float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);
    
	int numActualSamplesTaken = 0;
    
#if USE_ADAPTIVE_SAMPLING
	int targetBucketInTexel = threadIdx.x * NumBucketPhi + threadIdx.y;
	int targetBucket = blockIdx.x * TotalBucketPerTexel + targetBucketInTexel;
    
	int totalRayInThisBucket;
	if (targetBucketInTexel == TotalBucketPerTexel - 1)
		totalRayInThisBucket = TotalSamplePerTexel - BucketRayStartOffsetInTexel[targetBucket];
	else
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket + 1] - BucketRayStartOffsetInTexel[targetBucket];
	
	int numSampleTheta = sqrtf(totalRayInThisBucket / 4.0f);
	numSampleTheta = max(numSampleTheta, 1);
	int numSamplePhi = totalRayInThisBucket / numSampleTheta;
	numSamplePhi = max(numSamplePhi, 1);
	numActualSamplesTaken = numSampleTheta * numSamplePhi;
#endif

	GatheredLightSample LightSample;
	LightSample.Reset();
	float3 radiance = (2 * 3.1415926f) * make_float3(BucketRadiance[(TotalBucketPerTexel * blockIdx.x + threadIdx.x * NumBucketPhi + threadIdx.y)]) / (TotalSamplePerBucket + numActualSamplesTaken) / TotalBucketPerTexel;

	LightSample.PointLightWorldSpacePreweighted(radiance, RayInLocalSpace, RayInWorldSpace);

	if (isTwoSided)
		LightSample.PointLightWorldSpacePreweighted(radiance, RayInLocalSpace, -RayInWorldSpace);

	LightSample = blockReduceSumToThread0(LightSample);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		OutLightmapData[targetTexel] = LightSample;
	}
}

__host__ void LaunchFinalGather(
	float4* WorldPositionTexture,
	float4* WorldNormalTexture,
	float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
	GatheredLightSample* OutLightmapData)
{
	float4* cudaWorldPositionTexture;
	float4* cudaWorldNormalTexture;
	float*	cudaTexelRadiusTexture;
	GatheredLightSample* cudaOutLightmapData;

	cudaCheck(cudaMalloc((void**)&cudaWorldPositionTexture, SizeX * SizeY * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaWorldPositionTexture, WorldPositionTexture, SizeX * SizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaWorldNormalTexture, SizeX * SizeY * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaWorldNormalTexture, WorldNormalTexture, SizeX * SizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTexelRadiusTexture, SizeX * SizeY * sizeof(float)));
	cudaCheck(cudaMemcpy(cudaTexelRadiusTexture, TexelRadiusTexture, SizeX * SizeY * sizeof(float), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaOutLightmapData, SizeX * SizeY * sizeof(GatheredLightSample)));

	cudaCheck(cudaMemset(cudaOutLightmapData, 0, SizeX * SizeY * sizeof(GatheredLightSample)));
    
    
    MortonHash6* cudaMortonHashes;
    cudaCheck(cudaMalloc((void**)&cudaMortonHashes, SizeX * SizeY * sizeof(MortonHash6)));
    
    int numMappedTexel = rtComputeSurfelMortonHash(
        cudaWorldPositionTexture,
        cudaWorldNormalTexture,
        cudaTexelRadiusTexture,
        SizeX,
        SizeY,
        cudaMortonHashes);
    
    Ray* cudaRayBuffer;
	RayStartInfo* cudaRayStartInfoBuffer;
	RayResult* cudaRayHitResultBuffer;
	cudaCheck(cudaMalloc(&cudaRayBuffer, sizeof(Ray) * ImageBlockSize * TotalSamplePerTexel));
	cudaCheck(cudaMalloc(&cudaRayStartInfoBuffer, sizeof(RayStartInfo) * ImageBlockSize * TotalSamplePerTexel));
	cudaCheck(cudaMalloc(&cudaRayHitResultBuffer, sizeof(RayResult) * ImageBlockSize * TotalSamplePerTexel));
    
    int* cudaRayCount;
    cudaCheck(cudaMalloc(&cudaRayCount, sizeof(int)));
    
    double numTotalRay = 0.0;
    
    float elapsedTime = 0.0f;
    cudaEvent_t startEvent, stopEvent;
    cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));
	cudaCheck(cudaEventRecord(startEvent, 0));
    
    for (int i = 0; i < divideAndRoundup(numMappedTexel, ImageBlockSize); i++)
	{
        //progress_bar((float)(i + 1) / divideAndRoundup(numMappedTexel, ImageBlockSize));
        //printf("  Block %d/%d", i + 1, divideAndRoundup(numMappedTexel, ImageBlockSize));
		
        int rayCount = 0;
        
        cudaMemset(cudaRayCount, 0, sizeof(int));

		void* cudaBucketRadianceMap;
		cudaCheck(cudaGetSymbolAddress(&cudaBucketRadianceMap, BucketRadiance));
		cudaCheck(cudaMemset(cudaBucketRadianceMap, 0, sizeof(float4) * ImageBlockSize * TotalBucketPerTexel));
		void* cudaBucketRayStartOffsetInTexel;
		cudaCheck(cudaGetSymbolAddress(&cudaBucketRayStartOffsetInTexel, BucketRayStartOffsetInTexel));
		cudaCheck(cudaMemset(cudaBucketRayStartOffsetInTexel, 0, sizeof(int) * ImageBlockSize * TotalBucketPerTexel));

		{
			dim3 blockDim(32 * 2, 1);
			dim3 gridDim(divideAndRoundup(ImageBlockSize * TotalSamplePerTexel, blockDim.x), 1);

			BucketRayGenKernel << < gridDim, blockDim >> > (
				cudaWorldPositionTexture,
				cudaWorldNormalTexture,
				cudaTexelRadiusTexture,
				SizeX,
				SizeY,
				i * ImageBlockSize,
				cudaMortonHashes,
                numMappedTexel,
                cudaRayCount,
                cudaRayBuffer,
                cudaRayStartInfoBuffer
				);
			cudaPostKernelLaunchCheck
		}

		cudaCheck(cudaMemcpy(&rayCount, cudaRayCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));

		numTotalRay += rayCount;

		rtTrace(cudaRayBuffer, cudaRayHitResultBuffer, rayCount);

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(ImageBlockSize * TotalBucketPerTexel, 1);
			BucketGatherKernel << < gridDim, blockDim >> > (
                cudaRayHitResultBuffer,
                cudaRayStartInfoBuffer
            );
			cudaPostKernelLaunchCheck
		}

#if USE_ADAPTIVE_SAMPLING
		{
			dim3 blockDim(TotalBucketPerTexel, 1);
			dim3 gridDim(divideAndRoundup(ImageBlockSize, ImportanceImageFilteringFactor), 1);
			DownsampleImportanceKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		{
			dim3 blockDim(TotalBucketPerTexel, 1);
			dim3 gridDim(ImageBlockSize, 1);
			ScatterImportanceAndCalculateSampleNumKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		rayCount = 0;
        cudaMemset(cudaRayCount, 0, sizeof(int));

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(ImageBlockSize * TotalBucketPerTexel, 1);
			BucketAdaptiveRayGenKernel << < gridDim, blockDim >> > (
				cudaWorldPositionTexture,
				cudaWorldNormalTexture,
				cudaTexelRadiusTexture,
				SizeX,
				SizeY,
				i * ImageBlockSize,
				cudaMortonHashes,
                numMappedTexel,
                cudaRayCount,
                cudaRayBuffer,
                cudaRayStartInfoBuffer
				);
			cudaPostKernelLaunchCheck
		}

		cudaCheck(cudaMemcpy(&rayCount, cudaRayCount, 1 * sizeof(int), cudaMemcpyDeviceToHost));

		numTotalRay += rayCount;

		rtTrace(cudaRayBuffer, cudaRayHitResultBuffer, rayCount);

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(ImageBlockSize * TotalBucketPerTexel, 1);
			BucketGatherKernel << < gridDim, blockDim >> > (
                cudaRayHitResultBuffer,
                cudaRayStartInfoBuffer
            );
			cudaPostKernelLaunchCheck
		}
#endif

		{
			dim3 blockDim(NumBucketTheta, NumBucketPhi);
			dim3 gridDim(ImageBlockSize, 1);

			BucketShadingKernel << < gridDim, blockDim >> > (
				cudaWorldPositionTexture,
				cudaWorldNormalTexture,
				cudaTexelRadiusTexture,
				SizeX,
				SizeY,
				i * ImageBlockSize,
				cudaMortonHashes,
                numMappedTexel,
                cudaOutLightmapData
				);
			cudaPostKernelLaunchCheck
		}
	}
	
    cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    
    progress_bar(1.0f);
	printf("  Block %d/%d\n", divideAndRoundup(numMappedTexel, ImageBlockSize), divideAndRoundup(numMappedTexel, ImageBlockSize));
    
    double MRaysPerSecond = numTotalRay / 1000000.0f / (elapsedTime / 1000.0f);

	printf("%.3fMS, %.2lfMRays/s, %d texels\n", elapsedTime, MRaysPerSecond, numMappedTexel);
    
	cudaCheck(cudaMemcpy(OutLightmapData, cudaOutLightmapData, SizeX * SizeY * sizeof(GatheredLightSample), cudaMemcpyDeviceToHost));
    
    cudaFree(cudaRayBuffer);
	cudaFree(cudaRayStartInfoBuffer);
	cudaFree(cudaRayHitResultBuffer);
    
    cudaFree(cudaMortonHashes);

    cudaCheck(cudaFree(cudaOutLightmapData));
	cudaCheck(cudaFree(cudaWorldPositionTexture));
	cudaCheck(cudaFree(cudaWorldNormalTexture));
	cudaCheck(cudaFree(cudaTexelRadiusTexture));
}

