#include <cuda_runtime.h>
#include <cfloat>
#include "helper_math.h"
#include "rtWarpAggregation.h"
#include "rtDebugFunc.h"
#include "rtHelperFunc.h"
#include "rtSurfelSorting.h"

struct AABB
{
	float3 Low;
	float3 High;
};

__device__ __inline__ void atomicMin(float* ptr, float value)
{
	unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
	while (value < int_as_float(curr))
	{
		unsigned int prev = curr;
		curr = atomicCAS((unsigned int*)ptr, curr, float_as_int(value));
		if (curr == prev)
			break;
	}
}

__device__ __inline__ void atomicMax(float* ptr, float value)
{
	unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
	while (value > int_as_float(curr))
	{
		unsigned int prev = curr;
		curr = atomicCAS((unsigned int*)ptr, curr, float_as_int(value));
		if (curr == prev)
			break;
	}
}

__global__ void ComputeAABB(
	float4* WorldPositionTexture,
	float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
    AABB* const OutWorldPositionAABB
)
{
	__shared__ AABB threadAABBs[64];

	threadAABBs[threadIdx.x].Low = make_float3(FLT_MAX);
	threadAABBs[threadIdx.x].High = make_float3(-FLT_MAX);

	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= SizeX * SizeY)
		return;

	float3 WorldPosition = make_float3(WorldPositionTexture[TargetTexelLocation]);
	float TexelRadius = TexelRadiusTexture[TargetTexelLocation];

	if (TexelRadius == 0.0f)
		return;

	threadAABBs[threadIdx.x].Low = WorldPosition;
	threadAABBs[threadIdx.x].High = WorldPosition;

	for (int stride = 64 >> 1; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			threadAABBs[threadIdx.x].Low = fminf(threadAABBs[threadIdx.x].Low, threadAABBs[threadIdx.x + stride].Low);
			threadAABBs[threadIdx.x].High = fmaxf(threadAABBs[threadIdx.x].High, threadAABBs[threadIdx.x + stride].High);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		atomicMin(&OutWorldPositionAABB->Low.x, threadAABBs[0].Low.x);
		atomicMin(&OutWorldPositionAABB->Low.y, threadAABBs[0].Low.y);
		atomicMin(&OutWorldPositionAABB->Low.z, threadAABBs[0].Low.z);
		atomicMax(&OutWorldPositionAABB->High.x, threadAABBs[0].High.x);
		atomicMax(&OutWorldPositionAABB->High.y, threadAABBs[0].High.y);
		atomicMax(&OutWorldPositionAABB->High.z, threadAABBs[0].High.z);
	}
}

__device__ __inline__ void collectBits(MortonHash6 &OutHash, int index, float value)
{
	for (int i = 0; i < 32; i++)
		OutHash.Hash[(index + i * 6) >> 5] |= ((static_cast<unsigned int>(value) >> i) & 1) << ((index + i * 6) & 31);
}

__global__ void ComputeMortonHash(
	float4* WorldPositionTexture,
	float4* WorldNormalTexture,
	float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
    const AABB* WorldPositionAABB,
	MortonHash6* OutHashes,
    int* const OutNumMappedTexel
)
{
	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= SizeX * SizeY)
		return;

	float3 WorldPosition = make_float3(WorldPositionTexture[TargetTexelLocation]);
	float3 WorldNormal = make_float3(WorldNormalTexture[TargetTexelLocation]);
	float TexelRadius = TexelRadiusTexture[TargetTexelLocation];

	if (TexelRadius == 0.0f)
		return;

	float3 a = (WorldPosition - WorldPositionAABB->Low) / fmaxf((WorldPositionAABB->High - WorldPositionAABB->Low), make_float3(0.00001f));
	float3 b = (normalize(WorldNormal) + 1.0f) * 0.5f;

	int OutSlot = atomicAggInc(OutNumMappedTexel);

	OutHashes[OutSlot].OriginalPosition = TargetTexelLocation;

	for (int i = 0; i < 6; i++)
		OutHashes[OutSlot].Hash[i] = 0;

	collectBits(OutHashes[OutSlot], 0, a.x * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 1, a.y * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 2, a.z * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 3, b.x * 32.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 4, b.y * 32.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 5, b.z * 32.0f * 65536.0f);
}

__host__ int compareMortonKey(const void* A, const void* B)
{
	const MortonHash6& a = *((const MortonHash6*)A);
	const MortonHash6& b = *((const MortonHash6*)B);
	if (a.Hash[5] != b.Hash[5]) return (a.Hash[5] < b.Hash[5] ? 1 : -1);
	if (a.Hash[4] != b.Hash[4]) return (a.Hash[4] < b.Hash[4] ? 1 : -1);
	if (a.Hash[3] != b.Hash[3]) return (a.Hash[3] < b.Hash[3] ? 1 : -1);
	if (a.Hash[2] != b.Hash[2]) return (a.Hash[2] < b.Hash[2] ? 1 : -1);
	if (a.Hash[1] != b.Hash[1]) return (a.Hash[1] < b.Hash[1] ? 1 : -1);
	if (a.Hash[0] != b.Hash[0]) return (a.Hash[0] < b.Hash[0] ? 1 : -1);
	return 0;
}

__host__ int rtComputeSurfelMortonHash(
    float4* cudaWorldPositionTexture,
	float4* cudaWorldNormalTexture,
	float* cudaTexelRadiusTexture,
	const int SizeX,
	const int SizeY,
    MortonHash6* const OutCudaMortonHashes
)
{
    AABB* cudaWorldPositionAABB;
    cudaCheck(cudaMalloc((void**)&cudaWorldPositionAABB, 1 * sizeof(AABB)));
    
    {
        AABB initAABB;
		initAABB.Low = make_float3(FLT_MAX);
		initAABB.High = make_float3(-FLT_MAX);

		cudaCheck(cudaMemcpy(cudaWorldPositionAABB, &initAABB, 1 * sizeof(AABB), cudaMemcpyHostToDevice));
    }
    
    {
        const int Stride = 64;
        dim3 blockDim(Stride, 1);
        dim3 gridDim(divideAndRoundup(SizeX * SizeY, Stride), 1);

        ComputeAABB <<< gridDim, blockDim >>> (
            cudaWorldPositionTexture,
            cudaTexelRadiusTexture,
            SizeX, 
            SizeY,
            cudaWorldPositionAABB);
    }
    
    int numMappedTexel;
    int* cudaNumMappedTexel;
    cudaCheck(cudaMalloc((void**)&cudaNumMappedTexel, 1 * sizeof(int)));
    
    {
        const int Stride = 64;
        dim3 blockDim(Stride, 1);
        dim3 gridDim(divideAndRoundup(SizeX * SizeY, Stride), 1);

        ComputeMortonHash << < gridDim, blockDim >> > (
            cudaWorldPositionTexture,
            cudaWorldNormalTexture,
            cudaTexelRadiusTexture,
            SizeX,
            SizeY,
            cudaWorldPositionAABB,
            OutCudaMortonHashes,
            cudaNumMappedTexel);
    }

    cudaCheck(cudaMemcpy(&numMappedTexel, cudaNumMappedTexel, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    MortonHash6* ComputedHashes;
    cudaHostAlloc(&ComputedHashes, numMappedTexel * sizeof(MortonHash6), 0);

    cudaCheck(cudaMemcpy(ComputedHashes, OutCudaMortonHashes, numMappedTexel * sizeof(MortonHash6), cudaMemcpyDeviceToHost));

    qsort(ComputedHashes, numMappedTexel, sizeof(MortonHash6), compareMortonKey);

    cudaCheck(cudaMemcpy(OutCudaMortonHashes, ComputedHashes, numMappedTexel * sizeof(MortonHash6), cudaMemcpyHostToDevice));
    
    cudaFreeHost(ComputedHashes);
    cudaFree(cudaWorldPositionAABB);
    cudaFree(cudaNumMappedTexel);
    
    return numMappedTexel;
}
