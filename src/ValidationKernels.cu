#include <vector>
#include <curand_kernel.h>
#include "helper_math.h"
#include "CUDAAssert.h"

__global__ void GenerateValidationTrianglesKernel(
	float3* OutVertexBuffer,
	int3* OutIndexBuffer,
	int Count
)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= Count) return;

	int triId = threadId;

	OutVertexBuffer[threadId * 3 + 0] = make_float3(- 1.1f, - 1.1f, triId);
	OutVertexBuffer[threadId * 3 + 1] = make_float3(+ 1.1f, - 1.1f, triId);
	OutVertexBuffer[threadId * 3 + 2] = make_float3(+ 1.1f, + 1.1f, triId);

	OutIndexBuffer[threadId] = make_int3(triId * 3 + 0, triId * 3 + 1, triId * 3 + 2);
}

__global__ void GenerateValidationRaysKernel(
	Ray* OutRayBuffer,
	int Count
)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= Count) return;

	int rayId = threadId;

	Ray ray;
	ray.origin_tmin = make_float4(0.5f, -0.5f, rayId * 12345678 % Count - 0.5f, 0.0f);
	ray.dir_tmax = make_float4(normalize(make_float3(-0.01, 0.5, 1)), 1e20);

	OutRayBuffer[threadId] = ray;
}

__device__ float2 ConcentricSampleDisk(const float2 &u)
{
	float2 uOffset = 2.f * u - make_float2(1, 1);
	if (uOffset.x == 0 && uOffset.y == 0) return make_float2(0, 0);

	float theta, r;
	if (uOffset.x * uOffset.x > uOffset.y * uOffset.y) {
		r = uOffset.x;
		theta = (3.1415926f / 4.0f) * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = (3.1415926f / 2.0f) - (3.1415926f / 4.0f) * (uOffset.x / uOffset.y);
	}
	return r * make_float2(cosf(theta), sinf(theta));
}

__device__ float3 LiftPoint2DToHemisphere(const float2& p)
{
	return make_float3(p.x, p.y, sqrtf(1 - p.x * p.x - p.y * p.y));
}

__global__ void GenerateAORayUniformKernel(
	int NumSurfels,
	int multiplier,
	float4* __restrict__ SurfelWorldPosition,
	float4* __restrict__ SurfelWorldNormal,
	float* SurfelRadius,
	Ray* OutRays
)
{
	const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= NumSurfels * multiplier) return;

	curandState randState;
	curand_init(threadId, 0, 0, &randState);

	float3 WorldPosition = make_float3(SurfelWorldPosition[threadId / multiplier]);
	float3 WorldNormal = make_float3(SurfelWorldNormal[threadId / multiplier]);
	float TexelRadius = SurfelRadius[threadId / multiplier];

	float3 Tangent1, Tangent2;

	Tangent1 = cross(WorldNormal, make_float3(0, 0, 1));
	Tangent1 = length(Tangent1) < 0.1 ? cross(WorldNormal, make_float3(0, 1, 0)) : Tangent1;
	Tangent1 = normalize(Tangent1);
	Tangent2 = normalize(cross(Tangent1, WorldNormal));

	float RandA = curand_uniform(&randState);
	float RandB = curand_uniform(&randState);

	float2 UniformPoint2D = make_float2(RandA, RandB);
	float2 ConcentricMappedPoint = ConcentricSampleDisk(UniformPoint2D);
	float3 RayInLocalSpace = LiftPoint2DToHemisphere(ConcentricMappedPoint);
	float3 RayInWorldSpace = normalize(Tangent1 * RayInLocalSpace.x + Tangent2 * RayInLocalSpace.y + WorldNormal * RayInLocalSpace.z);
	float3 RayOrigin = WorldPosition + WorldNormal * TexelRadius * 0.5f;

	OutRays[threadId].origin_tmin = make_float4(RayOrigin, 0.01f);
	OutRays[threadId].dir_tmax = make_float4(RayInWorldSpace, 1e20);
}

void GenerateValidationTriangles(int Count, std::vector<float3>& OutVertexBuffer, std::vector<int3>& OutIndexBuffer)
{
	float3* cudaVertexBuffer;
	int3* cudaIndexBuffer;

	cudaCheck(cudaMalloc(&cudaVertexBuffer, sizeof(float3) * 3 * Count));
	cudaCheck(cudaMalloc(&cudaIndexBuffer, sizeof(int3) * Count));

	dim3 blockDim(128);
	dim3 gridDim(idivCeil(Count, blockDim.x));

	GenerateValidationTrianglesKernel <<< gridDim, blockDim >>> (cudaVertexBuffer, cudaIndexBuffer, Count);

	OutVertexBuffer.resize(Count * 3);
	OutIndexBuffer.resize(Count);

	cudaCheck(cudaMemcpy(OutVertexBuffer.data(), cudaVertexBuffer, sizeof(float3) * 3 * Count, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(OutIndexBuffer.data(), cudaIndexBuffer, sizeof(int3) * Count, cudaMemcpyDeviceToHost));

	cudaFree(cudaVertexBuffer);
	cudaFree(cudaIndexBuffer);
}

Ray* GenerateValidationRays(int Count)
{
	Ray* cudaRays;

	cudaCheck(cudaMalloc(&cudaRays, sizeof(Ray) * Count));

	dim3 blockDim(128);
	dim3 gridDim(idivCeil(Count, blockDim.x));

	GenerateValidationRaysKernel <<< gridDim, blockDim >>> (cudaRays, Count);

	return cudaRays;
}

Ray* GenerateAORaysUniform(
	int NumTexels,
	int NumRaysPerTexel,
	float4* WorldPosition,
	float4* WorldNormal,
	float* TexelRadius
)
{
	Ray* cudaRays;

	cudaCheck(cudaMalloc(&cudaRays, sizeof(Ray) * NumTexels * NumRaysPerTexel));

	float4* cudaWorldPositionTexture;
	float4* cudaWorldNormalTexture;
	float*	cudaTexelRadiusTexture;

	cudaCheck(cudaMalloc((void**)&cudaWorldPositionTexture, NumTexels * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaWorldPositionTexture, WorldPosition, NumTexels * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaWorldNormalTexture, NumTexels * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaWorldNormalTexture, WorldNormal, NumTexels * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTexelRadiusTexture, NumTexels * sizeof(float)));
	cudaCheck(cudaMemcpy(cudaTexelRadiusTexture, TexelRadius, NumTexels * sizeof(float), cudaMemcpyHostToDevice));

	float elapsedTime;
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));
	cudaCheck(cudaEventRecord(startEvent, 0));
	
	dim3 blockDim(256);
	dim3 gridDim(idivCeil(NumTexels * NumRaysPerTexel, blockDim.x));

	GenerateAORayUniformKernel <<< gridDim, blockDim >>> (NumTexels, NumRaysPerTexel, cudaWorldPositionTexture, cudaWorldNormalTexture, cudaTexelRadiusTexture, cudaRays);

	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	Log("%.3fMS, %.2fMRays/s (RayGen)", elapsedTime, (float)NumTexels * NumRaysPerTexel / 1000000.0f / (elapsedTime / 1000.0f));
	
	cudaCheck(cudaFree(cudaWorldPositionTexture));
	cudaCheck(cudaFree(cudaWorldNormalTexture));
	cudaCheck(cudaFree(cudaTexelRadiusTexture));

	return cudaRays;
}
