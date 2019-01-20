#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "FastDeviceMinMax.h"

#include "Logger.h"
#include "CUDAAssert.h"

#include <cstdio>

__device__ float4* BVHTreeNodes;
__device__ float4* TriangleWoopCoordinates;
__device__ int* MappingFromTriangleAddressToIndex;

__device__ inline bool RayBoxIntersection(float3 Low, float3 High, float3 InvDir, float3 Ood, float TMin, float TMax, float& OutIntersectionDist)
{
	const float3 lo = Low * InvDir - Ood;
	const float3 hi = High * InvDir - Ood;
	const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);

	OutIntersectionDist = slabMin;

	return slabMin <= slabMax;
}

__global__ void rtTraceBVH2Plain(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	int* finishedRayCount
)
{
	const int EntrypointSentinel = 0x76543210;
	const int STACK_SIZE = 32;
	const float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number

	int traversalStack[STACK_SIZE];

	int     rayidx = blockIdx.x * blockDim.x + threadIdx.x;
	float3  idir;    // 1 / ray direction
	float3	ood;
	float2  triangleuv;

	if (rayidx >= rayCount)
		return;

	float3 RayOrigin = make_float3(rayBuffer[rayidx].origin_tmin);
	float3 RayDirection = make_float3(rayBuffer[rayidx].dir_tmax);
	float tmin = rayBuffer[rayidx].origin_tmin.w;
	float hitT = rayBuffer[rayidx].dir_tmax.w;

	// ooeps is very small number, used instead of raydir xyz component when that component is near zero
	idir.x = 1.0f / (fabsf(RayDirection.x) > ooeps ? RayDirection.x : copysignf(ooeps, RayDirection.x)); // inverse ray direction
	idir.y = 1.0f / (fabsf(RayDirection.y) > ooeps ? RayDirection.y : copysignf(ooeps, RayDirection.y)); // inverse ray direction
	idir.z = 1.0f / (fabsf(RayDirection.z) > ooeps ? RayDirection.z : copysignf(ooeps, RayDirection.z)); // inverse ray direction
	ood = RayOrigin * idir;

	// Setup traversal + initialisation

	traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
	int* stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
	int nodeAddr = 0;   // Start from the root.
	int hitAddr = -1;  // No triangle intersected so far.
	int leafAddr = 0;

	const float4* localBVHTreeNodes = BVHTreeNodes;
	const float4* localTriangleWoopCoordinates = TriangleWoopCoordinates;
	// Traversal loop.

	while (nodeAddr != EntrypointSentinel)
	{
		leafAddr = 0;

		while (nodeAddr != EntrypointSentinel && nodeAddr >= 0)
		{
			const float4 n0xy = __ldg(localBVHTreeNodes + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = __ldg(localBVHTreeNodes + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 n01z = __ldg(localBVHTreeNodes + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
			float4 tmp = BVHTreeNodes[nodeAddr + 3]; // child_index0, child_index1
			int2  cnodes = *(int2*)&tmp;

			const float3 c0lo = make_float3(n0xy.x, n0xy.z, n01z.x);
			const float3 c0hi = make_float3(n0xy.y, n0xy.w, n01z.y);

			const float3 c1lo = make_float3(n1xy.x, n1xy.z, n01z.z);
			const float3 c1hi = make_float3(n1xy.y, n1xy.w, n01z.w);

			float c0dist, c1dist;
			bool traverseChild0 = RayBoxIntersection(c0lo, c0hi, idir, ood, tmin, hitT, c0dist);
			bool traverseChild1 = RayBoxIntersection(c1lo, c1hi, idir, ood, tmin, hitT, c1dist);

			bool swp = c1dist < c0dist;

			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *stackPtr;
				stackPtr--;
			}
			else
			{
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

				if (traverseChild0 && traverseChild1)
				{
					if (swp)
						swap(nodeAddr, cnodes.y);

					stackPtr++;
					*stackPtr = cnodes.y;
				}
			}

			if (nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				nodeAddr = *stackPtr;
				stackPtr--;
			}

			if (!__any_sync(__activemask(), leafAddr >= 0))
				break;
		}

		while (leafAddr < 0)
		{
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{
				float4 v00 = __ldg(localTriangleWoopCoordinates + triAddr + 0);
				float4 v11 = __ldg(localTriangleWoopCoordinates + triAddr + 1);
				float4 v22 = __ldg(localTriangleWoopCoordinates + triAddr + 2);

				if (__float_as_int(v00.x) == 0x80000000)
					break;

				float Oz = v00.w - RayOrigin.x * v00.x - RayOrigin.y * v00.y - RayOrigin.z * v00.z;
				float invDz = 1.0f / (RayDirection.x*v00.x + RayDirection.y*v00.y + RayDirection.z*v00.z);
				float t = Oz * invDz;

				if (t > tmin && t < hitT)
				{
					float Ox = v11.w + RayOrigin.x * v11.x + RayOrigin.y * v11.y + RayOrigin.z * v11.z;
					float Dx = RayDirection.x * v11.x + RayDirection.y * v11.y + RayDirection.z * v11.z;
					float u = Ox + t * Dx;

					if (u >= 0.0f && u <= 1.0f)
					{
						float Oy = v22.w + RayOrigin.x * v22.x + RayOrigin.y * v22.y + RayOrigin.z * v22.z;
						float Dy = RayDirection.x * v22.x + RayDirection.y * v22.y + RayDirection.z * v22.z;
						float v = Oy + t*Dy;

						if (v >= 0.0f && u + v <= 1.0f)
						{
							triangleuv.x = u;
							triangleuv.y = v;

							hitT = t;
							hitAddr = triAddr;
						}
					}
				}
			}

			leafAddr = nodeAddr;
			if (nodeAddr < 0)
			{
				nodeAddr = *stackPtr;
				stackPtr--;
			}
		}
	}

	rayResultBuffer[rayidx].t_triId_u_v = make_float4(
		hitT,
		int_as_float(hitAddr),
		triangleuv.x,
		triangleuv.y
	);
}

__host__ void rtBindBVH2Data(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex)
{
	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));
}

__host__ void rtTraceBVH2(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount
)
{
	float elapsedTime;
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));

	int* cudaFinishedRayCount;
	cudaCheck(cudaMalloc(&cudaFinishedRayCount, sizeof(int)));
	cudaMemset(cudaFinishedRayCount, 0, sizeof(int));

	dim3 blockDim(128, 1);
	dim3 gridDim(idivCeil(rayCount, blockDim.x), 1);

	cudaProfilerStart();
	cudaCheck(cudaEventRecord(startEvent, 0));

	rtTraceBVH2Plain <<< gridDim, blockDim >>> (
		rayBuffer,
		rayResultBuffer,
		rayCount,
		cudaFinishedRayCount
		);

	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	Log("%.3fMS, %.2lfMRays/s (rtTraceBVH2 No Dynamic Fetch)", elapsedTime, (double)rayCount / 1000000.0f / (elapsedTime / 1000.0f));

	cudaProfilerStop();

	cudaFree(cudaFinishedRayCount);
}
