#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "FastDeviceMinMax.h"

#include "Logger.h"
#include "CUDAAssert.h"

__device__ unsigned __bfind(unsigned i) { unsigned b; asm volatile("bfind.u32 %0, %1; " : "=r"(b) : "r"(i)); return b; }

__device__ __inline__ uint sign_extend_s8x4(uint i) { uint v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

__device__ __inline__ uint extract_byte(uint i, uint n) { return (i >> (n * 8)) & 0xFF; }

__device__ const float4* BVHTreeNodes;
__device__ const float4* TriangleWoopCoordinates;
__device__ const int* MappingFromTriangleAddressToIndex;

#define DYNAMIC_FETCH 1
#define TRIANGLE_POSTPONING 1

#define STACK_POP(X) { --stackPtr; if (stackPtr < SM_STACK_SIZE) X = traversalStackSM[threadIdx.x][threadIdx.y][stackPtr]; else X = traversalStack[stackPtr - SM_STACK_SIZE]; }
#define STACK_PUSH(X) { if (stackPtr < SM_STACK_SIZE) traversalStackSM[threadIdx.x][threadIdx.y][stackPtr] = X; else traversalStack[stackPtr - SM_STACK_SIZE] = X; stackPtr++; }

__global__ void rtTraceCWBVHDynamicFetch(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount,
	int* finishedRayCount
)
{
	const float ooeps = exp2f(-80.0f);

	const int STACK_SIZE = 32;
	uint2 traversalStack[STACK_SIZE];

	const int SM_STACK_SIZE = 8;
	__shared__ uint2 traversalStackSM[32][2][SM_STACK_SIZE];

	int rayidx;

	float3 orig, dir;
	float tmin, tmax;
	float idirx, idiry, idirz;
	uint octinv;
	uint2 nodeGroup = make_uint2(0);
	uint2 triangleGroup = make_uint2(0);
	char stackPtr = 0;
	int hitAddr = -1;
	float2 triangleuv;

	__shared__ int nextRayArray[2];

	const float4* localBVHTreeNodes = BVHTreeNodes;
	const float4* localTriangleWoopCoordinates = TriangleWoopCoordinates;

	do
	{
		int& rayBase = nextRayArray[threadIdx.y];

		bool				terminated = stackPtr == 0 && nodeGroup.y <= 0x00FFFFFF;
		const unsigned int	maskTerminated = __ballot_sync(__activemask(), terminated);
		const int			numTerminated = __popc(maskTerminated);
		const int			idxTerminated = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(finishedRayCount, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= rayCount)
				break;

			orig = make_float3(rayBuffer[rayidx].origin_tmin);
			dir = make_float3(rayBuffer[rayidx].dir_tmax);
			tmin = rayBuffer[rayidx].origin_tmin.w;
			tmax = rayBuffer[rayidx].dir_tmax.w;
			idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x)); // inverse ray direction
			idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y)); // inverse ray direction
			idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z)); // inverse ray direction
			octinv = ((dir.x < 0 ? 1 : 0) << 2) | ((dir.y < 0 ? 1 : 0) << 1) | ((dir.z < 0 ? 1 : 0) << 0);
			octinv = 7 - octinv;
			nodeGroup = make_uint2(0, 0b10000000000000000000000000000000);
			triangleGroup = make_uint2(0);
			stackPtr = 0;
			hitAddr = -1;
		}
		
	#if DYNAMIC_FETCH
		int lostLoopIterations = 0;
	#endif

		do
		{
			if (nodeGroup.y > 0x00FFFFFF)
			{
				const unsigned int hits = nodeGroup.y;
				const unsigned int imask = nodeGroup.y;
				const unsigned int child_bit_index = __bfind(hits);
				const unsigned int child_node_base_index = nodeGroup.x;

				nodeGroup.y &= ~(1 << child_bit_index);

				if (nodeGroup.y > 0x00FFFFFF)
				{
					STACK_PUSH(nodeGroup);
				}

				{
					const unsigned int slot_index = (child_bit_index - 24) ^ octinv;
					const unsigned int octinv4 = octinv * 0x01010101u;
					const unsigned int relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
					const unsigned int child_node_index = child_node_base_index + relative_index;

					float4 n0, n1, n2, n3, n4;

					n0 = __ldg(localBVHTreeNodes + child_node_index * 5 + 0);
					n1 = __ldg(localBVHTreeNodes + child_node_index * 5 + 1);
					n2 = __ldg(localBVHTreeNodes + child_node_index * 5 + 2);
					n3 = __ldg(localBVHTreeNodes + child_node_index * 5 + 3);
					n4 = __ldg(localBVHTreeNodes + child_node_index * 5 + 4);

					float3 p = make_float3(n0);
					int3 e;
					e.x = *((char*)&n0.w + 0);
					e.y = *((char*)&n0.w + 1);
					e.z = *((char*)&n0.w + 2);

					nodeGroup.x = float_as_uint(n1.x);
					triangleGroup.x = float_as_uint(n1.y);
					triangleGroup.y = 0;
					unsigned int hitmask = 0;

					const float adjusted_idirx = uint_as_float((e.x + 127) << 23) * idirx;
					const float adjusted_idiry = uint_as_float((e.y + 127) << 23) * idiry;
					const float adjusted_idirz = uint_as_float((e.z + 127) << 23) * idirz;
					const float origx = -(orig.x - p.x) * idirx;
					const float origy = -(orig.y - p.y) * idiry;
					const float origz = -(orig.z - p.z) * idirz;

					{
						// First 4
						const unsigned int meta4 = float_as_uint(n1.z);
						const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
						const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
						const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
						const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

						uint swizzledLox = (idirx < 0) ? float_as_uint(n3.z) : float_as_uint(n2.x);
						uint swizzledHix = (idirx < 0) ? float_as_uint(n2.x) : float_as_uint(n3.z);

						uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.x) : float_as_uint(n2.z);
						uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.z) : float_as_uint(n4.x);

						uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.z) : float_as_uint(n3.x);
						uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.x) : float_as_uint(n4.z);

						float tminx[4];
						float tminy[4];
						float tminz[4];
						float tmaxx[4];
						float tmaxy[4];
						float tmaxz[4];

						tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
						tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
						tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
						tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

						tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
						tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
						tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
						tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

						tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
						tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
						tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
						tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

						tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
						tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
						tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
						tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

						tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
						tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
						tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
						tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

						tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
						tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
						tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
						tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

						for (int childIndex = 0; childIndex < 4; childIndex++)
						{
							const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
							const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

							bool intersected = cmin <= cmax;

							if (intersected)
							{
								const unsigned int child_bits = extract_byte(child_bits4, childIndex);
								const unsigned int bit_index = extract_byte(bit_index4, childIndex);
								hitmask |= child_bits << bit_index;
							}
						}
					}

					{
						// Second 4
						const unsigned int meta4 = float_as_uint(n1.w);
						const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
						const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
						const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
						const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

						uint swizzledLox = (idirx < 0) ? float_as_uint(n3.w) : float_as_uint(n2.y);
						uint swizzledHix = (idirx < 0) ? float_as_uint(n2.y) : float_as_uint(n3.w);

						uint swizzledLoy = (idiry < 0) ? float_as_uint(n4.y) : float_as_uint(n2.w);
						uint swizzledHiy = (idiry < 0) ? float_as_uint(n2.w) : float_as_uint(n4.y);

						uint swizzledLoz = (idirz < 0) ? float_as_uint(n4.w) : float_as_uint(n3.y);
						uint swizzledHiz = (idirz < 0) ? float_as_uint(n3.y) : float_as_uint(n4.w);

						float tminx[4];
						float tminy[4];
						float tminz[4];
						float tmaxx[4];
						float tmaxy[4];
						float tmaxz[4];

						tminx[0] = ((swizzledLox >>  0) & 0xFF) * adjusted_idirx + origx;
						tminx[1] = ((swizzledLox >>  8) & 0xFF) * adjusted_idirx + origx;
						tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
						tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

						tminy[0] = ((swizzledLoy >>  0) & 0xFF) * adjusted_idiry + origy;
						tminy[1] = ((swizzledLoy >>  8) & 0xFF) * adjusted_idiry + origy;
						tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
						tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

						tminz[0] = ((swizzledLoz >>  0) & 0xFF) * adjusted_idirz + origz;
						tminz[1] = ((swizzledLoz >>  8) & 0xFF) * adjusted_idirz + origz;
						tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
						tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

						tmaxx[0] = ((swizzledHix >>  0) & 0xFF) * adjusted_idirx + origx;
						tmaxx[1] = ((swizzledHix >>  8) & 0xFF) * adjusted_idirx + origx;
						tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
						tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

						tmaxy[0] = ((swizzledHiy >>  0) & 0xFF) * adjusted_idiry + origy;
						tmaxy[1] = ((swizzledHiy >>  8) & 0xFF) * adjusted_idiry + origy;
						tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
						tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

						tmaxz[0] = ((swizzledHiz >>  0) & 0xFF) * adjusted_idirz + origz;
						tmaxz[1] = ((swizzledHiz >>  8) & 0xFF) * adjusted_idirz + origz;
						tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
						tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

						for (int childIndex = 0; childIndex < 4; childIndex++)
						{
							const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
							const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

							bool intersected = cmin <= cmax;

							if (intersected)
							{
								const unsigned int child_bits = extract_byte(child_bits4, childIndex);
								const unsigned int bit_index = extract_byte(bit_index4, childIndex);
								hitmask |= child_bits << bit_index;
							}
						}
					}

					nodeGroup.y = (hitmask & 0xFF000000) | (*((byte*)&n0.w + 3));
					triangleGroup.y = hitmask & 0x00FFFFFF;
				}
			}
			else
			{
				triangleGroup = nodeGroup;
				nodeGroup = make_uint2(0);
			}
			
		#if TRIANGLE_POSTPONING
			const int totalThreads = __popc(__activemask());
		#endif

			while (triangleGroup.y != 0)
			{
			#if TRIANGLE_POSTPONING
				const float Rt = 0.2;
				const int threshold = totalThreads * Rt;
				const int numActiveThreads = __popc(__activemask());
				if (numActiveThreads < threshold)
				{
					STACK_PUSH(triangleGroup);
					break;
				}
			#endif
					
				int triangleIndex = __bfind(triangleGroup.y);

				int triAddr = triangleGroup.x * 3 + triangleIndex * 3;

				float4 v00 = __ldg(localTriangleWoopCoordinates + triAddr + 0);
				float4 v11 = __ldg(localTriangleWoopCoordinates + triAddr + 1);
				float4 v22 = __ldg(localTriangleWoopCoordinates + triAddr + 2);

				float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
				float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
				float t = Oz * invDz;

				float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
				float Dx = dir.x * v11.x + dir.y * v11.y + dir.z * v11.z;
				float u = Ox + t * Dx;

				float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
				float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
				float v = Oy + t*Dy;

				if (t > tmin && t < tmax)
				{
					if (u >= 0.0f && u <= 1.0f)
					{
						if (v >= 0.0f && u + v <= 1.0f)
						{
							triangleuv.x = u;
							triangleuv.y = v;

							tmax = t;
							hitAddr = triAddr;
						}
					}
				}

				triangleGroup.y &= ~(1 << triangleIndex);
			}

			if (nodeGroup.y <= 0x00FFFFFF)
			{
				if (stackPtr > 0)
				{
					STACK_POP(nodeGroup);
				}
				else
				{
					rayResultBuffer[rayidx].t_triId_u_v = make_float4(tmax, int_as_float(hitAddr), triangleuv.x, triangleuv.y);
					break;
				}
			}
		
		#if DYNAMIC_FETCH
			const int Nd = 4;
			const int Nw = 16;
			lostLoopIterations += __popc(__activemask()) - Nd;
			if (lostLoopIterations >= Nw)
				break;
		#endif
		} while (true);

	} while (true);
}

__host__ void rtBindCWBVHData(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex)
{
	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));
}

__host__ void rtTraceCWBVH(
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

	dim3 blockDim(32, 2);
	dim3 gridDim(32, 32);

	cudaProfilerStart();
	cudaCheck(cudaEventRecord(startEvent, 0));

	{
		cudaMemset(cudaFinishedRayCount, 0, sizeof(int));
		rtTraceCWBVHDynamicFetch <<< gridDim, blockDim >>> (
			rayBuffer,
			rayResultBuffer,
			rayCount,
			cudaFinishedRayCount
			);
	}

	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	Log("%.3fMS, %.2fMRays/s (rtTraceCWBVH Dynamic Fetch)", elapsedTime, (float)rayCount / 1000000.0f / (elapsedTime / 1000.0f));

	cudaProfilerStop();

	cudaFree(cudaFinishedRayCount);
}
