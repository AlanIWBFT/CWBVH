#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "helper_math.h"
#include "rtDebugFunc.h"
#include "rtHelperFunc.h"
#include "rtTrace.h"
#include "sh_lightsample.h"
#include "sh_warpreduction.h"

texture<float4, 1, cudaReadModeElementType> CWBVHNodeDataTexture;
texture<float4, 1, cudaReadModeElementType> WoopifiedTriangleTexture;
texture<int, 1, cudaReadModeElementType> TriangleIndexTexture;

__device__ float4* CWBVHTreeNodes;
__device__ float4* WoopifiedTriangles;
__device__ int* TriangleIndices;

texture<float4, 1, cudaReadModeElementType> BVHTreeNodesTexture;
texture<float4, 1, cudaReadModeElementType> TriangleWoopCoordinatesTexture;
texture<int, 1, cudaReadModeElementType> MappingFromTriangleAddressToIndexTexture;

__device__ float4* BVHTreeNodes;
__device__ float4* TriangleWoopCoordinates;
__device__ int* MappingFromTriangleAddressToIndex;

#define DYNAMIC_FETCH 0
#define TRIANGLE_POSTPONING 1
#define MEMORY_STATISTICS 0

__device__ unsigned long long NodesFetched;
__device__ unsigned long long TrianglesFetched;

__global__ void rtTraceCWBVHDynamicFetch(
	Ray* rayBuffer,
	RayResult* rayResultBuffer,
	int rayCount,
	int* finishedRayCount
)
{
	const int STACK_SIZE = 32;
	const int SM_STACK_SIZE = 12;
	const float ooeps = exp2f(-80.0f);
	uint2 traversalStack[STACK_SIZE - SM_STACK_SIZE];
	__shared__ uint2 traversalStackSM[32][2][SM_STACK_SIZE];
	char stackPtr = 0;

	__shared__ int nextRayArray[2];

	uint2 nodeGroup = make_uint2(0);
	uint2 triangleGroup = make_uint2(0);

	// 	float2  triangleuv;
	// 	float3	trianglenormal;

	float3 orig;
	float3 dir;
	float tmin;
	float tmax;
	float idirx;
	float idiry;
	float idirz;
	int oct;
	int rayidx;
	int hitAddr;

#if MEMORY_STATISTICS
	unsigned long long nodesFetchedThisThread = 0;
	unsigned long long trianglesFetchedThisThread = 0;
#endif

	do // Dynamic fetch loop
	{
		int& rayBase = nextRayArray[threadIdx.y];

		// Fetch new rays from the global pool using lane 0.

		bool          terminated = stackPtr == 0 && (nodeGroup.y & 0xFF000000) == 0;
		const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(finishedRayCount, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= rayCount)
				break;

			orig = make_float3(rayBuffer[rayidx].RayOriginAndNearClip);
			dir = make_float3(rayBuffer[rayidx].RayDirectionAndFarClip);
			tmin = rayBuffer[rayidx].RayOriginAndNearClip.w;
			tmax = rayBuffer[rayidx].RayDirectionAndFarClip.w;
			idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x)); // inverse ray direction
			idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y)); // inverse ray direction
			idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z)); // inverse ray direction
			oct = (dir.x < 0) | ((dir.y < 0) << 1) | ((dir.z < 0) << 2);

			nodeGroup = make_uint2(0, 0b10000000000000000000000000000000 | (1 << ((31 - 24) ^ (7 - oct)))); // initial node group contains the root
			stackPtr = 0;

			hitAddr = -1;
		}

		// Traversal loop

	#if DYNAMIC_FETCH
		int lostLoopIterations = 0;
	#endif

		do
		{
			uint2 triangleGroup = make_uint2(0);

			if ((nodeGroup.y & 0xFF000000) != 0)
			{
				// G represents a node group
				// n <- GetClosestNode(G, r)
				const unsigned int hits = nodeGroup.y;
				const unsigned int imask = nodeGroup.y;
				const unsigned int bit_index = 31 - __clz(hits);
				const unsigned int child_node_base_index = nodeGroup.x;

				// Clear corresponding bit in hits field
				// G <- G / n
				nodeGroup.y &= ~(1 << bit_index);

				if ((nodeGroup.y & 0xFF000000) > 0)
				{
					if (stackPtr < SM_STACK_SIZE)
						traversalStackSM[threadIdx.x][threadIdx.y][stackPtr] = nodeGroup;
					else
						traversalStack[stackPtr - SM_STACK_SIZE] = nodeGroup;

					stackPtr++;
				}

				// Intersect with n
				// G, Gt <- IntersectChildren(n, r)
				{
				#if MEMORY_STATISTICS
					nodesFetchedThisThread++;
				#endif

					const unsigned int slot_index = (bit_index - 24) ^ (7 - oct);
					const unsigned int relative_index = __popc(imask & ~(-1 << slot_index));
					const unsigned int child_node_index = child_node_base_index + relative_index;

					// Load node n
					float4 n0, n1, n2, n3, n4;

					n0 = tex1Dfetch(CWBVHNodeDataTexture, child_node_index * 5 + 0);
					n1 = tex1Dfetch(CWBVHNodeDataTexture, child_node_index * 5 + 1);
					n2 = tex1Dfetch(CWBVHNodeDataTexture, child_node_index * 5 + 2);
					n3 = tex1Dfetch(CWBVHNodeDataTexture, child_node_index * 5 + 3);
					n4 = tex1Dfetch(CWBVHNodeDataTexture, child_node_index * 5 + 4);

					//n0 = CWBVHTreeNodes[child_node_index * 5 + 0];
					//n1 = CWBVHTreeNodes[child_node_index * 5 + 1];
					//n2 = CWBVHTreeNodes[child_node_index * 5 + 2];
					//n3 = CWBVHTreeNodes[child_node_index * 5 + 3];
					//n4 = CWBVHTreeNodes[child_node_index * 5 + 4];

					float3 p = make_float3(n0);
					int3 e;
					e.x = *((char*)&n0.w + 0);
					e.y = *((char*)&n0.w + 1);
					e.z = *((char*)&n0.w + 2);

					nodeGroup.x = float_as_uint(n1.x);
					triangleGroup.x = float_as_uint(n1.y);
					triangleGroup.y = 0;

					const unsigned int octinv4 = (7 - oct) * 0x01010101;

					const float adjusted_idirx = uint_as_float((e.x + 127) << 23) * idirx;
					const float adjusted_idiry = uint_as_float((e.y + 127) << 23) * idiry;
					const float adjusted_idirz = uint_as_float((e.z + 127) << 23) * idirz;
					const float origx = (p.x - orig.x) * idirx;
					const float origy = (p.y - orig.y) * idiry;
					const float origz = (p.z - orig.z) * idirz;

					float qlox[8];
					float qloy[8];
					float qloz[8];
					float qhix[8];
					float qhiy[8];
					float qhiz[8];

					qlox[0] = ((float_as_uint(n2.x) >>  0) & 0xFF) * adjusted_idirx + origx;
					qlox[1] = ((float_as_uint(n2.x) >>  8) & 0xFF) * adjusted_idirx + origx;
					qlox[2] = ((float_as_uint(n2.x) >> 16) & 0xFF) * adjusted_idirx + origx;
					qlox[3] = ((float_as_uint(n2.x) >> 24) & 0xFF) * adjusted_idirx + origx;
					qlox[4] = ((float_as_uint(n2.y) >>  0) & 0xFF) * adjusted_idirx + origx;
					qlox[5] = ((float_as_uint(n2.y) >>  8) & 0xFF) * adjusted_idirx + origx;
					qlox[6] = ((float_as_uint(n2.y) >> 16) & 0xFF) * adjusted_idirx + origx;
					qlox[7] = ((float_as_uint(n2.y) >> 24) & 0xFF) * adjusted_idirx + origx;

					qloy[0] = ((float_as_uint(n2.z) >>  0) & 0xFF) * adjusted_idiry + origy;
					qloy[1] = ((float_as_uint(n2.z) >>  8) & 0xFF) * adjusted_idiry + origy;
					qloy[2] = ((float_as_uint(n2.z) >> 16) & 0xFF) * adjusted_idiry + origy;
					qloy[3] = ((float_as_uint(n2.z) >> 24) & 0xFF) * adjusted_idiry + origy;
					qloy[4] = ((float_as_uint(n2.w) >>  0) & 0xFF) * adjusted_idiry + origy;
					qloy[5] = ((float_as_uint(n2.w) >>  8) & 0xFF) * adjusted_idiry + origy;
					qloy[6] = ((float_as_uint(n2.w) >> 16) & 0xFF) * adjusted_idiry + origy;
					qloy[7] = ((float_as_uint(n2.w) >> 24) & 0xFF) * adjusted_idiry + origy;


					qloz[0] = ((float_as_uint(n3.x) >>  0) & 0xFF) * adjusted_idirz + origz;
					qloz[1] = ((float_as_uint(n3.x) >>  8) & 0xFF) * adjusted_idirz + origz;
					qloz[2] = ((float_as_uint(n3.x) >> 16) & 0xFF) * adjusted_idirz + origz;
					qloz[3] = ((float_as_uint(n3.x) >> 24) & 0xFF) * adjusted_idirz + origz;
					qloz[4] = ((float_as_uint(n3.y) >>  0) & 0xFF) * adjusted_idirz + origz;
					qloz[5] = ((float_as_uint(n3.y) >>  8) & 0xFF) * adjusted_idirz + origz;
					qloz[6] = ((float_as_uint(n3.y) >> 16) & 0xFF) * adjusted_idirz + origz;
					qloz[7] = ((float_as_uint(n3.y) >> 24) & 0xFF) * adjusted_idirz + origz;

					qhix[0] = ((float_as_uint(n3.z) >>  0) & 0xFF) * adjusted_idirx + origx;
					qhix[1] = ((float_as_uint(n3.z) >>  8) & 0xFF) * adjusted_idirx + origx;
					qhix[2] = ((float_as_uint(n3.z) >> 16) & 0xFF) * adjusted_idirx + origx;
					qhix[3] = ((float_as_uint(n3.z) >> 24) & 0xFF) * adjusted_idirx + origx;
					qhix[4] = ((float_as_uint(n3.w) >>  0) & 0xFF) * adjusted_idirx + origx;
					qhix[5] = ((float_as_uint(n3.w) >>  8) & 0xFF) * adjusted_idirx + origx;
					qhix[6] = ((float_as_uint(n3.w) >> 16) & 0xFF) * adjusted_idirx + origx;
					qhix[7] = ((float_as_uint(n3.w) >> 24) & 0xFF) * adjusted_idirx + origx;

					qhiy[0] = ((float_as_uint(n4.x) >>  0) & 0xFF) * adjusted_idiry + origy;
					qhiy[1] = ((float_as_uint(n4.x) >>  8) & 0xFF) * adjusted_idiry + origy;
					qhiy[2] = ((float_as_uint(n4.x) >> 16) & 0xFF) * adjusted_idiry + origy;
					qhiy[3] = ((float_as_uint(n4.x) >> 24) & 0xFF) * adjusted_idiry + origy;
					qhiy[4] = ((float_as_uint(n4.y) >>  0) & 0xFF) * adjusted_idiry + origy;
					qhiy[5] = ((float_as_uint(n4.y) >>  8) & 0xFF) * adjusted_idiry + origy;
					qhiy[6] = ((float_as_uint(n4.y) >> 16) & 0xFF) * adjusted_idiry + origy;
					qhiy[7] = ((float_as_uint(n4.y) >> 24) & 0xFF) * adjusted_idiry + origy;

					qhiz[0] = ((float_as_uint(n4.z) >>  0) & 0xFF) * adjusted_idirz + origz;
					qhiz[1] = ((float_as_uint(n4.z) >>  8) & 0xFF) * adjusted_idirz + origz;
					qhiz[2] = ((float_as_uint(n4.z) >> 16) & 0xFF) * adjusted_idirz + origz;
					qhiz[3] = ((float_as_uint(n4.z) >> 24) & 0xFF) * adjusted_idirz + origz;
					qhiz[4] = ((float_as_uint(n4.w) >>  0) & 0xFF) * adjusted_idirz + origz;
					qhiz[5] = ((float_as_uint(n4.w) >>  8) & 0xFF) * adjusted_idirz + origz;
					qhiz[6] = ((float_as_uint(n4.w) >> 16) & 0xFF) * adjusted_idirz + origz;
					qhiz[7] = ((float_as_uint(n4.w) >> 24) & 0xFF) * adjusted_idirz + origz;

					unsigned int hitmask = 0;

				#pragma unroll 8
					for (int childIndex = 0; childIndex < 8; childIndex++)
					{
						const float cmin = spanBeginKepler(qlox[childIndex], qhix[childIndex], qloy[childIndex], qhiy[childIndex], qloz[childIndex], qhiz[childIndex], tmin);
						const float cmax = spanEndKepler  (qlox[childIndex], qhix[childIndex], qloy[childIndex], qhiy[childIndex], qloz[childIndex], qhiz[childIndex], tmax);

						bool intersected = cmin <= cmax;

						const unsigned int meta4 = childIndex < 4 ? float_as_uint(n1.z) : float_as_uint(n1.w);
						const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
						const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
						const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
						const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

						if (intersected)
						{
							const unsigned int child_bits = extract_byte(child_bits4, childIndex % 4);
							const unsigned int bit_index = extract_byte(bit_index4, childIndex % 4);
							hitmask |= child_bits << bit_index;
						}
					}


					nodeGroup.y = (hitmask & 0xFF000000) | (*((byte*)&n0.w + 3));
					triangleGroup.y = hitmask & 0x00FFFFFF;
				}
			}
			else
			{
				// G represents a triangle group, move it to Gt
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
					if (stackPtr < SM_STACK_SIZE)
						traversalStackSM[threadIdx.x][threadIdx.y][stackPtr] = triangleGroup;
					else
						traversalStack[stackPtr - SM_STACK_SIZE] = triangleGroup;
					stackPtr++;
					break;
				}
			#endif

			#if MEMORY_STATISTICS
				trianglesFetchedThisThread++;
			#endif

				int triangleIndex = __ffs(triangleGroup.y) - 1;

				triangleGroup.y &= ~(1 << triangleIndex);

				int triAddr = (triangleGroup.x + triangleIndex) * 3;

				float4 v00 = tex1Dfetch(WoopifiedTriangleTexture, triAddr);
				float4 v11 = tex1Dfetch(WoopifiedTriangleTexture, triAddr + 1);
				float4 v22 = tex1Dfetch(WoopifiedTriangleTexture, triAddr + 2);

				float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
				float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
				float t = Oz * invDz;

				if (t > tmin && t < tmax)
				{
					float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
					float Dx = dir.x * v11.x + dir.y * v11.y + dir.z * v11.z;
					float u = Ox + t * Dx;

					if (u >= 0.0f && u <= 1.0f)
					{
						float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
						float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
						float v = Oy + t*Dy;

						if (v >= 0.0f && u + v <= 1.0f)
						{
							//trianglenormal = cross(make_float3(v22.x, v22.y, v22.z), make_float3(v11.x, v11.y, v11.z));
							//triangleuv.x = u;
							//triangleuv.y = v;

							tmax = t;
							hitAddr = triAddr;
						}
					}
				}
			}

			if ((nodeGroup.y & 0xFF000000) == 0)
			{
				if (stackPtr == 0)
				{
					terminated = true;
					break;
				}

				--stackPtr;
				if (stackPtr < SM_STACK_SIZE)
					nodeGroup = traversalStackSM[threadIdx.x][threadIdx.y][stackPtr];
				else
					nodeGroup = traversalStack[stackPtr - SM_STACK_SIZE];
			}

		#if DYNAMIC_FETCH
			const int Nd = 4;
			const int Nw = 16;
			lostLoopIterations += __popc(__activemask()) - Nd;
			if (lostLoopIterations >= Nw)
				break;
		#endif
		} while (true);

		if (terminated)
		{
			//rayResultBuffer[rayidx].TriangleUV = triangleuv;
			rayResultBuffer[rayidx].HitTriangleIndex = hitAddr;
			//rayResultBuffer[rayidx].TriangleNormalUnnormalized = trianglenormal;
		}
	} while (true);

#if MEMORY_STATISTICS
	nodesFetchedThisThread = blockReduceSumToThread0(nodesFetchedThisThread);
	trianglesFetchedThisThread = blockReduceSumToThread0(trianglesFetchedThisThread);
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(&NodesFetched, nodesFetchedThisThread);
		atomicAdd(&TrianglesFetched, trianglesFetchedThisThread);
	}
#endif
}

__global__ void rtTraceBVH2DynamicFetch(
	Ray* rayBuffer,
	RayResult* rayResultBuffer,
	int rayCount,
	int* finishedRayCount
)
{
	const int EntrypointSentinel = 0x76543210;
	const int STACK_SIZE = 32;

	int traversalStack[STACK_SIZE];

	int     rayidx;
	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / ray direction
	float   oodx, oody, oodz;       // ray origin / ray direction
// 	float2  triangleuv;
// 	float3	trianglenormal;

	int*    stackPtr;               // Current position in traversal stack.
	int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
	int     nodeAddr = EntrypointSentinel;
	int     hitAddr;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.

	__shared__ volatile int nextRayArray[2];

#if MEMORY_STATISTICS
	unsigned long long nodesFetchedThisThread = 0;
	unsigned long long trianglesFetchedThisThread = 0;
#endif

	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		// Fetch new rays from the global pool using lane 0.

		const bool          terminated = nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(finishedRayCount, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= rayCount)
				break;

			float4 RayOrigin = rayBuffer[rayidx].RayOriginAndNearClip;
			float4 RayDirection = rayBuffer[rayidx].RayDirectionAndFarClip;
			origx = RayOrigin.x;
			origy = RayOrigin.y;
			origz = RayOrigin.z;
			dirx = RayDirection.x;
			diry = RayDirection.y;
			dirz = RayDirection.z;
			tmin = RayOrigin.w;

			// ooeps is very small number, used instead of raydir xyz component when that component is near zero
			float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
			idirx = 1.0f / (fabsf(RayDirection.x) > ooeps ? RayDirection.x : copysignf(ooeps, RayDirection.x)); // inverse ray direction
			idiry = 1.0f / (fabsf(RayDirection.y) > ooeps ? RayDirection.y : copysignf(ooeps, RayDirection.y)); // inverse ray direction
			idirz = 1.0f / (fabsf(RayDirection.z) > ooeps ? RayDirection.z : copysignf(ooeps, RayDirection.z)); // inverse ray direction
			oodx = origx * idirx;  // ray origin / ray direction
			oody = origy * idiry;  // ray origin / ray direction
			oodz = origz * idirz;  // ray origin / ray direction

								   // Setup traversal + initialisation

			traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
			stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
			leafAddr = 0;   // No postponed leaf.
			nodeAddr = 0;   // Start from the root.
			hitAddr = -1;  // No triangle intersected so far.
			hitT = RayDirection.w; // tmax  
		}

		// Traversal loop.

		while (nodeAddr != EntrypointSentinel)
		{
			leafAddr = 0;

			// Traverse internal nodes until all SIMD lanes have found a leaf.

			while (nodeAddr != EntrypointSentinel && nodeAddr >= 0)   // functionally equivalent, but faster
			{
				// Fetch AABBs of the two child nodes.

				//const float4 n0xy = BVHTreeNodes[nodeAddr + 0]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				//const float4 n1xy = BVHTreeNodes[nodeAddr + 1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				//const float4 nz = BVHTreeNodes[nodeAddr + 2]; // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				//float4 tmp = BVHTreeNodes[nodeAddr + 3]; // child_index0, child_index1
				//int2  cnodes = *(int2*)&tmp;

			#if MEMORY_STATISTICS
				nodesFetchedThisThread++;
			#endif

				const float4 n0xy = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const float4 n1xy = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const float4 nz = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				float4 tmp = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 3); // child_index0, child_index1
				int2  cnodes = *(int2*)&tmp;

				// Intersect the ray against the child nodes.

				const float c0lox = n0xy.x * idirx - oodx;
				const float c0hix = n0xy.y * idirx - oodx;
				const float c0loy = n0xy.z * idiry - oody;
				const float c0hiy = n0xy.w * idiry - oody;
				const float c0loz = nz.x   * idirz - oodz;
				const float c0hiz = nz.y   * idirz - oodz;
				const float c1loz = nz.z   * idirz - oodz;
				const float c1hiz = nz.w   * idirz - oodz;
				const float c0min = tMinFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
				const float c0max = tMaxFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
				const float c1lox = n1xy.x * idirx - oodx;
				const float c1hix = n1xy.y * idirx - oodx;
				const float c1loy = n1xy.z * idiry - oody;
				const float c1hiy = n1xy.w * idiry - oody;
				const float c1min = tMinFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
				const float c1max = tMaxFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

				bool swp = (c1min < c0min);

				bool traverseChild0 = (c0max >= c0min);
				bool traverseChild1 = (c1max >= c1min);

				// Neither child was intersected => pop stack.

				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *stackPtr;
					stackPtr--;
				}

				// Otherwise => fetch child pointers.

				else
				{
					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.

					if (traverseChild0 && traverseChild1)
					{
						if (swp)
							swap(nodeAddr, cnodes.y);
						stackPtr++;
						*stackPtr = cnodes.y;
					}
				}

				// First leaf => postpone and continue traversal.

				if (nodeAddr < 0 && leafAddr >= 0)     // Postpone max 1
													   //              if (nodeAddr < 0 && leafAddr2 >= 0)     // Postpone max 2
				{
					//leafAddr2= leafAddr;          // postpone 2
					leafAddr = nodeAddr;
					nodeAddr = *stackPtr;
					stackPtr--;
				}

				// All SIMD lanes have found a leaf? => process them.

				if (!__any_sync(__activemask(), leafAddr >= 0))
					break;
			}

			// Process postponed leaf nodes.

			while (leafAddr < 0)
			{
				for (int triAddr = ~leafAddr;; triAddr += 3)
				{
				#if MEMORY_STATISTICS
					trianglesFetchedThisThread++;
				#endif

					float4 v00 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr);
					float4 v11 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr + 1);
					float4 v22 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr + 2);

					if (__float_as_int(v00.x) == 0x80000000)
						break;

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;

					if (t > tmin && t < hitT)
					{
						float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
						float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;
						float u = Ox + t * Dx;

						if (u >= 0.0f && u <= 1.0f)
						{
							float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
							float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
							float v = Oy + t*Dy;

							if (v >= 0.0f && u + v <= 1.0f)
							{
								//trianglenormal = cross(make_float3(v22.x, v22.y, v22.z), make_float3(v11.x, v11.y, v11.z));
								//triangleuv.x = u;
								//triangleuv.y = v;

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

		if (terminated)
		{
			//rayResultBuffer[rayidx].TriangleUV = triangleuv;
			rayResultBuffer[rayidx].HitTriangleIndex = hitAddr != -1 ? tex1Dfetch(MappingFromTriangleAddressToIndexTexture, hitAddr) : -1;
			//rayResultBuffer[rayidx].TriangleNormalUnnormalized = trianglenormal;
		}

	} while (true);

#if MEMORY_STATISTICS
	nodesFetchedThisThread = blockReduceSumToThread0(nodesFetchedThisThread);
	trianglesFetchedThisThread = blockReduceSumToThread0(trianglesFetchedThisThread);
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(&NodesFetched, nodesFetchedThisThread);
		atomicAdd(&TrianglesFetched, trianglesFetchedThisThread);
	}
#endif
}

__host__ void rtBindCWBVHData(
	const float4* InBVHTreeNodes,
	const float4* InWoopifiedTriangles,
	const int* InTriangleIndices,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize)
{
	printf("GPU CWBVH video memory size: %.2fMB, triangle payload size: %.2fMB\n",
		BVHSize * sizeof(float4) / 1024.0f / 1024.0f,
		(TriangleWoopSize * sizeof(float4) + TriangleIndicesSize * sizeof(int)) / 1024.0f / 1024.0f
	);

	cudaCheck(cudaMemcpyToSymbol(TriangleIndices, &InTriangleIndices, 1 * sizeof(int*)));
	cudaCheck(cudaMemcpyToSymbol(WoopifiedTriangles, &InWoopifiedTriangles, 1 * sizeof(float4*)));
	cudaCheck(cudaMemcpyToSymbol(CWBVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(float4*)));

	cudaChannelFormatDesc channelDescFloat4 = cudaCreateChannelDesc<float4>();
	cudaCheck(cudaBindTexture(NULL, &CWBVHNodeDataTexture, InBVHTreeNodes, &channelDescFloat4, BVHSize * sizeof(float4)));
	cudaCheck(cudaBindTexture(NULL, &WoopifiedTriangleTexture, InWoopifiedTriangles, &channelDescFloat4, TriangleWoopSize * sizeof(float4)));

	cudaChannelFormatDesc channelDescInt = cudaCreateChannelDesc<int>();
	cudaCheck(cudaBindTexture(NULL, &TriangleIndexTexture, InTriangleIndices, &channelDescInt, TriangleIndicesSize * sizeof(int)));
}

__host__ void rtBindBVH2Data(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize)
{
	printf("GPU BVH2 video memory size: %.2fMB, triangle payload size: %.2fMB\n",
		BVHSize * sizeof(float4) / 1024.0f / 1024.0f,
		(TriangleWoopSize * sizeof(float4) + TriangleIndicesSize * sizeof(int)) / 1024.0f / 1024.0f
	);

	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));

	cudaChannelFormatDesc channelDescFloat4 = cudaCreateChannelDesc<float4>();
	cudaCheck(cudaBindTexture(NULL, &BVHTreeNodesTexture, InBVHTreeNodes, &channelDescFloat4, BVHSize * sizeof(float4)));
	cudaCheck(cudaBindTexture(NULL, &TriangleWoopCoordinatesTexture, InTriangleWoopCoordinates, &channelDescFloat4, TriangleWoopSize * sizeof(float4)));

	cudaChannelFormatDesc channelDescInt = cudaCreateChannelDesc<int>();
	cudaCheck(cudaBindTexture(NULL, &MappingFromTriangleAddressToIndexTexture, InMappingFromTriangleAddressToIndex, &channelDescInt, TriangleIndicesSize * sizeof(int)));
}

__host__ void rtTraceCWBVH(
	Ray* rayBuffer,
	RayResult* rayResultBuffer,
	int rayCount
)
{
	int* cudaFinishedRayCount;
	cudaCheck(cudaMalloc(&cudaFinishedRayCount, sizeof(int)));
	cudaMemset(cudaFinishedRayCount, 0, sizeof(int));

	dim3 blockDim(32, 2);
	dim3 gridDim(32 * 32, 1);

	cudaProfilerStart();

	rtTraceCWBVHDynamicFetch << < gridDim, blockDim >> > (
		rayBuffer,
		rayResultBuffer,
		rayCount,
		cudaFinishedRayCount
		);

	cudaPostKernelLaunchCheck

	cudaProfilerStop();

	cudaFree(cudaFinishedRayCount);
}

__host__ void rtTraceBVH2(
	Ray* rayBuffer,
	RayResult* rayResultBuffer,
	int rayCount
)
{
	int* cudaFinishedRayCount;
	cudaCheck(cudaMalloc(&cudaFinishedRayCount, sizeof(int)));
	cudaMemset(cudaFinishedRayCount, 0, sizeof(int));

	dim3 blockDim(32, 2);
	dim3 gridDim(32 * 32, 1);

	cudaProfilerStart();

	rtTraceBVH2DynamicFetch << < gridDim, blockDim >> > (
		rayBuffer,
		rayResultBuffer,
		rayCount,
		cudaFinishedRayCount
		);

	cudaProfilerStop();

	cudaPostKernelLaunchCheck

	cudaFree(cudaFinishedRayCount);
}

__host__ void rtTrace(
	Ray* rayBuffer,
	RayResult* rayResultBuffer,
	int rayCount
)
{
#if MEMORY_STATISTICS
	unsigned long long nodesFetched = 0;
	unsigned long long trianglesFetched = 0;

	cudaMemcpyToSymbol(NodesFetched, &nodesFetched, sizeof(nodesFetched));
	cudaMemcpyToSymbol(TrianglesFetched, &trianglesFetched, sizeof(nodesFetched));
#endif

	rtTraceCWBVH(rayBuffer, rayResultBuffer, rayCount);

#if MEMORY_STATISTICS
	cudaMemcpyFromSymbol(&nodesFetched, NodesFetched, sizeof(nodesFetched));
	cudaMemcpyFromSymbol(&trianglesFetched, TrianglesFetched, sizeof(trianglesFetched));

	printf("%llu, %llu ", nodesFetched, trianglesFetched);

	unsigned long long nodesFetched2 = 0;
	unsigned long long trianglesFetched2 = 0;

	cudaMemcpyToSymbol(NodesFetched, &nodesFetched2, sizeof(nodesFetched));
	cudaMemcpyToSymbol(TrianglesFetched, &trianglesFetched2, sizeof(nodesFetched));

	rtTraceBVH2(rayBuffer, rayResultBuffer, rayCount);

	cudaMemcpyFromSymbol(&nodesFetched2, NodesFetched, sizeof(nodesFetched));
	cudaMemcpyFromSymbol(&trianglesFetched2, TrianglesFetched, sizeof(trianglesFetched));

	printf("%llu, %llu ", nodesFetched2, trianglesFetched2);

	printf("%lf, %lf\n", (double)nodesFetched2 / nodesFetched, (double)trianglesFetched2 / trianglesFetched);
#endif

	//exit(0);
}
