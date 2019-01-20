#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "helper_math.h"
#include "CUDAAssert.h"

#include "ValidationKernels.h"
#include "EmbreeBVHBuilder.h"
#include "GPUBVHConverter.h"

#include "TraversalKernelBVH2.h"
#include "TraversalKernelCWBVH.h"

#include <optix_prime/optix_primepp.h>

int main()
{
	int numTriangles = 3000000;
	int numRays = 3000000;

	std::vector<float3> VertexBuffer;
	std::vector<int3> IndexBuffer;

	Ray* cudaRays = nullptr;

#if 1 // Use a mesh loaded from file
	{
		std::unique_ptr<std::FILE, decltype(&std::fclose)> meshFile(std::fopen("mesh.dat", "rb"), &std::fclose);
		int NumVertices, NumTriangles;

		fread(&NumVertices, sizeof(int), 1, meshFile.get());
		fread(&NumTriangles, sizeof(int), 1, meshFile.get());

		printf("Mesh: %d vertices, %d triangles\n", NumVertices, NumTriangles);

		std::unique_ptr<float3[]> VertexWorldPositionBuffer{ new float3[NumVertices]() };
		std::unique_ptr<int3[]> TriangleIndexBuffer{ new int3[NumTriangles]() };
		std::unique_ptr<int[]> TriangleMaterialIndex{ new int[NumTriangles]() };

		fread(VertexWorldPositionBuffer.get(), sizeof(float3), NumVertices, meshFile.get());
		fread(TriangleIndexBuffer.get(), sizeof(int3), NumTriangles, meshFile.get());
		fread(TriangleMaterialIndex.get(), sizeof(int), NumTriangles, meshFile.get());

		VertexBuffer = std::vector<float3>(VertexWorldPositionBuffer.get(), VertexWorldPositionBuffer.get() + NumVertices);
		IndexBuffer = std::vector<int3>(TriangleIndexBuffer.get(), TriangleIndexBuffer.get() + NumTriangles);
	}

	{
		std::unique_ptr<std::FILE, decltype(&std::fclose)> sampleFile(std::fopen("samplecache.dat", "rb"), &std::fclose);
		int SizeX, SizeY;

		fread(&SizeX, sizeof(SizeX), 1, sampleFile.get());
		fread(&SizeY, sizeof(SizeY), 1, sampleFile.get());

		std::unique_ptr<float4[]> WorldPositionTexture{ new float4[SizeX * SizeY]() };
		std::unique_ptr<float4[]> WorldNormalTexture{ new float4[SizeX * SizeY]() };
		std::unique_ptr<float[]> TexelRadiusTexture{ new float[SizeX * SizeY]() };

		fread(WorldPositionTexture.get(), sizeof(float4), SizeX * SizeY, sampleFile.get());
		fread(WorldNormalTexture.get(), sizeof(float4), SizeX * SizeY, sampleFile.get());
		fread(TexelRadiusTexture.get(), sizeof(float), SizeX * SizeY, sampleFile.get());

		numRays = SizeX * SizeY;
		
		const int numRaysPerTexel = 32;

		printf("Generating AO rays\n");

		cudaRays = GenerateAORaysUniform(numRays, numRaysPerTexel, WorldPositionTexture.get(), WorldNormalTexture.get(), TexelRadiusTexture.get());
		
		numRays *= numRaysPerTexel;

		Log("numRays: %d", numRays);
	}
#else
	printf("Generating validation triangles\n");

	GenerateValidationTriangles(numTriangles, VertexBuffer, IndexBuffer);

	printf("%d validation triangles generated\n", numTriangles);

	cudaRays = GenerateValidationRays(numRays);
#endif

	{
		printf("Building BVH2\n");

	#if 0 // BVH-ESC (Early Split Clipping)
		BVH2Node* RootBVHNode = BuildBVH2AABB((int)IndexBuffer.size(), [&](int PrimitiveIndex, float3& OutLower, float3& OutUpper)
		{
			OutLower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
			OutLower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
			OutLower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
			OutUpper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
			OutUpper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
			OutUpper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
		});
	#else // SBVH
		BVH2Node* RootBVHNode = BuildBVH2Triangle(IndexBuffer, VertexBuffer);
	#endif

		GPUBVHIntermediates BVHData;

		printf("Converting to GPU BVH2\n");

		ConvertToGPUBVH2(RootBVHNode,
			[&](int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)
		{
			float4 V0, V1, V2;
			WoopifyTriangle(
				VertexBuffer[IndexBuffer[PrimitiveIndex].x],
				VertexBuffer[IndexBuffer[PrimitiveIndex].y],
				VertexBuffer[IndexBuffer[PrimitiveIndex].z],
				V0, V1, V2
			);
			InlinedPrimitives.push_back(V0);
			InlinedPrimitives.push_back(V1);
			InlinedPrimitives.push_back(V2);
		},
			BVHData);

		RootBVHNode->Release();

		float4* cudaBVHNodeData;
		float4* cudaInlinedPrimitives;
		int* cudaPrimitiveIndices;
		cudaMalloc(&cudaBVHNodeData, BVHData.BVHNodeData.size() * sizeof(float4));
		cudaMalloc(&cudaInlinedPrimitives, BVHData.InlinedPrimitives.size() * sizeof(float4));
		cudaMalloc(&cudaPrimitiveIndices, BVHData.PrimitiveIndices.size() * sizeof(int));

		cudaMemcpy(cudaBVHNodeData, BVHData.BVHNodeData.data(), BVHData.BVHNodeData.size() * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaInlinedPrimitives, BVHData.InlinedPrimitives.data(), BVHData.InlinedPrimitives.size() * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaPrimitiveIndices, BVHData.PrimitiveIndices.data(), BVHData.PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

		rtBindBVH2Data(
			cudaBVHNodeData,
			cudaInlinedPrimitives,
			cudaPrimitiveIndices
		);
	}

	{
		printf("Building CWBVH\n");

	#if 0 // BVH-ESC (Early Split Clipping)
		BVH8Node* RootBVHNode = BuildBVH8AABB((int)IndexBuffer.size(), [&](int PrimitiveIndex, float3& OutLower, float3& OutUpper)
		{
			OutLower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
			OutLower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
			OutLower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
			OutUpper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
			OutUpper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
			OutUpper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
		});
	#else // SBVH
		BVH8Node* RootBVHNode = BuildBVH8Triangle(IndexBuffer, VertexBuffer);
	#endif

		GPUBVHIntermediates BVHData;

		printf("Converting to GPU CWBVH\n");

		ConvertToGPUCompressedWideBVH(RootBVHNode,
			[&](int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)
		{
			float4 V0, V1, V2;
			WoopifyTriangle(
				VertexBuffer[IndexBuffer[PrimitiveIndex].x],
				VertexBuffer[IndexBuffer[PrimitiveIndex].y],
				VertexBuffer[IndexBuffer[PrimitiveIndex].z],
				V0, V1, V2
			);
			InlinedPrimitives.push_back(V0);
			InlinedPrimitives.push_back(V1);
			InlinedPrimitives.push_back(V2);
		},
			BVHData);

		RootBVHNode->Release();

		float4* cudaBVHNodeData;
		float4* cudaInlinedPrimitives;
		int* cudaPrimitiveIndices;
		cudaMalloc(&cudaBVHNodeData, BVHData.BVHNodeData.size() * sizeof(float4));
		cudaMalloc(&cudaInlinedPrimitives, BVHData.InlinedPrimitives.size() * sizeof(float4));
		cudaMalloc(&cudaPrimitiveIndices, BVHData.PrimitiveIndices.size() * sizeof(int));

		cudaMemcpy(cudaBVHNodeData, BVHData.BVHNodeData.data(), BVHData.BVHNodeData.size() * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaInlinedPrimitives, BVHData.InlinedPrimitives.data(), BVHData.InlinedPrimitives.size() * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaPrimitiveIndices, BVHData.PrimitiveIndices.data(), BVHData.PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

		rtBindCWBVHData(
			cudaBVHNodeData,
			cudaInlinedPrimitives,
			cudaPrimitiveIndices
		);
	}

	Hit* cudaHits;
	cudaCheck(cudaMalloc(&cudaHits, sizeof(Hit) *  numRays));

	printf("Launching kernels\n");

	rtTraceBVH2(cudaRays, cudaHits, numRays);
	rtTraceCWBVH(cudaRays, cudaHits, numRays);

#if 1 // Optional OptiX Prime comparison
	{
		optix::prime::Context OptiXContext = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
		optix::prime::Model SceneModel = OptiXContext->createModel();
		SceneModel->setTriangles(IndexBuffer.size(), RTP_BUFFER_TYPE_HOST, IndexBuffer.data(), VertexBuffer.size(), RTP_BUFFER_TYPE_HOST, VertexBuffer.data());
		SceneModel->update(RTP_MODEL_HINT_NONE);
		SceneModel->finish();

		optix::prime::Query query = SceneModel->createQuery(RTP_QUERY_TYPE_CLOSEST);
		query->setRays(numRays, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, cudaRays);
		query->setHits(numRays, RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, cudaHits);

		cudaProfilerStart();
		{
			float elapsedTime;
			cudaEvent_t startEvent, stopEvent;
			cudaCheck(cudaEventCreate(&startEvent));
			cudaCheck(cudaEventCreate(&stopEvent));
			cudaCheck(cudaEventRecord(startEvent, 0));
			query->execute(0);
			cudaCheck(cudaEventRecord(stopEvent, 0));
			cudaCheck(cudaEventSynchronize(stopEvent));
			cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

			Log("%.3fMS, %.2fMRays/s (OptiXPrime)", elapsedTime, (float)numRays / 1000000.0f / (elapsedTime / 1000.0f));
		}
		cudaProfilerStop();
	}
#endif

	std::vector<Hit> hostHits(numRays);
	cudaCheck(cudaMemcpy(hostHits.data(), cudaHits, sizeof(Hit) * numRays, cudaMemcpyDeviceToHost));

	// Print out the first 10 results to validate by eye
	for (int rayIndex = 0; rayIndex < 10; rayIndex++)
		printf("%.2f %d\t", hostHits[rayIndex].t_triId_u_v.x, *(int*)&hostHits[rayIndex].t_triId_u_v.y);

	return 0;
}
