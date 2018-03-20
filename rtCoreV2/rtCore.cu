#include <cuda_runtime.h>
#include <memory>
#include <cstdio>
#include "helper_math.h"
#include "sh_lightsample.h"
#include "bvhbuilder.h"
#include "rtTrace.h"
#include "rtDebugFunc.h"

void BindCWBVHData(
	std::vector<float4>& NodeData,
	std::vector<float4>& WoopifiedTriangles,
	std::vector<int>& TriangleIndices)
{
	float4* cudaBVHNodes = NULL;
	float4* cudaTriangleWoopCoordinates = NULL;
	int*    cudaMappingFromTriangleAddressToIndex = NULL;

	cudaCheck(cudaMalloc((void**)&cudaBVHNodes, NodeData.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaBVHNodes, NodeData.data(), NodeData.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTriangleWoopCoordinates, WoopifiedTriangles.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaTriangleWoopCoordinates, WoopifiedTriangles.data(), WoopifiedTriangles.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaMappingFromTriangleAddressToIndex, TriangleIndices.size() * sizeof(int)));
	cudaCheck(cudaMemcpy(cudaMappingFromTriangleAddressToIndex, TriangleIndices.data(), TriangleIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

	rtBindCWBVHData(cudaBVHNodes, cudaTriangleWoopCoordinates, cudaMappingFromTriangleAddressToIndex, NodeData.size(), WoopifiedTriangles.size(), TriangleIndices.size());
}

void BindBVH2Data(
	std::vector<float4>& NodeData,
	std::vector<float4>& WoopifiedTriangles,
	std::vector<int>& TriangleIndices)
{
	float4* cudaBVHNodes = NULL;
	float4* cudaTriangleWoopCoordinates = NULL;
	int*    cudaMappingFromTriangleAddressToIndex = NULL;

	cudaCheck(cudaMalloc((void**)&cudaBVHNodes, NodeData.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaBVHNodes, NodeData.data(), NodeData.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTriangleWoopCoordinates, WoopifiedTriangles.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaTriangleWoopCoordinates, WoopifiedTriangles.data(), WoopifiedTriangles.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaMappingFromTriangleAddressToIndex, TriangleIndices.size() * sizeof(int)));
	cudaCheck(cudaMemcpy(cudaMappingFromTriangleAddressToIndex, TriangleIndices.data(), TriangleIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

	rtBindBVH2Data(cudaBVHNodes, cudaTriangleWoopCoordinates, cudaMappingFromTriangleAddressToIndex, NodeData.size(), WoopifiedTriangles.size(), TriangleIndices.size());
}

void LaunchFinalGather(
	float4* WorldPositionTexture,
	float4* WorldNormalTexture,
	float* TexelRadiusTexture,
	const int SizeX,
	const int SizeY,
	GatheredLightSample* OutLightmapData);

void WriteHDR(std::string fileName, const float4* buffer, int Width, int Height);

int main()
{
    {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> sampleFile(std::fopen("samplecache.dat", "rb"), &std::fclose);
        std::unique_ptr<std::FILE, decltype(&std::fclose)> meshFile(std::fopen("mesh.dat", "rb"), &std::fclose);
        if(sampleFile && meshFile)
        {
            printf("Building BVH\n");

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
        
            {
                EmbreeBVHBuilder builder(NumVertices, NumTriangles, VertexWorldPositionBuffer.get(), TriangleIndexBuffer.get());
            
                {
                    BVH8Node* root = builder.BuildBVH8Direct();
                    std::vector<float4> nodeData;
                    std::vector<float4> woopifiedTriangles;
                    std::vector<int> triangleIndices;
                    builder.ConvertToCUDACompressedWideBVH(root, TriangleMaterialIndex.get(), nodeData, woopifiedTriangles, triangleIndices);
                    BindCWBVHData(nodeData, woopifiedTriangles, triangleIndices);
                    root->Release();
                }
                
                {
                    EmbreeBVHBuilder builder(NumVertices, NumTriangles, VertexWorldPositionBuffer.get(), TriangleIndexBuffer.get());
                    BVH2Node* root = builder.BuildBVH2();
                    std::vector<float4> nodeData;
                    std::vector<float4> woopifiedTriangles;
                    std::vector<int> triangleIndices;
                    builder.ConvertToCUDABVH2(root, TriangleMaterialIndex.get(), nodeData, woopifiedTriangles, triangleIndices);
                    BindBVH2Data(nodeData, woopifiedTriangles, triangleIndices);
                    root->Release();
                } 
            }
            
            int SizeX, SizeY;

            fread(&SizeX, sizeof(SizeX), 1, sampleFile.get());
            fread(&SizeY, sizeof(SizeY), 1, sampleFile.get());

            std::unique_ptr<float4[]> WorldPositionTexture{ new float4[SizeX * SizeY]() };
            std::unique_ptr<float4[]> WorldNormalTexture{ new float4[SizeX * SizeY]() };
            std::unique_ptr<float[]> TexelRadiusTexture{ new float[SizeX * SizeY]() };

            fread(WorldPositionTexture.get(), sizeof(float4), SizeX * SizeY, sampleFile.get());
            fread(WorldNormalTexture.get(), sizeof(float4), SizeX * SizeY, sampleFile.get());
            fread(TexelRadiusTexture.get(), sizeof(float), SizeX * SizeY, sampleFile.get());
            
            std::unique_ptr<GatheredLightSample[]> OutLightmapData{ new GatheredLightSample[SizeX * SizeY]() };
            
            printf("Launching kernel\n");
            
            for(int i = 0; i < 10; i++)
                LaunchFinalGather(
                    WorldPositionTexture.get(),
                    WorldNormalTexture.get(),
                    TexelRadiusTexture.get(),
                    SizeX,
                    SizeY,
                    OutLightmapData.get()
                );

            std::unique_ptr<float4[]> OutColor{ new float4[SizeX * SizeY]() };

            for (int Y = 0; Y < SizeY; Y++)
                for (int X = 0; X < SizeX; X++)
                    OutColor[Y * SizeX + X] = make_float4(OutLightmapData[Y * SizeX + X].IncidentLighting / 3.1415926f);

            WriteHDR("output.hdr", OutColor.get(), SizeX, SizeY);
        }
        else
        {
            if(!sampleFile)
                printf("samplecache.dat not found\n");
            if(!meshFile)
                printf("mesh.dat not found\n");
            return 1;
        }
    }
    return 0;
}
 
