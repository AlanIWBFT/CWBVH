#pragma once

#include <embree3/rtcore.h>
#include <embree3/rtcore_builder.h>
#include <vector>

template <int N>
class BVHNNode
{
public:
	float3 ChildrenMin[N];
	float3 ChildrenMax[N];
	BVHNNode* Children[N] = {};
	int ChildCount = 0;
	int* LeafPrimitiveRefs = nullptr;
	int LeafPrimitiveCount = 0;

	void Release()
	{
		for (int i = 0; i < N; i++)
			if (Children[i] != nullptr)
			{
				Children[i]->Release();
				delete Children[i];
				Children[i] = nullptr;
			}

		if (LeafPrimitiveCount > 0)
		{
			delete[] LeafPrimitiveRefs;
			LeafPrimitiveRefs = nullptr;
		}
	}
};

typedef BVHNNode<2> BVH2Node;
typedef BVHNNode<8> BVH8Node;

class EmbreeBVHBuilder
{
public:
	EmbreeBVHBuilder(
		const int InNumVertices,
		const int InNumTriangles,
		const float3 InVertexWorldPositionBuffer[],
		const int3 InTriangleIndexBuffer[]);

	~EmbreeBVHBuilder();

	BVH2Node* BuildBVH2();
	BVH8Node* BuildBVH8Direct();

	void ConvertToCUDABVH2(
		const BVH2Node * root,
		const int TriangleMaterialIndex[],
		std::vector<float4>& OutNodeData,
		std::vector<float4>& OutWoopifiedTriangles,
		std::vector<int>& OutTriangleIndices);

	void ConvertToCUDACompressedWideBVH(
		BVH8Node * root,
		const int TriangleMaterialIndex[],
		std::vector<float4>& OutNodeData,
		std::vector<float4>& OutWoopifiedTriangles,
		std::vector<int>& OutTriangleIndices);

private:
	RTCDevice EmbreeDevice;

	const int NumVertices;
	const int NumTriangles;
	const float3* VertexWorldPositionBuffer;
	const int3* TriangleIndexBuffer;
};
