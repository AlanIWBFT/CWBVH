#pragma once

#include <vector>
#include <functional>
#include "helper_math.h"

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

BVH2Node* BuildBVH2AABB(int NumPrimitives, std::function<void(int PrimitiveIndex, float3& OutLower, float3& OutUpper)> GetPrimitiveBoundsFunc);
BVH8Node* BuildBVH8AABB(int NumPrimitives, std::function<void(int PrimitiveIndex, float3& OutLower, float3& OutUpper)> GetPrimitiveBoundsFunc);

BVH2Node* BuildBVH2Triangle(std::vector<int3>& IndexBuffer, std::vector<float3> VertexBuffer);
BVH8Node* BuildBVH8Triangle(std::vector<int3>& IndexBuffer, std::vector<float3> VertexBuffer);

struct MeshDescription
{
	std::vector<int3>* IndexBuffer;
	std::vector<float3>* VertexBuffer;
};
