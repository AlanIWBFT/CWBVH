#undef NDEBUG
#include <cassert>
#include <random>
#include <float.h>
#include "helper_math.h"
#include "WoopTriangleHelper.h"
#include "GPUBVHConverter.h"
#include "Logger.h"

void ConvertToGPUBVH2(
	BVH2Node*& root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
)
{
	const BVH2Node* stackNodePtr[32];
	int stackNodeParentIdx[32];
	int stackNodeIdxInParent[32];
	int stackPtr = 0;

	// Handle degenrated case: root is a leaf
	if (root->LeafPrimitiveCount != 0)
	{
		BVH2Node* newRoot = new BVH2Node();
		newRoot->Children[0] = root;
		newRoot->Children[1] = new BVH2Node();
		newRoot->ChildrenMin[0] = make_float3(-FLT_MAX);
		newRoot->ChildrenMax[0] = make_float3(FLT_MAX);
		newRoot->ChildrenMin[1] = make_float3(FLT_MAX);
		newRoot->ChildrenMax[1] = make_float3(-FLT_MAX);
		newRoot->ChildCount = 2;

		newRoot->Children[1]->LeafPrimitiveCount = 1;
		newRoot->Children[1]->LeafPrimitiveRefs = new int[1]();
		newRoot->Children[1]->LeafPrimitiveRefs[0] = root->LeafPrimitiveRefs[0];

		root = newRoot;
	}

	stackNodePtr[stackPtr] = root;
	stackNodeParentIdx[stackPtr] = -1;
	stackNodeIdxInParent[stackPtr] = 0;
	stackPtr++;

	while (stackPtr > 0)
	{
		stackPtr--;
		const BVH2Node* node = stackNodePtr[stackPtr];
		const int parentNodeAddr = stackNodeParentIdx[stackPtr];
		const int nodeIdxInParent = stackNodeIdxInParent[stackPtr];

		if (node->ChildCount != 0)
		{
			// Internal node
			assert(node->Children[0] != nullptr && node->Children[1] != nullptr);

			int currentNodeAddr = (int)OutIntermediates.BVHNodeData.size();
			assert(currentNodeAddr % 4 == 0);
			if (parentNodeAddr >= 0)
			{
				if (nodeIdxInParent == 0)
					OutIntermediates.BVHNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (nodeIdxInParent == 1)
					OutIntermediates.BVHNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			OutIntermediates.BVHNodeData.push_back(make_float4(node->ChildrenMin[0].x, node->ChildrenMax[0].x, node->ChildrenMin[0].y, node->ChildrenMax[0].y));
			OutIntermediates.BVHNodeData.push_back(make_float4(node->ChildrenMin[1].x, node->ChildrenMax[1].x, node->ChildrenMin[1].y, node->ChildrenMax[1].y));
			OutIntermediates.BVHNodeData.push_back(make_float4(node->ChildrenMin[0].z, node->ChildrenMax[0].z, node->ChildrenMin[1].z, node->ChildrenMax[1].z));
			OutIntermediates.BVHNodeData.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f)); // children indices, will be filled in by the children themselves

			assert(currentNodeAddr < OutIntermediates.BVHNodeData.size());

			stackNodePtr[stackPtr] = node->Children[0];
			stackNodeParentIdx[stackPtr] = currentNodeAddr;
			stackNodeIdxInParent[stackPtr] = 0;
			stackPtr++;

			stackNodePtr[stackPtr] = node->Children[1];
			stackNodeParentIdx[stackPtr] = currentNodeAddr;
			stackNodeIdxInParent[stackPtr] = 1;
			stackPtr++;
		}
		else
		{
			// Leaf node, can be either top-level or bottom level
			assert(node->Children[0] == nullptr && node->Children[1] == nullptr);
			int currentNodeAddr = (int)OutIntermediates.InlinedPrimitives.size();
			currentNodeAddr = ~currentNodeAddr;

			assert(currentNodeAddr < 0);

			if (parentNodeAddr >= 0)
			{
				if (nodeIdxInParent == 0)
					OutIntermediates.BVHNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (nodeIdxInParent == 1)
					OutIntermediates.BVHNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			for (int i = 0; i < node->LeafPrimitiveCount; i++)
			{
				int primitiveIndex = node->LeafPrimitiveRefs[i];

				AppendPrimitiveFunc(primitiveIndex, OutIntermediates.InlinedPrimitives);

				OutIntermediates.PrimitiveIndices.push_back(primitiveIndex);
				OutIntermediates.PrimitiveIndices.push_back(-1);
				OutIntermediates.PrimitiveIndices.push_back(-1);
			}

			uint32_t leafTerminator = 0x80000000;
			OutIntermediates.InlinedPrimitives.push_back(make_float4(*(float*)(&leafTerminator), 0, 0, 0));
			OutIntermediates.PrimitiveIndices.push_back(-1);
		}
	}
}

void ConvertToGPUCompressedWideBVH(
	BVH8Node * root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
)
{
	BVH8Node* stackNodePtr[256];
	int stackNodeAddr[256];
	int stackPtr = 0;

	stackNodePtr[stackPtr] = root;
	stackNodeAddr[stackPtr] = 0;
	stackPtr++;

	// Potentially the same degenrated case as BVH2: root is a leaf
	// Handle it by yourself

	// Preallocate space for root
	OutIntermediates.BVHNodeData.push_back(make_float4(0.0f));
	OutIntermediates.BVHNodeData.push_back(make_float4(0.0f));
	OutIntermediates.BVHNodeData.push_back(make_float4(0.0f));
	OutIntermediates.BVHNodeData.push_back(make_float4(0.0f));
	OutIntermediates.BVHNodeData.push_back(make_float4(0.0f));

	while (stackPtr > 0)
	{
		assert(stackPtr < 256);
		stackPtr--;
		BVH8Node* node = stackNodePtr[stackPtr];

		assert(node->ChildCount == 0 || node->LeafPrimitiveCount == 0);

		const int currentNodeAddr = stackNodeAddr[stackPtr];

		// Calculate current node's bounding box (origin point p and Bhi, i.e. low and high respectively) on the fly
		float3 nodeLo = make_float3(FLT_MAX);
		float3 nodeHi = make_float3(-FLT_MAX);

		for (int childIndex = 0; childIndex < 8; childIndex++)
		{
			if (node->Children[childIndex] == nullptr)
				continue;

			nodeLo = fminf(nodeLo, node->ChildrenMin[childIndex]);
			nodeHi = fmaxf(nodeHi, node->ChildrenMax[childIndex]);
		}

		// Greedy child node ordering
		// Should be 99.8% effective as the auction method used by the paper
		{
			const float3 nodeCentroid = (nodeLo + nodeHi) * 0.5f;

			float cost[8][8];
			int assignment[8];
			bool isSlotEmpty[8];

			for (int s = 0; s < 8; s++)
				isSlotEmpty[s] = true;

			for (int childIndex = 0; childIndex < 8; childIndex++)
				assignment[childIndex] = -1;

			for (int s = 0; s < 8; s++)
			{
				float3 ds = make_float3(
					(((s >> 2) & 1) == 1) ? -1.0f : 1.0f,
					(((s >> 1) & 1) == 1) ? -1.0f : 1.0f,
					(((s >> 0) & 1) == 1) ? -1.0f : 1.0f);

				for (int childIndex = 0; childIndex < 8; childIndex++)
				{
					if (node->Children[childIndex] == nullptr)
					{
						cost[s][childIndex] = FLT_MAX;
					}
					else
					{
						float3 childCentroid = (node->ChildrenMin[childIndex] + node->ChildrenMax[childIndex]) * 0.5f;
						cost[s][childIndex] = dot(childCentroid - nodeCentroid, ds);
					}
				}
			}

			while (true)
			{
				float minCost = FLT_MAX;
				int2 minEntry = make_int2(-1);

				for (int s = 0; s < 8; s++)
				{
					for (int childIndex = 0; childIndex < 8; childIndex++)
					{
						if (assignment[childIndex] == -1 && isSlotEmpty[s] && cost[s][childIndex] < minCost)
						{
							minCost = cost[s][childIndex];
							minEntry = make_int2(s, childIndex);
						}
					}
				}

				if (minEntry.x != -1 || minEntry.y != -1)
				{
					assert(minEntry.x != -1 && minEntry.y != -1);
					isSlotEmpty[minEntry.x] = false;
					assignment[minEntry.y] = minEntry.x;
				}
				else
				{
					assert(minEntry.x == -1 && minEntry.y == -1);
					break;
				}
			}

			for (int childIndex = 0; childIndex < 8; childIndex++)
			{
				if (assignment[childIndex] == -1)
				{
					for (int s = 0; s < 8; s++)
					{
						if (isSlotEmpty[s])
						{
							isSlotEmpty[s] = false;
							assignment[childIndex] = s;
							break;
						}
					}
				}
			}

			BVH8Node oldNode = *node;
			for (int childIndex = 0; childIndex < 8; childIndex++)
			{
				assert(assignment[childIndex] != -1);
				   node->Children[assignment[childIndex]] =    oldNode.Children[childIndex];
				node->ChildrenMin[assignment[childIndex]] = oldNode.ChildrenMin[childIndex];
				node->ChildrenMax[assignment[childIndex]] = oldNode.ChildrenMax[childIndex];
			}
		}

	#if 0 // Optional random permutation to destroy the order, serving as comparison
		static std::default_random_engine generator;

		for (int i = 0; i < 7; i++)
		{
			std::uniform_int_distribution<int> distribution(i + 1, 7);
			int index = distribution(generator);
			auto tempChildren    = node->Children[index];
			auto tempChildrenMin = node->ChildrenMin[index];
			auto tempChildrenMax = node->ChildrenMax[index];
			node->Children[index]    = node->Children[i];
			node->ChildrenMin[index] = node->ChildrenMin[i];
			node->ChildrenMax[index] = node->ChildrenMax[i];
			node->Children[i]    = tempChildren;
			node->ChildrenMin[i] = tempChildrenMin;
			node->ChildrenMax[i] = tempChildrenMax;
		}
	#endif

		// Calculate quantization parameters for each axis respectively
		const float Nq = 8;
		char3 e; // implicit quantization
		e.x = (char)ceilf(log2f((nodeHi.x - nodeLo.x) / (powf(2, Nq) - 1)));
		e.y = (char)ceilf(log2f((nodeHi.y - nodeLo.y) / (powf(2, Nq) - 1)));
		e.z = (char)ceilf(log2f((nodeHi.z - nodeLo.z) / (powf(2, Nq) - 1)));

		// Encode output
		int internalChildCount = 0;
		int leafChildPrimitiveCount = 0;

		unsigned char imask = 0;
		int childBaseIndex = 0;
		int triangleBaseIndex = 0;

		for (int childIndex = 0; childIndex < 8; childIndex++)
		{
			if (node->Children[childIndex] == nullptr)
				continue;

			int3 qlo, qhi;

			qlo.x = (int)floorf((node->ChildrenMin[childIndex].x - nodeLo.x) / powf(2, e.x));
			qlo.y = (int)floorf((node->ChildrenMin[childIndex].y - nodeLo.y) / powf(2, e.y));
			qlo.z = (int)floorf((node->ChildrenMin[childIndex].z - nodeLo.z) / powf(2, e.z));
			qhi.x =  (int)ceilf((node->ChildrenMax[childIndex].x - nodeLo.x) / powf(2, e.x));
			qhi.y =  (int)ceilf((node->ChildrenMax[childIndex].y - nodeLo.y) / powf(2, e.y));
			qhi.z =  (int)ceilf((node->ChildrenMax[childIndex].z - nodeLo.z) / powf(2, e.z));

			unsigned char* const childBoundsBaseAddr = (unsigned char*)&OutIntermediates.BVHNodeData[currentNodeAddr + 2];
			childBoundsBaseAddr[childIndex + 0] = qlo.x;
			childBoundsBaseAddr[childIndex + 8] = qlo.y;
			childBoundsBaseAddr[childIndex + 16] = qlo.z;
			childBoundsBaseAddr[childIndex + 24] = qhi.x;
			childBoundsBaseAddr[childIndex + 32] = qhi.y;
			childBoundsBaseAddr[childIndex + 40] = qhi.z;

			if (node->Children[childIndex]->LeafPrimitiveCount == 0)
			{
				// Internal child, set params accordingly and push onto stack
				const int childNodeAddr = (int)OutIntermediates.BVHNodeData.size();
				if (internalChildCount == 0)
					childBaseIndex = childNodeAddr / 5;

				// Preallocate space for child
				OutIntermediates.BVHNodeData.push_back(make_float4(0.0f)); // p, e and imask
				OutIntermediates.BVHNodeData.push_back(make_float4(0.0f)); // child, triangle base index and meta field
				OutIntermediates.BVHNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box
				OutIntermediates.BVHNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box
				OutIntermediates.BVHNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box

				imask |= 1 << childIndex;
				// Set the meta field. This calculation assumes children are stored contiguously
				unsigned char* const childMetaField = ((unsigned char*)&OutIntermediates.BVHNodeData[currentNodeAddr + 1]) + 8;
				childMetaField[childIndex] = (1 << 5) | ((24 + childIndex) & 0b11111);
				internalChildCount++;

				stackNodePtr[stackPtr] = node->Children[childIndex];
				stackNodeAddr[stackPtr] = childNodeAddr;
				stackPtr++;
			}
			else
			{
				// Leaf child
				if (leafChildPrimitiveCount == 0)
					triangleBaseIndex = (int)OutIntermediates.InlinedPrimitives.size() / 3;

				imask |= 0 << childIndex; // Not actually doing anything, just for symmetry
				assert(node->Children[childIndex]->LeafPrimitiveCount <= 3 && node->Children[childIndex]->LeafPrimitiveCount > 0);
				int unaryEncodedPrimitiveCount = node->Children[childIndex]->LeafPrimitiveCount == 1 ? 0b001 : node->Children[childIndex]->LeafPrimitiveCount == 2 ? 0b011 : 0b111;
				// Set the meta field. This calculation assumes children are stored contiguously
				unsigned char* const childMetaField = ((unsigned char*)&OutIntermediates.BVHNodeData[currentNodeAddr + 1]) + 8;
				childMetaField[childIndex] = (unaryEncodedPrimitiveCount << 5) | (leafChildPrimitiveCount & 0b11111);
				leafChildPrimitiveCount += node->Children[childIndex]->LeafPrimitiveCount;

				for (int i = 0; i < node->Children[childIndex]->LeafPrimitiveCount; i++)
				{
					int primitiveIndex = node->Children[childIndex]->LeafPrimitiveRefs[i];
					AppendPrimitiveFunc(primitiveIndex, OutIntermediates.InlinedPrimitives);

					OutIntermediates.PrimitiveIndices.push_back(primitiveIndex);
					OutIntermediates.PrimitiveIndices.push_back(-1);
					OutIntermediates.PrimitiveIndices.push_back(-1);
				}
			}
		}
		
		unsigned char exyzAndimask[4];
		exyzAndimask[0] = *(unsigned char*)&e.x;
		exyzAndimask[1] = *(unsigned char*)&e.y;
		exyzAndimask[2] = *(unsigned char*)&e.z;
		exyzAndimask[3] = imask;
		OutIntermediates.BVHNodeData[currentNodeAddr + 0] = make_float4(nodeLo, *(float*)&exyzAndimask);
		OutIntermediates.BVHNodeData[currentNodeAddr + 1].x = *(float*)&childBaseIndex;
		OutIntermediates.BVHNodeData[currentNodeAddr + 1].y = *(float*)&triangleBaseIndex;
		// z and w are already set for meta field
	}
}

void WoopifyTriangle(
	float3 v0,
	float3 v1,
	float3 v2,
	float4& OutWoopifiedV0,
	float4& OutWoopifiedV1,
	float4& OutWoopifiedV2
)
{
	Mat4f mtx;
	float4 col0 = make_float4(v0 - v2, 0.0f);
	float4 col1 = make_float4(v1 - v2, 0.0f);
	float4 col2 = make_float4(cross(v0 - v2, v1 - v2), 0.0f);
	float4 col3 = make_float4(v2, 1.0f);
	mtx.setCol(0, Vec4f(col0));
	mtx.setCol(1, Vec4f(col1));
	mtx.setCol(2, Vec4f(col2));
	mtx.setCol(3, Vec4f(col3));
	mtx = invert(mtx);

	OutWoopifiedV0 = make_float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
	OutWoopifiedV1 = make_float4(mtx.getRow(0).x, mtx.getRow(0).y, mtx.getRow(0).z, mtx.getRow(0).w);
	OutWoopifiedV2 = make_float4(mtx.getRow(1).x, mtx.getRow(1).y, mtx.getRow(1).z, mtx.getRow(1).w);
}
