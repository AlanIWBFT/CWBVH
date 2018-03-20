#include "linear_math.h"
#include "helper_math.h"
#define EMBREE_STATIC_LIB
#include "bvhbuilder.h"
#include <tbb/parallel_for.h>
#undef NDEBUG
#include <cassert>
#include <cfloat>

#pragma comment(lib, "embree_avx.lib")
#pragma comment(lib, "embree_avx2.lib")
#pragma comment(lib, "embree_sse42.lib")
#pragma comment(lib, "embree3.lib")
#pragma comment(lib, "lexers.lib")
#pragma comment(lib, "math.lib")
#pragma comment(lib, "simd.lib")
#pragma comment(lib, "sys.lib")
#pragma comment(lib, "tasking.lib")

/* Callback to create a node */
template <int N>
void* CreateNode(RTCThreadLocalAllocator allocator, unsigned int childCount, void* userPtr)
{
	BVHNNode<N>* newNode = new BVHNNode<N>;
	return newNode;
}

/* Callback to set the pointer to all children */
template <int N>
void SetNodeChildren(void* nodePtr, void** children, unsigned int childCount, void* userPtr)
{
	assert(childCount > 0);
	BVHNNode<N>* node = (BVHNNode<N>*)nodePtr;
	for (int i = 0; i < childCount; i++)
		node->Children[i] = ((BVHNNode<N>**)children)[i];
	node->ChildCount = childCount;
}

/* Callback to set the bounds of all children */
template <int N>
void SetNodeBounds(void* nodePtr, const struct RTCBounds** bounds, unsigned int childCount, void* userPtr)
{
	BVHNNode<N>* node = (BVHNNode<N>*)nodePtr;
	for (int i = 0; i < childCount; i++)
	{
		node->ChildrenMin[i].x = bounds[i]->lower_x;
		node->ChildrenMin[i].y = bounds[i]->lower_y;
		node->ChildrenMin[i].z = bounds[i]->lower_z;
		node->ChildrenMax[i].x = bounds[i]->upper_x;
		node->ChildrenMax[i].y = bounds[i]->upper_y;
		node->ChildrenMax[i].z = bounds[i]->upper_z;
	}
}

/* Callback to create a leaf node */
template <int N>
void* CreateLeaf(RTCThreadLocalAllocator allocator, const struct RTCBuildPrimitive* primitives, size_t primitiveCount, void* userPtr)
{
	assert(primitiveCount > 0);
	BVHNNode<N>* newNode = new BVHNNode<N>;
	newNode->LeafPrimitiveRefs = new int[primitiveCount]();
	for (int i = 0; i < primitiveCount; i++)
		newNode->LeafPrimitiveRefs[i] = primitives[i].primID;
	newNode->LeafPrimitiveCount = primitiveCount;
	return newNode;
}

/* Callback to split a build primitive */
void SplitPrimitive(const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr)
{
	leftBounds->lower_x = primitive->lower_x;
	leftBounds->lower_y = primitive->lower_y;
	leftBounds->lower_z = primitive->lower_z;
	leftBounds->upper_x = primitive->upper_x;
	leftBounds->upper_y = primitive->upper_y;
	leftBounds->upper_z = primitive->upper_z;

	rightBounds->lower_x = primitive->lower_x;
	rightBounds->lower_y = primitive->lower_y;
	rightBounds->lower_z = primitive->lower_z;
	rightBounds->upper_x = primitive->upper_x;
	rightBounds->upper_y = primitive->upper_y;
	rightBounds->upper_z = primitive->upper_z;

	switch (dimension)
	{
	case 0:
		leftBounds->upper_x = rightBounds->lower_x = position;
		break;
	case 1:
		leftBounds->upper_y = rightBounds->lower_y = position;
		break;
	case 2:
		leftBounds->upper_z = rightBounds->lower_z = position;
		break;
	default:
		assert(0);
	}
}

EmbreeBVHBuilder::EmbreeBVHBuilder(
	const int InNumVertices,
	const int InNumTriangles,
	const float3 InVertexWorldPositionBuffer[],
	const int3 InTriangleIndexBuffer[])
	:
	NumVertices(InNumVertices),
	NumTriangles(InNumTriangles),
	VertexWorldPositionBuffer(InVertexWorldPositionBuffer),
	TriangleIndexBuffer(InTriangleIndexBuffer)
{
	EmbreeDevice = rtcNewDevice(nullptr);
}

EmbreeBVHBuilder::~EmbreeBVHBuilder()
{
	rtcReleaseDevice(EmbreeDevice);
}

BVH2Node * EmbreeBVHBuilder::BuildBVH2()
{
	RTCBuildPrimitive* Primitives = new RTCBuildPrimitive[NumTriangles * 2]();

	tbb::parallel_for(0, NumTriangles, [&](int PrimitiveIndex)
	{
		Primitives[PrimitiveIndex].lower_x = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].lower_y = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].lower_z = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].upper_x = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].upper_y = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].upper_z = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].geomID = 0;
		Primitives[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = Primitives;
	args.primitiveCount = NumTriangles;
	args.primitiveArrayCapacity = NumTriangles * 2;
	args.createNode = CreateNode<2>;
	args.setNodeChildren = SetNodeChildren<2>;
	args.setNodeBounds = SetNodeBounds<2>;
	args.createLeaf = CreateLeaf<2>;
	args.splitPrimitive = SplitPrimitive;

	BVH2Node* root = (BVH2Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);

	delete[] Primitives;
	return root;
}

void EmbreeBVHBuilder::ConvertToCUDABVH2(
	const BVH2Node * root,
	const int TriangleMaterialIndex[],
	std::vector<float4>& OutNodeData,
	std::vector<float4>& OutWoopifiedTriangles,
	std::vector<int>& OutTriangleIndices
)
{
	const BVH2Node* stackNodePtr[32];
	int stackNodeParentIdx[32];
	int stackNodeIdxInParent[32];
	int stackPtr = 0;

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

		if (node->LeafPrimitiveCount == 0)
		{
			assert(node->Children[0] != nullptr && node->Children[1] != nullptr);

			int currentNodeAddr = OutNodeData.size();
			if (parentNodeAddr >= 0)
			{
				if (nodeIdxInParent == 0)
					OutNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (nodeIdxInParent == 1)
					OutNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			OutNodeData.push_back(make_float4(node->ChildrenMin[0].x, node->ChildrenMax[0].x, node->ChildrenMin[0].y, node->ChildrenMax[0].y));
			OutNodeData.push_back(make_float4(node->ChildrenMin[1].x, node->ChildrenMax[1].x, node->ChildrenMin[1].y, node->ChildrenMax[1].y));
			OutNodeData.push_back(make_float4(node->ChildrenMin[0].z, node->ChildrenMax[0].z, node->ChildrenMin[1].z, node->ChildrenMax[1].z));
			OutNodeData.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f)); // children indices, will be filled in by the children themselves

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
			assert(node->Children[0] == nullptr && node->Children[1] == nullptr);
			// Leaf node
			int currentNodeAddr = OutWoopifiedTriangles.size();
			currentNodeAddr = ~currentNodeAddr;

			if (parentNodeAddr >= 0)
			{
				if (nodeIdxInParent == 0)
					OutNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (nodeIdxInParent == 1)
					OutNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			for (int i = 0; i < node->LeafPrimitiveCount; i++)
			{
				int triangleIndex = node->LeafPrimitiveRefs[i];
				float3 v0 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].x];
				float3 v1 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].y];
				float3 v2 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].z];

				Mat4f mtx;
				float4 col0 = make_float4(v0 - v2, 0.0f);
				float4 col1 = make_float4(v1 - v2, 0.0f);
				float4 col2 = make_float4(cross(v0 - v2, v1 - v2), 0.0f);
				float4 col3 = make_float4(v2, 1.0f);
				mtx.setCol(0, Vec4f(col0)); // sets matrix column 0 equal to a Vec4f(Vec3f, 0.0f )
				mtx.setCol(1, Vec4f(col1));
				mtx.setCol(2, Vec4f(col2));
				mtx.setCol(3, Vec4f(col3));
				mtx = invert(mtx);

				float4 WoopifiedVertices[3];
				WoopifiedVertices[0] = make_float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
				WoopifiedVertices[1] = make_float4(mtx.getRow(0).x, mtx.getRow(0).y, mtx.getRow(0).z, mtx.getRow(0).w);
				WoopifiedVertices[2] = make_float4(mtx.getRow(1).x, mtx.getRow(1).y, mtx.getRow(1).z, mtx.getRow(1).w);

				OutWoopifiedTriangles.push_back(WoopifiedVertices[0]);
				OutWoopifiedTriangles.push_back(WoopifiedVertices[1]);
				OutWoopifiedTriangles.push_back(WoopifiedVertices[2]);

				OutTriangleIndices.push_back(triangleIndex);
				OutTriangleIndices.push_back(TriangleMaterialIndex[triangleIndex]);
				OutTriangleIndices.push_back(0);
			}

			unsigned int leafTerminator = 0x80000000;
			OutWoopifiedTriangles.push_back(make_float4(*(float*)(&leafTerminator), 0, 0, 0));

			OutTriangleIndices.push_back(0);
		}
	}
}

BVH8Node * EmbreeBVHBuilder::BuildBVH8Direct()
{
	RTCBuildPrimitive* Primitives = new RTCBuildPrimitive[NumTriangles * 8]();

	tbb::parallel_for(0, NumTriangles, [&](int PrimitiveIndex)
	{
		Primitives[PrimitiveIndex].lower_x = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].lower_y = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].lower_z = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].upper_x = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].upper_y = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].upper_z = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].geomID = 0;
		Primitives[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.maxBranchingFactor = 8;
	args.maxLeafSize = 3;
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = Primitives;
	args.primitiveCount = NumTriangles;
	args.primitiveArrayCapacity = NumTriangles * 8;
	args.createNode = CreateNode<8>;
	args.setNodeChildren = SetNodeChildren<8>;
	args.setNodeBounds = SetNodeBounds<8>;
	args.createLeaf = CreateLeaf<8>;
	args.splitPrimitive = SplitPrimitive;

	BVH8Node* root = (BVH8Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);

	delete[] Primitives;
	return root;
}

void EmbreeBVHBuilder::ConvertToCUDACompressedWideBVH(BVH8Node * root, const int TriangleMaterialIndex[], std::vector<float4>& OutNodeData, std::vector<float4>& OutWoopifiedTriangles, std::vector<int>& OutTriangleIndices)
{
	BVH8Node* stackNodePtr[256];
	int stackNodeAddr[256];
	int stackPtr = 0;

	stackNodePtr[stackPtr] = root;
	stackNodeAddr[stackPtr] = 0;
	stackPtr++;

	// Preallocate space for root
	OutNodeData.push_back(make_float4(0.0f));
	OutNodeData.push_back(make_float4(0.0f));
	OutNodeData.push_back(make_float4(0.0f));
	OutNodeData.push_back(make_float4(0.0f));
	OutNodeData.push_back(make_float4(0.0f));

	std::random_device rd;
	std::mt19937 gen(rd());

	while (stackPtr > 0)
	{
		assert(stackPtr < 256);
		stackPtr--;
		BVH8Node* node = stackNodePtr[stackPtr];

        assert(node->ChildCount == 0 || node->LeafPrimitiveCount == 0);
        
		const int currentNodeAddr = stackNodeAddr[stackPtr];

		// Calculate current node's bounding box (origin point p and Bhi, i.e. low and high respectively) on the fly
		float3 p = make_float3(FLT_MAX);
		float3 Bhi = make_float3(-FLT_MAX);
		
		for (int childIndex = 0; childIndex < 8; childIndex++)
		{
			if (node->Children[childIndex] == nullptr)
				continue;

			p = fminf(p, node->ChildrenMin[childIndex]);
			Bhi = fmaxf(Bhi, node->ChildrenMax[childIndex]);
		}
		
		// Child node ordering
		{
			float3 nodeCentroid = (p + Bhi) * 0.5f;
			
			float cost[8][8] = {};
			int assignment[8] = {};
			bool isSlotEmpty[8];
			
			for(int s = 0; s < 8; s++)
				isSlotEmpty[s] = true;
			
			for (int childIndex = 0; childIndex < 8; childIndex++)
				assignment[childIndex] = -1;
			
			for(int s = 0; s < 8; s++)
			{
				float3 ds = make_float3((s & 1) ? -1.0f : 1.0f, ((s >> 1) & 1) ? -1.0f : 1.0f, ((s >> 2) & 1) ? -1.0f : 1.0f);
				
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
			
			while(true)
			{
				// ExtractMin
				float minCost = FLT_MAX;
				int2 minEntry = make_int2(-1);
				
				for(int s = 0; s < 8; s++)
				{
					for (int childIndex = 0; childIndex < 8; childIndex++)
					{
						if(assignment[childIndex] == -1 && isSlotEmpty[s] && cost[s][childIndex] < minCost)
						{
							minCost = cost[s][childIndex];
							minEntry = make_int2(s, childIndex);
						}
					}
				}
				
				if(minEntry.x != -1 || minEntry.y != -1)
				{
					assert(minEntry.x != -1 && minEntry.y != -1);
					isSlotEmpty[minEntry.x] = false;
					assignment[minEntry.y] = minEntry.x;
				}
				else
					break;
			}
			
			for (int childIndex = 0; childIndex < 8; childIndex++)
			{
				if(assignment[childIndex] == -1)
				{
					int s = 0;
					for(; s < 8; s++)
						if(isSlotEmpty[s])
						{
							isSlotEmpty[s] = false;
							assignment[childIndex] = s;
							break;
						}
					if(s == 8)
						assert(0);
				}
			}
			
			BVH8Node oldNode = *node;
			for (int childIndex = 0; childIndex < 8; childIndex++)
			{
				assert(assignment[childIndex] != -1);
				node->Children[childIndex] = oldNode.Children[assignment[childIndex]];
				node->ChildrenMin[childIndex] = oldNode.ChildrenMin[assignment[childIndex]];
				node->ChildrenMax[childIndex] = oldNode.ChildrenMax[assignment[childIndex]];
			}
		}

		// Calculate quantization parameters for each axis respectively
		const int Nq = 8;
		char3 e; // implicit quantization
		assert(ceilf(log2f((Bhi.x - p.x) / (pow(2, Nq) - 1))) <= 127);
		assert(ceilf(log2f((Bhi.y - p.y) / (pow(2, Nq) - 1))) <= 127);
		assert(ceilf(log2f((Bhi.z - p.z) / (pow(2, Nq) - 1))) <= 127);
		e.x = ceilf(log2f((Bhi.x - p.x) / (pow(2, Nq) - 1)));
		e.y = ceilf(log2f((Bhi.y - p.y) / (pow(2, Nq) - 1)));
		e.z = ceilf(log2f((Bhi.z - p.z) / (pow(2, Nq) - 1)));

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

			int3 qlo, qhi; // implicit quantization

			qlo.x = floorf((node->ChildrenMin[childIndex].x - p.x) / pow(2, e.x));
			qlo.y = floorf((node->ChildrenMin[childIndex].y - p.y) / pow(2, e.y));
			qlo.z = floorf((node->ChildrenMin[childIndex].z - p.z) / pow(2, e.z));

			qhi.x = ceilf((node->ChildrenMax[childIndex].x - p.x) / pow(2, e.x));
			qhi.y = ceilf((node->ChildrenMax[childIndex].y - p.y) / pow(2, e.y));
			qhi.z = ceilf((node->ChildrenMax[childIndex].z - p.z) / pow(2, e.z));

			unsigned char* const childBoundsBaseAddr = (unsigned char*)&OutNodeData[currentNodeAddr + 2];
			childBoundsBaseAddr[childIndex + 0] = qlo.x;
			childBoundsBaseAddr[childIndex + 8] = qlo.y;
			childBoundsBaseAddr[childIndex + 16] = qlo.z;
			childBoundsBaseAddr[childIndex + 24] = qhi.x;
			childBoundsBaseAddr[childIndex + 32] = qhi.y;
			childBoundsBaseAddr[childIndex + 40] = qhi.z;

			if (node->Children[childIndex]->LeafPrimitiveCount == 0)
			{
				// Internal child, set params accordingly and push onto stack
				const int childNodeAddr = OutNodeData.size();
				if (internalChildCount == 0)
					childBaseIndex = childNodeAddr / 5;

				// Preallocate space for child
				OutNodeData.push_back(make_float4(0.0f)); // p, e and imask
				OutNodeData.push_back(make_float4(0.0f)); // child, triangle base index and meta field
				OutNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box
				OutNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box
				OutNodeData.push_back(make_float4(0.0f)); // child meta field and quantized bounding box

				imask |= 1 << childIndex;
				// Set the meta field. This calculation assumes children are stored contiguously
				unsigned char* const childMetaField = ((unsigned char*)&OutNodeData[currentNodeAddr + 1]) + 8;
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
					triangleBaseIndex = OutWoopifiedTriangles.size() / 3;

				imask |= 0 << childIndex; // Not actually doing anything, just for symmetry
				assert(node->Children[childIndex]->LeafPrimitiveCount <= 3);
				int unaryEncodedPrimitiveCount = node->Children[childIndex]->LeafPrimitiveCount == 1 ? 0b001 : node->Children[childIndex]->LeafPrimitiveCount == 2 ? 0b011 : 0b111;
				// Set the meta field. This calculation assumes children are stored contiguously
				unsigned char* const childMetaField = ((unsigned char*)&OutNodeData[currentNodeAddr + 1]) + 8;
				childMetaField[childIndex] = (unaryEncodedPrimitiveCount << 5) | (leafChildPrimitiveCount & 0b11111);
				leafChildPrimitiveCount += node->Children[childIndex]->LeafPrimitiveCount;

				for (int i = 0; i < node->Children[childIndex]->LeafPrimitiveCount; i++)
				{
					int triangleIndex = node->Children[childIndex]->LeafPrimitiveRefs[i];
					float3 v0 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].x];
					float3 v1 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].y];
					float3 v2 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].z];

					// Prepare triangles for woop intersection test
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

					float4 WoopifiedVertices[3];
					WoopifiedVertices[0] = make_float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
					WoopifiedVertices[1] = make_float4(mtx.getRow(0).x, mtx.getRow(0).y, mtx.getRow(0).z, mtx.getRow(0).w);
					WoopifiedVertices[2] = make_float4(mtx.getRow(1).x, mtx.getRow(1).y, mtx.getRow(1).z, mtx.getRow(1).w);

					OutWoopifiedTriangles.push_back(WoopifiedVertices[0]);
					OutWoopifiedTriangles.push_back(WoopifiedVertices[1]);
					OutWoopifiedTriangles.push_back(WoopifiedVertices[2]);

					OutTriangleIndices.push_back(triangleIndex);
					OutTriangleIndices.push_back(TriangleMaterialIndex[triangleIndex]);
					OutTriangleIndices.push_back(0);
				}
			}
		}

		unsigned char exyzAndimask[4];
		exyzAndimask[0] = e.x;
		exyzAndimask[1] = e.y;
		exyzAndimask[2] = e.z;
		exyzAndimask[3] = imask;
		OutNodeData[currentNodeAddr + 0] = make_float4(p, *(float*)&exyzAndimask);
		OutNodeData[currentNodeAddr + 1].x = *(float*)&childBaseIndex;
		OutNodeData[currentNodeAddr + 1].y = *(float*)&triangleBaseIndex;
		// z and w are already set for meta field
	}
}
