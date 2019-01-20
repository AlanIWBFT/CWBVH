#undef NDEBUG
#include <cassert>
#include <cfloat>

#include "helper_math.h"
#include "EmbreeBVHBuilder.h"
#include "Logger.h"
#include <tbb/parallel_for.h>

#include <embree3/rtcore.h>
#include <embree3/rtcore_builder.h>

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
	for (uint32_t i = 0; i < childCount; i++)
		node->Children[i] = ((BVHNNode<N>**)children)[i];
	node->ChildCount = childCount;
}

/* Callback to set the bounds of all children */
template <int N>
void SetNodeBounds(void* nodePtr, const struct RTCBounds** bounds, unsigned int childCount, void* userPtr)
{
	BVHNNode<N>* node = (BVHNNode<N>*)nodePtr;
	for (uint32_t i = 0; i < childCount; i++)
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
	newNode->LeafPrimitiveCount = (int)primitiveCount;
	return newNode;
}

/* Callback to split a build primitive */
void SplitPrimitiveAABB(const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr)
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

float GetDimension(float3 v, unsigned int dimension)
{
	switch (dimension)
	{
	case 0:
		return v.x;
	case 1:
		return v.y;
	case 2:
		return v.z;
	default:
		assert(0);
		return 0;
	}
}

void SplitTriangle(std::vector<int3>& IndexBuffer, std::vector<float3>& VertexBuffer, int TriangleIndex, unsigned int Dimension, float Position, AABB& InOutLeft, AABB& InOutRight)
{
	AABB NewLeft;
	AABB NewRight;
	NewLeft.Low = make_float3(FLT_MAX);
	NewLeft.High = make_float3(-FLT_MAX);
	NewRight.Low = make_float3(FLT_MAX);
	NewRight.High = make_float3(-FLT_MAX);

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].z];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].x];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].x];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].y];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].y];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].z];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	InOutLeft.Low = fmaxf(InOutLeft.Low, NewLeft.Low);
	InOutLeft.High = fminf(InOutLeft.High, NewLeft.High);
	InOutRight.Low = fmaxf(InOutRight.Low, NewRight.Low);
	InOutRight.High = fminf(InOutRight.High, NewRight.High);
}

void SplitPrimitiveTriangle(const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr)
{
	AABB Left;
	Left.Low.x = primitive->lower_x;
	Left.Low.y = primitive->lower_y;
	Left.Low.z = primitive->lower_z;
	Left.High.x = primitive->upper_x;
	Left.High.y = primitive->upper_y;
	Left.High.z = primitive->upper_z;

	AABB Right;
	Right.Low.x  = primitive->lower_x;
	Right.Low.y  = primitive->lower_y;
	Right.Low.z  = primitive->lower_z;
	Right.High.x = primitive->upper_x;
	Right.High.y = primitive->upper_y;
	Right.High.z = primitive->upper_z;

	MeshDescription* Mesh = (MeshDescription*)userPtr;

	SplitTriangle(*Mesh->IndexBuffer, *Mesh->VertexBuffer, primitive->primID, dimension, position, Left, Right);

	leftBounds->lower_x = Left.Low.x;
	leftBounds->lower_y = Left.Low.y;
	leftBounds->lower_z = Left.Low.z;
	leftBounds->upper_x = Left.High.x;
	leftBounds->upper_y = Left.High.y;
	leftBounds->upper_z = Left.High.z;

	rightBounds->lower_x = Right.Low.x;
	rightBounds->lower_y = Right.Low.y;
	rightBounds->lower_z = Right.Low.z;
	rightBounds->upper_x = Right.High.x;
	rightBounds->upper_y = Right.High.y;
	rightBounds->upper_z = Right.High.z;
}

BVH2Node* BuildBVH2AABB(int NumAABBs, std::function<void(int PrimitiveIndex, float3& OutLower, float3& OutUpper)> GetPrimitiveBoundsFunc)
{
	RTCBuildPrimitive* AABBs = (RTCBuildPrimitive*)_aligned_malloc(sizeof(RTCBuildPrimitive) * NumAABBs * 2, 32);

	tbb::parallel_for(0, NumAABBs, [&](int PrimitiveIndex)
	{
		float3 lower = make_float3(0);
		float3 upper = make_float3(0);

		GetPrimitiveBoundsFunc(PrimitiveIndex, lower, upper);

		AABBs[PrimitiveIndex].lower_x = lower.x;
		AABBs[PrimitiveIndex].lower_y = lower.y;
		AABBs[PrimitiveIndex].lower_z = lower.z;
		AABBs[PrimitiveIndex].upper_x = upper.x;
		AABBs[PrimitiveIndex].upper_y = upper.y;
		AABBs[PrimitiveIndex].upper_z = upper.z;
		AABBs[PrimitiveIndex].geomID = 0;
		AABBs[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCDevice EmbreeDevice = rtcNewDevice(nullptr);
	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = AABBs;
	args.primitiveCount = NumAABBs;
	args.primitiveArrayCapacity = NumAABBs * 2;
	args.createNode = CreateNode<2>;
	args.setNodeChildren = SetNodeChildren<2>;
	args.setNodeBounds = SetNodeBounds<2>;
	args.createLeaf = CreateLeaf<2>;
	args.splitPrimitive = SplitPrimitiveAABB;

	args.maxDepth = 35;

	BVH2Node* root = (BVH2Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);
	rtcReleaseDevice(EmbreeDevice);

	_aligned_free(AABBs);
	return root;
}

BVH2Node * BuildBVH2Triangle(std::vector<int3>& IndexBuffer, std::vector<float3> VertexBuffer)
{
	RTCBuildPrimitive* AABBs = (RTCBuildPrimitive*)_aligned_malloc(sizeof(RTCBuildPrimitive) * IndexBuffer.size() * 2, 32);

	tbb::parallel_for(0, (int)IndexBuffer.size(), [&](int PrimitiveIndex)
	{
		float3 lower = make_float3(0);
		float3 upper = make_float3(0);

		lower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
		lower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
		lower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
		upper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
		upper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
		upper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);

		AABBs[PrimitiveIndex].lower_x = lower.x;
		AABBs[PrimitiveIndex].lower_y = lower.y;
		AABBs[PrimitiveIndex].lower_z = lower.z;
		AABBs[PrimitiveIndex].upper_x = upper.x;
		AABBs[PrimitiveIndex].upper_y = upper.y;
		AABBs[PrimitiveIndex].upper_z = upper.z;
		AABBs[PrimitiveIndex].geomID = 0;
		AABBs[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCDevice EmbreeDevice = rtcNewDevice(nullptr);
	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = AABBs;
	args.primitiveCount = IndexBuffer.size();
	args.primitiveArrayCapacity = IndexBuffer.size() * 2;
	args.createNode = CreateNode<2>;
	args.setNodeChildren = SetNodeChildren<2>;
	args.setNodeBounds = SetNodeBounds<2>;
	args.createLeaf = CreateLeaf<2>;

	MeshDescription Mesh { &IndexBuffer, &VertexBuffer };
	args.userPtr = &Mesh;
	args.splitPrimitive = SplitPrimitiveTriangle;

	BVH2Node* root = (BVH2Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);
	rtcReleaseDevice(EmbreeDevice);

	_aligned_free(AABBs);
	return root;
}


BVH8Node * BuildBVH8AABB(int NumAABBs, std::function<void(int PrimitiveIndex, float3& OutLower, float3& OutUpper)> GetPrimitiveBoundsFunc)
{
	RTCBuildPrimitive* AABBs = (RTCBuildPrimitive*)_aligned_malloc(sizeof(RTCBuildPrimitive) * NumAABBs * 2, 32);

	tbb::parallel_for(0, NumAABBs, [&](int PrimitiveIndex)
	{
		float3 lower = make_float3(0);
		float3 upper = make_float3(0);

		GetPrimitiveBoundsFunc(PrimitiveIndex, lower, upper);

		AABBs[PrimitiveIndex].lower_x = lower.x;
		AABBs[PrimitiveIndex].lower_y = lower.y;
		AABBs[PrimitiveIndex].lower_z = lower.z;
		AABBs[PrimitiveIndex].upper_x = upper.x;
		AABBs[PrimitiveIndex].upper_y = upper.y;
		AABBs[PrimitiveIndex].upper_z = upper.z;
		AABBs[PrimitiveIndex].geomID = 0;
		AABBs[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCDevice EmbreeDevice = rtcNewDevice(nullptr);
	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.maxBranchingFactor = 8;
	args.maxLeafSize = 3;
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = AABBs;
	args.primitiveCount = NumAABBs;
	args.primitiveArrayCapacity = NumAABBs * 2;
	args.createNode = CreateNode<8>;
	args.setNodeChildren = SetNodeChildren<8>;
	args.setNodeBounds = SetNodeBounds<8>;
	args.createLeaf = CreateLeaf<8>;
	args.splitPrimitive = SplitPrimitiveAABB;

	BVH8Node* root = (BVH8Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);
	rtcReleaseDevice(EmbreeDevice);

	_aligned_free(AABBs);
	return root;
}

BVH8Node * BuildBVH8Triangle(std::vector<int3>& IndexBuffer, std::vector<float3> VertexBuffer)
{
	RTCBuildPrimitive* AABBs = (RTCBuildPrimitive*)_aligned_malloc(sizeof(RTCBuildPrimitive) * IndexBuffer.size() * 8, 32);

	tbb::parallel_for(0, (int)IndexBuffer.size(), [&](int PrimitiveIndex)
	{
		float3 lower = make_float3(0);
		float3 upper = make_float3(0);

		lower.x = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
		lower.y = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
		lower.z = fminf(fminf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);
		upper.x = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].x, VertexBuffer[IndexBuffer[PrimitiveIndex].y].x), VertexBuffer[IndexBuffer[PrimitiveIndex].z].x);
		upper.y = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].y, VertexBuffer[IndexBuffer[PrimitiveIndex].y].y), VertexBuffer[IndexBuffer[PrimitiveIndex].z].y);
		upper.z = fmaxf(fmaxf(VertexBuffer[IndexBuffer[PrimitiveIndex].x].z, VertexBuffer[IndexBuffer[PrimitiveIndex].y].z), VertexBuffer[IndexBuffer[PrimitiveIndex].z].z);

		AABBs[PrimitiveIndex].lower_x = lower.x;
		AABBs[PrimitiveIndex].lower_y = lower.y;
		AABBs[PrimitiveIndex].lower_z = lower.z;
		AABBs[PrimitiveIndex].upper_x = upper.x;
		AABBs[PrimitiveIndex].upper_y = upper.y;
		AABBs[PrimitiveIndex].upper_z = upper.z;
		AABBs[PrimitiveIndex].geomID = 0;
		AABBs[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCDevice EmbreeDevice = rtcNewDevice(nullptr);
	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.maxBranchingFactor = 8;
	args.maxLeafSize = 3;
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.bvh = bvh;
	args.primitives = AABBs;
	args.primitiveCount = IndexBuffer.size();
	args.primitiveArrayCapacity = IndexBuffer.size() * 8;
	args.createNode = CreateNode<8>;
	args.setNodeChildren = SetNodeChildren<8>;
	args.setNodeBounds = SetNodeBounds<8>;
	args.createLeaf = CreateLeaf<8>;

	MeshDescription Mesh { &IndexBuffer, &VertexBuffer };
	args.userPtr = &Mesh;
	args.splitPrimitive = SplitPrimitiveTriangle;

	BVH8Node* root = (BVH8Node*)rtcBuildBVH(&args);
	rtcReleaseBVH(bvh);
	rtcReleaseDevice(EmbreeDevice);

	_aligned_free(AABBs);
	return root;
}
