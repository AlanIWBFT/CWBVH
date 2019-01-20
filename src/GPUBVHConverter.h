#pragma once

#include <functional>
#include "EmbreeBVHBuilder.h"

struct GPUBVHIntermediates
{
	std::vector<float4> BVHNodeData;
	std::vector<float4> InlinedPrimitives;
	std::vector<int>    PrimitiveIndices;
};

void ConvertToGPUBVH2(
	BVH2Node*& root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
);

void ConvertToGPUCompressedWideBVH(
	BVH8Node * root,
	std::function<void(int PrimitiveIndex, std::vector<float4>& InlinedPrimitives)> AppendPrimitiveFunc,
	GPUBVHIntermediates& OutIntermediates
);

void WoopifyTriangle(
	float3 v0,
	float3 v1,
	float3 v2,
	float4& OutWoopifiedV0,
	float4& OutWoopifiedV1,
	float4& OutWoopifiedV2
);
