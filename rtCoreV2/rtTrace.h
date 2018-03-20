#pragma once

struct Ray
{
	float4 RayOriginAndNearClip;
	float4 RayDirectionAndFarClip;
};

struct RayResult
{
	float3 TriangleNormalUnnormalized;
	int HitTriangleIndex;
	float2 TriangleUV;
};

__host__ void rtBindCWBVHData(
	const float4* InBVHTreeNodes,
	const float4* InWoopifiedTriangles,
	const int* InTriangleIndicies,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize);

__host__ void rtBindBVH2Data(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize);

__host__ void rtTrace(
    Ray* rayBuffer,
    RayResult* rayResultBuffer,
    int rayCount);
