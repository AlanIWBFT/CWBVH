#pragma once

void rtBindBVH2Data(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex);

void rtTraceBVH2(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount);
