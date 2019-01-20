#pragma once

void rtBindCWBVHData(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex);

void rtTraceCWBVH(
	Ray* rayBuffer,
	Hit* rayResultBuffer,
	int rayCount);
