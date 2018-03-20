#pragma once

struct MortonHash6
{
	int OriginalPosition;
	unsigned int Hash[6];
};

__host__ int rtComputeSurfelMortonHash(
    float4* cudaWorldPositionTexture,
	float4* cudaWorldNormalTexture,
	float* cudaTexelRadiusTexture,
	const int SizeX,
	const int SizeY,
    MortonHash6* const OutCudaMortonHashes
);
