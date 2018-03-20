#pragma once

#undef NDEBUG
#include <cassert>
#include <cstdio>
#define cudaCheck(x) \
	{ \
		cudaError_t err = (x); \
		if (err != cudaSuccess) { \
			printf("Line %d: cudaCheckError: %s", __LINE__, cudaGetErrorString(err)); \
			assert(0); \
		} \
	}

#define cudaPostKernelLaunchCheck \
{ \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) \
	{ \
		printf("Line %d: PostKernelLaunchError: %s", __LINE__, cudaGetErrorString(err)); \
		exit(1); \
	} \
}
