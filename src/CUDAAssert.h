#pragma once
#undef NDEBUG
#include <cassert>
#include "Logger.h"
#define cudaCheck(x) \
	{ \
		cudaError_t err = (x); \
		if (err != cudaSuccess) { \
			Log("Line %d: cudaCheckError: %s", __LINE__, cudaGetErrorString(err)); \
			assert(0); \
		} \
	}
