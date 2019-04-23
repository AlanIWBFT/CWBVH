#include <cuda_runtime.h>
#include "helper_math.h"

inline __host__ __device__ int2 abs(int2 v)
{
  return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
  return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
  return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

template<typename T>
__inline__ __device__
T warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(__activemask(), val, offset);
  return val;
}

template<typename T>
__inline__ __device__
T blockReduceSumToThread0(T val) {

  static __shared__ T shared[32]; // Shared mem for 32 partial sums
  int lane = (threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int wid = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

                  //read from shared memory only if that warp existed
  val = ((threadIdx.y * blockDim.x + threadIdx.x) < blockDim.x * blockDim.y / warpSize) ? shared[lane] : 0;

  if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}