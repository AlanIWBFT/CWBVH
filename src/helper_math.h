/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /*
  *  This file implements common mathematical operations on vector types
  *  (float3, float4 etc.) since these are not provided as standard by CUDA.
  *
  *  The syntax is modeled on the Cg standard library.
  *
  *  This is part of the Helper library includes
  *
  *    Thanks to Linh Hah for additions and fixes.
  */

#pragma once

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
	return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline int max(int a, int b)
{
	return a > b ? a : b;
}

inline int min(int a, int b)
{
	return a < b ? a : b;
}

inline float rsqrtf(float x)
{
	return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
	return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
	return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
	return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
	return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
	return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
	return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
	return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
	return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
	return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
	return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
	return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
	return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
	return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
	return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
	return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
	return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
	return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
	return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
	return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
	return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
	return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
	return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
	return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
	return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
	return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
	return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
	return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
	return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
	return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
	return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
	return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
	return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
	return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
	return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
	return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
	return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
	a.x += b;
	a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
	return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
	return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
	return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
	return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
	return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
	return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
	return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
	return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
	return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
	return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
	return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
	return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
	return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
	return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
	return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
	return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
	return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
	return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
	return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
	return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
	return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
	return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
	return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
	return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
	return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
	return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
	return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
	return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
	return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
	return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
	return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
	return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
	return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
	return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
	a.x /= b;
	a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
	return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
	return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
	return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
	return make_int2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
	return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
	return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
	return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
	return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
	return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
	return make_int2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
	return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
	return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
	return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
	return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
	return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
	return a + t*(b - a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
	return a + t*(b - a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
	return a + t*(b - a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
	return a + t*(b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
	return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
	return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
	return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
	return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
	return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
	return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
	return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
	return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
	return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
	return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
	return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
	return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
	return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
	return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
	return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
	return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
	return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
	return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
	return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
	return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
	return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
	return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
	return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
	return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
	return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
	return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
	return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
	return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
	return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
	return make_float2(fabsf(v.x), fabsf(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
	return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
	return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

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

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
	return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
	float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
	float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
	float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

typedef unsigned char byte;

inline __device__ __host__ int idivCeil(int x, int y) { return (x + y - 1)/y; }

struct matrix4x4
{
	// Stored by column major, m[0][.] = col 0, m[1][.] = col 1, etc
	float m[4][4];

	__device__ __host__ static matrix4x4 Identity()
	{
		matrix4x4 r = 
		{
			{
				{ 1.0f, 0.0f, 0.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f, 0.0f },
				{ 0.0f, 0.0f, 1.0f, 0.0f },
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			}
		};
		return r;
	}

	__device__ __host__ static matrix4x4 From3Rows(float4 row0, float4 row1, float4 row2)
	{
		matrix4x4 r =
		{
			{
				{ row0.x,  row1.x,  row2.x, 0.0f },
				{ row0.y,  row1.y,  row2.y, 0.0f },
				{ row0.z,  row1.z,  row2.z, 0.0f },
				{ row0.w,  row1.w,  row2.w, 1.0f },
			}
		};
		return r;
	}

	__device__ __host__ static matrix4x4 From3Cols(float4 col0, float4 col1, float4 col2)
	{
		matrix4x4 r =
		{
			{
				{ col0.x, col0.y, col0.z, col0.w },
				{ col1.x, col1.y, col1.z, col1.w },
				{ col2.x, col2.y, col2.z, col2.w },
				{ 0.0f,   0.0f,   0.0f,   1.0f   },
			}
		};
		return r;
	}

	__device__ __host__ float4 row(int i) const { return make_float4(m[0][i], m[1][i], m[2][i], m[3][i]); };
	__device__ __host__ float4 col(int i) const { return make_float4(m[i][0], m[i][1], m[i][2], m[i][3]); };

	__device__ __host__ float3 transformPoint3(const float3 v) const { return make_float3(col(0) * v.x + col(1) * v.y + col(2) * v.z + col(3)); };
	__device__ __host__ float3 transformVector3(const float3 v) const { return make_float3(col(0) * v.x + col(1) * v.y + col(2) * v.z); };

	__host__ matrix4x4 inverted()
	{
		float inv[16];
		float v[16] = { m[0][0], m[1][0], m[2][0], m[3][0],
						m[0][1], m[1][1], m[2][1], m[3][1],
						m[0][2], m[1][2], m[2][2], m[3][2],
						m[0][3], m[1][3], m[2][3], m[3][3] };
							  
		inv[0]  =  v[5] * v[10] * v[15] - v[5] * v[11] * v[14] - v[9] * v[6] * v[15] + v[9] * v[7] * v[14] + v[13] * v[6] * v[11] - v[13] * v[7] * v[10];
		inv[4]  = -v[4] * v[10] * v[15] + v[4] * v[11] * v[14] + v[8] * v[6] * v[15] - v[8] * v[7] * v[14] - v[12] * v[6] * v[11] + v[12] * v[7] * v[10];
		inv[8]  =  v[4] * v[9] *  v[15] - v[4] * v[11] * v[13] - v[8] * v[5] * v[15] + v[8] * v[7] * v[13] + v[12] * v[5] * v[11] - v[12] * v[7] * v[9];
		inv[12] = -v[4] * v[9] *  v[14] + v[4] * v[10] * v[13] + v[8] * v[5] * v[14] - v[8] * v[6] * v[13] - v[12] * v[5] * v[10] + v[12] * v[6] * v[9];
		inv[1]  = -v[1] * v[10] * v[15] + v[1] * v[11] * v[14] + v[9] * v[2] * v[15] - v[9] * v[3] * v[14] - v[13] * v[2] * v[11] + v[13] * v[3] * v[10];
		inv[5]  =  v[0] * v[10] * v[15] - v[0] * v[11] * v[14] - v[8] * v[2] * v[15] + v[8] * v[3] * v[14] + v[12] * v[2] * v[11] - v[12] * v[3] * v[10];
		inv[9]  = -v[0] * v[9] * v[15] + v[0] * v[11] * v[13] + v[8] * v[1] * v[15] - v[8] * v[3] * v[13] - v[12] * v[1] * v[11] + v[12] * v[3] * v[9];
		inv[13] =  v[0] * v[9] * v[14] - v[0] * v[10] * v[13] - v[8] * v[1] * v[14] + v[8] * v[2] * v[13] + v[12] * v[1] * v[10] - v[12] * v[2] * v[9];
		inv[2]  =  v[1] * v[6] * v[15] - v[1] * v[7] * v[14] - v[5] * v[2] * v[15] + v[5] * v[3] * v[14] + v[13] * v[2] * v[7] - v[13] * v[3] * v[6];
		inv[6]  = -v[0] * v[6] * v[15] + v[0] * v[7] * v[14] + v[4] * v[2] * v[15] - v[4] * v[3] * v[14] - v[12] * v[2] * v[7] + v[12] * v[3] * v[6];
		inv[10] =  v[0] * v[5] * v[15] - v[0] * v[7] * v[13] - v[4] * v[1] * v[15] + v[4] * v[3] * v[13] + v[12] * v[1] * v[7] - v[12] * v[3] * v[5];
		inv[14] = -v[0] * v[5] * v[14] + v[0] * v[6] * v[13] + v[4] * v[1] * v[14] - v[4] * v[2] * v[13] - v[12] * v[1] * v[6] + v[12] * v[2] * v[5];
		inv[3]  = -v[1] * v[6] * v[11] + v[1] * v[7] * v[10] + v[5] * v[2] * v[11] - v[5] * v[3] * v[10] - v[9] * v[2] * v[7] + v[9] * v[3] * v[6];
		inv[7]  =  v[0] * v[6] * v[11] - v[0] * v[7] * v[10] - v[4] * v[2] * v[11] + v[4] * v[3] * v[10] + v[8] * v[2] * v[7] - v[8] * v[3] * v[6];
		inv[11] = -v[0] * v[5] * v[11] + v[0] * v[7] * v[9] + v[4] * v[1] * v[11] - v[4] * v[3] * v[9] - v[8] * v[1] * v[7] + v[8] * v[3] * v[5];
		inv[15] =  v[0] * v[5] * v[10] - v[0] * v[6] * v[9] - v[4] * v[1] * v[10] + v[4] * v[2] * v[9] + v[8] * v[1] * v[6] - v[8] * v[2] * v[5];

		float det = v[0] * inv[0] + v[1] * inv[4] + v[2] * inv[8] + v[3] * inv[12];

		if (det == 0)
			return matrix4x4::Identity();

		det = 1.f / det;
		matrix4x4 inverse;
		for (int i = 0; i < 16; i++)
			*((float*)inverse.m + i) = inv[i] * det;

		return inverse;
	}
};

struct AABB
{
	float3 Low;
	float3 High;
};

__device__ __host__ __inline__ float getLuminance(const float3& v)
{
	return v.x * 0.3f + v.y * 0.59f + v.z * 0.11f;
}

struct SHVector2
{
	float v[4] ={ 0 };

	__device__ __host__ static SHVector2 basisFunction(const float3 vector)
	{
		SHVector2 result;
		result.v[0] = 0.282095f;
		result.v[1] = -0.488603f * vector.y;
		result.v[2] = 0.488603f * vector.z;
		result.v[3] = -0.488603f * vector.x;
		return result;
	}

	__device__ __host__ SHVector2 operator*(const float& b) const
	{
		SHVector2 result;
		result.v[0] = v[0] * b;
		result.v[1] = v[1] * b;
		result.v[2] = v[2] * b;
		result.v[3] = v[3] * b;
		return result;
	}

	__device__ __host__ SHVector2& operator+=(const SHVector2& rhs)
	{
		v[0] += rhs.v[0];
		v[1] += rhs.v[1];
		v[2] += rhs.v[2];
		v[3] += rhs.v[3];
		return *this;
	}

	__device__ __host__ SHVector2 operator+(const SHVector2& rhs)
	{
		SHVector2 result;
		result.v[0] = v[0] + rhs.v[0];
		result.v[1] = v[1] + rhs.v[1];
		result.v[2] = v[2] + rhs.v[2];
		result.v[3] = v[3] + rhs.v[3];
		return result;
	}

	__device__ __host__ void reset()
	{
		v[0] = 0.0f;
		v[1] = 0.0f;
		v[2] = 0.0f;
		v[3] = 0.0f;
	}

	__device__ __host__ inline SHVector2& addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
	{
		*this += SHVector2::basisFunction(worldSpaceDirection) * (getLuminance(incomingRadiance) * weight);
		return *this;
	}
};

struct SHVector3
{
	float v[9] ={ 0 };

	__device__ __host__ static SHVector3 basisFunction(const float3 vector)
	{
		SHVector3 result;
		result.v[0] = 0.282095f;
		result.v[1] = -0.488603f * vector.y;
		result.v[2] = 0.488603f * vector.z;
		result.v[3] = -0.488603f * vector.x;

		float3 vectorsquared = vector * vector;
		result.v[4] = 1.092548f * vector.x * vector.y;
		result.v[5] = -1.092548f * vector.y * vector.z;
		result.v[6] = 0.315392f * (3.0f * vectorsquared.z - 1.0f);
		result.v[7] = -1.092548f * vector.x * vector.z;
		result.v[8] = 0.546274f * (vectorsquared.x - vectorsquared.y);

		return result;
	}

	__device__ __host__ SHVector3 operator*(const float& b) const
	{
		SHVector3 result;
		for (int i = 0; i < 9; i++)
			result.v[i] = v[i] * b;

		return result;
	}

	__device__ __host__ SHVector3& operator+=(const SHVector3& rhs)
	{
		for (int i = 0; i < 9; i++)
			v[i] += rhs.v[i];

		return *this;
	}

	__device__ __host__ SHVector3 operator+(const SHVector3& rhs)
	{
		SHVector3 result;
		for (int i = 0; i < 9; i++)
			result.v[i] = v[i] + rhs.v[i];

		return result;
	}

	__device__ __host__ void reset()
	{
		for (int i = 0; i < 9; i++)
			v[i] = 0.0f;
	}

	__device__ __host__ inline SHVector3& addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
	{
		*this += SHVector3::basisFunction(worldSpaceDirection) * (getLuminance(incomingRadiance) * weight);
		return *this;
	}
};

struct SHVectorRGB
{
	SHVector2 r;
	SHVector2 g;
	SHVector2 b;

	__device__ __host__ SHVectorRGB& operator+=(const SHVectorRGB& rhs)
	{
		r += rhs.r;
		g += rhs.g;
		b += rhs.b;
		return *this;
	}

	__device__ __host__ SHVectorRGB operator+(const SHVectorRGB& rhs)
	{
		SHVectorRGB result;
		result.r = r + rhs.r;
		result.g = g + rhs.g;
		result.b = b + rhs.b;
		return result;
	}

	__device__ __host__ SHVectorRGB operator*(const float s) const
	{
		SHVectorRGB result;
		result.r = r*s;
		result.g = g*s;
		result.b = b*s;
		return result;
	}

	__device__ __host__ void addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection);
};

struct SHVectorRGB3
{
	SHVector3 r;
	SHVector3 g;
	SHVector3 b;

	__device__ __host__ SHVectorRGB3& operator+=(const SHVectorRGB3& rhs)
	{
		r += rhs.r;
		g += rhs.g;
		b += rhs.b;
		return *this;
	}

	__device__ __host__ SHVectorRGB3 operator+(const SHVectorRGB3& rhs)
	{
		SHVectorRGB3 result;
		result.r = r + rhs.r;
		result.g = g + rhs.g;
		result.b = b + rhs.b;
		return result;
	}

	__device__ __host__ SHVectorRGB3 operator*(const float s) const
	{
		SHVectorRGB3 result;
		result.r = r*s;
		result.g = g*s;
		result.b = b*s;
		return result;
	}

	__device__ __host__ void addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection);
};

__device__ __host__ inline SHVectorRGB operator*(SHVector2 A, float3 color)
{
	SHVectorRGB result;
	result.r = A * color.x;
	result.g = A * color.y;
	result.b = A * color.z;

	return result;
}

__device__ __host__ inline SHVectorRGB3 operator*(SHVector3 A, float3 color)
{
	SHVectorRGB3 result;
	result.r = A * color.x;
	result.g = A * color.y;
	result.b = A * color.z;

	return result;
}

__device__ __host__ inline void SHVectorRGB::addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
{
	*this += SHVector2::basisFunction(worldSpaceDirection) * (incomingRadiance * weight);
}

__device__ __host__ inline void SHVectorRGB3::addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
{
	*this += SHVector3::basisFunction(worldSpaceDirection) * (incomingRadiance * weight);
}

__align__(16)
struct GatheredLightSample
{
	SHVector2 SHVector;
	float SHCorrection;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float AverageDistance;
	float NumBackfaceHits;

	__device__ __host__ GatheredLightSample& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			SHVector.addIncomingRadiance(Color, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += (Color.x * 0.3f + Color.y * 0.59f + Color.z * 0.11f) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += Color * TangentDirection.z;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample& PointLightWorldSpacePreweighted(const float3 PreweightedColor, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			SHVector.addIncomingRadiance(PreweightedColor, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += getLuminance(PreweightedColor) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += PreweightedColor;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample operator*(float Scalar) const
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector * Scalar;
		Result.SHCorrection = SHCorrection * Scalar;
		Result.IncidentLighting = IncidentLighting * Scalar;
		Result.SkyOcclusion = SkyOcclusion * Scalar;
		Result.AverageDistance = AverageDistance * Scalar;
		Result.NumBackfaceHits = NumBackfaceHits * Scalar;
		return Result;
	}

	__device__ __host__ GatheredLightSample& operator+=(const GatheredLightSample& rhs)
	{
		SHVector += rhs.SHVector;
		SHCorrection += rhs.SHCorrection;
		IncidentLighting += rhs.IncidentLighting;
		SkyOcclusion += rhs.SkyOcclusion;
		AverageDistance += rhs.AverageDistance;
		NumBackfaceHits += rhs.NumBackfaceHits;
		return *this;
	}

	__device__ __host__ GatheredLightSample operator+(const GatheredLightSample& rhs)
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector + rhs.SHVector;
		Result.SHCorrection = SHCorrection + rhs.SHCorrection;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		Result.SkyOcclusion = SkyOcclusion + rhs.SkyOcclusion;
		Result.AverageDistance = AverageDistance + rhs.AverageDistance;
		Result.NumBackfaceHits = NumBackfaceHits + rhs.NumBackfaceHits;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		SHVector.reset();
		IncidentLighting = make_float3(0);
		SkyOcclusion = make_float3(0);
		SHCorrection = 0.0f;
		AverageDistance = 0.0f;
		NumBackfaceHits = 0;
	}
};

struct Ray
{
	float4 origin_tmin;
	float4 dir_tmax;
};

struct Hit
{
	float4 t_triId_u_v;
	//float4 triNormal_instId;
	//float4 geom_normal;
	//float4 texcoord;
};

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

#endif
