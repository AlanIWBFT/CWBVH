template <class T> __device__ __inline__ void swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

__device__ __inline__ float min4(float a, float b, float c, float d)
{
	return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d)
{
	return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

__device__ __inline__ float min3(float a, float b, float c)
{
	return fminf(fminf(a, b), c);
}

__device__ __inline__ float max3(float a, float b, float c)
{
	return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
__inline__ __device__ float fminf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2 < b2 ? a2 : b2);
}

__inline__ __device__ float fmaxf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2 > b2 ? a2 : b2);
}

// Using video instructions
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
//__device__ __inline__ float fmin_fmin(float a, float b, float c) { return fminf(fminf(a, b), c); }
//__device__ __inline__ float fmin_fmax(float a, float b, float c) { return fmaxf(fminf(a, b), c); }
//__device__ __inline__ float fmax_fmin(float a, float b, float c) { return fminf(fmaxf(a, b), c); }
//__device__ __inline__ float fmax_fmax(float a, float b, float c) { return fmaxf(fmaxf(a, b), c); }

__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmin_fmax(a0, a1, d);
	float t2 = fmin_fmax(b0, b1, t1);
	float t3 = fmin_fmax(c0, c1, t2);
	return t3;
}

__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmax_fmin(a0, a1, d);
	float t2 = fmax_fmin(b0, b1, t1);
	float t3 = fmax_fmin(c0, c1, t2);
	return t3;
}

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

// Same for Fermi.
__device__ __inline__ float tMinFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_max7(a0, a1, b0, b1, c0, c1, d); }
__device__ __inline__ float tMaxFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_min7(a0, a1, b0, b1, c0, c1, d); }

__device__ __host__ __inline__ int divideAndRoundup(int i, int divisor) { return (i + divisor - 1) / divisor; }

__device__ __inline__ int sign_extend_s8x4(unsigned int i) { unsigned int v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

__device__ __inline__ int extract_byte(unsigned int i, unsigned int n) { return (i >> (n * 8)) & 0xFF; }
