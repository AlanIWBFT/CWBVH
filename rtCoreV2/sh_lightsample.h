#pragma once

struct SHVector2
{
	float v[4] = { 0 };

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
};


struct SHVector3
{
	float v[9] = { 0 };

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

__device__ __host__ inline SHVectorRGB operator*(SHVector2 A, float3 color)
{
	SHVectorRGB result;
	result.r = A * color.x;
	result.g = A * color.y;
	result.b = A * color.z;

	return result;
}


__device__ __host__ inline void SHVectorRGB::addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
{
	*this += SHVector2::basisFunction(worldSpaceDirection) * (incomingRadiance * weight);
}

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

__device__ __host__ inline SHVectorRGB3 operator*(SHVector3 A, float3 color)
{
	SHVectorRGB3 result;
	result.r = A * color.x;
	result.g = A * color.y;
	result.b = A * color.z;

	return result;
}

__device__ __host__ inline void SHVectorRGB3::addIncomingRadiance(const float3& incomingRadiance, float weight, const float3& worldSpaceDirection)
{
	*this += SHVector3::basisFunction(worldSpaceDirection) * (incomingRadiance * weight);
}

__device__ __host__ __inline__ float getLuminance(const float3& v)
{
	return v.x * 0.3f + v.y * 0.59f + v.z * 0.11f;
}

struct GatheredLightSampleSH
{
	SHVectorRGB SHVector;
	float SHCorrection;
	float3 IncidentLighting;
	float3 SkyOcclusion;

	__device__ __host__ GatheredLightSampleSH& PointLightWorldSpacePreweighted(const float3 PreweightedColor, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			float3 UnweightedRadiance = PreweightedColor / TangentDirection.z;
			SHVector.addIncomingRadiance(UnweightedRadiance, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += (UnweightedRadiance.x * 0.3f + UnweightedRadiance.y * 0.59f + UnweightedRadiance.z * 0.11f) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += PreweightedColor;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSampleSH& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
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

	__device__ __host__ GatheredLightSampleSH operator*(float Scalar) const
	{
		GatheredLightSampleSH Result;
		Result.SHVector = SHVector * Scalar;
		Result.SHCorrection = SHCorrection * Scalar;
		Result.IncidentLighting = IncidentLighting * Scalar;
		Result.SkyOcclusion = SkyOcclusion * Scalar;
		return Result;
	}

	__device__ __host__ GatheredLightSampleSH& operator+=(const GatheredLightSampleSH& rhs)
	{
		SHVector += rhs.SHVector;
		SHCorrection += rhs.SHCorrection;
		IncidentLighting += rhs.IncidentLighting;
		SkyOcclusion += rhs.SkyOcclusion;
		return *this;
	}

	__device__ __host__ GatheredLightSampleSH operator+(const GatheredLightSampleSH& rhs)
	{
		GatheredLightSampleSH Result;
		Result.SHVector = SHVector + rhs.SHVector;
		Result.SHCorrection = SHCorrection + rhs.SHCorrection;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		Result.SkyOcclusion = SkyOcclusion + rhs.SkyOcclusion;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		SHVector.r.reset();
		SHVector.g.reset();
		SHVector.b.reset();
		IncidentLighting = make_float3(0);
		SkyOcclusion = make_float3(0);
		SHCorrection = 0.0f;
	}
};

struct GatheredLightSampleFloat
{
	float IncidentLighting;

	__device__ __host__ GatheredLightSampleFloat& PointLightWorldSpacePreweighted(const float3 PreweightedColor, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			IncidentLighting += PreweightedColor.x;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSampleFloat& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			IncidentLighting += Color.x * TangentDirection.z;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSampleFloat operator*(float Scalar) const
	{
		GatheredLightSampleFloat Result;
		Result.IncidentLighting = IncidentLighting * Scalar;
		return Result;
	}

	__device__ __host__ GatheredLightSampleFloat& operator+=(const GatheredLightSampleFloat& rhs)
	{
		IncidentLighting += rhs.IncidentLighting;
		return *this;
	}

	__device__ __host__ GatheredLightSampleFloat operator+(const GatheredLightSampleFloat& rhs)
	{
		GatheredLightSampleFloat Result;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		IncidentLighting = 0.0f;
	}
};

#define GatheredLightSample GatheredLightSampleFloat
