void GenerateValidationTriangles(int Count, std::vector<float3>& OutVertexBuffer, std::vector<int3>& OutIndexBuffer);
Ray* GenerateValidationRays(int Count);
Ray* GenerateAORaysUniform(
	int NumTexels,
	int NumRaysPerTexel,
	float4* WorldPosition,
	float4* WorldNormal,
	float* TexelRadius
);
