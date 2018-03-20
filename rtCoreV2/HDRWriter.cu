#include <string>
#include "linear_math.h"

void float2rgbe(unsigned char rgbe[4], float red, float green, float blue)
{
	float v;
	int e;

	v = red;
	if (green > v) v = green;
	if (blue > v) v = blue;
	if (v < 1e-32) {
		rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
	}
	else {
		v = frexp(v, &e) * 256.0f / v;
		rgbe[0] = (unsigned char)(red * v);
		rgbe[1] = (unsigned char)(green * v);
		rgbe[2] = (unsigned char)(blue * v);
		rgbe[3] = (unsigned char)(e + 128);
	}
}

void WriteHDR(std::string fileName, const float4* buffer, int Width, int Height)
{
	FILE* outputFile = fopen(fileName.c_str(), "wb");

	fprintf(outputFile, "#?RADIANCE\n"); 
	fprintf(outputFile, "FORMAT=32-bit_rle_rgbe\n\n");
	fprintf(outputFile, "-Y %d +X %d\n", Height, Width);

	for (int Y = Height - 1; Y >= 0; Y--)
	for (int X = 0; X < Width; X++)
	{
		unsigned char rgbe[4];
		float2rgbe(rgbe, buffer[Y * Width + X].x, buffer[Y * Width + X].y, buffer[Y * Width + X].z);
		fwrite(rgbe, sizeof(rgbe), 1, outputFile);
	}
	fclose(outputFile);
}

void WriteNormal(std::string fileName, const float4* buffer, int Width, int Height)
{
	FILE* outputFile = fopen(fileName.c_str(), "wb");

	fprintf(outputFile, "#?RADIANCE\n");
	fprintf(outputFile, "FORMAT=32-bit_rle_rgbe\n\n");
	fprintf(outputFile, "-Y %d +X %d\n", Height, Width);

	for (int Y = Height - 1; Y >= 0; Y--)
		for (int X = 0; X < Width; X++)
		{
			unsigned char rgbe[4];
			float2rgbe(rgbe, 
				(buffer[Y * Width + X].x + 1) / 2, 
				(buffer[Y * Width + X].y + 1) / 2, 
				(buffer[Y * Width + X].z + 1) / 2);
			fwrite(rgbe, sizeof(rgbe), 1, outputFile);
		}
	fclose(outputFile);
}
