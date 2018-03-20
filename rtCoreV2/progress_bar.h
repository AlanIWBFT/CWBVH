#pragma once
static void progress_bar(float v, int len = 50)
{
	printf("\r");
	printf("[");
	for (int i = 0; i < roundf(v*len); i++)
		printf(">");
	for (int i = roundf(v*len); i < len; i++)
		printf("-");
	printf("]");
}
