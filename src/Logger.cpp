#include <cstdarg>
#include <cstdio>
#include <string>
#include "Logger.h"

void Log(const char* format, ...)
{
	va_list argptr;
	va_start(argptr, format);
	vprintf((std::string(format) + "\n").c_str(), argptr);
	va_end(argptr);
}
