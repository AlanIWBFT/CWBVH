#pragma once

extern void(*LogMessageWriter)(const wchar_t* message);

void Log(const char* format, ...);
