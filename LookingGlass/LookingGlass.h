#pragma once

#ifdef _DEBUG
#pragma comment(lib, "cudaFunctions_debug.lib")
#endif
#ifndef _DEBUG
#pragma comment(lib, "cudaFunctions.lib")
#endif

#include "globals.h"
#include "DevicesAndShaderscpp.h"
#include "HwndsAndWindowManagement.h"
#include "rtdetr.h"