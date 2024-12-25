#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <device_launch_parameters.h>

namespace KCuda {
	float* TextureToRGBPlanar(const cudaArray* Mapped2dTexture, int width, int height);
}

