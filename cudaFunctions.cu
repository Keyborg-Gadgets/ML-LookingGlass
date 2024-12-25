#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(_err)              \
                      << " (error code " << _err << ") at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                   \
            return nullptr;                                                       \
        }                                                                          \
    } while (0)

__global__
void bgra_to_rgb_planar_normalized(const unsigned char* __restrict__ bgra,
    float* __restrict__ dst,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float B = static_cast<float>(bgra[idx * 4 + 0]) / 255.0f;
    float G = static_cast<float>(bgra[idx * 4 + 1]) / 255.0f;
    float R = static_cast<float>(bgra[idx * 4 + 2]) / 255.0f;

    int planeSize = width * height;
    dst[0 * planeSize + idx] = R;  
    dst[1 * planeSize + idx] = G;
    dst[2 * planeSize + idx] = B;
}

namespace KCuda {
    float* TextureToRGBPlanar(
        const cudaArray* Mapped2dTexture,
        int width,
        int height
    ) {
        size_t bgraSize = static_cast<size_t>(width) * height * 4 * sizeof(unsigned char);
        unsigned char* d_bgra = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_bgra), bgraSize);

        cudaMemcpy2DFromArray(
            d_bgra,                             // dst pointer in device memory
            width * 4 * sizeof(unsigned char),  // dpitch (bytes in each row of dst)
            Mapped2dTexture,                    // source cudaArray
            0,                                  // wOffset in array
            0,                                  // hOffset in array
            width * 4 * sizeof(unsigned char),  // width in bytes to copy per row
            height,                             // height (number of rows)
            cudaMemcpyDeviceToDevice            // both in device memory
        );

        size_t rgbPlanarSize = static_cast<size_t>(width) * height * 3 * sizeof(float);
        float* d_rgbPlanar = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_rgbPlanar), rgbPlanarSize);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        bgra_to_rgb_planar_normalized << <grid, block >> > (d_bgra, d_rgbPlanar, width, height);
        cudaDeviceSynchronize();  

        cudaFree(d_bgra);

        return d_rgbPlanar;
    }

} 
