#include "orbd.h"
#include <device_launch_parameters.h>

namespace orb
{
#define X1            64
#define X2            32

#define MAX_OCTAVE      5
#define FAST_PATTERN    16
#define HARRIS_SIZE     7
#define MAX_PATCH       31
#define K               (FAST_PATTERN / 2)
#define N               (FAST_PATTERN + K + 1)
#define HARRIS_K        (0.04f)
#define MAX_DIST        64

#define GR              3
#define R2              6
#define R4              12
#define DX              (X2 - R2)

    __constant__ int d_max_num_points;
    __constant__ float d_scale_sq_sq;
    __device__ unsigned int d_point_counter;
    __constant__ int dpixel[25 * MAX_OCTAVE];
    __constant__ unsigned char dthresh_table[512];
    __constant__ int d_umax[MAX_PATCH / 2 + 2];
    __constant__ int2 d_pattern[512];
    __constant__ float d_gauss[GR + 1];
    __constant__ int ofs[HARRIS_SIZE * HARRIS_SIZE];
    __constant__ int angle_param[MAX_OCTAVE * 2];


    void setMaxNumPoints(const int num)
    {
        CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
    }

    void getPointCounter(void** addr)
    {
        CHECK(cudaGetSymbolAddress(addr, d_point_counter));
    }

    void setFastThresholdLUT(int fast_threshold)
    {
        unsigned char hthreshold_tab[512];
        for (int i = -255, j = 0; i <= 255; i++, j++)
            hthreshold_tab[j] = (unsigned char)(i < -fast_threshold ? 1 : i > fast_threshold ? 2 : 0);
        CHECK(cudaMemcpyToSymbol(dthresh_table, hthreshold_tab, 512 * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    }

    void setUmax(const int patch_size)
    {
        int half_patch = patch_size / 2;
        int* h_umax = new int[half_patch + 2];
        h_umax[half_patch + 1] = 0;

        float v = half_patch * sqrtf(2.f) / 2;
        int vmax = (int)floorf(v + 1);
        int vmin = (int)ceilf(v);
        for (int i = 0; i <= vmax; i++)
        {
            h_umax[i] = (int)roundf(sqrtf(half_patch * half_patch - i * i));
        }

        for (int i = half_patch, v0 = 0; i >= vmin; --i)
        {
            while (h_umax[v0] == h_umax[v0 + 1])
                ++v0;
            h_umax[i] = v0;
            ++v0;
        }

        CHECK(cudaMemcpyToSymbol(d_umax, h_umax, sizeof(int) * (half_patch + 2), 0, cudaMemcpyHostToDevice));
        delete[] h_umax;
    }

    void setPattern(const int patch_size, const int wta_k)
    {
        int bit_pattern_31_[256 * 4] = {
            8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13, 2,-13, 2,12,
            1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8, -13,-3, -12,-9,
            10,4, 11,9, -13,-8, -8,-9, -11,7, -9,12, 7,7, 12,6, -4,-5, -3,0,
            -13,2, -12,-3, -9,0, -7,5, 12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8,
            11,-13, 12,-8, 4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2,
            -2,11, -1,-10, -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7, -10,-12, -6,11,
            5,-12, 6,-7, 5,-6, 7,-1, 1,0, 4,-5, 9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4,
            -4,-12, -2,7, -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2,
            -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3, 7,-5, 11,-10, -13,-6, -11,0,
            10,7, 12,1, -6,-3, -6,12, 10,-9, 12,-4, -13,8, -8,-12, -13,0, -8,-4,
            3,3, 7,8, 5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6, 2,-4, 3,-10, -13,0, -13,5,
            -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7, 6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12,
            -12,5, -7,5, 3,-10, 8,-13, -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11, -11,-13, -5,-13,
            -1,6, 0,-1, 5,-3, 5,2, -4,-13, -4,12, -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3,
            7,12, 12,12, -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1, -13,11, -12,5,
            -3,7, -2,-6, 7,1, 8,-6, 1,-1, 3,12, 9,1, 12,6, -1,-9, -1,3, -13,-13, -10,5,
            7,7, 10,12, 12,-5, 12,9, 6,3, 7,11, 5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13,
            9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8, 6,-9, 7,-4, -2,1, -1,-4,
            11,-6, 12,-11, -12,-9, -6,4, 3,7, 7,12, 5,5, 10,8, 0,-4, 2,8, -9,12, -5,-13,
            0,7, 2,12, -1,2, 1,7, 5,11, 7,-9, 3,5, 6,-8, -13,-4, -8,9, -5,9, -3,-3, -4,-7,
            -3,-12, 6,5, 8,0, -7,6, -6,12, -13,6, -5,-2, 1,-10, 3,10, 4,1, 8,-4, -2,-2, 2,-13,
            2,-12, 12,12, -2,-13, 0,-6, 4,1, 9,3, -6,-10, -3,-5, -3,-13, -1,1, 7,5, 12,-11,
            4,-13, 5,-1, -9,9, -4,3, 0,3, 3,-9, -12,1, -6,1, 3,2, 4,-8, -1,4, 0,10, 3,-6, 4,5,
            -13,0, -10,5, 5,8, 12,11, 8,-9, 9,-6, -1,-8, 1,-2, 7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11,
            -10,-10, -5,-7, -10,-8, -8,-13, 4,-6, 8,5, 3,12, 8,-13, -4,2, -3,-3, 5,-13, 10,-12,
            -9,9, -4,3, 0,3, 3,-9, -12,1, -6,1, 3,2, 4,-8, -1,4, 0,10, 3,-6, 4,5, -13,0, -10,5
        };

        const int npoints = 512;
        int2 patternbuf[npoints];
        const int2* pattern0 = (const int2*)bit_pattern_31_;
        if (patch_size != 31)
        {
            pattern0 = patternbuf;
            srand(0x34985739);
            for (int i = 0; i < npoints; i++)
            {
                patternbuf[i].x = rand() % patch_size - patch_size / 2;
                patternbuf[i].y = rand() % patch_size - patch_size / 2;
            }
        }

        if (wta_k == 2)
        {
            CHECK(cudaMemcpyToSymbol(d_pattern, pattern0, npoints * sizeof(int2), 0, cudaMemcpyHostToDevice));
        }
        else
        {
            srand(0x12345678);
            int ntuples = 32 * 4;
            int2* pattern = new int2[ntuples * wta_k];
            for (int i = 0; i < ntuples; i++)
            {
                for (int k = 0; k < wta_k; k++)
                {
                    while (true)
                    {
                        int idx = rand() % npoints;
                        int2 pt = pattern0[idx];
                        int k1;
                        for (k1 = 0; k1 < k; k1++)
                        {
                            int2 pt1 = pattern[wta_k * i + k1];
                            if (pt.x == pt1.x && pt.y == pt1.y)
                                break;
                        }
                        if (k1 == k)
                        {
                            pattern[wta_k * i + k] = pt;
                            break;
                        }
                    }
                }
            }
            CHECK(cudaMemcpyToSymbol(d_pattern, pattern, ntuples * wta_k * sizeof(int2), 0, cudaMemcpyHostToDevice));
            delete[] pattern;
        }
    }

    void setGaussianKernel()
    {
        const float sigma = 2.f;
        const float svar = -1.f / (2.f * sigma * sigma);
        float kernel[GR + 1];
        float kersum = 0.f;
        for (int i = 0; i <= GR; i++)
        {
            kernel[i] = expf(i * i * svar);
            kersum = kersum + (kernel[i] + (i == 0 ? 0 : kernel[i]));
        }
        kersum = 1.f / kersum;
        for (int i = 0; i <= GR; i++)
        {
            kernel[i] *= kersum;
        }
        CHECK(cudaMemcpyToSymbol(d_gauss, kernel, (GR + 1) * sizeof(float), 0, cudaMemcpyHostToDevice));
    }

    void setScaleSqSq()
    {
        float scale = 1.f / (4 * HARRIS_SIZE * 255.f);
        float scale_sq_sq = scale * scale * scale * scale;
        CHECK(cudaMemcpyToSymbol(d_scale_sq_sq, &scale_sq_sq, sizeof(float), 0, cudaMemcpyHostToDevice));
    }

    void setHarrisOffsets(const int pitch)
    {
        static int p = -1;
        if (p != pitch)
        {
            int hofs[HARRIS_SIZE * HARRIS_SIZE];
            for (int i = 0; i < HARRIS_SIZE; i++)
            {
                for (int j = 0; j < HARRIS_SIZE; j++)
                {
                    hofs[i * HARRIS_SIZE + j] = i * pitch + j;
                }
            }
            CHECK(cudaMemcpyToSymbol(ofs, hofs, HARRIS_SIZE * HARRIS_SIZE * sizeof(int), 0, cudaMemcpyHostToDevice));
            p = pitch;
        }
    }

    void makeOffsets(int* pitchs, int noctaves)
    {
#if (FAST_PATTERN == 16)
        const int offsets[16][2] = {
            {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
            {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
        };
#elif (FAST_PATTERN == 12)
        const int offsets[12][2] = {
            {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
            {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
        };
#elif (FAST_PATTERN == 8)
        const int offsets[8][2] = {
            {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
            {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
        };
#endif

        int* hpixel = new int[25 * noctaves];
        int* temp_pixel = hpixel;
        for (int i = 0; i < noctaves; i++)
        {
            int k = 0;
            for (; k < FAST_PATTERN; k++)
                temp_pixel[k] = offsets[k][0] + offsets[k][1] * pitchs[i];
            for (; k < 25; k++)
                temp_pixel[k] = temp_pixel[k - FAST_PATTERN];
            temp_pixel += 25;
        }

        CHECK(cudaMemcpyToSymbol(dpixel, hpixel, 25 * noctaves * sizeof(int), 0, cudaMemcpyHostToDevice));
        delete[] hpixel;
    }

    __global__ void convertRGBToGray(const float* d_rgb, unsigned char* d_gray, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < width && y < height)
        {
            int idx = (y * width + x) * 3;
            float R = d_rgb[idx];
            float G = d_rgb[idx + 1];
            float B = d_rgb[idx + 2];
            float gray = 0.299f * R + 0.587f * G + 0.114f * B;
            d_gray[y * width + x] = (unsigned char)(gray * 255.0f);
        }
    }

    void hConvertRGBToGray(const float* d_rgb, unsigned char* d_gray, int width, int height)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        convertRGBToGray << <grid, block >> > (d_rgb, d_gray, width, height);
        cudaDeviceSynchronize();
    }

    void hFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* vmem, OrbData& result, int* oszp,
        int noctaves, int threshold, int border, bool harris_score)
    {
        if (border < 3)
            border = 3;

        int* osizes = oszp;
        int* widths = osizes + noctaves;
        int* heights = widths + noctaves;
        int* pitchs = heights + noctaves;
        int* offsets = pitchs + noctaves;

        float* vmap = vmem;
        int* layer_map = (int*)(vmap + osizes[0]);

        dim3 block(X2, X2);
        dim3 grid1;
        CHECK(cudaMemcpy(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

        int factor = 1;
        for (int i = 1; i < noctaves; i++)
        {
            grid1.x = (widths[i] + X2 * 4 - 1) / (X2 * 4);
            grid1.y = (heights[i] + X2 - 1) / X2;
            gDownSampleUnroll4 << <grid1, block >> > (image, octave_images + offsets[i], factor, widths[i], heights[i], pitchs[i], pitchs[0]);
            cudaDeviceSynchronize();
            factor++;
        }

        dim3 grid2;
        for (int i = 0; i < noctaves; i++)
        {
            if (harris_score)
            {
                setHarrisOffsets(pitchs[i]);
            }
            grid2.x = (widths[i] - 6 + X2 - 1) / X2;
            grid2.y = (heights[i] - 6 + X2 - 1) / X2;
            gCalcExtramaMap << <grid2, block >> > (octave_images + offsets[i], vmap, layer_map, threshold, i,
                harris_score, widths[i], heights[i], pitchs[i], pitchs[0]);
            cudaDeviceSynchronize();
        }

        // Non-maximum suppression.
        int total_border = border + border;
        dim3 grid3((widths[0] - total_border + X2 * 4 - 1) / (X2 * 4), (heights[0] - total_border + X2 - 1) / X2);
        gNmsUnroll4 << <grid3, block >> > (result.d_data, vmap, layer_map, border, widths[0], heights[0], pitchs[0]);
        cudaDeviceSynchronize();
    }

    void hComputeAngle(unsigned char* octave_images, OrbData& result, int* oszp, int noctaves, int patch_size)
    {
        int* aparams = oszp + noctaves * 3;
        CHECK(cudaMemcpyToSymbol(angle_param, aparams, noctaves * 2 * sizeof(int), 0, cudaMemcpyHostToDevice));

        dim3 block(X1);
        dim3 grid((result.num_pts + X1 - 1) / X1);
        angleIC << <grid, block >> > (octave_images, result.d_data, patch_size / 2, noctaves);
        cudaDeviceSynchronize();
    }

    void hGassianBlur(unsigned char* octave_images, int* oszp, int noctaves)
    {
        int* osizes = oszp;
        int* widths = osizes + noctaves;
        int* heights = widths + noctaves;
        int* pitchs = heights + noctaves;
        int* offsets = pitchs + noctaves;

        dim3 block(X2, X2), grid;
        for (int i = 0; i < noctaves; i++)
        {
            unsigned char* mem = octave_images + offsets[i];
            grid.x = (widths[i] + DX - 1) / DX;
            grid.y = (heights[i] + X2 - 1) / X2;
            gConv2dUnroll << <grid, block >> > (mem, mem, widths[i], heights[i], pitchs[i]);
        }
        cudaDeviceSynchronize();
    }

    void hDescribe(unsigned char* octave_images, OrbData& result, unsigned char* desc, int wta_k, int noctaves)
    {
        dim3 block(X2);
        dim3 grid(result.num_pts);
        gDescrible << <grid, block >> > (octave_images, result.d_data, desc, wta_k, noctaves);
        cudaDeviceSynchronize();
    }

    void hMatch(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2)
    {
        dim3 block(X2);
        dim3 grid(result1.num_pts);
        gHammingMatch << <grid, block >> > (result1.d_data, desc1, desc2, result1.num_pts, result2.num_pts);
        cudaDeviceSynchronize();
    }

} 
