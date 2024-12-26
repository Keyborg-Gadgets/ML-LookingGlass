#include "cudaFunctions.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(_err)              \
                      << " (error code " << _err << ") at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                   \
            break;                                                       \
        }                                                                          \
    } while (0)

#define THREADS_PER_BLOCK_NMS 64

__device__ float IoU(const Box & a, const Box & b) {
    float x1 = fmaxf(a.l, b.l);
    float y1 = fmaxf(a.t, b.t);
    float x2 = fminf(a.r, b.r);
    float y2 = fminf(a.b, b.b);

    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_a = (a.r - a.l) * (a.b - a.t);
    float area_b = (b.r - b.l) * (b.b - b.t);

    float union_area = area_a + area_b - intersection + 1e-5f;
    return intersection / union_area;
}

__global__ void nmsKernel(Box * boxes, int num_boxes, float iou_threshold, int max_output_boxes_per_class) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    Box current_box = boxes[idx];
    int is_suppressed = 0;

    for (int i = 0; i < num_boxes; ++i) {
        if (i == idx) continue;
        if (boxes[i].label != current_box.label) continue;

        if (boxes[i].score > current_box.score) {
            float iou = IoU(current_box, boxes[i]);
            if (iou > iou_threshold) {
                is_suppressed = 1;
                break;
            }
        }
        else if (boxes[i].score == current_box.score && i < idx) {
            float iou = IoU(current_box, boxes[i]);
            if (iou > iou_threshold) {
                is_suppressed = 1;
                break;
            }
        }
    }

    if (is_suppressed) {
        boxes[idx].score = -1.0f;
    }
}

__global__ void processDetectionsKernel(
    float* d_output0, float* d_output1, Box * d_boxes, int* d_box_count,
    int maxdet, int numClass, int numCoords,
    float conf_thr, float scaleWidth, float scaleHeight,
    int imgsz, int enginesz
) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= maxdet) return;

    float* bbox = d_output1 + position * numCoords;
    float l = fminf(fmaxf(bbox[0], 0.0f), 1.0f) * enginesz * scaleWidth;
    float t = fminf(fmaxf(bbox[1], 0.0f), 1.0f) * enginesz * scaleHeight;
    float r = fminf(fmaxf(bbox[2], 0.0f), 1.0f) * enginesz * scaleWidth;
    float b = fminf(fmaxf(bbox[3], 0.0f), 1.0f) * enginesz * scaleHeight;

    float* class_scores = d_output0 + position * numClass;

    int best_class = -1;
    float best_score = -1.0f;
    for (int class_id = 0; class_id < numClass; ++class_id) {
        float score = class_scores[class_id];
        if (score > best_score) {
            best_score = score;
            best_class = class_id;
        }
    }

    if (best_score < conf_thr)
        return;

    int index = atomicAdd(d_box_count, 1);

    d_boxes[index] = { l, t, r, b, best_score, best_class};
}


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
    Detections processDetections(
        float* d_output0,        // Device pointer to class scores
        float* d_output1,        // Device pointer to bounding box coordinates
        int maxdet,              // Maximum number of detections
        int numClass,            // Number of classes
        int numCoords,           // Number of coordinates per box
        float conf_thr,          // Confidence threshold
        int imgsz,               // Original image size
        int enginesz             // Engine input size
    ) {
        // Scaling factors
        float scaleWidth = static_cast<float>(imgsz) / enginesz;
        float scaleHeight = static_cast<float>(imgsz) / enginesz;

        // Allocate device memory for boxes and box count
        Box* d_boxes;
        int* d_box_count;
        CUDA_CHECK(cudaMalloc((void**)&d_boxes, maxdet * sizeof(Box)));
        CUDA_CHECK(cudaMalloc((void**)&d_box_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_box_count, 0, sizeof(int)));

        // Set up kernel launch parameters
        int threadsPerBlock = 256;
        int blocksPerGrid = (maxdet + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel to process detections
        processDetectionsKernel << <blocksPerGrid, threadsPerBlock >> > (
            d_output0, d_output1, d_boxes, d_box_count,
            maxdet, numClass, numCoords,
            conf_thr, scaleWidth, scaleHeight,
            imgsz, enginesz
            );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy the number of boxes back to host
        int h_box_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_box_count == 0) {
            // No detections above the confidence threshold
            Detections detections;
            // Free device memory
            cudaFree(d_boxes);
            cudaFree(d_box_count);
            return detections;
        }

        // Now, run NMS on the device
        float iou_threshold = 0.5f;  // Adjust IoU threshold as needed
        int max_output_boxes_per_class = maxdet; // You can adjust as needed

        // Set up NMS kernel launch parameters
        threadsPerBlock = THREADS_PER_BLOCK_NMS;
        blocksPerGrid = (h_box_count + threadsPerBlock - 1) / threadsPerBlock;

        nmsKernel << <blocksPerGrid, threadsPerBlock >> > (
            d_boxes, h_box_count, iou_threshold, max_output_boxes_per_class
            );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy the boxes back to host
        std::vector<Box> boxes(h_box_count);
        CUDA_CHECK(cudaMemcpy(boxes.data(), d_boxes, h_box_count * sizeof(Box), cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_boxes);
        cudaFree(d_box_count);

        // Filter out suppressed boxes (score set to -1.0f)
        Detections detections;
        for (const auto& box : boxes) {
            if (box.score >= 0.0f) {
                detections.dets.push_back(box);
            }
        }

        return detections;
    }

    float* TextureToRGBPlanar(
        const cudaArray* Mapped2dTexture,
        int width,
        int height
    ) {
        size_t bgraSize = static_cast<size_t>(width) * height * 4 * sizeof(unsigned char);
        unsigned char* d_bgra = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bgra), bgraSize));

        CUDA_CHECK(cudaMemcpy2DFromArray(
            d_bgra,                             // dst pointer in device memory
            width * 4 * sizeof(unsigned char),  // dpitch (bytes in each row of dst)
            Mapped2dTexture,                    // source cudaArray
            0,                                  // wOffset in array
            0,                                  // hOffset in array
            width * 4 * sizeof(unsigned char),  // width in bytes to copy per row
            height,                             // height (number of rows)
            cudaMemcpyDeviceToDevice            // both in device memory
        ));

        size_t rgbPlanarSize = static_cast<size_t>(width) * height * 3 * sizeof(float);
        float* d_rgbPlanar = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgbPlanar), rgbPlanarSize));

        dim3 block(32, 32);
        dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        bgra_to_rgb_planar_normalized << <grid, block >> > (d_bgra, d_rgbPlanar, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_bgra);

        return d_rgbPlanar;
    }
}
