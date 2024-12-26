#pragma once
#include <cuda.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <device_launch_parameters.h>

struct Box {
    float l, t, r, b;
    float score;
    int label;
};

struct Detections {
    std::vector<Box> dets;
};

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
    );
	float* TextureToRGBPlanar(const cudaArray* Mapped2dTexture, int width, int height);
}

