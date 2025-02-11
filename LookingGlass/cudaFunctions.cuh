#pragma once
#include <cuda.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <device_launch_parameters.h>

#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_SIZE 62
#define MAX_LEVELS 5
#define PHRASE_BUFFER_SIZE 4096   

struct BoxHash {
    char chars[MAX_LEVELS];
};

struct Box {
    float l, t, r, b;
    float score;
    int label;
};

struct Detections {
    std::vector<Box> dets;
};

struct Phrase {
    int index; // index (for detection or bounding box)
    char phrase[PHRASE_BUFFER_SIZE]; // concatenated string of keypoint words (no spaces)
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

