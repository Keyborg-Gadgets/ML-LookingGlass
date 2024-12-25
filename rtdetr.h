#ifndef RTDETR_H
#define RTDETR_H
#include "globals.h"
#include <dwrite.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

class glogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            //std::cout << "[WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[INFO] " << msg << std::endl;
            break;
        default:
            std::cout << "[UNKNOWN] " << msg << std::endl;
            break;
        }
    }

    int64_t volume(const nvinfer1::Dims& dims) const {
        int64_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            vol *= dims.d[i];
        }
        return vol;
    }

    int64_t dataTypeSize(nvinfer1::DataType dtype) const {
        switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        default:
            throw std::runtime_error("Unknown data type.");
        }
    }
};

glogger logger;
std::string engine_file;
int batchSize;
int imageWidth;
int imageHeight;

struct Box {
    float l, t, r, b;
    float score;
    int label;
};

struct Detections {
    std::vector<Box> dets;
};

nvinfer1::ICudaEngine* engine;
nvinfer1::IExecutionContext* context;

const std::vector<std::string> coco = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

static ID2D1SolidColorBrush* cocoBrushes[80] = { nullptr };

inline void HSVtoRGB(float h, float s, float v, float* r, float* g, float* b)
{
    while (h < 0.0f)   h += 360.0f;
    while (h >= 360.f) h -= 360.0f;

    int   i = static_cast<int>(h / 60.0f) % 6;
    float f = (h / 60.0f) - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i)
    {
    case 0: *r = v; *g = t; *b = p; break;
    case 1: *r = q; *g = v; *b = p; break;
    case 2: *r = p; *g = v; *b = t; break;
    case 3: *r = p; *g = q; *b = v; break;
    case 4: *r = t; *g = p; *b = v; break;
    case 5: *r = v; *g = p; *b = q; break;
    default:
        *r = 1.0f;
        *g = 0.0f;
        *b = 0.0f;
        break;
    }
}

HRESULT InitCocoBrushes()
{
    if (!d2dContext)
        return E_INVALIDARG;

    for (int i = 0; i < 80; ++i)
    {
        float hue = i * (360.0f / 80.0f);  
        float r, g, b;
        HSVtoRGB(hue, 1.0f, 1.0f, &r, &g, &b);

        D2D1_COLOR_F color = D2D1::ColorF(r, g, b, 1.0f);

        HRESULT hr = d2dContext->CreateSolidColorBrush(color, &cocoBrushes[i]);
        if (FAILED(hr))
        {
            return hr;
        }
    }
    return S_OK;
}

HRESULT DrawDetectionsOnBitmap(const Detections& detections) {
    // Ensure the context target is set as needed, avoiding unnecessary release
    d2dContext->BeginDraw();
    float strokeThickness = 5.0f;

    for (const auto& box : detections.dets) {
        auto brush = cocoBrushes[box.label];

        // Draw the rectangle
        D2D1_RECT_F rect = D2D1::RectF(box.l + (float)xOfWindow, box.t + (float)yOfWindow, 
                                       box.r + (float)xOfWindow, box.b + (float)yOfWindow);
        d2dContext->DrawRectangle(rect, brush, strokeThickness);
    }

    HRESULT hr = d2dContext->EndDraw();
    return hr;
}

void ReadTrtFile() {
    std::string cached_engine;
    std::fstream file;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engine_file, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        std::cout << "read file error: " << engine_file << std::endl;
        cached_engine = "";
        exit(1);
    }

    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();
    trtRuntime = nvinfer1::createInferRuntime(logger);
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
}

void LoadEngine() {
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        ReadTrtFile();
    }
    context = engine->createExecutionContext();  
    assert(context != nullptr);

    int64_t shape0[] = { 1, 300, 80 };
    int64_t shape1[] = { 1, 300, 4 };

    int output0_dims[] = { 1, 300, 80 };
    int output1_dims[] = { 1, 300, 4 };

    size_t output0_size = 1 * 300 * 80 * sizeof(float);
    size_t output1_size = 1 * 300 * 4 * sizeof(float);

    cudaMalloc(&d_output0, output0_size);
    cudaMalloc(&d_output1, output1_size);
}

inline void InitRTdetr() {
    engine_file = exeDir + "/rtdetr_r18vd_6x_coco-fp16.engine";
    batchSize = 1;
    imageWidth = 640;
    imageHeight = 640;
    InitCocoBrushes();
    LoadEngine();
}

#undef min
#undef max
inline Detections CopyAndProcessDetections(float* d_output0, float* d_output1){
    const int maxdet = 300;
    const int numClass = 80;
    const int numCoords = 4;
    float conf_thr = .5;

    std::vector<float> h_output0(maxdet * numClass);
    std::vector<float> h_output1(maxdet * numCoords);

    size_t output0_size = maxdet * numClass * sizeof(float);
    size_t output1_size = maxdet * numCoords * sizeof(float);

    CUDA_CHECK(cudaMemcpy(h_output0.data(), d_output0, output0_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output1.data(), d_output1, output1_size, cudaMemcpyDeviceToHost));

    Detections detections;
    float scaleWidth = static_cast<float>(imgsz) / enginesz; 
    float scaleHeight = static_cast<float>(imgsz) / enginesz;
    for (int position = 0; position < maxdet; ++position) {
        float* bbox = h_output1.data() + position * numCoords; 
        float x = std::clamp(bbox[0], 0.0f, 1.0f) * enginesz * scaleWidth;
        float y = std::clamp(bbox[1], 0.0f, 1.0f) * enginesz * scaleHeight;
        float w = std::clamp(bbox[2], 0.0f, 1.0f) * enginesz * scaleWidth;
        float h = std::clamp(bbox[3], 0.0f, 1.0f) * enginesz * scaleHeight;

        float* class_scores = h_output0.data() + position * numClass;

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
            continue;
        if (std::abs(x - w) > imgsz / 2 || std::abs(y - h) > imgsz / 2)
            continue;
        Box box = { x, y, w, h, best_score, best_class};
        detections.dets.push_back(box);
    }
#ifdef _DEBUG
    for (const auto& box : detections.dets) {
        std::cout << "Class: " << coco[box.label] << " - Score: " << box.score
            << " - Box: (" << box.l << ", " << box.t << ", " << box.r << ", " << box.b << ")\n";
    }
#endif

    cudaMemset(d_output0, 0, output0_size);
    cudaMemset(d_output1, 0, output1_size);

    return detections;
}

#undef min
inline Detections Detect() {
    float* dPlanar = nullptr;
    dPlanar = GetRGBPlanar(cudaTexture, enginesz, enginesz);

    void* gpu_buffers[3];
    gpu_buffers[0] = dPlanar;
    gpu_buffers[1] = d_output0;
    gpu_buffers[2] = d_output1;
    bool success = context->enqueue(batchSize, gpu_buffers, 0, nullptr);
    if (!success) {
        std::cerr << "[ERROR] Enqueue inference failed!\n";
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dPlanar);
    Detections detections = CopyAndProcessDetections(d_output0, d_output1);
    DrawDetectionsOnBitmap(detections);
    return detections;
}
#endif
