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
std::string onnx_file;
int batchSize;
int imageWidth;
int imageHeight;

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



void BuildTrtFile(const std::string& onnxFile, const std::string& engineFile)
{
    glogger gLogger;

    // Create builder
    std::unique_ptr<nvinfer1::IBuilder, void(*)(nvinfer1::IBuilder*)> builder(
        nvinfer1::createInferBuilder(gLogger),
        [](nvinfer1::IBuilder* p) { p->destroy(); }
    );
    if (!builder)
    {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return;
    }

    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );
    std::unique_ptr<nvinfer1::INetworkDefinition, void(*)(nvinfer1::INetworkDefinition*)> network(
        builder->createNetworkV2(explicitBatch),
        [](nvinfer1::INetworkDefinition* p) { p->destroy(); }
    );
    if (!network)
    {
        std::cerr << "Failed to create TensorRT network." << std::endl;
        return;
    }

    // Create ONNX parser
    std::unique_ptr<nvonnxparser::IParser, void(*)(nvonnxparser::IParser*)> parser(
        nvonnxparser::createParser(*network, gLogger),
        [](nvonnxparser::IParser* p) { p->destroy(); }
    );
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser." << std::endl;
        return;
    }

    // Parse ONNX
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "Failed to parse ONNX file: " << onnxFile << std::endl;
        return;
    }

    // Create BuilderConfig
    std::unique_ptr<nvinfer1::IBuilderConfig, void(*)(nvinfer1::IBuilderConfig*)> config(
        builder->createBuilderConfig(),
        [](nvinfer1::IBuilderConfig* p) { p->destroy(); }
    );
    if (!config)
    {
        std::cerr << "Failed to create builder config." << std::endl;
        return;
    }
    config->setMaxWorkspaceSize(1ULL << 30); // e.g. 1GB
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // optional, if your GPU supports it

    //------------------------------------------------------------------------------
    // Force the input to a static shape of [1 x 300 x 80] using an optimization profile
    //------------------------------------------------------------------------------

    // We assume the network has exactly one input, but adjust if you have more.
    nvinfer1::ITensor* input = network->getInput(0);
    if (!input)
    {
        std::cerr << "Network has no inputs? Aborting." << std::endl;
        return;
    }

    // Create an optimization profile so that TensorRT knows to treat the input
    // as having a fixed shape of [1 x 300 x 80].
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile)
    {
        std::cerr << "Failed to create optimization profile." << std::endl;
        return;
    }

    // The input name must match the ONNX input tensor name.
    // You can verify it via `std::cout << input->getName() << std::endl;`
    const char* inputName = input->getName();

    // Set min, opt, max to the same shape => effectively "static"
    profile->setDimensions(
        inputName,
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4(1, 3, enginesz, enginesz)
    );
    profile->setDimensions(
        inputName,
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(1, 3, enginesz, enginesz)
    );
    profile->setDimensions(
        inputName,
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(1, 3, enginesz, enginesz)
    );

    // Add the profile to the config
    config->addOptimizationProfile(profile);

    //------------------------------------------------------------------------------
    // Build the engine
    //------------------------------------------------------------------------------

    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine(
        builder->buildEngineWithConfig(*network, *config),
        [](nvinfer1::ICudaEngine* p) { p->destroy(); }
    );
    if (!engine)
    {
        std::cerr << "Engine building failed." << std::endl;
        return;
    }

    // Serialize engine to memory
    std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)> serializedEngine(
        engine->serialize(),
        [](nvinfer1::IHostMemory* p) { p->destroy(); }
    );
    if (!serializedEngine)
    {
        std::cerr << "Failed to serialize engine." << std::endl;
        return;
    }

    // Write engine to file
    std::ofstream engineOutputFile(engineFile, std::ios::binary);
    if (!engineOutputFile)
    {
        std::cerr << "Cannot open file to write engine: " << engineFile << std::endl;
        return;
    }
    engineOutputFile.write(
        static_cast<const char*>(serializedEngine->data()),
        serializedEngine->size()
    );
    engineOutputFile.close();

    std::cout << "Engine file created: " << engineFile << std::endl;
}

bool fileExists(const std::string& filename)
{
    return std::filesystem::exists(std::filesystem::path(filename));
}



void ReadTrtFile() {
    std::string cached_engine;
    std::fstream file;
    nvinfer1::IRuntime *trtRuntime;

    file.open(engine_file, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        std::cout << "read file error: " << engine_file << std::endl;
        cached_engine = "";
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
    if (!fileExists(engine_file))
        BuildTrtFile(onnx_file, engine_file);

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
    onnx_file = exeDir + "/modified_out.sim.onnx";
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


    size_t output0_size = maxdet * numClass * sizeof(float);
    size_t output1_size = maxdet * numCoords * sizeof(float);

//#ifdef _DEBUG
//    for (const auto& box : detections.dets) {
//        std::cout << "Class: " << coco[box.label] << " - Score: " << box.score
//            << " - Box: (" << box.l << ", " << box.t << ", " << box.r << ", " << box.b << ")\n";
//    }
//#endif
    Detections dets = KCuda::processDetections(
        d_output0,        // Device pointer to class scores
        d_output1,        // Device pointer to bounding box coordinates
        maxdet,              // Maximum number of detections
        numClass,            // Number of classes
        numCoords,           // Number of coordinates per box
        conf_thr,          // Confidence threshold
        imgsz,               // Original image size
        enginesz             // Engine input size
    );
    cudaMemset(d_output0, 0, output0_size);
    cudaMemset(d_output1, 0, output1_size);
    return dets;
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
