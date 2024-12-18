#pragma once
#include "globals.h"
#include "cudaFunctions.cuh"

std::vector<std::string> readLabels(const std::string& labelPath) {
    std::vector<std::string> labels;
    std::ifstream infile(labelPath);
    std::string line;
    while (std::getline(infile, line)) {
        labels.push_back(line);
    }
    infile.close();
    return labels;
}

wchar_t* stringToWcharT(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wideStr = converter.from_bytes(str);
    wchar_t* wideCStr = new wchar_t[wideStr.size() + 1];
    std::copy(wideStr.begin(), wideStr.end(), wideCStr);
    wideCStr[wideStr.size()] = L'\0';
    return wideCStr;
}

float* TextureToCuda(ID3D11Texture2D* texture, unsigned int width, unsigned int height) {
    cudaArray* mappedArray;
    cudaGraphicsMapResources(1, &cudaResource);
    cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResource, 0, 0);
    auto rgbCudaArray = KCuda::RemoveAlphaAndFlip(mappedArray, width, height);
    cudaGraphicsUnmapResources(1, &cudaResource);
    cudaFree(mappedArray);

    return rgbCudaArray;
}

void CreateOnnxValueFromTexture(ID3D11Texture2D* texture) {
    D3D11_TEXTURE2D_DESC d;
    texture->GetDesc(&d);
    unsigned int width = d.Width;
    unsigned int height = d.Height;
    rgbCudaArray = TextureToCuda(texture, width, height);
    int64_t shape[] = { 1, 3, height, width };
    size_t shape_len = sizeof(shape) / sizeof(shape[0]);
    size_t p_data_len = sizeof(float) * 1 * 3 * height * width;

    status = api->CreateTensorWithDataAsOrtValue(info_cuda,
        reinterpret_cast<float*>(rgbCudaArray),
        p_data_len,
        shape,
        shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &cudaTextureOrt
    );
    if (status != nullptr) {
        std::cerr << "Error creating tensor: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        exit(1);
    }
}

void UpdateOnnxValueFromTexture(ID3D11Texture2D* texture) {
    D3D11_TEXTURE2D_DESC d;
    texture->GetDesc(&d);
    unsigned int width = d.Width;
    unsigned int height = d.Height;
    cudaArray* mappedArray;
    cudaGraphicsMapResources(1, &cudaResource);
    cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResource, 0, 0);
    KCuda::UpdateArray(mappedArray, rgbCudaArray, width, height);
    cudaGraphicsUnmapResources(1, &cudaResource);
    cudaFree(mappedArray);
}

void Detect() {
    OrtAllocator* allocator;
    status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        std::cerr << "failed to get allocator" << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }
    size_t numInputNodes, numOutputNodes;
    status = api->SessionGetInputCount(session, &numInputNodes);
    if (status != nullptr) {
        std::cerr << "bad inputs" << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }
    status = api->SessionGetOutputCount(session, &numOutputNodes);
    if (status != nullptr) {
        std::cerr << "bad outputs" << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }
    std::vector<char*> inputNames(numInputNodes);
    std::vector<OrtValue*> inputTensors(numInputNodes);

    for (size_t i = 0; i < numInputNodes; ++i) {
        status = api->SessionGetInputName(session, i, allocator, &inputNames[i]);
        inputTensors[i] = cudaTextureOrt;
    }

    std::vector<char*> outputNames(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; ++i) {
        status = api->SessionGetOutputName(session, i, allocator, &outputNames[i]);
    }
    std::vector<OrtValue*> outputTensors(numOutputNodes);

    std::cout << "here\n";

    status = api->Run(session, nullptr, inputNames.data(), inputTensors.data(), numInputNodes,
        outputNames.data(), numOutputNodes, outputTensors.data());
    if (status != nullptr) {
        std::cerr << "fail detecting" << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }

    for (size_t i = 0; i < numOutputNodes; ++i) {
        OrtTensorTypeAndShapeInfo* info;
        status = api->GetTensorTypeAndShape(outputTensors[i], &info);

        ONNXTensorElementDataType type;
        status = api->GetTensorElementType(info, &type);
        size_t numDims;
        status = api->GetDimensionsCount(info, &numDims);
        std::vector<int64_t> outputDims(numDims);
        status = api->GetDimensions(info, outputDims.data(), numDims);

        std::cout << "Output " << i << " shape: [";
        for (size_t j = 0; j < numDims; ++j) {
            std::cout << outputDims[j] << (j < numDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        api->ReleaseTensorTypeAndShapeInfo(info);
    }
}

void InitializeOnnx() {
    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;

    std::string labelPath = exeDir + "/labels.txt";
    std::string modelPath = exeDir + "/rtdetr_r18vd_6x_coco-modify.onnx";
    std::string instanceName = "rtdetr-r18";
    size_t deviceId = 0;
    size_t batchSize = 1;
    float confThreshold = 0.3;
    std::vector<std::string> labels = readLabels(labelPath);
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instanceName.c_str(), &env);
    if (status != nullptr) {
        std::cerr << "Error creating environment: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        exit(1);
    }

    OrtSessionOptions* session_options = nullptr;
    status = api->CreateSessionOptions(&session_options);
    if (status != nullptr) {
        std::cerr << "Error creating session options: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        api->ReleaseEnv(env);
        exit(1);
    }

    api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
    api->SetIntraOpNumThreads(session_options, 1);

    status = api->CreateSession(env, stringToWcharT(modelPath), session_options, &session);
    if (status != nullptr) {
        std::cerr << "Error creating session: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        api->ReleaseSessionOptions(session_options);
        api->ReleaseEnv(env);
        exit(1);
    }

    const char* allocator_name = "Cuda";
    OrtAllocatorType allocator_type = OrtArenaAllocator;
    int device_id = 0; 
    OrtMemType mem_type = OrtMemTypeDefault;

    status = api->CreateMemoryInfo(allocator_name, allocator_type, device_id, mem_type, &info_cuda);
    if (status != nullptr) {
        std::cerr << "Error creating memory info: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        exit(1);
    }

    std::cout << "[INFO] Model was initialized." << std::endl;
}