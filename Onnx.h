#pragma once
#include "globals.h"
#include "cudaFunctions.cuh"

bool bound = false;

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
        rgbCudaArray,
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
    if (!bound) {
        int64_t shape0[] = { 1, 300, 80 };
        int64_t shape1[] = { 1, 300, 4 };

        int output0_dims[] = { 1, 300, 80 };
        int output1_dims[] = { 1, 300, 4 };

        size_t output0_size = 1 * 300 * 80 * sizeof(float);
        size_t output1_size = 1 * 300 * 4 * sizeof(float);

        cudaMalloc(&d_output0, output0_size);
        cudaMalloc(&d_output1, output1_size);

        status = api->CreateTensorWithDataAsOrtValue(info_cuda, d_output0, output0_size, 
            shape0, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &scoresValue);
        if (status != nullptr) {
            std::cerr << "failed scores tensor" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }
        status = api->CreateTensorWithDataAsOrtValue(info_cuda, d_output1, output1_size,
            shape1, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &boxesValue);
        if (status != nullptr) {
            std::cerr << "failed boxes tensor" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }

        status = api->CreateIoBinding(session, &io_binding);
        if (status != nullptr) {
            std::cerr << "bad input binding" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }
      
        status = api->BindInput(io_binding, "image", cudaTextureOrt);
        if (status != nullptr) {
            std::cerr << "bad binding input" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }
        status = api->BindOutput(io_binding, "boxes", boxesValue);
        if (status != nullptr) {
            std::cerr << "bad binding output boxes" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }
        status = api->BindOutput(io_binding, "scores", scoresValue);
        if (status != nullptr) {
            std::cerr << "bad output boxes" << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
            cleanup();
            exit(1);
        }
        bound = true;
    }

    std::vector<const char*> outputNames = {"scores", "boxes"};
    std::vector<const char*> inputNames = { "image" };
    std::vector<OrtValue*> inputTensors = { cudaTextureOrt };
    std::vector<OrtValue*> outputTensors = { scoresValue , boxesValue };
    status = api->RunWithBinding(session, Ort::RunOptions(), io_binding);
    if (status != nullptr) {
        std::cerr << "Run failed: " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }
    else {
        std::cout << "doot";
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

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = deviceId;

    status =  api->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options);
    if (status != nullptr) {
        std::cerr << "Error appending provider: " << api->GetErrorMessage(status) << std::endl;
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

    status = api->CreateAllocator(session, info_cuda, &allocator);
    if (status != nullptr) {
        std::cerr << "failed to get allocator " << api->GetErrorMessage(status) << std::endl;
        api->ReleaseStatus(status);
        cleanup();
        exit(1);
    }

    std::cout << "[INFO] Model was initialized." << std::endl;
}