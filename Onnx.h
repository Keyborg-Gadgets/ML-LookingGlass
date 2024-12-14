#pragma once
#include "globals.h"
using namespace Ort;

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


size_t vectorProduct(const std::vector<int64_t>& vector) {
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

vector<vector<float>> bbox_cxcywh_to_xyxy(const vector<vector<float>>& boxes)
{
    vector<vector<float>> xyxy_boxes;
    for (const auto& box : boxes)
    {
        float x1 = box[0] - box[2] / 2.0f;
        float y1 = box[1] - box[3] / 2.0f;
        float x2 = box[0] + box[2] / 2.0f;
        float y2 = box[1] + box[3] / 2.0f;
        xyxy_boxes.push_back({ x1, y1, x2, y2 });
    }
    return xyxy_boxes;
}


bool is_normalized(const std::vector<std::vector<float>>& values) {
    for (const auto& row : values) {
        for (const auto& val : row) {
            if (val <= 0 || val >= 1) {
                return false;
            }
        }
    }
    return true;
}

void normalize_scores(std::vector<std::vector<float>>& scores) {
    for (auto& row : scores) {
        for (auto& val : row) {
            val = 1 / (1 + std::exp(-val));
        }
    }
}

vector<vector<int>> generate_class_colors(int num_classes) {
    vector<vector<int>> class_colors(num_classes, vector<int>(3));
    for (int i = 0; i < num_classes; ++i) {
        class_colors[i][0] = rand() % 256;
        class_colors[i][1] = rand() % 256;
        class_colors[i][2] = rand() % 256;
    }
    return class_colors;
}

void draw_boxes_and_save_image(
    const std::vector<int>& labels,
    const std::vector<float>& scores,
    const std::vector<std::vector<float>>& boxes,
    const std::string& save_path,
    const std::vector<std::string>& CLASS_NAMES,
    cv::Mat& im0
) {
    vector<vector<int>> CLASS_COLORS = generate_class_colors(CLASS_NAMES.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        int label = labels[i];
        float score = scores[i];
        std::ostringstream oss;
        oss << CLASS_NAMES[label] << ": " << std::fixed << std::setprecision(2) << score;
        std::string label_text = oss.str();
        cv::Rect rect((int)boxes[i][0], (int)boxes[i][1], (int)(boxes[i][2] - boxes[i][0]), (int)(boxes[i][3] - boxes[i][1]));
        cv::Scalar color(CLASS_COLORS[label][0], CLASS_COLORS[label][1], CLASS_COLORS[label][2]);
        cv::rectangle(im0, rect, color, 2);
        cv::putText(im0, label_text, cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite(save_path, im0);
}

inline void InitOnnx() {

    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;

    std::string labelPath = exeDir + "/labels.txt";
    std::string modelPath = exeDir + "/rtdetr_r50vd_6x_coco_cvhub.onnx";
    std::string instanceName = "rtdetr";
    size_t deviceId = 0;
    size_t batchSize = 1;
    float confThreshold = 0.3;

    std::vector<std::string> labels = readLabels(labelPath);

    if (labels.empty()) {
        throw std::runtime_error("No labels found!");
    }

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = deviceId;
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    // Sets graph optimization level [Available levels are as below]
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    Ort::Session ortSession(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = ortSession.GetInputCount();
    size_t numOutputNodes = ortSession.GetOutputCount();

    std::vector <std::string> inputNodeNames;
    std::vector <vector <int64_t>> inputNodeDims;
    for (int i = 0; i < numInputNodes; i++) {
        auto inputName = ortSession.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(inputName.get());
        Ort::TypeInfo inputTypeInfo = ortSession.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputDims = inputTensorInfo.GetShape();
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
        inputNodeDims.push_back(inputDims);

        std::cout << "[INFO] Input name and shape is: " << inputName.get() << " [";
        for (size_t j = 0; j < inputDims.size(); j++) {
            std::cout << inputDims[j];
            if (j != inputDims.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }

    std::vector <std::string> outputNodeNames;
    std::vector <vector <int64_t>> outputNodeDims;
    for (int i = 0; i < numOutputNodes; i++) {
        auto outputName = ortSession.GetOutputNameAllocated(i, allocator);
        outputNodeNames.push_back(outputName.get());
        Ort::TypeInfo outputTypeInfo = ortSession.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();

        if (outputDims.at(0) == -1)
        {
            std::cout << "[Warning] Got dynamic batch size. Setting output batch size to "
                << batchSize << "." << std::endl;
            outputDims.at(0) = batchSize;
        }

        outputNodeDims.push_back(outputDims);

        std::cout << "[INFO] Output name and shape is: " << outputName.get() << " [";
        for (size_t j = 0; j < outputDims.size(); j++) {
            std::cout << outputDims[j];
            if (j != outputDims.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }
    std::cout << "[INFO] Model was initialized." << std::endl;

    float* blob = nullptr;
    blob = new float[resizedImageNormRGB.cols * resizedImageNormRGB.rows * resizedImageNormRGB.channels()];
    cv::Size floatImageSize{ resizedImageNormRGB.cols, resizedImageNormRGB.rows };
    std::vector<cv::Mat> chw(resizedImageNormRGB.channels());
    for (int i = 0; i < resizedImageNormRGB.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(resizedImageNormRGB, chw);
    std::cout << "[INFO] [Preprocess] HWC to CHW" << std::endl;

    std::vector<int64_t> inputTensorShape = { 1, 3, inputHeight, inputWidth };

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
        ));
    std::cout << "[INFO] [Preprocess] CHW to NCHW" << std::endl;

    for (const auto& inputNodeName : inputNodeNames) {
        if (std::string(inputNodeName).empty()) {
            std::cerr << "Empty input node name found." << std::endl;
            return 1;
        }
    }

    std::vector<const char*> inputNodeNamesCStr;
    for (const auto& inputName : inputNodeNames) {
        inputNodeNamesCStr.push_back(inputName.c_str());
    }
    std::vector<const char*> outputNodeNamesCStr;
    for (const auto& outputName : outputNodeNames) {
        outputNodeNamesCStr.push_back(outputName.c_str());
    }

    std::vector<Ort::Value> outputTensors = ortSession.Run(
        Ort::RunOptions{ nullptr },
        inputNodeNamesCStr.data(),
        inputTensors.data(),
        inputTensors.size(),
        outputNodeNamesCStr.data(),
        1
    );
    std::cout << "[INFO] [Inference] Successfully!" << std::endl;


    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);
    std::cout << "[INFO] [Postprocess] Get output results" << std::endl;

    int num_boxes = outputShape[1];
    int num_classes = labels.size();
    vector<vector<float>> boxes(num_boxes, vector<float>(4));
    vector<vector<float>> scores(num_boxes, vector<float>(num_classes));
    int score_start_index = 4;
    int score_end_index = 4 + num_classes;
    for (int i = 0; i < num_boxes; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            boxes[i][j] = rawOutput[i * score_end_index + j];
        }
        for (int j = score_start_index; j < score_end_index; ++j)
        {
            scores[i][j - score_start_index] = rawOutput[i * score_end_index + j];
        }
    }
    std::cout << "[INFO] [Postprocess] Extract boxes and scores " << std::endl;

    vector<vector<float>> xyxy_boxes = bbox_cxcywh_to_xyxy(boxes);

    if (!is_normalized(scores)) {
        normalize_scores(scores);
    }

    std::vector<float> max_scores;
    for (const auto& score_row : scores) {
        auto max_score = *std::max_element(score_row.begin(), score_row.end());
        max_scores.push_back(max_score);
    }

    std::vector<bool> mask;
    for (const auto& max_score : max_scores) {
        mask.push_back(max_score > confThreshold);
    }

    std::vector<std::vector<float>> filtered_boxes, filtered_scores;
    for (std::size_t i = 0; i < xyxy_boxes.size(); ++i) {
        if (mask[i]) {
            filtered_boxes.push_back(xyxy_boxes[i]);
            filtered_scores.push_back(scores[i]);
        }
    }

    std::vector<int> filtered_labels;
    std::vector<float> max_filtered_scores;
    for (const auto& score_row : filtered_scores) {
        auto max_score_it = std::max_element(score_row.begin(), score_row.end());
        auto max_score = *max_score_it;
        auto label = std::distance(score_row.begin(), max_score_it);
        filtered_labels.push_back(label);
        max_filtered_scores.push_back(max_score);
    }


    std::vector<float> \
        x1(filtered_boxes.size()), y1(filtered_boxes.size()), \
        x2(filtered_boxes.size()), y2(filtered_boxes.size());
    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = filtered_boxes[i][0];
        y1[i] = filtered_boxes[i][1];
        x2[i] = filtered_boxes[i][2];
        y2[i] = filtered_boxes[i][3];
    }

    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = std::floor(std::min(std::max(1.0f, x1[i] * imageWidth), imageWidth - 1.0f));
        y1[i] = std::floor(std::min(std::max(1.0f, y1[i] * imageHeight), imageHeight - 1.0f));
        x2[i] = std::ceil(std::min(std::max(1.0f, x2[i] * imageWidth), imageWidth - 1.0f));
        y2[i] = std::ceil(std::min(std::max(1.0f, y2[i] * imageHeight), imageHeight - 1.0f));
    }

    std::vector<std::vector<float>> new_boxes(filtered_boxes.size(), std::vector<float>(4));
    for (int i = 0; i < filtered_boxes.size(); i++) {
        new_boxes[i][0] = x1[i];
        new_boxes[i][1] = y1[i];
        new_boxes[i][2] = x2[i];
        new_boxes[i][3] = y2[i];
    }
    filtered_boxes = new_boxes;

    draw_boxes_and_save_image(
        filtered_labels,
        max_filtered_scores,
        filtered_boxes,
        savePath,
        labels,
        imageBGR
    );
    std::cout << "[INFO] [Postprocess] Done! " << std::endl;
}