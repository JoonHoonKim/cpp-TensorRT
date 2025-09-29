#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>   // __half, __half2float
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

using namespace nvinfer1;

union HalfBits { __half h; uint16_t u; };

// -------------------- Logger --------------------
class Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kINFO) std::cout << "[TRT] " << msg << std::endl;
    }
};

#define CHECK_CUDA(x) \
    do { cudaError_t err = (x); if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
        std::abort(); }} while(0)

// -------------------- Detection struct ----------
struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

// -------------------- IoU + NMS -----------------
static float IoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int iw = std::max(0, x2 - x1);
    int ih = std::max(0, y2 - y1);
    int inter = iw * ih;
    int uni = std::max(1, a.area() + b.area() - inter);
    return static_cast<float>(inter) / static_cast<float>(uni);
}

static std::vector<Detection> NMS(const std::vector<Detection>& dets, float nms_thr) {
    std::vector<Detection> sorted = dets;
    std::sort(sorted.begin(), sorted.end(),
        [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<Detection> keep;
    std::vector<char> removed(sorted.size(), 0);
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (!removed[j] && IoU(sorted[i].box, sorted[j].box) > nms_thr) removed[j] = 1;
        }
    }
    return keep;
}

// -------------------- Helper --------------------
static int64_t volume(const Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

// -------------------- Main ----------------------
int main(int argc, char** argv) {
    std::string enginePath;
    std::string imagePath;

    if (argc == 3) {
        enginePath = argv[1];
        imagePath = argv[2];
    }
    else {
        // 기본 경로
        enginePath = "./model/yolov5s_fp16.engine";
        imagePath = "./images/test.jpg";
    }

    std::filesystem::path p(enginePath);
    std::string filename = p.stem().string(); // "yolov5s_fp16"

    // '_' 기준으로 잘라서 마지막 토큰 추출
    size_t pos = filename.find_last_of('_');
    std::string d_type = (pos != std::string::npos) ? filename.substr(pos + 1) : filename;

    const int INPUT_W = 1280, INPUT_H = 640, NUM_CLASSES = 80;
    const float CONF_THR = 0.25f, NMS_THR = 0.45f;

    Logger logger;

    // 엔진 로드
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) { std::cerr << "Failed to open engine\n"; return -1; }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg(); file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), size));
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    // IO 텐서
    const char* inName = engine->getIOTensorName(0);
    const char* outName = engine->getIOTensorName(1);

    // 입력 준비
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) { std::cerr << "Image load failed\n"; return -1; }

    cv::Mat resized; cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1 / 255.0);

    // fp32, INT8
    //std::vector<float> inputHost(3 * INPUT_H * INPUT_W);
    //int idx = 0;
    //for (int c = 0; c < 3; c++) for (int y = 0; y < INPUT_H; y++) for (int x = 0; x < INPUT_W; x++)
    //    inputHost[idx++] = resized.at<cv::Vec3f>(y, x)[c];
    
    // fp16
    std::vector<uint16_t> inputHost(3 * INPUT_H * INPUT_W);
    int idx = 0;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < INPUT_H; y++)
            for (int x = 0; x < INPUT_W; x++) {
                float v = resized.at<cv::Vec3f>(y, x)[c]; // CV_32F로 읽음
                HalfBits hb; hb.h = __float2half(v);
                inputHost[idx++] = hb.u;
            }


    Dims4 inShape{ 1,3,INPUT_H,INPUT_W };
    context->setInputShape(inName, inShape);

    // 출력 shape/dtype
    Dims outDims = context->getTensorShape(outName);
    int num_preds = outDims.d[1];
    int kStride = outDims.d[2];
    int64_t outElems = volume(outDims);
    auto dtype = engine->getTensorDataType(outName);

    size_t elemSize = (dtype == DataType::kFLOAT ? 4 : (dtype == DataType::kHALF ? 2 : 0));
    size_t outputBytes = outElems * elemSize;

    //std::cout << "outElems=" << outElems << " elemSize=" << elemSize << " outputBytes=" << outputBytes << std::endl;

    // GPU 메모리
    void* dInput = nullptr; void* dOutput = nullptr;

    //fp32, INT8
    //CHECK_CUDA(cudaMalloc(&dInput, inputHost.size() * sizeof(float)));

    //fp16
    CHECK_CUDA(cudaMalloc(&dInput, inputHost.size() * sizeof(uint16_t)));

    CHECK_CUDA(cudaMalloc(&dOutput, outputBytes));
    cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));

    context->setTensorAddress(inName, dInput);
    context->setTensorAddress(outName, dOutput);

    // H2D
    // fp32, INT8
    //CHECK_CUDA(cudaMemcpyAsync(dInput, inputHost.data(),
    //    inputHost.size() * sizeof(float),
    //    cudaMemcpyHostToDevice, stream));
    
    // fp16
    CHECK_CUDA(cudaMemcpyAsync(dInput, inputHost.data(),
        inputHost.size() * sizeof(uint16_t),
        cudaMemcpyHostToDevice, stream));

    // 실행
    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n"; return -1;
    }

    // 출력 가져오기 (FP16/FP32 대응)
    std::vector<float> outputHostFloat(outElems);
    if (dtype == DataType::kFLOAT) {
        CHECK_CUDA(cudaMemcpyAsync(outputHostFloat.data(), dOutput,
            outputBytes, cudaMemcpyDeviceToHost, stream));
    }
    else if (dtype == DataType::kHALF) {
        std::vector<uint16_t> outputHostHalf(outElems);
        CHECK_CUDA(cudaMemcpyAsync(outputHostHalf.data(), dOutput,
            outputBytes, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < outElems; i++) {
            __half h = *reinterpret_cast<__half*>(&outputHostHalf[i]);
            outputHostFloat[i] = __half2float(h);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 값 확인
    //for (int i = 0; i < 30; i++) {
    //    std::cout << outputHostFloat[i] << " ";
    //}
    //std::cout << std::endl;

    // ---------------- YOLO 디코딩 ----------------
    std::vector<Detection> dets;
    for (int i = 0; i < num_preds; ++i) {
        const float* p = &outputHostFloat[i * kStride];
        float obj = p[4];
        if (obj < CONF_THR) continue;

        int clsId = -1; float clsConf = 0.f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float v = p[5 + c];
            if (v > clsConf) { clsConf = v; clsId = c; }
        }
        float conf = obj * clsConf;
        if (conf < CONF_THR) continue;

        float cx = p[0], cy = p[1], w = p[2], h = p[3];
        int x = static_cast<int>((cx - w / 2.f) * img.cols / INPUT_W);
        int y = static_cast<int>((cy - h / 2.f) * img.rows / INPUT_H);
        int ww = static_cast<int>(w * img.cols / INPUT_W);
        int hh = static_cast<int>(h * img.rows / INPUT_H);

        if (ww <= 0 || hh <= 0) continue;
        dets.push_back({ cv::Rect(x,y,ww,hh), conf, clsId });
    }

    auto finalDet = NMS(dets, NMS_THR);

    // ---------------- 시각화 ----------------
    for (const auto& d : finalDet) {
        cv::rectangle(img, d.box, { 0,255,0 }, 2);
        std::string text = std::to_string(d.class_id) + " " + cv::format("%.2f", d.conf);
        cv::putText(img, text, d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, { 0,0,255 }, 2);
    }

    cv::imwrite("result.jpg", img);
    std::cout << "Done. Saved result.jpg with " << finalDet.size() << " detections\n";

    // 정리
    CHECK_CUDA(cudaFree(dInput)); CHECK_CUDA(cudaFree(dOutput));
    cudaStreamDestroy(stream);
    return 0;
}
