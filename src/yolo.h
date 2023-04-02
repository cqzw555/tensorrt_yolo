#ifndef YOLOV8_H
#define YOLOV8_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <NvInfer.h>
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "cuda_utils.h"
#include "config.h"
#include "types.h"
using namespace std;
using namespace nvinfer1;



class yolo
{
private:
    unique_ptr<IRuntime> runtime = nullptr;
    shared_ptr<ICudaEngine> engine = nullptr;
    shared_ptr<IExecutionContext> context = nullptr;
    samplesCommon::BufferManager *buffers = nullptr;
    ILogger *logger_ = nullptr;
    
    uint8_t *img_buffer_device = nullptr;
    uint8_t *img_buffer_device_gpu = nullptr;
    inline int getSize(Dims32 dim);
    inline int dataTypeToSize(DataType dataType);
    std::vector<unsigned char> load_engine_file(const std::string &file_name)
    {
        std::vector<unsigned char> engine_data;
        std::ifstream engine_file(file_name, std::ios::binary);
        assert(engine_file.is_open() && "Unable to load engine file.");
        engine_file.seekg(0, engine_file.end);
        int length = engine_file.tellg();
        engine_data.resize(length);
        engine_file.seekg(0, engine_file.beg);
        engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
        return engine_data;
    }
    cv::Rect get_rect(cv::Mat &img, float bbox[4])
    {

        float scale = std::min(kInputH / float(img.cols), kInputW / float(img.rows));
        int offsetx = (kInputW - img.cols * scale) / 2;
        int offsety = (kInputH - img.rows * scale) / 2;

        size_t output_width = img.cols;
        size_t output_height = img.rows;

        float x1 = (bbox[0] - offsetx) / scale;
        float y1 = (bbox[1] - offsety) / scale;
        float x2 = (bbox[2] - offsetx) / scale;
        float y2 = (bbox[3] - offsety) / scale;

        x1 = clamp(x1, 0, output_width);
        y1 = clamp(y1, 0, output_height);
        x2 = clamp(x2, 0, output_width);
        y2 = clamp(y2, 0, output_height);

        auto width = clamp(x2 - x1, 0, output_width);
        auto height = clamp(y2 - y1, 0, output_height);

        return cv::Rect(x1, y1, width, height);
    }
    float clamp(const float val, const float minVal, const float maxVal)
    {
        assert(minVal <= maxVal);
        return std::min(maxVal, std::max(minVal, val));
    }
    void make_pipe(cv::Mat &img){
        int size = img.size().width * img.size().height;
        // cout << img.size().width << endl;
        CUDA_CHECK(cudaMalloc((void **)&this->img_buffer_device, size * 3));
    }
    void remove_pipe(){
        CUDA_CHECK(cudaFree(this->img_buffer_device));
    }

public:
    yolo(string trtFile, bool end2end);
    ~yolo();
    void infer();
    void draw_objects(cv::Mat &, vector<Detection> &, const vector<vector<unsigned int>> &, const vector<string> &);
    void draw_objects(cv::Mat &, vector<Detection> &);
    void getResults(vector<Detection>&);
    void copy_from_Mat_CPU(cv::Mat &img);
    void copy_from_Mat_GPU(cv::Mat &img);
    vector<Detection> predict_CPU(cv::Mat &);
    vector<Detection> predict_CPU(string &);
    vector<Detection> predict_GPU(cv::Mat &);
    vector<Detection> predict_GPU(string &);
    inline cv::Mat letterbox(cv::Mat &src);
};

inline int yolo::getSize(Dims32 dim)
{
    int res = 1;
    for (int i = 0; i < dim.nbDims; i++)
        res *= dim.d[i];
    return res;
}

inline int yolo::dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

#endif