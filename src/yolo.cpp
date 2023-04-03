#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include "buffers.h"
#include "logger.h"
#include "logging.h"
#include "common.h"
#include "yolo.h"
#include "preprocess.h"

using namespace std;

yolo::yolo(string trtFile, bool end2end)
{
    extern sample::Logger logger;
    this->logger_ = &logger.getTRTLogger();
    auto plan = this->load_engine_file(trtFile);
    initLibNvInferPlugins(this->logger_, "");
    this->runtime = unique_ptr<IRuntime>(createInferRuntime(*this->logger_));
    assert(this->runtime);

    this->engine = shared_ptr<ICudaEngine>(this->runtime->deserializeCudaEngine((void *)plan.data(), plan.size()));
    assert(this->engine);

    this->context = shared_ptr<IExecutionContext>(this->engine->createExecutionContext());
    assert(this->context);

    this->buffers = new samplesCommon::BufferManager(this->engine);
}

yolo::~yolo()
{
    delete this->buffers;
    this->context.reset();
    this->engine.reset();
    this->runtime.reset();
}

void yolo::copy_from_Mat_CPU(cv::Mat &img)
{
    this->make_pipe(img);
    auto warp_dst = this->letterbox(img);

    warp_dst.convertTo(warp_dst, CV_32FC3, 1.0 / 255);
    cv::cvtColor(warp_dst, warp_dst, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> warp_dst_nchw_channels;
    cv::split(warp_dst, warp_dst_nchw_channels);
    for (auto &img : warp_dst_nchw_channels)
        img = img.reshape(1, 1);
    cv::Mat warp_dst_nchw;
    cv::hconcat(warp_dst_nchw_channels, warp_dst_nchw);
    // 将处理后的图片数据拷贝到GPU
    CUDA_CHECK(cudaMemcpy(this->buffers->getDeviceBuffer(kInputTensorName), warp_dst_nchw.ptr(), kInputH * kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void yolo::copy_from_Mat_GPU(cv::Mat &img)
{
    this->make_pipe(img);
    cuda_preprocess(this->img_buffer_device,img.ptr(), img.cols, img.rows,
                    (float *)this->buffers->getDeviceBuffer(kInputTensorName), kInputW, kInputH);
}

void yolo::infer()
{
    context->executeV2(this->buffers->getDeviceBindings().data());
    this->buffers->copyOutputToHost();
}

void yolo::getResults(vector<Detection> &results)
{
    results.clear();
    int32_t *num_det = (int32_t *)buffers->getHostBuffer(kOutNumDet); // 检测到的目标个数
    int32_t *cls = (int32_t *)buffers->getHostBuffer(kOutDetCls);     // 检测到的目标类别
    float *conf = (float *)buffers->getHostBuffer(kOutDetScores);     // 检测到的目标置信度
    float *bbox = (float *)buffers->getHostBuffer(kOutDetBBoxes);     // 检测到的目标框
    std::map<int32_t, std::vector<Detection>> m;
    for (int i = 0; i < num_det[0]; i++)
    {
        Detection det;
        // convert xywh to xyxy
        det.bbox[0] = bbox[i * 4 + 0] - bbox[i * 4 + 2] / 2;
        det.bbox[1] = bbox[i * 4 + 1] - bbox[i * 4 + 3] / 2;
        det.bbox[2] = bbox[i * 4 + 0] + bbox[i * 4 + 2] / 2;
        det.bbox[3] = bbox[i * 4 + 1] + bbox[i * 4 + 3] / 2;
        det.conf = conf[i];
        det.class_id = cls[i];
        results.push_back(det);
    }
}

void yolo::draw_objects(cv::Mat &frame, vector<Detection> &bboxs, const vector<vector<unsigned int>> &COLORS, const vector<string> &CLASS_NAMES)
{
    for (auto bbox : bboxs)
    {
        int i = bbox.class_id;
        if (i < 0 || i > COLORS.size())
            continue;
        cv::Rect rect = get_rect(frame, bbox.bbox);
        cv::Scalar color{COLORS[i][0], COLORS[i][1], COLORS[i][2]};
        cv::rectangle(frame, rect, color, 2);
        char name[256];
        sprintf(name, "%s %.1f%%", CLASS_NAMES[bbox.class_id].c_str(), bbox.conf * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(name, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::putText(frame, name, cv::Point(rect.x, rect.y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}


inline cv::Mat yolo::letterbox(cv::Mat &src)
{
    float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);

    int offsetx = (kInputW - src.cols * scale) / 2;
    int offsety = (kInputH - src.rows * scale) / 2;

    cv::Point2f srcTri[3]; // 计算原图的三个点：左上角、右上角、左下角
    srcTri[0] = cv::Point2f(0.f, 0.f);
    srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
    cv::Point2f dstTri[3]; // 计算目标图的三个点：左上角、右上角、左下角
    dstTri[0] = cv::Point2f(offsetx, offsety);
    dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
    dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);       // 计算仿射变换矩阵
    cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type()); // 创建目标图
    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());        // 进行仿射变换
    return warp_dst;
}

vector<Detection> yolo::predict_CPU(cv::Mat &img)
{
    this->copy_from_Mat_CPU(img);
    this->infer();
    vector<Detection> results;
    this->getResults(results);
    this->remove_pipe();
    return results;
}

vector<Detection> yolo::predict_GPU(cv::Mat &img)
{
    this->copy_from_Mat_GPU(img);
    this->infer();
    vector<Detection> results;
    this->getResults(results);
    this->remove_pipe();
    return results;
}

vector<Detection> yolo::predict_CPU(string &img_path)
{
    auto img = cv::imread(img_path, cv::IMREAD_COLOR);
    return this->predict_CPU(img);
}

vector<Detection> yolo::predict_GPU(string &img_path)
{
    auto img = cv::imread(img_path, cv::IMREAD_COLOR);
    return this->predict_CPU(img);
}
