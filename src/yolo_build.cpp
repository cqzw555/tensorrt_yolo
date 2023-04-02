#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cassert>

#include "logger.h"
#include "config.h"

sample::Logger logger;
using namespace std;
using nvinfer1::Dims3;
using nvinfer1::Dims4;
using nvinfer1::PluginField;
using std::cout;

int main(int agrc, char **argv)
{
    if (agrc != 3)
    {
        cerr << "usage: ./build [onnx_file_path] [output_file_path]" << endl;
        return -1;
    }
    char *output_file_path = argv[2];
    char *onnx_file_path = argv[1];

    // =========== 1. 创建builder ===========
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger.getTRTLogger()));
    assert(builder && "Failed to create builder");
    cout << "create builder done!" << endl;

    // ========== 2. 创建network：builder--->network ==========
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    assert(network && "Failed to create network");
    cout << "create network done!" << endl;

    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger.getTRTLogger()));
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(logger.getReportableSeverity()));
    assert(parsed && "Failed to parser onnx file");
    cout << "parser onnx file done!" << endl;

    // 配置网络参数
    // 我们需要告诉tensorrt我们最终运行时，输入图像的范围，batch size的范围。这样tensorrt才能对应为我们进行模型构建与优化。                                                                                  // 获取输入节点
    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile(); // 创建profile，用于设置输入的动态尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                           Dims4{1, 3, kInputW, kInputH}); // 设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                           Dims4{1, 3, kInputW, kInputH}); // 设置最优尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                           Dims4{1, 3, kInputW, kInputH}); // 设置最大尺寸

    // 开始往后面增加一个自定义层
    vector<nvinfer1::ITensor *> outputs;
    int nOutput = network->getNbOutputs();
    for (int i = 0; i < nOutput; i++)
        outputs.push_back(network->getOutput(i));
    auto previous_output = outputs[0];

    // previous_output的初始形状为 (1,84,8400)
    auto shuffleLayer = network->addShuffle(*previous_output);
    shuffleLayer->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    // 现在previous_output的形状变为 (1,8400,84)
    // batchnms无法处理超过4096的数据，因此有些识别结果无法显示

    // 先把box提取出来
    // 形状定义，起始点，步长
    auto dim = shuffleLayer->getOutput(0)->getDimensions();
    Dims3 shapes{dim.d[0], dim.d[1], 4};
    Dims3 starts{0, 0, 0};
    Dims3 strides{1, 1, 1};
    auto boxLayer = network->addSlice(*shuffleLayer->getOutput(0), starts, shapes, strides);
    auto box = network->addShuffle(*boxLayer->getOutput(0));
    box->setReshapeDimensions(Dims4{0, 0, 1, 4});

    // 置信度 提取
#ifdef YOLOV8
    starts.d[2] = 4;
    shapes.d[2] = dim.d[2] - 4;
    auto scorelayer = network->addSlice(*shuffleLayer->getOutput(0), starts, shapes, strides);
    cout << "extract box and scores done" << endl;
#elif defined(YOLOV5)
    starts.d[2] = 4;
    shapes.d[2] = 1;
    auto objlayer = network->addSlice(*shuffleLayer->getOutput(0), starts, shapes, strides);

    starts.d[2] = 5;
    shapes.d[2] = dim.d[2] - 5;
    auto objscorelayer = network->addSlice(*shuffleLayer->getOutput(0), starts, shapes, strides);
    auto scorelayer = network->addElementWise(*objlayer->getOutput(0), *objscorelayer->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kPROD);
    cout << "extract box and scores done" << endl;
#endif

#ifdef USING_EFFICIENTNMS_TRT
    // work well done
    // the home page of  EfficientNMS_TRT shows some issue as follows:

    // The EfficientNMS_ONNX_TRT plugin's output may not always be sufficiently sized to capture all NMS-ed boxes.
    // This is because it ignores the number of classes in the calculation of the output size (it produces an output
    // of size (batch_size * max_output_boxes_per_class, 3) when in general, a tensor of size
    // (batch_size * max_output_boxes_per_class * num_classes, 3)) would be required. This was a compromise
    // made to keep the output size from growing uncontrollably since it lacks an attribute similar to max_output_boxes to
    // control the number of output boxes globally.
    // Due to this reason, please use TensorRT's inbuilt INMSLayer instead of the EfficientNMS_ONNX_TRT plugin
    // wherever possible.

    // Although the plugin I use is EfficientNMS_TRT, I have tried the addNMS function but there is no sample.
    auto creator = getPluginRegistry()->getPluginCreator("EfficientNMS_TRT", "1");

    vector<PluginField> fields;
    fields.emplace_back(PluginField("max_output_boxes", (void const *)new int32_t(max_output_boxes),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("background_class", (void const *)new int32_t(background_class),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("score_threshold", (void const *)new float(score_threshold),
                                    nvinfer1::PluginFieldType::kFLOAT32, 1));
    fields.emplace_back(PluginField("iou_threshold", (void const *)new float(iou_threshold),
                                    nvinfer1::PluginFieldType::kFLOAT32, 1));
    fields.emplace_back(PluginField("box_coding", (void const *)new int32_t(box_coding),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("score_activation", (void const *)new int32_t(score_activation),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("class_agnostic", (void const *)new int32_t(class_agnostic),
                                    nvinfer1::PluginFieldType::kINT32, 1));

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = fields.size();
    fc.fields = fields.data();
    auto nms_layer = creator->createPlugin("nms_layer", &fc);
    cout << "begin to add plugin !";

    nvinfer1::ITensor *const nms_inputs[2] = {box->getOutput(0), scorelayer->getOutput(0)};
    auto plugin_layer = network->addPluginV2(nms_inputs, 2, *nms_layer);
    cout << "done!" << endl;

    cout << "begin to set plugin's output name!" << endl;

    // 设置plugin的输出名
    plugin_layer->getOutput(0)->setName(kOutNumDet);
    plugin_layer->getOutput(1)->setName(kOutDetBBoxes);
    plugin_layer->getOutput(2)->setName(kOutDetScores);
    plugin_layer->getOutput(3)->setName(kOutDetCls);
    cout << "done!" << endl;
    // 将plugin的输出设置为模型输出
    for (int i = 0; i < 4; i++)
        network->markOutput(*plugin_layer->getOutput(i));
    cout << "add EfficientNMS_TRT plugin done!" << endl;

#elif defined(USING_BatchedNMS_TRT)
    // work well done
    // some outputs which's id is bigger than 4096,can not be input to BatchedNMS_TRT plugin
    // thus some detection results are missing.
    auto creator = getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
    vector<PluginField> fields;
    // parameter for BatchedNMS_TRT, max_input_size(topK) is 4096
    fields.emplace_back(PluginField("shareLocation", (void const *)new int32_t(shareLocation),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("backgroundLabelId", (void const *)new int32_t(backgroundLabelId),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("numClasses", (void const *)new int32_t(numClasses),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("topK", (void const *)new int32_t(topK),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("keepTopK", (void const *)new int32_t(keepTopK),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("scoreThreshold", (void const *)new float(scoreThreshold),
                                    nvinfer1::PluginFieldType::kFLOAT32, 1));
    fields.emplace_back(PluginField("iouThreshold", (void const *)new float(iouThreshold),
                                    nvinfer1::PluginFieldType::kFLOAT32, 1));
    fields.emplace_back(PluginField("isNormalized", (void const *)new int32_t(isNormalized),
                                    nvinfer1::PluginFieldType::kINT32, 1));
    fields.emplace_back(PluginField("clipBoxes", (void const *)new int32_t(clipBoxes),
                                    nvinfer1::PluginFieldType::kINT32, 1));

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = fields.size();
    fc.fields = fields.data();
    auto nms_layer = creator->createPlugin("nms_layer", &fc);
    cout << "begin to add plugin !";

    nvinfer1::ITensor *const nms_inputs[2] = {box->getOutput(0), scorelayer->getOutput(0)};
    auto plugin_layer = network->addPluginV2(nms_inputs, 2, *nms_layer);
    cout << "done!" << endl;

    cout << "begin to set plugin's output name!" << endl;
    // 设置plugin的输出名
    plugin_layer->getOutput(0)->setName(kOutNumDet);
    plugin_layer->getOutput(1)->setName(kOutDetBBoxes);
    plugin_layer->getOutput(2)->setName(kOutDetScores);
    plugin_layer->getOutput(3)->setName(kOutDetCls);
    cout << "done!" << endl;
    // 将plugin的输出设置为模型输出
    for (int i = 0; i < 4; i++)
        network->markOutput(*plugin_layer->getOutput(i));
    cout << "add BatchedNMS_TRT plugin done!" << endl;

#elif defined(USING_NMS)
    // this part is wrong, attemp to fix
    int32_t const maxOutputBoxesPerClassDefault = 0;
    auto constantLayer = network->addConstant(nvinfer1::Dims{0, {}}, nvinfer1::Weights{nvinfer1::DataType::kINT32,
                                                                                       &maxOutputBoxesPerClassDefault, 1});
    auto maxOutputboxesPtr = constantLayer->getOutput(0);
    auto nms_layer = network->addNMS(*box->getOutput(0), *scorelayer->getOutput(0), *maxOutputboxesPtr);
    nms_layer->setBoundingBoxFormat(nvinfer1::BoundingBoxFormat::kCORNER_PAIRS); // xyxy
    nms_layer->setTopKBoxLimit(topK);
    cout << nms_layer->getNbOutputs() << endl;
    cout << nms_layer->getNbInputs() << endl;
    nms_layer->getOutput(0)->setName(kOutNumDet);
    nms_layer->getOutput(1)->setName(kOutDetBBoxes);
    nms_layer->getOutput(2)->setName(kOutDetScores);
    nms_layer->getOutput(3)->setName(kOutDetCls);
    for (int i = 0; i < 4; i++)
        network->markOutput(*nms_layer->getOutput(i));
    cout << "add NMS layer done!" << endl;

#endif
    // 顺便把输出输出名改了
    previous_output->setName(kOutputTensorName);
    network->getInput(0)->setName(kInputTensorName);
    // 原本输出屏蔽掉部分
    for (int i = 1; i < nOutput; i++)
        network->unmarkOutput(*outputs[i]);

    // ========== 3. 创建config配置：builder--->config ==========
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    assert(config && "Failed to create config");
    cout << "create config done!" << endl;

    // 使用addOptimizationProfile方法添加profile，用于设置输入的动态尺寸
    config->addOptimizationProfile(profile);
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // ========== 4. 创建engine：builder--->engine(*nework, *config) ==========
    auto plan = unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    assert(plan && "Failed to serialize engine");
    cout << "serialize engine done!" << endl;

    // ========== 5. 序列化保存engine ==========
    cout << output_file_path << endl;
    ofstream engine_file(output_file_path, ios::binary | ios::out);
    assert(engine_file.is_open() && "Failed to open engine file");
    cout << "open engine file done!" << endl;

    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    cout << "Engine build success!" << endl;

    return 0;
}