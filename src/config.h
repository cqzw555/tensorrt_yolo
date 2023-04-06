#ifndef CONFIG_H
#define CONFIG_H

static int kInputH = 640;
static int kInputW = 640;

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "prob";
const static char* kOutNumDet = "DetectionNUm";
const static char* kOutDetScores = "DetectionScores";
const static char* kOutDetBBoxes = "DetectionBoxes";
const static char* kOutDetCls = "DetectionClasses";

const static float kNmsThresh = 0.5f;
const static float kConfThresh = 0.5f;

// parameters for BatchedNMSPlugin
const int32_t shareLocation = 1;
const int32_t backgroundLabelId = -1;
const int32_t numClasses = 80;
const int32_t topK = 4096;
const int32_t keepTopK = 100;
const float scoreThreshold = 0.2;
const float iouThreshold = 0.7;
const int32_t clipBoxes = 0;
const int32_t isNormalized = 0;

// parameters for EfficientNMSPlugin
const int max_output_boxes = 100;
const float iou_threshold = 0.5;
const float score_threshold = 0.5;
const int background_class = -1;
const int32_t score_activation = 0; // 1 refer to true, 0 refer to false
const int32_t class_agnostic = 1;
const int box_coding = 0; // 1 is xywh, 0 is xyxy
#endif