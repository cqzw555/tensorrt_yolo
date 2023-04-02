#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_utils.h"
#include "config.h"
uint8_t* img_buffer_device_gpu = nullptr;

void cuda_pure_preprocess(uint8_t* img_buffer_device,uint8_t *src, float *dst, int dst_width, int dst_height);
void cuda_preprocess(uint8_t* img_buffer_device,uint8_t *src, int src_width, int src_height,float *dst, int dst_width, int dst_height);
// void cuda_preprocess_init();
// void cuda_preprocess_destory();