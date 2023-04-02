# tensorrt_yolo
一个将yolov8模型转化成tensorrt引擎并运行的示例代码，

## 已完成：

- 在原有模型后增加plugin，直接输出识别结果。无需后处理代码
  
  BatchedNMS_TRT:输入边框数量需要小于4096，后面的边框估计会被忽略

  EfficientNMS_TRT: 

- 现在仅支持yolov8 和 yolov5，其他版本需要测试

- 预处理过程可以在GPU上完成，也可以全在cpu上完成，测试两者速度相差2ms左右

  在GPU上预处理部分的代码来自[github/enpeizhao](https://github.com/enpeizhao/CVprojects.git)
 

## TODO:

- 增加后处理代码，直接从原有模型输出结果中获得识别结果，估计也是借鉴别人的代码

- 测试其他版本的yolo

- 这段时间很忙，估计找一段时间再弄把

## 文件结构

- src：程序以及头文件源码

- common: comes from [github/TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/common)
