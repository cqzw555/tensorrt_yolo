# tensorrt_yolo

一个将yolo 模型转化成tensorrt引擎并运行的示例代码。

现在支持的yolo版本：

- yolov5:[github/ultralytics](https://github.com/ultralytics/yolov5)

- yolov8:[github/ultralytics](https://github.com/ultralytics/ultralytics)


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

- **src**：程序以及头文件源码

- **src/options.h**: 解析命令行的库，来自[github/absop](https://github.com/absop/ThreadPool/blob/main/Options.h)

- **common**: comes from [github/TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/common)

## 运行平台

- **ubuntu18**: cuda-11-8 tensorrt 8.5.3.1 cudnn 8.6 从[cuda/repos](https://developer.download.nvidia.cn/compute/cuda/repos/)处下载 测试可以运行

- **wsl ubuntu20.04**: cuda-12.1 (windows端 cuda12.1) [cuda/repos](https://developer.download.nvidia.cn/compute/cuda/repos/wsl-ubuntu)处下载 cudnn8.8 tensorrt8.6 EA 需要手动将common文件夹替换成TensorRT下的release/8.6分支 可以运行 速度和纯ubuntu18下无明显速度区别。
  
  **PS**. 原本windows上cuda是11.8的，wsl也是，但是编译成功后，需要链接libcudaLt.so.12，不知道原因，所以装12.1的cuda，保持版本一致。
  另外在wsl上跑的原因是我测试其他yolo模型方便，ubuntu18里一直没配成功过

## 用法

- **build**程序需要将所有的参数都输入进去，确保模型转换正确

  eg:./build -i yolov5l.onnx -o=yolov5l_plugin.trt -v=5 -n=80 --EfficientNMS
