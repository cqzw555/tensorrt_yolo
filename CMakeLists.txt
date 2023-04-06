cmake_minimum_required(VERSION 3.10.1)
project(tensorrt-yolo LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(jsoncpp REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/common/)

message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# preprocess的库，不知怎么，只要预处理文件和cuda文件在一起就不能通过编译，就单独成共享库了
add_library(preprocess SHARED src/preprocess.cu src/preprocess.h)
target_link_libraries(preprocess cudart opencv_core)

# infer
add_executable(infer src/yolo_infer.cpp src/yolo.cpp)
target_link_libraries(infer nvinfer nvinfer_plugin nvonnxparser cudart ${OpenCV_LIBS} jsoncpp)
target_link_libraries(infer -Wl,--no-as-needed preprocess)
# 使用include_directories，在编译的时候preprocess库就会报错，所以就infer单独加
target_include_directories(infer PRIVATE /usr/local/cuda-11.8/include)

# build
add_executable(build src/yolo_build.cpp)
target_link_libraries(build nvinfer nvinfer_plugin nvonnxparser cudart ${OpenCV_LIBS})
