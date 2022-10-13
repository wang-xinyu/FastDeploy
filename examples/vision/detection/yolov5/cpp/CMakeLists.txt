PROJECT(infer_demo C CXX CUDA)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${CUDA_DIRECTORY}/include)

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc ${PROJECT_SOURCE_DIR}/preprocess.cu)
# 添加FastDeploy库依赖
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
