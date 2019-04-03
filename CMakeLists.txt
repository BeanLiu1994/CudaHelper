cmake_minimum_required(VERSION 2.8)

project(CudaHelper_Test)

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++14" "-G")

file(GLOB PATHS_SRC *.cu *.cpp)
CUDA_ADD_EXECUTABLE(test ${PATHS_SRC})
set_property(TARGET test PROPERTY CXX_STANDARD 17)
