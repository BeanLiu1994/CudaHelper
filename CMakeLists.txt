cmake_minimum_required(VERSION 2.8)

project(CudaHelper_Test)

find_package(CUDA REQUIRED)


file(GLOB PATHS_SRC *.cu *.cpp)
CUDA_ADD_EXECUTABLE(test ${PATHS_SRC})