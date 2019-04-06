cmake_minimum_required(VERSION 3.7)

set(PROJECTNAME "CudaHelper_Test")
project(${PROJECTNAME})

# find cuda here. (before VERSION 3.10)
# The FindCUDA module has been superseded by first-class support for the CUDA language in CMake. 
# use  enable_language()  to enable cuda support, find_package is no longer necessary.
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)
find_package(CUDA REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 5.2 6.0)  
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "-std=c++14" "-rdc=true")
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-G")

# grab files, and add our target "test", Executable.
include_directories (include)
file(GLOB_RECURSE PATHS_HEADERS ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/include/*.cuh)
file(GLOB_RECURSE PATHS_SOURCES src/*.cpp src/*.cu)

CUDA_ADD_EXECUTABLE(test ${PATHS_HEADERS} ${PATHS_SOURCES})
set_property(TARGET test PROPERTY CXX_STANDARD 14)
target_link_libraries(test ${CUDA_cudadevrt_LIBRARY})