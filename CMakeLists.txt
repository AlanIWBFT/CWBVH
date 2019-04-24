cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RTLib LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
enable_language(CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -use_fast_math")
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_50; -std=c++11;")

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/src/CMake ${CMAKE_MODULE_PATH})
SET(TBB_ROOT_DIR "Path to TBB root dir" CACHE PATH "Intel TBB Library")
SET(EMBREE_ROOT_DIR "Path to Embree root dir" CACHE PATH "Intel Embree Library")
SET(OPTIX_INCLUDE_DIR "~/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/include" CACHE PATH "NVIDIA OptiX Prime Library")
SET(OPTIX_PRIME_LIB_FILE "~/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/liboptix_prime.so.5.1.0" CACHE PATH "NVIDIA OptiX Prime Library")
find_package(TBB REQUIRED)
find_package(Embree REQUIRED)


cuda_add_executable(RTLib
       src/main.cpp
       src/Logger.cpp
       src/Logger.h
       src/EmbreeBVHBuilder.cpp
       src/EmbreeBVHBuilder.h
       src/GPUBVHConverter.cpp
       src/GPUBVHConverter.h
       src/WoopTriangleHelper.h
       src/TraversalKernelBVH2.cu
       src/TraversalKernelBVH2.h
       src/TraversalKernelCWBVH.cu
       src/TraversalKernelCWBVH.h
       src/helper_math.h
       src/helper_math.cu
       src/CUDAAssert.h
       src/FastDeviceMinMax.h
       src/ValidationKernels.h
       src/ValidationKernels.cu
)

target_include_directories(RTLib PRIVATE ${TBB_INCLUDE_DIRS})
target_compile_definitions(RTLib PRIVATE ${TBB_DEFINITIONS})
target_include_directories(RTLib PRIVATE ${EMBREE_INCLUDE_DIR})

set_target_properties(RTLib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

add_library(CudaPTX OBJECT src/TraversalKernelCWBVH.cu)
set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)
add_custom_target(ptx ALL
    DEPENDS CudaPTX src/TraversalKernelCWBVH.cu
    SOURCES src/TraversalKernelCWBVH.cu)

target_link_libraries(RTLib ${TBB_LIBRARIES_DEBUG})
target_link_libraries(RTLib ${TBB_LIBRARIES_RELEASE})
target_link_libraries(RTLib ${EMBREE_LIBRARIES})

target_include_directories(RTLib PRIVATE ${OPTIX_INCLUDE_DIR})
target_link_libraries(RTLib ${OPTIX_PRIME_LIB_FILE})