cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RTLib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -Xcompiler=/wd4819") 

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/src/CMake ${CMAKE_MODULE_PATH})
SET(TBB_ROOT_DIR "Path to TBB root dir" CACHE PATH "Intel TBB Library")
SET(EMBREE_ROOT_DIR "Path to Embree root dir" CACHE PATH "Intel Embree Library")
SET(OPTIX_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0/include" CACHE PATH "NVIDIA OptiX Prime Library")
SET(OPTIX_PRIME_LIB_FILE "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0/lib64/optix_prime.1.lib" CACHE PATH "NVIDIA OptiX Prime Library")
find_package(TBB REQUIRED)
find_package(Embree REQUIRED)

add_executable(RTLib
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
