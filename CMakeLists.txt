cmake_minimum_required(VERSION 3.5)
project(cuda_perf_test CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 尝试查找 cublas 库（优先在 CUDA 安装目录下查找）
find_library(CUBLAS_LIBRARY cublas
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
        NO_DEFAULT_PATH)

if(NOT CUBLAS_LIBRARY)
    message(STATUS "cublas not found via find_library, falling back to -lcublas")
    set(CUBLAS_LIBRARY cublas)
else()
    message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
endif()

add_library(mm_kernel SHARED mm_kernel.cu)
set_target_properties(mm_kernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
target_compile_options(mm_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)

# 链接 cuBLAS 和 cuDART 到共享库，确保符号在 libmm_kernel.so 中解析
target_link_libraries(mm_kernel PRIVATE ${CUDA_CUDART_LIBRARY} ${CUBLAS_LIBRARY})

add_executable(cuda_perf perf.cpp)
set_target_properties(cuda_perf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
target_link_libraries(cuda_perf PRIVATE mm_kernel)
