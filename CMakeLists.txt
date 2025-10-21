cmake_minimum_required(VERSION 3.5)
project(cuda_perf_test CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(mm_kernel SHARED mm_kernel.cu)
set_target_properties(mm_kernel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
target_compile_options(mm_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)

add_executable(cuda_perf perf.cpp)
set_target_properties(cuda_perf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
target_link_libraries(cuda_perf mm_kernel ${CUDA_CUDART_LIBRARY})
