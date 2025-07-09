cmake_minimum_required(VERSION 3.10)
project(benchlab)

option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_HIP "Enable HIP backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)
    enable_language(CUDA)

    add_executable(run_benchmarks
        src/run_benchmarks_cuda.cpp
        src/benchmarks/matmul/matmul_cuda.cu
        src/benchmarks/vector_add/vector_add_cuda.cu
        src/benchmarks/conv1d/conv1d_cuda.cu
        src/benchmarks/reduction/reduction_cuda.cu
        src/benchmarks/bandwidth/bandwidth_cuda.cu
    )
endif()

if(ENABLE_HIP)
    find_package(HIP REQUIRED)

    add_executable(run_benchmarks
        src/run_benchmarks_hip.cpp
        src/benchmarks/matmul/matmul_hip.cpp
        src/benchmarks/vector_add/vector_add_hip.cpp
        src/benchmarks/conv1d/conv1d_hip.cpp
        src/benchmarks/reduction/reduction_hip.cpp
        src/benchmarks/bandwidth/bandwidth_hip.cpp
    )

    target_link_libraries(run_benchmarks PRIVATE HIP::device)
endif()

