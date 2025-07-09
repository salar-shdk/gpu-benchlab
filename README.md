# BenchLab

BenchLab is a modular GPU benchmarking suite written in C++ using both **CUDA** and **HIP** backends. It provides comparative performance benchmarks for common GPU operations across NVIDIA and AMD GPUs.

## Features

- Unified interface for **HIP** (AMD) and **CUDA** (NVIDIA)
- Modular benchmarks:
  - Vector Addition
  - Matrix Multiplication
  - 1D Convolution
  - Reduction (Sum)
  - Memory Bandwidth Test
---

## Build Instructions

### Prerequisites

- CMake ≥ 3.10
- A supported compiler (GCC, Clang, or MSVC)
- **For CUDA backend**:
  - NVIDIA GPU with CUDA Toolkit installed
- **For HIP backend**:
  - AMD GPU with ROCm installed

### Build

Use the provided build.sh script:

```bash
# Build for CUDA
./build.sh -f --backend=CUDA

# Build for HIP
./build.sh -f --backend=HIP

# If you don't provide the backend, it will choose the backend based on installed drivers
./build.sh -f
```

Or build manually with Cmake:

```bash
# CUDA
cmake -DENABLE_CUDA=ON -DENABLE_HIP=OFF -B build
cmake --build build -j

# HIP
cmake -DENABLE_CUDA=OFF -DENABLE_HIP=ON -B build
cmake --build build -j
```

### Run

```bash
./build/run_benchmarks
```
