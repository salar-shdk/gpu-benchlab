#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

extern "C" float run_vector_add_hip() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(vector_add, blocks, threads, 0, 0, d_A, d_B, d_C, N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    std::chrono::duration<float, std::milli> duration = end - start;

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return duration.count();
}

