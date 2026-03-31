#include <hip/hip_runtime.h>
#include <iostream>

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

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, d_A, d_B, d_C, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);

    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return ms;
}
