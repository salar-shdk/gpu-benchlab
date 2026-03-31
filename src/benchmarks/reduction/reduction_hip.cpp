#include <hip/hip_runtime.h>
#include <iostream>

__global__ void reduce_sum(const float* input, float* output, int N) {
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += stride)
        sum += input[i];

    atomicAdd(output, sum);
}

extern "C" float run_reduction_hip() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float* d_input;
    float* d_output;
    float h_output = 0.0f;

    hipMalloc(&d_input,  size);
    hipMalloc(&d_output, sizeof(float));
    hipMemcpy(d_input,  h_input,   size,          hipMemcpyHostToDevice);
    hipMemcpy(d_output, &h_output, sizeof(float),  hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(256);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(reduce_sum, grid, block, 0, 0, d_input, d_output, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);

    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    std::cout << "HIP Reduction Sum = " << h_output << "\n";

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;

    return ms;
}
