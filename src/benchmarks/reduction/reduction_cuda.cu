#include <cuda_runtime.h>
#include <iostream>

__global__ void reduce_sum(const float* input, float* output, int N) {
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += stride)
        sum += input[i];

    atomicAdd(output, sum);
}

extern "C" float run_reduction_cuda() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float* d_input;
    float* d_output;
    float h_output = 0.0f;

    cudaMalloc(&d_input,  size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input,  h_input,   size,         cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(256);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_sum<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "CUDA Reduction Sum = " << h_output << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return ms;
}
