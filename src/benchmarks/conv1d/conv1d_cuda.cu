#include <cuda_runtime.h>
#include <iostream>

__global__ void conv1d(const float* input, const float* kernel, float* output, int N, int K) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int k_half = K / 2;

    if (idx < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            int i = idx + j - k_half;
            if (i >= 0 && i < N)
                sum += input[i] * kernel[j];
        }
        output[idx] = sum;
    }
}

extern "C" float run_conv1d_cuda() {
    const int N = 1 << 20;
    const int K = 7;
    size_t input_size  = N * sizeof(float);
    size_t kernel_size = K * sizeof(float);

    float *h_input  = new float[N];
    float *h_kernel = new float[K];
    float *h_output = new float[N];

    for (int i = 0; i < N; ++i) h_input[i]  = 1.0f;
    for (int i = 0; i < K; ++i) h_kernel[i] = 0.1f;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input,  input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, input_size);

    cudaMemcpy(d_input,  h_input,  input_size,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1d<<<grid, block>>>(d_input, d_kernel, d_output, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    delete[] h_input; delete[] h_kernel; delete[] h_output;

    return ms;
}
