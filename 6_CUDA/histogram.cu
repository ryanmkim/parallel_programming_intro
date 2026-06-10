// Parallel histogram with shared-memory privatization to cut global atomic contention.
// Ryan Kim

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void histogramShared(const int *data, int *hist, int n, int bins, int maxVal) {
    extern __shared__ int local[];

    for (int i = threadIdx.x; i < bins; i += blockDim.x)
        local[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // data is 1..maxVal, map to [0, bins)
        int bin = (data[idx] - 1) * bins / maxVal;
        if (bin >= bins) bin = bins - 1;
        if (bin < 0) bin = 0;
        atomicAdd(&local[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bins; i += blockDim.x)
        atomicAdd(&hist[i], local[i]);
}

int main() {
    const int N = 10000000;
    const int bins = 10;
    const int maxVal = 1000;

    std::vector<int> h_data(N);
    std::vector<int> h_hist(bins, 0);

    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i)
        h_data[i] = (std::rand() % maxVal) + 1;

    int *d_data, *d_hist;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, bins * sizeof(int));

    cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bins * sizeof(int));

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    size_t shmem = bins * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    histogramShared<<<blocks, tpb, shmem>>>(d_data, d_hist, N, bins, maxVal);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(h_hist.data(), d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "N = " << N << ", bins = " << bins << "\n";
    std::cout << "kernel time: " << ms << " ms\n\n";

    int total = 0;
    for (int i = 0; i < bins; ++i) {
        int lo = (i * maxVal) / bins + 1;
        int hi = ((i + 1) * maxVal) / bins;
        std::cout << "[" << lo << "-" << hi << "]\t" << h_hist[i] << "\n";
        total += h_hist[i];
    }
    std::cout << "\ntotal: " << total << " (expected " << N << ")\n";

    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
