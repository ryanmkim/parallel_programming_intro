#include <iostream>
#include <vector>
#include <random>
#include <cstdint>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error " << cudaGetErrorString(err_)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <int REPLICAS>
__global__ void histogramShared(const int *__restrict__ data,
                                unsigned int *__restrict__ hist,
                                int n, int bins, int maxVal) {
    extern __shared__ unsigned int smem[];
    const int totalSlots = REPLICAS * bins;

    for (int i = threadIdx.x; i < totalSlots; i += blockDim.x)
        smem[i] = 0u;
    __syncthreads();

    const int base = (threadIdx.x % REPLICAS) * bins;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        int bin = (data[idx] - 1) * bins / maxVal;
        bin = bin < 0 ? 0 : (bin >= bins ? bins - 1 : bin);
        atomicAdd(&smem[base + bin], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        unsigned int sum = 0u;
        for (int r = 0; r < REPLICAS; ++r)
            sum += smem[r * bins + b];
        atomicAdd(&hist[b], sum);
    }
}

static void cpuHistogram(const std::vector<int> &data, std::vector<unsigned int> &hist,
                         int bins, int maxVal) {
    std::fill(hist.begin(), hist.end(), 0u);
    for (int v : data) {
        int bin = (v - 1) * bins / maxVal;
        bin = bin < 0 ? 0 : (bin >= bins ? bins - 1 : bin);
        ++hist[bin];
    }
}

int main() {
    const int N      = 10'000'000;
    const int bins   = 10;
    const int maxVal = 1000;
    constexpr int REPLICAS = 8;
    const int    ITERS  = 20;

    std::vector<int> h_data(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, maxVal);
    for (int i = 0; i < N; ++i)
        h_data[i] = dist(gen);

    int          *d_data = nullptr;
    unsigned int *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, bins * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));

    const int tpb = 256;
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int coverage = (N + tpb - 1) / tpb;
    int blocks   = std::min(coverage, numSMs * 32);
    size_t shmem = static_cast<size_t>(REPLICAS) * bins * sizeof(unsigned int);

    CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(unsigned int)));
    histogramShared<REPLICAS><<<blocks, tpb, shmem>>>(d_data, d_hist, N, bins, maxVal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalMs = 0.0f;
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventRecord(start));
        histogramShared<REPLICAS><<<blocks, tpb, shmem>>>(d_data, d_hist, N, bins, maxVal);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned int> h_hist(bins, 0u);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, bins * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned int> ref(bins, 0u);
    cpuHistogram(h_data, ref, bins, maxVal);
    bool ok = (h_hist == ref);

    const float avgMs = totalMs / ITERS;
    const double gelems = N / (avgMs * 1.0e6);
    const double gbps   = (N * sizeof(int)) / (avgMs * 1.0e6);

    std::cout << "N = " << N << ", bins = " << bins
              << ", replicas = " << REPLICAS
              << ", blocks = " << blocks << " (" << numSMs << " SMs)\n";
    std::cout << "kernel time: " << avgMs << " ms (avg over " << ITERS << " runs)\n";
    std::cout << "throughput:  " << gelems << " Gelem/s, "
              << gbps << " GB/s read\n\n";

    unsigned long long total = 0;
    for (int i = 0; i < bins; ++i) {
        int lo = (i * maxVal) / bins + 1;
        int hi = ((i + 1) * maxVal) / bins;
        std::cout << "[" << lo << "-" << hi << "]\t" << h_hist[i] << "\n";
        total += h_hist[i];
    }
    std::cout << "\ntotal: " << total << " (expected " << N << ")\n";
    std::cout << "verify: " << (ok ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    return ok ? 0 : 1;
}
