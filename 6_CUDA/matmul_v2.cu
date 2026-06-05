#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

typedef unsigned int uint;

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint innerColA = threadIdx.x % BK; 
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; 
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        int c_idx = (threadRow * TM + resIdx) * N + threadCol;
        C[c_idx] = alpha * threadResults[resIdx] + beta * C[c_idx];
    }
}

// naive kernel
__global__ void naiveSgemm(int M, int N, int K, float alpha, 
                           const float *A, const float *B, float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

bool verifyResult(const std::vector<float>& ref, const std::vector<float>& test, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        // Use an absolute error threshold because floating-point addition order varies
        if (std::abs(ref[i] - test[i]) > 1e-2) {
            printf("Mismatch at index %d: Expected %f, Got %f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_naive(M * N, 0.0f);
    std::vector<float> h_C_tiled(M * N, 0.0f);

    for(int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C_naive, sizeC);
    cudaMalloc(&d_C_tiled, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);
    cudaMemset(d_C_naive, 0, sizeC);
    cudaMemset(d_C_tiled, 0, sizeC);

    dim3 blockNaive(32, 32);
    dim3 gridNaive(CEIL_DIV(N, 32), CEIL_DIV(M, 32));

    const int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 blockTiled(BM * BK); // 512 threads per block
    dim3 gridTiled(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float naiveTime, tiledTime;

    naiveSgemm<<<gridNaive, blockNaive>>>(M, N, K, alpha, d_A, d_B, beta, d_C_naive);
    sgemm1DBlocktiling<BM, BN, BK, TM><<<gridTiled, blockTiled>>>(M, N, K, alpha, d_A, d_B, beta, d_C_tiled);
    cudaDeviceSynchronize();

    cudaMemset(d_C_naive, 0, sizeC); // Reset to ensure beta=0 works properly
    cudaEventRecord(start);
    naiveSgemm<<<gridNaive, blockNaive>>>(M, N, K, alpha, d_A, d_B, beta, d_C_naive);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaMemset(d_C_tiled, 0, sizeC); // Reset
    cudaEventRecord(start);
    sgemm1DBlocktiling<BM, BN, BK, TM><<<gridTiled, blockTiled>>>(M, N, K, alpha, d_A, d_B, beta, d_C_tiled);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiledTime, start, stop);

    cudaMemcpy(h_C_naive.data(), d_C_naive, sizeC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled.data(), d_C_tiled, sizeC, cudaMemcpyDeviceToHost);

    bool correct = verifyResult(h_C_naive, h_C_tiled, M, N);

    std::cout << "Result Matching: " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "Naive Execution Time: " << naiveTime << " ms\n";
    std::cout << "Tiled Execution Time: " << tiledTime << " ms\n";
    std::cout << "Speedup: " << naiveTime / tiledTime << "x\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}