#include <stdio.h>

__global__ void helloKernel() {
	printf("hello world from thread %d, block%d\n",
		threadIdx.x, blockIdx.x);
}

int main() {
// 4 blocks, 32 threads per block = 128 threads = 4 warps
	helloKernel<<<4, 32>>>();
	cudaDeviceSynchronize();
	return 0;
}