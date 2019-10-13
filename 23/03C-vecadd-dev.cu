#include <cstdio>

#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (cudaSuccess != e) { \
			printf("cuda failure \"%s\" at %s:%d\n", \
			       cudaGetErrorString(e), \
			       __FILE__, __LINE__); \
			exit(1); \
		} \
	} while (0)
#endif


// kernel program for the device (GPU): compiled by NVCC
__global__ void addKernel(int* c, const int* a, const int* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


// main program for the CPU: compiled by MS-VC++
int main(void) {
	// host-side data
	const int SIZE = 5;
	const int a[SIZE] = { 1, 2, 3, 4, 5 };
	const int b[SIZE] = { 10, 20, 30, 40, 50 };
	int c[SIZE] = { 0 };
	// device-side data
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	// allocate device memory
	CUDA_CHECK( cudaMalloc((void**)&dev_a, SIZE * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_b, SIZE * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_c, SIZE * sizeof(int)) );
	// copy from host to device
	CUDA_CHECK( cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice) ); // dev_a = a;
	CUDA_CHECK( cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice) ); // dev_b = b;
	// launch a kernel on the GPU with one thread for each element.
	addKernel<<< 1, SIZE>>>( dev_c, dev_a, dev_b );		// dev_c = dev_a + dev_b;
	CUDA_CHECK( cudaPeekAtLastError() );
	// copy from device to host
	CUDA_CHECK( cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost) ); // c = dev_c;
	// free device memory
	CUDA_CHECK( cudaFree(dev_c) );
	CUDA_CHECK( cudaFree(dev_a) );
	CUDA_CHECK( cudaFree(dev_b) );
	// print the result
	printf("{%d,%d,%d,%d,%d} + {%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d}\n",
	       a[0], a[1], a[2], a[3], a[4],
	       b[0], b[1], b[2], b[3], b[4],
	       c[0], c[1], c[2], c[3], c[4]);
	// done
	return 0;
}

