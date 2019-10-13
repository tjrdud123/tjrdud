#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <windows.h> // for high-resolution performance counter

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

#define WIDTH		(1 * 1024)	// total width is 1024*1024
#define	TILE_WIDTH	32		// block will be (TILE_WIDTH,TILE_WIDTH)
#define	GRID_WIDTH	(WIDTH / TILE_WIDTH)	// grid will be (GRID_WDITH,GRID_WDITH)


void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}


__global__ void matmul(float* c, const float* a, const float* b, const int width) {
	// c[y][x] = sum_k a[y][k] * b[k][x]
	// c[y * WIDTH + x] = sum_k a[y*WIDTH + k] * b[k*WIDTH + x]
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0F;
	for (register int k = 0; k < width; ++k) {
		float lhs = a[y * width + k];
		float rhs = b[k * width + x];
		sum += lhs * rhs;
	}
	c[y * width + x] = sum;
}


int main(void) {
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	// malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));
//	printf("pA, pB, pC = %#x %#x %#x\n", pA, pB, pC);
	// generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);
	// CUDA: allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;
	CUDA_CHECK( cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float)) );
//	printf("pAdev, pBdev, pCdev = %#x %#x %#x\n", pAdev, pBdev, pCdev);
	// CUDA: copy from host to device
	CUDA_CHECK( cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
	// start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul <<< dimGrid, dimBlock>>>(pCdev, pAdev, pBdev, WIDTH);
	// end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	CUDA_CHECK( cudaPeekAtLastError() );
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));
	// CUDA: copy from device to host
	CUDA_CHECK( cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
	// print sample cases
	int i, j;
	i = 0; j = 0; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	// CUDA: free the memory
	CUDA_CHECK( cudaFree(pAdev) );
	CUDA_CHECK( cudaFree(pBdev) );
	CUDA_CHECK( cudaFree(pCdev) );
	// free the memory
	free(pA);
	free(pB);
	free(pC);
}

