#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <io.h> // for open(), write(), close() in WIN32
#include <fcntl.h> // for open(), write()
#include <sys/stat.h>
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

#define GRIDSIZE	(8 * 1024)
#define BLOCKSIZE	1024
#define TOTALSIZE	(GRIDSIZE * BLOCKSIZE)


void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}


void getDiff(float* dst, const float* src, unsigned int size) {
	for (register int i = 1; i < size; ++i) {
		dst[i] = src[i] - src[i - 1];
	}
}


void writeData(char* filename, const float* src, unsigned int size) {
	int fd = open(filename, O_WRONLY | O_BINARY | O_CREAT, S_IREAD | S_IWRITE);
	write(fd, src, size * sizeof(float));
	close(fd);
	printf("data written to \"%s\"\n", filename);
}


__global__ void adj_diff_shared(float* result, float* input) {
	__shared__ float s_data[BLOCKSIZE];
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	register float answer;
	s_data[tx] = input[i];
	__syncthreads();
	if (tx > 0) {
		answer = s_data[tx] - s_data[tx - 1];
	} else if (i > 0) {
		answer = s_data[tx] - input[i - 1];
	}
	result[i] = answer;
}


int main(void) {
	float* pSource = NULL;
	float* pResult = NULL;
	int i;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	// malloc memories on the host-side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));
	// generate source data
	genData(pSource, TOTALSIZE);
	// CUDA: allocate device memory
	float* pSourceDev = NULL;
	float* pResultDev = NULL;
	CUDA_CHECK( cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float)) );
	// CUDA: copy from host to device
	CUDA_CHECK( cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice) );
	// start timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// CUDA: launch the kernel: result[i] = input[i] - input[i-1]
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	adj_diff_shared <<< dimGrid, dimBlock>>>(pResultDev, pSourceDev);
	// end timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	printf("elapsed time = %f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));
	// CUDA: copy from device to host
	CUDA_CHECK( cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost) );
	// write the result on the disk
	// writeData("host.out", pResult, TOTALSIZE);
	// print sample cases
	i = 0;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE - 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE / 2;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	// CUDA: free the memory
	CUDA_CHECK( cudaFree(pSourceDev) );
	CUDA_CHECK( cudaFree(pResultDev) );
	// free the memory
	free(pSource);
	free(pResult);
}

