#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <windows.h> // for high-resolution performance counter

#define WIDTH		(1 * 1024)	// total width is 1024*1024
#define	TILE_WIDTH	32		// block will be (TILE_WIDTH,TILEWIDTH)
#define	GRID_WIDTH	(WIDTH / TILE_WIDTH)	// grid will be (GRID_WDITH,GRID_WDITH)

void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}


void matmul(float* c, const float* a, const float* b, const int width) {
	// c[y][x] = sum_k a[y][k] * b[k][x]
	// c[y * WIDTH + x] = sum_k a[y*WIDTH + k] * b[k*WIDTH + x]
	for (register int y = 0; y < width; ++y) {
		for (register int x = 0; x < width; ++x) {
			register float sum = 0.0F;
			for (register int k = 0; k < width; ++k) {
				sum += a[y * width + k] * b[k * width + x];
			}
			c[y * width + x] = sum;
		}
	}
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
	// generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);
	// start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// perform the action: C = A * B
	matmul(pC, pA, pB, WIDTH);
	// end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));
	// print sample cases
	int i, j;
	i = 0; j = 0; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	// free the memory
	free(pA);
	free(pB);
	free(pC);
}

