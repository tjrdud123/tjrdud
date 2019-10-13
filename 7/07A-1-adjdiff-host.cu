#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <io.h> // for open(), write(), close() in WIN32
#include <fcntl.h> // for open(), write()
#include <sys/stat.h>
#include <windows.h> // for high-resolution performance counter


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
	// start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// perform the action: result[i] = input[i] - input[i-1]
	pResult[0] = 0.0F; // exceptional case
	getDiff(pResult, pSource, TOTALSIZE);
	// end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	printf("elapsed time = %f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));
	// write the result on the disk
	// writeData("host.out", pResult, TOTALSIZE);
	// print sample cases
	i = 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE - 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE / 2;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	// free the memory
	free(pSource);
	free(pResult);
}

