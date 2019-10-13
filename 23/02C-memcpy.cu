#include <cstdio>

// main program for the CPU: compiled by MS-VC++
int main(void) {
	// host-side data
	const int SIZE = 5;
	const int a[SIZE] = { 1, 2, 3, 4, 5 };
	int b[SIZE] = { 0, 0, 0, 0, 0 };
	// print source
	printf("a = {%d,%d,%d,%d,%d}\n", a[0], a[1], a[2], a[3], a[4]);
	// device-side data
	int* dev_a = 0;
	int* dev_b = 0;
	// allocate device memory
	cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
	// copy from host to device
	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice); // dev_a = a;
	// copy from device to device
	cudaMemcpy(dev_b, dev_a, SIZE * sizeof(int), cudaMemcpyDeviceToDevice); // dev_b = dev_a;
	// copy from device to host
	cudaMemcpy(b, dev_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost); // b = dev_b;
	// free device memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	// print the result
	printf("b = {%d,%d,%d,%d,%d}\n", b[0], b[1], b[2], b[3], b[4]);
	// done
	return 0;
}

