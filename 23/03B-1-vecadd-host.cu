#include <cstdio>

// main program for the CPU: compiled by MS-VC++
int main(void) {
	// host-side data
	const int SIZE = 5;
	const int a[SIZE] = { 1, 2, 3, 4, 5 };
	const int b[SIZE] = { 10, 20, 30, 40, 50 };
	int c[SIZE] = { 0 };
	// calculate the addition
	for (register int i = 0; i < SIZE; ++i) {
		c[i] = a[i] + b[i];
	}
	// print the result
	printf("{%d,%d,%d,%d,%d} + {%d,%d,%d,%d,%d} "
	       " = {%d,%d,%d,%d,%d}\n",
	       a[0], a[1], a[2], a[3], a[4],
	       b[0], b[1], b[2], b[3], b[4],
	       c[0], c[1], c[2], c[3], c[4]);
	// done
	return 0;
}

