#include <cstdio>

int main(void) {
	// host-side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };
	// make a, b matrices
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			a[y][x] = y + x;
			b[y][x] = y + x;
		}
	}
	// print the result
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			printf("%5d", a[y][x]);
		}
		printf("\n");
	}
	// calculate
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			int sum = 0;
			for (int k = 0; k < WIDTH; ++k) {
				sum += a[y][k] * b[k][x];
			}
			c[y][x] = sum;
		}
	}
	// print the result
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			printf("%5d", c[y][x]);
		}
		printf("\n");
	}
	// done
	return 0;
}

