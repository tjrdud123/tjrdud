// common.h

#if ! defined(MY_COMMON_H)
#define MY_COMMON_H

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

#endif // MY_COMMON_H
