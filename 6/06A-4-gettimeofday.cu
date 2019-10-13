#include <sys/time.h>
#include <unistd.h>

int main(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	printf("%d.%06d\n", tv.tv_sec, tv.tv_usec);
	
	gettimeofday(&tv, NULL);
	printf("%d.%06d\n", tv.tv_sec, tv.tv_usec);
}

