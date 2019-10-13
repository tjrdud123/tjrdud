#include <stdio.h>
#include <time.h>

int main(void) {
	time_t t;
	struct tm* ptm;
	int hour, minute, second;
	time( &t );
	ptm = localtime(&t);
	hour = ptm->tm_hour;
	minute = ptm->tm_min;
	second = ptm->tm_sec;
	printf("%02d:%02d:%02d\n", hour, minute, second);
}

