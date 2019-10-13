#include <stdio.h>
#include <time.h>
#include <windows.h>

int main(void) {
	time_t t;
	struct tm* ptm;
	int hour, minute, second;
	while (1) {
		time( &t );
		ptm = localtime(&t);
		hour = ptm->tm_hour;
		minute = ptm->tm_min;
		second = ptm->tm_sec;
		printf("%02d:%02d:%02d\n", hour, minute, second);
		fflush(stdout);
		Sleep(1000);
	}
}

