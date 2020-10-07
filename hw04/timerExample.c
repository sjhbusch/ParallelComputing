#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        perror("Command-line usage: executableName <# vector size>");
        exit(1);
    }
    long nx = atoi(argv[ 1 ]);

    int    i;
    double start, finish, flop, elapsedTime;

    float *u = malloc(nx * sizeof(*u));

    GET_TIME(start);

    u[ 0 ] = 1.f;

    for (i = 1; i < nx; i++) {
        u[ i ] = 2.f * u[ i - 1 ] + M_PI;
    }

    GET_TIME(finish);

    elapsedTime = finish - start;
    printf("Let's print something %5.3f\n", u[ 5 ]);

    printf("elapsed wall time = %.6f seconds\n", elapsedTime);

    free(u);

    return EXIT_SUCCESS;
}
