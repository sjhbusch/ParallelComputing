/*
Assignment #4:

Author: Shawn Hinnebusch

Date: 10/09/2020

To compile: gcc -DDOUBLEPREC -O3 -o hw4.exe  main.c -lm
Single Precision: gcc -O3 -o hw4.exe  main.c -lm
Example input:
*/

#include "timer.h"
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#ifdef DOUBLEPREC
typedef double   REAL;
typedef long int INT;
// NEED TO CHANGE BACK TO LONG_MAX
#define LOOP_MAX LONG_MAX
#define MACHINE_PRECISION DBL_EPSILON

#else
typedef float    REAL;
typedef int INT;
#define LOOP_MAX INT_MAX
#define MACHINE_PRECISION FLT_EPSILON
#endif

int main( )
{
    printf("Largest positive integer that can be represented on a 64-bit machine: %llu\n", ULLONG_MAX);
    printf("Epsilon in single precision: %e\n", FLT_EPSILON);
    printf("Epsilon in double precision: %e\n", DBL_EPSILON);

    REAL   sum = 0;
    REAL   residual;
    double start, finish, elapsedTime;
    
// ########################### Part 1 #########################################

    // Output the results to the file
    FILE *outputPart1;
    outputPart1 = fopen("Problem1Part1.dat", "w");
    if (!outputPart1) return 1;
#ifdef DOUBLEPREC
    fprintf(outputPart1,"Double Precision: Infinite Series summation of 1/n\n");
#else
    fprintf(outputPart1,"Single Precision: Infinite Series summation of 1/n\n");
#endif
    fprintf(outputPart1,"Iteration\tResidual\tsum\n");

    GET_TIME(start);
    INT n = 1;
    for (n = 1; n < LOOP_MAX; n++) {
        sum      = sum + 1.0 / (REAL)n;
        residual = (1.0 / (REAL)n) / sum;

        if (residual < MACHINE_PRECISION) { break; }

#ifdef DOUBLEPREC
        if (n % 10000000 == 0) {
            fprintf(outputPart1,"%12ld\t%e\t%10.15lf\n",n,residual,sum);
 //           fprintf(outputPart1,"%3ld %.15e || %.15e || %.15e\n",n,residual,sum);
        }
#else
        if (n % 10000 == 0) {
            fprintf(outputPart1,"%d\t\t%e\t%9.6f\n",n,residual,sum);
        }
#endif
    }

#ifdef DOUBLEPREC
            fprintf(outputPart1,"%12ld\t%e\t%10.15lf\n",n,residual,sum);
           // fprintf(outputPart1,"%3ld %.15e || %.15e || %.15e\n",n,residual,sum);
#else
            fprintf(outputPart1,"%d\t\t%e\t%9.6f\n",n,residual,sum);
#endif

    // Finish time and display time
    GET_TIME(finish);
    elapsedTime = finish - start;
    fprintf(outputPart1,"Total time 1/n = %f seconds\n", elapsedTime);

    fclose(outputPart1);

// ########################### Part 2 #########################################
    sum = 0;
    GET_TIME(start);

    // Output the results to the file
    FILE *output;
    output = fopen("Problem1Part2.dat", "w");
    if (!output) return 1;

#ifdef DOUBLEPREC
    fprintf(output,"Double Precision: Infinite Series summation of 1/n^2\n");
#else
    fprintf(output,"Single Precision: Infinite Series summation of 1/n^2\n");
#endif
    fprintf(output,"Iteration\tResidual\tsum\n");

    for ( n = 1; n < LOOP_MAX; n++) {
        sum      = sum + 1.0 / (n * n);
        residual = (1.0 / (n * n)) / sum;

        if (residual < MACHINE_PRECISION) { break; }
#ifdef DOUBLEPREC
                if (n % 10000 == 0) {
            fprintf(output,"%10ld\t%e\t%10.15lf\n",n,residual,sum);
#else
                if (n % 100 == 0) {
            fprintf(output,"%d\t\t%e\t%9.6f\n",n,residual,sum);
#endif
        }
    } // end of Loop

// Print final values after convergence
#ifdef DOUBLEPREC
            fprintf(output,"%10ld\t%e\t%10.15lf\n",n,residual,sum);
#else
            fprintf(output,"%d\t\t%e\t%9.6f\n",n,residual,sum);
#endif
    GET_TIME(finish);
    elapsedTime = finish - start;
    fprintf(output,"Total time 1/n^2= %.6f seconds\n", elapsedTime);

    fclose(output);
    return 0;
}
