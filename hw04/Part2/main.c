/*
Assignment #4: 

Author: Shawn Hinnebusch

Date: 10/09/2020

Single Precision: gcc -O3 -o hw4.exe main.c -lm
Double Precision: gcc -DdoublePres -O3 -o hw4.exe main.c -lm
Example input:
*/

#include "timer.h"
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#ifdef doublePres
typedef double      REAL;
typedef int    INT;
#define LOOP_MAX INT_MAX
#define MACHINE_PRECISION DBL_EPSILON

#else 
typedef float   REAL;
typedef  int    INT;
#define LOOP_MAX INT_MAX
#define MACHINE_PRECISION FLT_EPSILON
#endif

void exponentialFunctin(INT a);

int main( )
{
    INT a[10] = {1, 5, 10, 15, 20, -1, -5, -10, -15, -20};

#ifdef doublePres
    printf("x\tResidual\tTaylor\t\t\texp(x)\t\t\tIterations\n");
#else 
    printf("x\tResidual\tTaylor\t\texp(x)\t\tIterations\n");
#endif

    for (int i = 0; i < 10; i++){
        exponentialFunctin(a[i]);
    }
    return 0;
}

void exponentialFunctin(INT factNum){
    REAL residual = 1;
    INT n;
    REAL stoppingCrit;
    REAL expNum = 1;
    REAL expNumNeg = 1;
    for (n = 1; n <= LOOP_MAX; n++){ 
        if(factNum > 0){
            residual = residual*factNum/n;
            expNum = expNum + residual;
            stoppingCrit = fabs(residual) / fabs(expNum);
        }
        else {
            residual = residual*(-1)*factNum/n;
            expNumNeg = expNumNeg + residual;
            expNum = 1/expNumNeg;           
            stoppingCrit = fabs(residual) / fabs(expNum);

        }

        if (stoppingCrit < MACHINE_PRECISION ){ break;}
    }

#ifdef doublePres
    printf("%d\t%e\t%.15e\t%.15e\t%d\n",factNum,residual,expNum,exp(factNum),n);
#else 
    printf("%d\t%e\t%.6e\t%.6e\t%d\n",factNum,residual,expNum,exp(factNum),n);
#endif
}