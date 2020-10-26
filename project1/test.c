
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>


typedef float REAL;
typedef int   INT;




/*
 int row = blockIdx.y * blockDim.y + threadIdx.y;
 int col = blockIdx.x * blockDim.x + threadIdx.x;
*/


INT main()
{
    float x = 100;


    printf("Does this work?\n");

    x = 200;

    return EXIT_SUCCESS;
}