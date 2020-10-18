/*
 * Simple CPU program to add two long vectors
 *
 * Author: Shawn Hinnebusch
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer_nv.h"

typedef float REAL;
typedef int INT;

__global__ void vector_add_gpu (const INT n, const REAL *a, const REAL *b, REAL *c) 
{
      INT tid = blockIdx.x*blockDim.x + threadIdx.x;

      if (tid <  n)
         c[tid] = a[tid] + b[tid];
}

__global__ void stencil_1d(INT *in, INT *out) {
    __shared__ INT temp[BLOCK_SIZE + 2*RADIUS];
    INT gindex = threadIdx.x + blockIdx.x * blockDim.x;
    INT lindex = threadIdx.x + RADIUS;

    // Read om[it e;e,emts omtp shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS){
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex+BLOCK_SIZE];
    }
}


void vector_add_cpu(const INT n, const REAL *a, const REAL *b, REAL *c)
{
  for (int i = 0; i<n; i++)
      c[i] = a[i] + b[i];
}


int main(INT argc, char *argv[])
{

    if (argc < 2) {
       perror("Command-line usage: executableName <vector size>");
       exit(1);
    }

    int n = atof(argv[1]);

    REAL *x, *y, *z;

    cudaMallocManaged(  &x, n * sizeof (*x));
    cudaMallocManaged(  &y, n * sizeof (*y));
    cudaMallocManaged(  &z, n * sizeof (*z));

    for (int i = 0; i < n; i++){
        x[i] = 3.5;
        y[i] = 1.5;
    }

    StartTimer();
    
    vector_add_cpu(n,x,y,z);
    printf("vector_add on the CPU. z[100] = %4.2f\n",z[100]);

    double cpu_elapsedTime = GetTimer(); //elapsed time is in seconds

    for (int i = 0; i < n; i++){
        z[i] = 0.0;
    }

    cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
    cudaEventCreate(&timeStart);
    cudaEventCreate(&timeStop);
    float gpu_elapsedTime; // make sure it is of type float, precision is milliseconds (ms) !!!

    int blockSize = 256;
    int nBlocks   = (n + blockSize -1) / blockSize; //round up if n is not a multiple of blocksize

    cudaEventRecord(timeStart, 0); //don't worry for the 2nd argument zero, it is about cuda streams

    vector_add_gpu <<< nBlocks, blockSize >>> (n, x, y, z);
    cudaDeviceSynchronize();

    printf("vector_add on the GPU. z[100] = %4.2f\n",z[100]);

    cudaEventRecord(timeStop, 0);
    cudaEventSynchronize(timeStop);

    //WARNING!!! do not simply print (timeStop-timeStart)!!

    cudaEventElapsedTime(&gpu_elapsedTime, timeStart, timeStop);

    printf("elapsed wall time (CPU) = %5.4f ms\n", cpu_elapsedTime*1000.);
    printf("elapsed wall time (GPU) = %5.4f ms\n\n", gpu_elapsedTime);

    cudaEventDestroy(timeStart);
    cudaEventDestroy(timeStop);

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return EXIT_SUCCESS;
}
