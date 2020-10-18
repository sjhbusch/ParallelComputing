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
#define RADIUS 3
#define BLOCK_SIZE 1024


__global__ void stencil_1d(REAL *in, REAL *out, INT n) {
    __shared__ REAL temp[BLOCK_SIZE + 2*RADIUS];
    INT gindex = threadIdx.x + blockIdx.x * blockDim.x;
    INT lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    __syncthreads();

    temp[lindex] = in[gindex];
    // Fills temp vector with previous value unless its the first block
    if (threadIdx.x < RADIUS && gindex > RADIUS) {
	temp[lindex - RADIUS] = in[gindex - RADIUS];
	// Check to not exceed the largest block size before filling in 
	// the last 2 ghost cells
        if(gindex+ BLOCK_SIZE < n){
	temp[lindex + BLOCK_SIZE] = in[gindex+BLOCK_SIZE];
	}
    }
    __syncthreads();

    // Apply the stencil
    REAL result = 0.0;
    if (gindex >= RADIUS && gindex < (n - RADIUS))
    {
        for (int offset = -RADIUS; offset <= RADIUS; offset++){
            result += temp[lindex + offset];
        }
    }

    // Store the result
    out[gindex] = result;
}

void stencil_1d_cpu(const REAL *in, REAL *out, INT n)
{
  for (int i = RADIUS; i<(n-RADIUS); i++){
      for (int offset = -RADIUS; offset <= RADIUS; offset++){
        out[i] += in[i + offset];
      }
    }
}



int main(INT argc, char *argv[])
{

    if (argc < 2) {
       perror("Command-line usage: executableName <vector size>");
       exit(1);
    }

    int n = atof(argv[1]);

    //int radius = atof(argv[2]);
    printf("N: %d\n",n);
    //printf("Radius is: %d\n",radius);

    REAL *x, *y;

    cudaMallocManaged(  &x, n * sizeof (*x));
    cudaMallocManaged(  &y, n * sizeof (*y));

    for (int i = 0; i < n; i++){
        x[i] = 1.0;
        //x[i] = i;
        y[i] = 0.0;
    }
    
    //printf("x[0] = %4.2f\n",x[0]);

    // CPU Run

    StartTimer();
    stencil_1d_cpu(x,y,n);
    double cpu_elapsedTime = GetTimer(); //elapsed time is in seconds

    // GPU Run
    cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
    cudaEventCreate(&timeStart);
    cudaEventCreate(&timeStop);
    float gpu_elapsedTime; // make sure it is of type float, precision is milliseconds (ms) !!!

    int nBlocks   = (n + BLOCK_SIZE -1) / BLOCK_SIZE; //round up if n is not a multiple of blocksize

    cudaEventRecord(timeStart, 0); //don't worry for the 2nd argument zero, it is about cuda streams

    printf("nBlocks: %d\n", nBlocks);
    stencil_1d <<< nBlocks, BLOCK_SIZE >>> (x, y, n);
    cudaDeviceSynchronize();

    cudaEventRecord(timeStop, 0);
    cudaEventSynchronize(timeStop);

    cudaEventElapsedTime(&gpu_elapsedTime, timeStart, timeStop);

    printf("elapsed wall time (CPU) = %5.4f ms\n", cpu_elapsedTime*1000.);
    printf("elapsed wall time (GPU) = %5.4f ms\n\n", gpu_elapsedTime);

    cudaEventDestroy(timeStart);
    cudaEventDestroy(timeStop);
    
    //Used to print final results of CPU or GPU
    //for (int i = 0; i < n; i++) {
    //    printf("%f\n", y[i]);
    //}

    cudaFree(x);
    cudaFree(y);

    return EXIT_SUCCESS;
}

