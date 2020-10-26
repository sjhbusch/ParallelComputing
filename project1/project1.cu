/* Project 1
 * Finite Difference Solution of a Vibrating 
 * 2D Membrane on a GPU
 * Author: Shawn Hinnebusch
 * Date: 10/30/2020
 *
 * Part 2 code is VECTORADD 1
 * Part 3-5 code is VECTORADD 0

 * To compile locally: nvcc -O3 -o hw5.exe  hw5.cu -lm

 * To compile on the CRC:
 * crc-interactive.py -g -u 1 -t 1 -p gtx1080
 * nvcc -O3 -arch=sm_61 -o hw5.exe  hw5.cu -lm

 * To run:
 * 
 * 

 * Create PDF: 
 * a2ps hw5.cu --pro=color --columns=2 -E --pretty-print='c' -o hw5.ps | ps2pdf hw5.ps

 * Compress: tar czvf Hinnebusch_proj1.tar.gz project1/
 */

#include "timer_nv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>


typedef float REAL;
typedef int   INT;
#define RADIUS 3
#define BLOCK_SIZE 256

void solveHeat_1D(REAL *unew, const REAL *u, const REAL *x)
{
    INT  i;
    REAL dxi = 1.f / (DX * DX);
    REAL xc, source;

    for (i = 1; i < NX - 1; i++) {
        xc     = x[ i ];
        source = -(xc * xc - 4.f * xc + 2.f) * exp(-xc); // source term
        unew[ i ]
        = (ALPHA * (u[ i + 1 ] - 2.0f * u[ i ] + u[ i - 1 ]) * dxi + source) * DT + u[ i ];
    }
}
void exactSolution(REAL *uExact, const REAL *x)
{
    INT i;
    for (i = 0; i < NX; i++) {
        uExact[ i ] = x[ i ] * x[ i ] * exp(-x[ i ]);
    }
}

void meshGrid(REAL *x)
{
    INT i;
    for (i = 0; i < NX; i++) {
        x[ i ] = DX * (( REAL ) i);
    }
}

/*
 int row = blockIdx.y * blockDim.y + threadIdx.y;
 int col = blockIdx.x * blockDim.x + threadIdx.x;

*/


void writeOutput(const REAL *x, const REAL *uExact, const REAL *u)
{
    INT   i;
    FILE *output;
    output = fopen("1d_heat.dat", "w");

    for (i = 0; i < NX; i++) {
        fprintf(output, "%10f %10f %10f\n", x[ i ], uExact[ i ], u[ i ]);
    }
    fclose(output);
}

INT main(INT argc, char *argv[])
{
    if (argc < 2) {
        perror("Command-line usage: executableName <end Time (seconds)>");
        exit(1);
    }

    REAL endTime = atof(argv[ 1 ]);

    REAL *uExact, *x;
    REAL *unew, *u, *tmp;
    //  allocate heap memory here for arrays needed in the solution algorithm

    // calculate the x coordinates of each computational point
    meshGrid(x);
    // compute the exact solution to the 1D heat conduction problem
    exactSolution(uExact, x);

    // apply boundary conditions (make sure to apply boundary conditions to both u and unew)
    u[ 0 ]         = 0.f;
    unew[ 0 ]      = 0.f;
    unew[ NX - 1 ] = uExact[ NX - 1 ];
    u[ NX - 1 ]    = uExact[ NX - 1 ];

    REAL time = 0.f;
    while (time < endTime) {
        // call the solveHeat_1D( ) function here with correct parameters
        // and necessary updates on the solution array

        time += DT;
    }

    // call the writeOutput( ) function here with correct parameters

    free(unew);
    free(u);
    free(uExact);
    free(x);

    return EXIT_SUCCESS;
}
