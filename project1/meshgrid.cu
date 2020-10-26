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

 #include <math.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <sys/resource.h>
 
 
 typedef float REAL;
 typedef int   INT;
 #define BLOCK_SIZE 256

#define PI M_PI 

 void writeOutput(const INT nx, const INT ny, const REAL *x)
{
    INT   i, j, ic;
    FILE *output;
    output = fopen("MatrixPrint.dat", "w");

    for ( j = 0; j < ny; j++){
        for ( i = 0; i < nx; i++){
            ic = j*nx + i;
            fprintf(output,"%f",x[ic]);
            if(i<(nx-1)){fprintf(output," ");}
        }
        fprintf(output,"\n");
    }

    fclose(output);
}

void printOutput(const INT nx, const INT ny, const REAL *x)
{
    INT   i, j, ic;

    for ( j = 0; j < ny; j++){ //ny
        for ( i = 0; i < 10; i++){ //nx
            ic = j*nx + i;
            printf("%f\t",x[ic]);
        }
        printf("\n");
    }
}

void printConstX(const INT nx, const INT ny, const REAL *x, const REAL dx)
{
    INT   i, j, ic;
    i = 10;

    for ( j = 0; j < ny; j++){ //ny
            ic = j*nx + i;
            REAL y = j * dx;
            printf("%f\t%f\n",x[ic],y);
        }
        printf("\n");
}

void initialize(REAL *matrix, const INT nx, const INT ny, const REAL dx, const REAL dy)
{
    INT   i, j, ic;

    for ( j = 1; j < (ny-1); j++){
        for ( i = 1; i < (nx-1); i++){
            ic = j*nx + i;
            REAL x = i * dx;
            REAL y = j * dy;
            matrix[ic] = 0.1*(4.0*x-x*x)*(2.0*y-y*y);
        }
    }
}

    void phiFirstIteration(const REAL *phiCurrent, REAL *phiPrev, const INT nx, const INT ny, const REAL h, const REAL dt)
    {
        INT   i, j, ic, IP1, IM1, jP1, jM1;
        REAL waveConst = 5.0*dt*dt/(2.0*h*h);
    
    
        for ( j = 1; j < (ny-1); j++){
            for ( i = 1; i < (nx-1); i++){
                ic = j*nx + i;
                IP1 = j*nx + (i+1);
                IM1 = j*nx + (i-1);
                jP1 = (j+1)*nx + i;
                jM1 = (j-1)*nx + i;
                phiPrev[ic] = phiCurrent[ic] + waveConst*(phiCurrent[IP1]+phiCurrent[IM1]+ phiCurrent[jM1] + phiCurrent[jP1] - 4.0*phiCurrent[ic]);
            }
        }

}


void phiNext(REAL * phiNew, REAL *phiCurrent, REAL *phiPrev, const INT nx, const INT ny, const REAL h, const REAL dt)
{
    INT   i, j, ic, IP1, IM1, jP1, jM1;
    REAL waveConst = 5.0*dt*dt/(h*h);


    for ( j = 1; j < (ny-1); j++){
        for ( i = 1; i < (nx-1); i++){
            ic = j*nx + i;
            IP1 = j*nx + (i+1);
            IM1 = j*nx + (i-1);
            jP1 = (j+1)*nx + i;
            jM1 = (j-1)*nx + i;
            phiNew[ic] = 2.0*phiCurrent[ic] - phiPrev[ic]
                       + waveConst*(phiCurrent[IP1]+phiCurrent[IM1]+ phiCurrent[jM1] + phiCurrent[jP1] - 4.0*phiCurrent[ic]);
        }
    }

}

__global__ void phiNextGPU(REAL * phiNew, REAL *phiCurrent, REAL *phiPrev, const INT nx, const INT ny, const REAL h, const REAL dt)
{

    INT   i, j, ic, IP1, IM1, jP1, jM1;
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    REAL waveConst = 5.0*dt*dt/(h*h);

    if (i != 0 && i < (nx-1) && j != 0 && j < (ny-1) ) {
    ic = j*nx + i;
    IP1 = j*nx + (i+1);
    IM1 = j*nx + (i-1);
    jP1 = (j+1)*nx + i;
    jM1 = (j-1)*nx + i;
    phiNew[ic] = 2.0*phiCurrent[ic] - phiPrev[ic]
                + waveConst*(phiCurrent[IP1]+phiCurrent[IM1]+ phiCurrent[jM1] + phiCurrent[jP1] - 4.0*phiCurrent[ic]);

    }

}

void writeSurface(REAL *matrix, const INT nx, const INT ny, const REAL dx, const REAL dy)
{
    INT   i, j, ic;

    FILE *output;
    output = fopen("Surface.dat", "w");

    for ( j = 0; j < ny; j++){
        for ( i = 0; i < nx; i++){
            ic = j*nx + i;
            REAL x = i * dx;
            REAL y = j * dy;
            fprintf(output,"%f,%f,%f\n",x,y,matrix[ic]);

        }
        //fprintf(output,"\n");
    }


    fclose(output);

}

REAL  phiInnerLoop( const REAL x, const REAL y, const REAL t) {
    REAL result = 0;
    for(int m = 1; m < 100; m+=2) {
        for (int n = 1; n < 100; n+=2) {
            result += (1.0/(m*m*m*n*n*n)*cos((t*sqrt(5.0)*PI*0.25)*sqrt(m*m+4.0*n*n))*sin(m*PI*x*0.25)*sin(n*PI*y*0.5));
        }
    }
    result = 0.426050*result;
    return result;
}

void analyticalSol(const INT nx, const INT ny, const REAL dx, const REAL dy, const REAL t, REAL *matrix)
{
    INT   ic;

    for (int j = 1; j < (ny-1); j++){
        for (int i = 1; i < (nx-1); i++){
            ic = j*nx + i;
            REAL x = i * dx;
            REAL y = j * dy;
            matrix[ic] = phiInnerLoop(x,y,t);
        }
    }
}

 
 INT main()
 {

     int ny = 21;
     int nx = 41;
     REAL length = 4.0;
     REAL width = 2.0;
     int size = ny*nx;
     REAL dx = length/(nx-1);
     REAL dy = width/(ny-1);
     REAL time = 1.0; //seconds
     REAL h = dx;
     REAL * temp;

     REAL dt = 0.1*h/sqrt(5.0); // NEEEEEEEEEEEEED TOOOOOOOOOOO CHANGEEEEEEEEEEEEEEE
     int numOfLoops = (int) ceil(time/dt);
     printf("dx = %f\n",dx);
     printf("dy = %f\n",dy);
     printf("dt = %f\n",dt);

    // Alloc memory for arrays
    REAL * phiCurrent, * phiPrev, *phiNew;
    cudaMallocManaged(&phiCurrent, size * sizeof(*phiCurrent));
    //cudaMallocManaged(&phiPrev, size * sizeof(*phiPrev));
    cudaMallocManaged(&phiNew, size * sizeof(*phiNew));

    // Memory Allocation for CPU only functions
    REAL *analyticSol      = (REAL*)calloc(nx*ny, sizeof(*analyticSol));
    REAL *phinew_CPU       = (REAL*)calloc(nx*ny, sizeof(*phinew_CPU));
    REAL *phiCurrent_CPU   = (REAL*)calloc(nx*ny, sizeof(*phiCurrent_CPU));
    REAL *phiPrev_CPU   = (REAL*)calloc(nx*ny, sizeof(*phiPrev_CPU));

    //analyticalSol(nx, ny, dx, dy, time, analyticSol);


    // GPU
	//dim3 threadsPerBlock(32, 32);
    //dim3 numBlocks((nx - 1) / threadsPerBlock.x + 1, (ny - 1) / threadsPerBlock.y + 1);
    //phiNextGPU<<<numBlocks,threadsPerBlock>>> (phiNew, phiCurrent, phiPrev,nx,ny,h, dt);

    initialize(phiCurrent_CPU,nx,ny,dx,dy);

    
    // CPU
    phiFirstIteration(phiCurrent_CPU, phiPrev_CPU,nx,ny,h, dt);

    int i = 0;
    for (i = 0; i < numOfLoops; i++){
        phiNext(phinew_CPU, phiCurrent_CPU, phiPrev_CPU,nx,ny,h, dt);
        temp = phiPrev_CPU;

        phiPrev_CPU = phiCurrent_CPU;

        phiCurrent_CPU = phinew_CPU;
        phinew_CPU = temp;
    }


    //printOutput(nx,ny,phinew_CPU);
    printConstX(nx,ny,phiCurrent_CPU,dx);

    //printf("\nAnalytic Sol\n");
    //printConstX(nx,ny,analyticSol,dx);

    //printOutput(nx,ny,analyticSol);


    //writeSurface(phiCurrent,nx,ny,dx,dy);
    writeOutput(nx,ny,phiCurrent_CPU);


     return EXIT_SUCCESS;
 }
 































