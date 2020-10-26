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
 
 

 
 /*
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
 */

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

    for ( j = 0; j < ny; j++){
        for ( i = 0; i < nx; i++){
            ic = j*nx + i;
            printf("%f\t",x[ic]);
        }
        printf("\n");
    }
}

void initialize(REAL *matrix, const INT nx, const INT ny, const REAL dx, const REAL dy)
{
    INT   i, j, ic;

    for ( j = 1; j < (ny-1); j++){
        for ( i = 1; i < (nx-1); i++){
            ic = j*nx + i;
            REAL x = i * dx;
            REAL y = j * dy;
            //printf("x = %f\n", x);
            //matrix[ic] = 0.1*x*x;
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
    REAL waveConst = 5.0*dt*dt/(2.0*h*h);


    for ( j = 1; j < (ny-1); j++){
        for ( i = 1; i < (nx-1); i++){
            ic = j*nx + i;
            IP1 = j*nx + (i+1);
            IM1 = j*nx + (i-1);
            jP1 = (j+1)*nx + i;
            jM1 = (j-1)*nx + i;
            phiNew[ic] = 2.0*phiCurrent[ic] - phiPrev[ic]
                        + waveConst *
                        phiCurrent[ic] + waveConst*(phiCurrent[IP1]+phiCurrent[IM1]+ phiCurrent[jM1] + phiCurrent[jP1] - 4.0*phiCurrent[ic]);
        }
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
            result += (1.0/(m*m*m*n*n)*cos(t*sqrt(5.0)*PI/4.0)*sqrt(m*m+4.0*n*n)*sin(m*PI*x/4.0)*sin(n*PI*y/2));
        }
    }
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

     int ny = 6;
     int nx = 11;
     //REAL dt = 0.1;
     REAL length = 4.0;
     REAL width = 2.0;
     int size = ny*nx;
     REAL dx = length/(nx-1);
     REAL dy = width/(ny-1);
     REAL time = 2.0; //seconds

    // Alloc memory for arrays
    REAL * phiCurrent, * phiPrev, *phiNew;
    cudaMallocManaged(&phiCurrent, size * sizeof(*phiCurrent));
    cudaMallocManaged(&phiPrev, size * sizeof(*phiPrev));
    cudaMallocManaged(&phiNew, size * sizeof(*phiNew));

    // Memory Allocation for CPU only functions
    REAL *analyticSol      = (REAL*)calloc(nx*ny, sizeof(*analyticSol));

    // Initialize the PDE to Equation 1 
    initialize(phiCurrent,nx,ny,dx,dy);

    analyticalSol(nx, ny, dx, dy, time, analyticSol);
    //phiFirstIteration(phiCurrent, phiPrev,nx,ny,h, dt);

    //for (int i = 0; i < 10; i++){
    //phiNext(phiNew, phiCurrent, phiPrev,nx,ny,h, dt);
    //}

    printOutput(nx,ny,analyticSol);
    //writeSurface(phiCurrent,nx,ny,dx,dy);
    writeOutput(nx,ny,phiCurrent);





/*
for (int j = 1; j < 2; j++){
    for (int i = 0; i < nx; i++){
        int ic = j*nx + i;
        phiNew[ic] = 1.0;
    }
    printf("\n");
}
*/




     return EXIT_SUCCESS;
 }
 