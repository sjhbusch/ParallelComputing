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
 
 

 
 /*
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
 */

 
 INT main()
 {

     int ny = 5;
     int nx = 6;

     REAL **mat;
     cudaMallocManaged(&mat, sizeof(REAL *) * ny);

     for (int i = 0; i < ny; i++) {
         cudaMallocManaged(&mat[i], sizeof(REAL) * nx);
     }

     REAL **phiNew;
     cudaMallocManaged(&mat, sizeof(REAL *) * ny);

     for (int i = 0; i < ny; i++) {
         cudaMallocManaged(&phiNew[i], sizeof(REAL) * nx);
     }

/*
    for (int row = 0; row < ny; row++) {
        for (int colm = 0; colm < nx; colm++) {
            mat[ row ][colm] = 1;
        }
    }
*/  


    for (int row = 0; row < ny; row++) {
        for (int colm = 0; colm < nx; colm++) {
            printf("%f\t",mat[ row ][colm]);
        }
        printf("\n");
    }

     return EXIT_SUCCESS;
 }
 