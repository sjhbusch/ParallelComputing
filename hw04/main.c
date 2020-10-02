/*
Assignment #2: 

Author: Shawn Hinnebusch

Date: 9/25/2020

To compile: gcc -o hw2.exe main.c -lm
Example input:
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>


int main( )
{

    printf("Epsilon in single precision: %e\n", FLT_EPSILON);
     printf("Epsilon in double precision: %e\n", DBL_EPSILON);


    return 0;
}
