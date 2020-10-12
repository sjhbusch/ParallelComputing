#!/bin/bash
#SBATCH --job-name=HW4
#SBATCH --output=HW4.o%j
#SBATCH --error=HW4.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --time=24:00:00

## Loading proper modules
module purge
module load gcc/8.2.0

## File names and directory (config file, exe, and output should be changed to match different hw assignments)
HOME=/ihome/me2054-2020f/sjh68/hw04

CONFIGFILE=main
EXE=testFile
OUTPUT=outputFile

run_on_exit(){
    pkill --uid=$SLURM_JOB_USER $EXE
}
echo SLURM_NTASKS $SLURM_NTASKS

## Compile code
# Single Precision

gcc -O3 -o $EXE.exe  $CONFIGFILE.c -lm
./$EXE.exe > $OUTPUT.txt

# Double Precision
gcc -DDOUBLEPREC -O3 -o $EXE.exe  $CONFIGFILE.c -lm
./$EXE.exe

## Running executable
#./$EXE.exe
#./$EXE.exe > $OUTPUT.txt

## GNU plotting


## Generating PDF file for final submission (pdftk can be used to combine multiple pdf files)
##a2ps $CONFIGFILE.c -r --columns=2 -chars-per-line=100 -highlight-level=normal --pro=color -o $CONFIGFILE.ps
##ps2pdf $CONFIGFILE.ps $CONFIGFILE.pdf

