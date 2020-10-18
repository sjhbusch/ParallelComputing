#!/bin/sh
nvcc -O3 1DStencil.cu

echo 'Results for: 1,000,000' > output.txt 
./a.out 1000000 >> output.txt
#./a.out 7 >> output.txt

echo 'Results for: 10,000,000' >> output.txt
./a.out 10000000 >> output.txt


echo 'Results for: 10,000,000' >> output.txt
./a.out 100000000 >> output.txt




