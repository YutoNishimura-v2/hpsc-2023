10_cavity: 10_cavity.cpp
	g++ 10_cavity.cpp -O3 -I/home/7/23R14101/hpsc-2023/14_pde -I/usr/include/python2.7 -lpython2.7

10_cavity_cu: 10_cavity.cu
	nvcc 10_cavity.cu -Xcompiler "-O3 -fopenmp" -lcublas