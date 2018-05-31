NVCC 				= 	/usr/local/cuda/bin/nvcc
NVCC_FLAG_GENCODE	=	-gencode arch=compute_35,code=sm_35

CC 					= 	g++
CFLAG_Warnless		=	-w
CFLAG_BASIC			=	-std=c++11 -m64 -O3
CFLAG_CUDA_INC 		= 	-I/usr/local/cuda/include/

all:test

test:main.o kernel.cu.obj CudaManager.cu.obj
	$(CC) $^ -o $@ -dlink -L/usr/local/cuda/lib64 -lcudart

main.o:main.cpp
	$(CC) $(CFLAG_BASIC) $(CFLAG_CUDA_INC) -c $^ -o $@ 

CudaManager.cu.obj:CudaManager.cu
	$(NVCC) $(CFLAG_BASIC) -c $^ -o $@ 

kernel.cu.obj:kernel.cu
	$(NVCC) $(CFLAG_BASIC) -c $^ -o $@ 

.PHONY:clean
clean:
	find ./ -name \*.o -exec rm -rf {} \;
	
