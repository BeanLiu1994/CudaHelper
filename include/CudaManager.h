#pragma once

#include <algorithm>

// these headers are from offical sdk
#if __CUDACC_VER_MAJOR__ == 8
#include "common/helper_cuda_80.h"
#elif __CUDACC_VER_MAJOR__ >= 9
#include "common/helper_cuda.h"
#else
#define DEVICE_RESET
#endif

inline cudaError_t gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		DEVICE_RESET
			if (abort) throw std::runtime_error(cudaGetErrorString(code));
	}
	return code;
}
#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)




class cudaInitializer
{
private:
	cudaInitializer();
public:
	static int Init();
	static int dev;
	static bool CudaOK();
};

inline void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
{
	numThreads = std::min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}
