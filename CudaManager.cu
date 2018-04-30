#include "CudaManager.h"

#ifdef UseCuda

#include <cuda_runtime.h>

#if __CUDACC_VER_MAJOR__ == 8
#include "common/helper_cuda_80.h"
#elif __CUDACC_VER_MAJOR__ == 9
#include "common/helper_cuda.h"
#else
#define DEVICE_RESET
#endif
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		DEVICE_RESET
		if (abort) throw std::runtime_error(cudaGetErrorString(code));
	}
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void CuPtr::CuMallocAndCopy()
{
	if (state == CuState::MallocFinisied)
		CuFree();
	gpuErrchk(cudaMalloc((void**)&d_Ptr, Size));
	gpuErrchk(cudaMemcpy(d_Ptr, Ptr, Size, cudaMemcpyHostToDevice));
	state = CuState::MallocFinisied;
}
void CuPtr::CuMallocAndSetVal(int val)
{
	if (state == CuState::MallocFinisied)
		CuFree();
	gpuErrchk(cudaMalloc((void**)&d_Ptr, Size));
	gpuErrchk(cudaMemset(d_Ptr, val, Size));
	state = CuState::MallocFinisied;
}
void CuPtr::CuGetResult(void* OutPtr)
{
	if (OutPtr == nullptr)
		OutPtr = Ptr;
	//gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(OutPtr, d_Ptr, Size, cudaMemcpyDeviceToHost));
}
void CuPtr::CuFree()
{
	if (state == CuState::MallocFinisied)
	{
		gpuErrchk(cudaFree(d_Ptr));
		state = CuState::Deleted;
	}
}
void CuPtr::CuSyncDevice()
{
	gpuErrchk(cudaDeviceSynchronize());
}




void CuPtr_Const::CuMallocAndCopy()
{
	if (state == CuState::MallocFinisied)
		CuFree();
	gpuErrchk(cudaMalloc((void**)&d_Ptr, Size));
	gpuErrchk(cudaMemcpy(d_Ptr, Ptr, Size, cudaMemcpyHostToDevice));
	state = CuState::MallocFinisied;
}
void CuPtr_Const::CuMallocAndSetVal(int val)
{
	if (state == CuState::MallocFinisied)
		CuFree();
	gpuErrchk(cudaMalloc((void**)&d_Ptr, Size));
	gpuErrchk(cudaMemset(d_Ptr, val, Size));
	state = CuState::MallocFinisied;
}
void CuPtr_Const::CuFree()
{
	if (state == CuState::MallocFinisied)
	{
		gpuErrchk(cudaFree(d_Ptr));
		state = CuState::Deleted;
	}
}
void CuPtr_Const::CuSyncDevice()
{
	gpuErrchk(cudaDeviceSynchronize());
}

cudaInitializer cudaInitializer::item = cudaInitializer();
cudaInitializer::~cudaInitializer()
{
	if (cudaInitializer::CudaOK())
	{
		gpuErrchk(cudaDeviceReset());
	}
}

int cudaInitializer::dev = -1;
cudaInitializer::cudaInitializer()
{

}
int cudaInitializer::Init()
{
	if (!cudaInitializer::CudaOK())
	{
		try
		{
			dev = findCudaDevice(0, nullptr);
		}
		catch (std::exception& e)
		{
			printf(("[Cuda Initial Failed] " + std::string(e.what()) + " .\n").c_str());
			return dev;
		}
		cudaDeviceProp deviceProp;
		gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));
		//printf("[Cuda Initial Succeed] GPU Device %d: \"%s\" with compute capability %d.%d\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	return dev;
}

bool cudaInitializer::CudaOK()
{
	return dev != -1;
}
#endif