#include "CudaManager.h"
// Made by BeanLiu
// njueebeanliu@hotmail.com
#include <iostream>

void CuPtr_Base::CuMalloc()
{
	if (state == CuState::MallocFinisied)
		CuFree();
	gpuErrchk(cudaMalloc((void**)&d_Ptr, Size));
	state = CuState::MallocFinisied;
}
void CuPtr_Base::CuGetResult(void* OutPtr)
{
	if (OutPtr == nullptr || state != CuState::MallocFinisied)
		throw std::runtime_error("copy with nullptr");
	//gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(OutPtr, d_Ptr, Size, cudaMemcpyDeviceToHost));
}
void CuPtr_Base::CuFree()
{
	if (state == CuState::MallocFinisied)
	{
		gpuErrchk(cudaFree(d_Ptr));
		state = CuState::Deleted;
	}
}
void CuPtr_Base::CuSyncDevice()
{
	gpuErrchk(cudaDeviceSynchronize());
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
			std::cerr<<std::string(e.what())<<std::endl;
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


