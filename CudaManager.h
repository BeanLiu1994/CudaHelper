#pragma once
// Made by BeanLiu
// njueebeanliu@hotmail.com


#include <cuda_runtime.h>
// these headers are from offical sdk
#if __CUDACC_VER_MAJOR__ == 8
#include "common/helper_cuda_80.h"
#elif __CUDACC_VER_MAJOR__ == 9
#include "common/helper_cuda.h"
#else
#define DEVICE_RESET
#endif

#include <algorithm>
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

enum CuState {
	NotInitialized,
	MallocFinisied,
	Deleted
};

class CuPtr_Base
{
protected:
	//仅用于处理GPU端的内存管理
	//能够产生指针并析构时释放空间
	//用户不要再次malloc或free
	CuState state;
	void* d_Ptr;
	//禁止复制
	CuPtr_Base(const CuPtr_Base& rhs) = delete;
	const CuPtr_Base& operator=(const CuPtr_Base& rhs) = delete;
	size_t _Size;
	size_t _TypeSize;
public:

	const size_t& Size;
	const size_t& TypeSize;

	//唯一初始化方式
	CuPtr_Base(size_t __Size) :_Size(__Size), Size(_Size), _TypeSize(1), TypeSize(_TypeSize), state(NotInitialized)
	{ CuMalloc(); }
	~CuPtr_Base() { CuFree(); }
	void CuMalloc();
	void CuGetResult(void* OutPtr);
	void CuFree();
	void* GetDevicePtr() { return d_Ptr; }
	static void CuSyncDevice();
};

template <typename _DataTy>
class CuPtr :public CuPtr_Base
{
	_DataTy* Ptr;
	void CuInitVal(int val);
	void CuCopyFromPtr();
public:
	CuPtr(_DataTy* _Ptr, size_t _Len) : CuPtr_Base(_Len * sizeof(_DataTy)), Ptr(_Ptr)
	{
		_TypeSize = sizeof(_DataTy);
		if (Ptr == nullptr)
			CuInitVal(0);
		else
			CuCopyFromPtr();
	}
	template<
		typename _DataTyNonConst = typename std::remove_const<_DataTy>::type,
		typename Check = std::enable_if<std::is_same<_DataTy, _DataTyNonConst>::value>
	>
	void CuGetResult(_DataTyNonConst* OutPtr = nullptr);
	_DataTy* GetDevicePtr() { return static_cast<_DataTy*>(d_Ptr); }
};


template<typename _DataTy>
void CuPtr<_DataTy>::CuCopyFromPtr()
{
	if (state != CuState::MallocFinisied)
		CuMalloc();
	gpuErrchk(cudaMemcpy(d_Ptr, Ptr, Size, cudaMemcpyHostToDevice));
}


template<typename _DataTy>
void CuPtr<_DataTy>::CuInitVal(int val)
{
	if (state != CuState::MallocFinisied)
		CuMalloc();
	gpuErrchk(cudaMemset(d_Ptr, val, Size));
}


template<typename _DataTy>
template<typename _DataTyNonConst, typename Check>
void CuPtr<_DataTy>::CuGetResult(_DataTyNonConst* OutPtr)
{
	if (state != CuState::MallocFinisied)
		throw std::runtime_error("copy with nullptr");
	if (OutPtr == nullptr)
		OutPtr = Ptr;
	//gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(OutPtr, d_Ptr, Size, cudaMemcpyDeviceToHost));
}


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
