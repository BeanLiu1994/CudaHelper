#pragma once
// Made by BeanLiu
#define UseCuda

// CuPtr(nullptr, size)

#ifdef UseCuda
#include <algorithm>
#include <cuda_runtime.h>

enum CuState {
	NotInitialized,
	MallocFinisied,
	Deleted
};

/* 传入结束后最好不要使用Ptr内容 */
/*  OutPtr一定要手动分配好内存  */
/*  或者用其他库已经分好的空间，大小要一致  */
struct CuPtr
{
private:
	CuState state;
	void* d_Ptr;
	CuPtr(const CuPtr& rhs) = delete;
	const CuPtr& operator=(const CuPtr& rhs) = delete;
	//CuPtr(CuPtr&& move) {};
public:
	void* Ptr;
	size_t Size;
	size_t TypeSize;
	template <class DataType>
	CuPtr(DataType* _Ptr, size_t _Len);
	CuPtr(void* _Ptr, size_t _Size) :Ptr(_Ptr), Size(_Size), TypeSize(1), state(NotInitialized)
	{
		if (Ptr == nullptr)
			CuMallocAndSetVal(0);
		else
			CuMallocAndCopy();
	}
	void CuMallocAndCopy();
	void CuMallocAndSetVal(int val);
	void CuGetResult(void* OutPtr = nullptr);
	void CuFree();
	void* GetDevicePtr() { return d_Ptr; }
	static void CuSyncDevice();
	~CuPtr() { CuFree(); }
};

template <class DataType>
CuPtr::CuPtr(DataType* _Ptr, size_t _Len) : Ptr(_Ptr), Size(_Len * sizeof(DataType)), state(NotInitialized), TypeSize(sizeof(DataType))
{
	if (Ptr == nullptr)
		CuMallocAndSetVal(0);
	else
		CuMallocAndCopy();
}

/* 传入结束后后续不能修改Ptr指向的内容 */
/* 一般传递不变的量，因为在gpu修改后不能直接复制到源地址 */
struct CuPtr_Const
{
private:
	CuState state;
	void* d_Ptr;
	CuPtr_Const(const CuPtr_Const& rhs) = delete;
	const CuPtr_Const& operator=(const CuPtr_Const& rhs) = delete;
	//CuPtr_Const(CuPtr_Const&& move) {};
public:
	const void* Ptr;
	size_t Size;
	size_t TypeSize;
	template <class DataType>
	CuPtr_Const(const DataType* _Ptr, size_t _Len);
	CuPtr_Const(const void* _Ptr, size_t _Size) :Ptr(_Ptr), Size(_Size), TypeSize(1), state(NotInitialized)
	{
		if (Ptr == nullptr)
			CuMallocAndSetVal(0);
		else
			CuMallocAndCopy();
	}
	void CuMallocAndCopy();
	void CuMallocAndSetVal(int val);
	void CuFree();
	void* GetDevicePtr() { return d_Ptr; }
	static void CuSyncDevice();
	~CuPtr_Const() { CuFree(); }
};

template <class DataType>
CuPtr_Const::CuPtr_Const(const DataType* _Ptr, size_t _Len) : Ptr(_Ptr), Size(_Len * sizeof(DataType)), state(NotInitialized), TypeSize(sizeof(DataType))
{
	if (Ptr == nullptr)
		CuMallocAndSetVal(0);
	else
		CuMallocAndCopy();
}


class cudaInitializer
{
private:
	cudaInitializer();
	static cudaInitializer item;
	~cudaInitializer();
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
#endif