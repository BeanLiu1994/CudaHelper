#include "kernel.h"
#include "CudaManager.h"
#include <vector>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void test1(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr_Const da(a, size);
	CuPtr_Const db(b, size);
	CuPtr dc(c, size);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(
		(int*)dc.GetDevicePtr(),
		(int*)da.GetDevicePtr(), 
		(int*)db.GetDevicePtr()
		);
    
	dc.CuGetResult();
}

void test2(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr_Const da(a, size);
	CuPtr_Const db(b, size);
	CuPtr dc(nullptr, size*sizeof(int));

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(
		(int*)dc.GetDevicePtr(),
		(int*)da.GetDevicePtr(),
		(int*)db.GetDevicePtr()
		);

	dc.CuGetResult(c);
}

void test3(std::vector<int>& c, const int *a, const int *b, unsigned int size)
{
	// do not change sizeof [c], while running this part
	CuPtr_Const da(a, size);
	CuPtr_Const db(b, size);
	CuPtr dc(&(c[0]), size);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(
		(int*)dc.GetDevicePtr(),
		(int*)da.GetDevicePtr(),
		(int*)db.GetDevicePtr()
		);

	dc.CuGetResult();
}