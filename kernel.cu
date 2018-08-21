#include "kernel.h"
#include "CudaManager.h"
#include <vector>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// 测试1 常见用法
void test1(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, c);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}

// 测试2 直接分配gpu空间,之后复制到c里
void test2(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, nullptr);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult(c);
}

// 测试3 使用其他库分出的空间
void test3(std::vector<int>& c, const int *a, const int *b, unsigned int size)
{
	// 运行过程中不要使c重分空间
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, &(c[0]));

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}
