#include "kernel.h"
#include "CudaManager.h"
#include <vector>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// ����1 �����÷�
void test1(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, c);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}

// ����2 ֱ�ӷ���gpu�ռ�,֮���Ƶ�c��
void test2(int *c, const int *a, const int *b, unsigned int size)
{
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, nullptr);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult(c);
}

// ����3 ʹ��������ֳ��Ŀռ�
void test3(std::vector<int>& c, const int *a, const int *b, unsigned int size)
{
	// ���й����в�Ҫʹc�طֿռ�
	CuPtr<const int> da(size, a);
	CuPtr<const int> db(size, b);
	CuPtr<int> dc(size, &(c[0]));

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}
