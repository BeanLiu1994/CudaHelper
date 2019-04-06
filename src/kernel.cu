#include "kernel.h"
#include "CuPtr.cuh"
#include <vector>

__global__ void addKernel(VarType *c, const VarType *a, const VarType *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// ����1 �����÷�
void test1(VarType *c, const VarType *a, const VarType *b, unsigned int size)
{
	CuPtr<const VarType> da(size, a);
	CuPtr<const VarType> db(size, b);
	CuPtr<VarType> dc(size, c);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}

// ����2 ֱ�ӷ���gpu�ռ�,֮���Ƶ�c��
void test2(VarType *c, const VarType *a, const VarType *b, unsigned int size)
{
	CuPtr<const VarType> da(size, a);
	CuPtr<const VarType> db(size, b);
	CuPtr<VarType> dc(size, nullptr);

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult(c);
}

// ����3 ʹ��������ֳ��Ŀռ�
void test3(std::vector<VarType>& c, const VarType *a, const VarType *b, unsigned int size)
{
	// ���й����в�Ҫʹc�طֿռ�
	CuPtr<const VarType> da(size, a);
	CuPtr<const VarType> db(size, b);
	CuPtr<VarType> dc(size, &(c[0]));

	addKernel << <1, size >> > (dc(), da(), db());

	dc.GetResult();
}
