
#include "kernel.h"
#include "CuPtr.cuh"
#include <stdexcept>
#include <vector>
using namespace std;

const int arraySize = 5;
const VarType a[arraySize] = { 1, 2, 3, 4, 5 };
const VarType b[arraySize] = { 10, 20, 30, 40, 50 };
const VarType c_expect[arraySize] = { 11, 22, 33, 44, 55 };

template<typename AccessArray_ty1, typename AccessArray_ty2>
bool cmp(AccessArray_ty1 v1, AccessArray_ty2 v2, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		if (v1[i] != v2[i])
			return false;
	}
	return true;
}


#define MakeTest_i(x) {\
try\
{\
	cout << "running test #" << x << " now." << endl;\
	test##x(c##x, a, b, arraySize);\
	if (!cmp(c##x, c_expect, arraySize))\
		return (x);\
}\
catch (runtime_error& e)\
{\
	cerr << e.what() << endl;\
	return (x);\
}\
cout << "success!" << endl;\
}\


int main()
{
	cudaInitializer::Init();
	if (!cudaInitializer::CudaOK())
		return 100;

	// 测试用变量
	VarType c1[arraySize] = { 0 };
	VarType c2[arraySize] = { 0 };
	vector<VarType> c3(arraySize, 0);

	//创建三个测试
	MakeTest_i(1);
	MakeTest_i(2);
	MakeTest_i(3);

	return 0;
}
