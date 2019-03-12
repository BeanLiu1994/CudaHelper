
#include "kernel.h"
#include "CuPtr.cuh"
#include <stdexcept>
#include <vector>
using namespace std;

const int arraySize = 5;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
const int c_expect[arraySize] = { 11, 22, 33, 44, 55 };

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

bool cmp(const vector<int>& v1, const int* v2, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		if (v1[i] != v2[i])
			return false;
	}
	return true;
}
bool cmp(const int* v1, const int* v2, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		if (v1[i] != v2[i])
			return false;
	}
	return true;
}

int main()
{
	cudaInitializer::Init();
	if (!cudaInitializer::CudaOK())
		return 100;

	// 测试用变量
	int c1[arraySize] = { 0 };
	int c2[arraySize] = { 0 };
	vector<int> c3(arraySize, 0);

	//创建三个测试
	MakeTest_i(1);
	MakeTest_i(2);
	MakeTest_i(3);

	return 0;
}
