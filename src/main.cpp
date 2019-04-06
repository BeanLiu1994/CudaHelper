
#include "kernel.h"
#include "CuPtr.cuh"
#include "ArraySum.h"
#include <stdexcept>
#include <vector>
#include <chrono>
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


int test_for_helper()
{
	// 测试用变量
	VarType c1[arraySize] = { 0 };
	VarType c2[arraySize] = { 0 };
	vector<VarType> c3(arraySize, 0);

	// 创建三个测试
	MakeTest_i(1);
	MakeTest_i(2);
	MakeTest_i(3);
	return 0;
}

void test_for_cudasum()
{
	// 以下是一个CUDA数组求SUM的测试
	vector<int> d_sizes{ 1,10,50,100,300,1000,10000,100000,1000000,10000000 };
	for (auto d_size : d_sizes)
	{
		cout << d_size << endl;
		// 测试我的数组求和计算方式
		std::vector<Ty> d(d_size, 0);
		Ty cpu_result = 0;
		auto start_cpu = chrono::steady_clock::now();
		for (int i = 0; i < d.size(); ++i)
		{
			d[i] = rand() / 10000.0;
			cpu_result += d[i];
		}
		auto end_cpu = chrono::steady_clock::now();
		cout << "cpu time(): " << std::chrono::duration_cast<chrono::nanoseconds>(end_cpu - start_cpu).count() << endl;

		CuPtr<Ty> d_d(d.size(), d.data());
		vector<int> sizes{ 10,50,100,150,200,250,300 };
		for (auto m : sizes)
		{
			Ty gpu_result = SumOfArray(d_d, m);
			cout << m << " diff is:\t" << gpu_result - cpu_result << endl;
		}

		cout << cpu_result << endl;
	}
}

int main()
{
	cudaInitializer::Init();
	if (!cudaInitializer::CudaOK())
		return 100;

	test_for_helper();

	test_for_cudasum();

	return 0;
}
