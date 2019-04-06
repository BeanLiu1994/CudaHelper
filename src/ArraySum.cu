// calculates the sum of array. using cuda.

#include "CuPtr.cuh"
#include "ArraySum.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <chrono>
using namespace std;

#define OUT

const int BlockDimX_max = 300;

// SumOfInput is a pointer to an array of output.
// threadIdx.x shows which part it is.
__global__ void SumOfArray_cuda(Ty* data, unsigned int len, OUT Ty* SumOfInput)
{
	// This is reserved space, so BlockSizeX must be not less than blockDim.x
	__shared__ Ty SumOfInput_middle[BlockDimX_max];
	int which_part = threadIdx.x;

	int ThreadsAval = blockDim.x;
	int PartLen = len / ThreadsAval;
	int StartPos = PartLen * which_part;
	int EndPos = StartPos + PartLen; // left close right open range.
	if (which_part == ThreadsAval - 1)
		EndPos = len;

	SumOfInput_middle[which_part] = 0;
	for (int i = StartPos; i < EndPos; ++i)
	{
		SumOfInput_middle[which_part] += data[i];
	}
	__syncthreads();
	// finally calclulate sum using the first thread.
	if (which_part == ThreadsAval - 1)
	{
		SumOfInput[0] = 0;
		for (int i = 0; i < ThreadsAval; ++i)
		{
			SumOfInput[0] += SumOfInput_middle[i];
		}
	}
}

Ty SumOfArray(CuPtr<Ty>& input_data, int BlockDimX)
{
	assert(BlockDimX_max >= BlockDimX);
	Ty* data_out;
	Ty output;
	cudaMalloc((void**)&data_out, sizeof(Ty));

	auto start_gpu = chrono::steady_clock::now();
	SumOfArray_cuda << <1, BlockDimX >> > (input_data(), input_data.Get_elem_size(), data_out);
	auto end_gpu = chrono::steady_clock::now();
	cout << "gpu time(): " << std::chrono::duration_cast<chrono::nanoseconds>(end_gpu - start_gpu).count() << endl;

	cudaMemcpy(&output, data_out, sizeof(Ty), cudaMemcpyDeviceToHost);
	cudaFree(data_out);

	return output;
}