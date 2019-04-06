#pragma once
#include "CuPtr.cuh"
using Ty = double;

Ty SumOfArray(CuPtr<Ty>& input_data, int BlockDimX);
