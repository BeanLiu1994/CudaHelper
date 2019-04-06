#pragma once

typedef int VarType;

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>

void test1(VarType *c, const VarType *a, const VarType *b, unsigned int size);
void test2(VarType *c, const VarType *a, const VarType *b, unsigned int size);
void test3(std::vector<VarType>& c, const VarType *a, const VarType *b, unsigned int size);