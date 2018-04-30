#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>

void test1(int *c, const int *a, const int *b, unsigned int size);
void test2(int *c, const int *a, const int *b, unsigned int size);
void test3(std::vector<int>& c, const int *a, const int *b, unsigned int size);