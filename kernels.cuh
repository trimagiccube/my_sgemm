#pragma once

#include "kernel/native.cuh"

//round up M/N
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
