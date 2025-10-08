#pragma once

#include "kernel/native.cuh"
#include "kernel/native_global_coalesce.cuh"
#include "kernel/shared_memory.cuh"

//round up M/N
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
