#pragma once

#include "kernel/native.cuh"
#include "kernel/native_global_coalesce.cuh"
#include "kernel/shared_memory.cuh"
#include "kernel/blocktile_1d_thread.cuh"
#include "kernel/blocktile_2d_thread.cuh"

//round up M/N
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
