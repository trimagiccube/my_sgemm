#pragma once
#include <cublas_v2.h>

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
