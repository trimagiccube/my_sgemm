#include <cublas_v2.h>
#include "kernels.cuh"


void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
#if 0
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		N/*c row*/, M/*c column*/, K, &alpha,
		B/*matrix b*/, CUDA_R_32F/*B is fp32*/, N/*B leading dimension*/,
		A, CUDA_R_32F/*A is fp32*/, K/*A leading dimension*/,
		&beta, C, CUDA_R_32F, N/*C leading dimension*/,
		CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		N, M, K, &alpha,
		B, CUDA_R_32F, N,
		A, CUDA_R_32F, K,
		&beta, C, CUDA_R_32F, N,
		CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
