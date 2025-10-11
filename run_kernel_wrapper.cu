#include <cublas_v2.h>
#include <stdexcept>
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

void run_native(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
	dim3 blockDim(32, 32);
	native<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_shared_memory_1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
	dim3 blockDim(32 , 32);
#if 0
	cudaFuncSetAttribute(shared_memory<32>,
			cudaFuncAttributePreferredSharedMemoryCarveout,
			cudaSharedmemCarveoutMaxShared);
#endif
	shared_memory_1<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_shared_memory_2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
	dim3 blockDim(32 * 32);
#if 0
	cudaFuncSetAttribute(shared_memory<32>,
			cudaFuncAttributePreferredSharedMemoryCarveout,
			cudaSharedmemCarveoutMaxShared);
#endif
	shared_memory_2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_blocktile_1d_thread(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const uint BM = 64;
	const uint BN = 64;
	const uint BK = 8;
	const uint TM = 8;
	dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
	dim3 blockDim((BM * BN) / TM);
	blocktile_1d_thread<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_blocktile_2d_thread(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const uint BM = 128;
	const uint BN = 128;
	const uint BK = 8;
	const uint TM = 8;
	const uint TN = 8;
	dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
	dim3 blockDim((BM * BN) / (TM * TN));
	blocktile_2d_thread<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_native_global_coalesce(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
	dim3 blockDim(32, 32);
	native_global_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	switch (kernel_num) {
		case 0:
			runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
			break;
		case 1:
			run_native(M, N, K, alpha, A, B, beta, C);
			break;
		case 2:
			run_native_global_coalesce(M, N, K, alpha, A, B, beta, C);
			break;
		case 3:
			run_shared_memory_1(M, N, K, alpha, A, B, beta, C);
			break;
		case 4:
			run_shared_memory_2(M, N, K, alpha, A, B, beta, C);
			break;
		case 5:
			run_blocktile_1d_thread(M, N, K, alpha, A, B, beta, C);
			break;
		case 6:
			run_blocktile_2d_thread(M, N, K, alpha, A, B, beta, C);
			break;

		default:
			throw std::invalid_argument("invalid kernel number");

	}

}
