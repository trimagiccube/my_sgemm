#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "run_kernel_wrapper.cuh"

void randomize_matrix(float *mat, int N)
{
	struct timeval time {};
	gettimeofday(&time, nullptr);
	srand(time.tv_usec);
	for (int i = 0; i < N; i++) {
		float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
		tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
		mat[i] = tmp;
	}
}

void print_matrix(const float* matrix, int rows, int cols, const char* name) {
	std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
	for (int i = 0; i < std::min(rows, 4); i++) {
		for (int j = 0; j < std::min(cols, 4); j++) {
			std::cout << matrix[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "..." << std::endl;
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

long  cal_total_flops(long m, long n, long k)
{
	/*c = alpha * (a * b) + beta * c*/
	long total_flops;
	long tmp1 = m * n * k;/*total a * b multiply, c matrix have  m * n element, every element k times multiply*/
	long tmp2 = m * n * (k-1);/*total a * b add , c matrix have m * n element, every element k -1 times add*/
	long tmp3 = m * n;/*total scalar multiply, every element in c matrix have 1 times alpha * (a * b)*/
	long tmp4 = m * n;/*every element in c matrix : beta * c*/
	long tmp5 = m * n;/*every element in c matrix : alpha * (a * b)     "+"     beta * c, here is "+"*/
	total_flops = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
	return total_flops;
}

int main(int argc, char **argv)
{
	long m,n,k;
	float *A = nullptr, *B = nullptr, *C = nullptr,
		  *C_ref = nullptr;/*host matrices*/
	float *dA = nullptr, *dB = nullptr, *dC = nullptr,
		  *dC_ref = nullptr;/*device matrices*/
	float alpha = 1.0f, beta = 0.0f;
	long test_size = 4096;
	cudaEvent_t start, stop;
	m = n = k = test_size;
	cublasHandle_t handle;
	int kernel_num = 0;

	if (argc != 2) {
		std::cout << "please input  ./sgemm 0(0 - 10)" << std::endl;
		exit(EXIT_FAILURE);
	}
	kernel_num = std::stoi(argv[1]);

	A = (float *)malloc(sizeof(float) * m * k);
	B = (float *)malloc(sizeof(float) * k * n);
	C = (float *)malloc(sizeof(float) * m * n);
	C_ref = (float *)malloc(sizeof(float) * m * n);

	cudaMalloc((void **)&dA, sizeof(float) * m * k);
	cudaMalloc((void **)&dB, sizeof(float) * k * n);
	cudaMalloc((void **)&dC, sizeof(float) * m * n);
	cudaMalloc((void **)&dC_ref, sizeof(float) * m * n);

	randomize_matrix(A, m * k);
	randomize_matrix(B, k * n);
	randomize_matrix(C, m * n);
	randomize_matrix(C_ref, m * n);

#if 0
	print_matrix(A, m, k, "matrx A");
	print_matrix(B, k, n, "matrx B");
	print_matrix(A, m, n, "matrx C");
#endif
	cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dC_ref, C_ref, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	if (cublasCreate(&handle)) {
		std::cout << "create cublas handle error" << std::endl;
		exit(EXIT_FAILURE);
	}
	/*for warmup*/
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n,dA, k, &beta, dC_ref, n);
	cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	run_kernel(kernel_num, handle, m, n, k, alpha, dA, dB, beta, dC);
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n,dA, k, &beta, dC, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
//	cudaEventSynchronize(start);
	float millseconds = 0.0f;
	cudaEventElapsedTime(&millseconds, start, stop);
	std::cout << "cublasGemmEx completed in " << millseconds << " ms" << std::endl;

	long total_flops;
	total_flops = cal_total_flops(m, n, k);
	double gflops = (total_flops / (millseconds / 1000.0) / 1e9);
	std::cout << "total flops is " << total_flops << std::endl << "gflops is " << gflops <<  std::endl;

	/*for check cublas cal right or not*/
	float ref_value = 0.0f;
	for (int k_index = 0; k_index < k; k_index++) {
		ref_value += A[0 * k + k_index] * B[k_index * n + 0];
	}

	ref_value = alpha * ref_value + beta * C[0];
	//std::cout << "C[0,0] reference: " << ref_value << std::endl;
	cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	//print_matrix(C, m, n, "result matrx C");

	if(!verify_matrix(C_ref, C, m * n)) {
		std::cout << "run failed" << std::endl;
	}

	free(A);
	free(B);
	free(C);
	free(C_ref);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(dC_ref);
	cublasDestroy(handle);
}
