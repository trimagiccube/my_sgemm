#pragma once

__global__ void native_global_coalesce(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const long cRow = blockIdx.y * blockDim.y + threadIdx.y;
	const long cColumn = blockIdx.x * blockDim.x + threadIdx.x;

	if (cRow < M && cColumn < N) {
		float tmp = 0.0f;
		for (long i = 0; i < K; i++)
			tmp += A[cRow * K + i] * B[i * N  + cColumn];

		tmp = alpha * tmp + beta * C[cRow * N + cColumn];
		C[cRow * N + cColumn] = tmp;
	}
}
