#pragma once

__global__ void native(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const long x = blockIdx.x * blockDim.x + threadIdx.x;
	const long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < M && y < N) {
		float tmp = 0.0f;
		for (long i = 0; i < K; i++)
			tmp += A[x * K + i] * B[i * N  + y];

		tmp = alpha * tmp + beta * C[x * N + y];
		C[x * N + y] = tmp;
	}
}
