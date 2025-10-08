#pragma once
/*
block(x,y)
block(0,0)	block(0,1)	block(0,2)	block(0,3)...
block(1,0)	block(1,1)	block(1,2)	block(1,3)...
block(2,0)	block(2,1)	block(2,2)	block(2,3)...
block(3,0)	block(3,1)	block(3,2)	block(3,3)...
...
 

thread(x,y)
thread(0,0)	thread(0,1)	thread(0,2)	thread(0,3)...
thread(1,0)	thread(1,1)	thread(1,2)	thread(1,3)...
thread(2,0)	thread(2,1)	thread(2,2)	thread(2,3)...
thread(3,0)	thread(3,1)	thread(3,2)	thread(3,3)...
...
 */
#if 1
template <const int BLOCKSIZE>
__global__ void shared_memory_1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const long cRow = blockIdx.x;
	const long cColumn = blockIdx.y;

	/*inner block use*/
	const long threadRow = threadIdx.x;
	const long threadColumn = threadIdx.y;

	float tmp = 0.0f;
	//should not dynamic assign
//	__shared__ float As[blockDim.x * blockDim.y];
//	__shared__ float As[blockDim.x * blockDim.y];
	__shared__ float As[BLOCKSIZE * BLOCKSIZE];
	__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

	/*input matrix should located at the block area*/
	A += cRow * blockDim.x * K;
	B += cColumn * blockDim.y;
	C += cRow * blockDim.x * N + cColumn * blockDim.y;

	/*block moved along k*/
	for (int blockIndex = 0; blockIndex < K; blockIndex += blockDim.y) {
		/*use cuda core load data, every time*/
		As[threadRow * blockDim.y + threadColumn] = A[threadRow * K + threadColumn];
		Bs[threadRow * blockDim.y + threadColumn] = B[threadRow * N + threadColumn];
		__syncthreads();

		A += blockDim.y;
		B += blockDim.x * N;

		for (int inner_index = 0; inner_index < blockDim.y; inner_index++) {
			tmp += As[threadRow * blockDim.y + inner_index] * Bs[inner_index * blockDim.y + threadColumn];
		}
		__syncthreads();
	}
	C[threadRow * N + threadColumn] = alpha * tmp + beta * C[threadRow * N + threadColumn];
}
#endif


template <const int BLOCKSIZE>
__global__ void shared_memory_2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	const long cRow = blockIdx.x;
	const long cColumn = blockIdx.y;

	/*inner block use*/
	const long threadRow = threadIdx.x / BLOCKSIZE;
	const long threadColumn = threadIdx.x % BLOCKSIZE;

//	printf("blockdim.x %d, y %d\n", blockDim.x, blockDim.y);
	float tmp = 0.0f;
	//should not dynamic assign
//	__shared__ float As[blockDim.x * blockDim.y];
//	__shared__ float As[blockDim.x * blockDim.y];
	__shared__ float As[BLOCKSIZE * BLOCKSIZE];
	__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

	/*input matrix should located at the block area*/
	A += cRow * BLOCKSIZE * K;
	B += cColumn * BLOCKSIZE;
	C += cRow * BLOCKSIZE * N + cColumn * BLOCKSIZE;

	/*block moved along k*/
	for (int blockIndex = 0; blockIndex < K; blockIndex += BLOCKSIZE) {
		/*use cuda core load data, every time*/
		As[threadRow * BLOCKSIZE + threadColumn] = A[threadRow * K + threadColumn];
		Bs[threadRow * BLOCKSIZE + threadColumn] = B[threadRow * N + threadColumn];
		__syncthreads();

		A += BLOCKSIZE;
		B += BLOCKSIZE * N;

		for (int inner_index = 0; inner_index < BLOCKSIZE; inner_index++) {
			tmp += As[threadRow * BLOCKSIZE + inner_index] * Bs[inner_index * BLOCKSIZE + threadColumn];
		}
		__syncthreads();
	}
	C[threadRow * N + threadColumn] = alpha * tmp + beta * C[threadRow * N + threadColumn];
}
