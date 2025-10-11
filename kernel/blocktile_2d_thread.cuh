#pragma once
#include <cassert>
/*
1> there are two layout of block 

griddim(x, y) is divided across C matrix
1)x along to M, y along to N
   ------------------------   N    ------------------------➡️
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(0,0) * block(0,1) * block(0,2) * block(0,3) *
  |    *            *            *            *            *
  |    *****************************************************
       *            *            *            *            *
  M    * block(1,0) * block(1,1) * block(1,2) * block(1,3) *
       *            *            *            *            *
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(2,0) * block(2,1) * block(2,2) * block(2,3) *
  |    *            *            *            *            *
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(3,0) * block(3,1) * block(3,2) * block(3,3) *
  |    *            *            *            *            *
 \ /   *****************************************************

2)x along to N, y along to M
   ------------------------   N    ------------------------➡️
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(0,0) * block(1,0) * block(2,0) * block(3,0) *
  |    *            *            *            *            *
  |    *****************************************************
       *            *            *            *            *
  M    * block(0,1) * block(1,1) * block(2,1) * block(3,1) *
       *            *            *            *            *
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(0,2) * block(1,2) * block(2,2) * block(3,2) *
  |    *            *            *            *            *
  |    *****************************************************
  |    *            *            *            *            *
  |    * block(0,3) * block(1,3) * block(2,3) * block(3,3) *
  |    *            *            *            *            *
 \ /   *****************************************************

layout 2) is more friendly
cause:
1. consective block access consective B matrix column  and share common A row

3> threadtile

from the calcaulate result of C matrix's point of view

   -----------------------------------------------BN---------------------------------------------------➡️
  |
  |    ********08****************08***********************************************************08*********
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |   08       t0        *       t1        *               ...                        *       t15       *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    ********08****************08***********************************************************08*********
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |   08       t16       *       t17       *               ...                        *       t31       *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    ********08****************08***********************************************************08*********
  |    *                 *                 *                                          *                 *
 BM    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *       ...       *       ...       *               ...                        *       ...       *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    ********08****************08***********************************************************08*********
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  |   08       t240      *       t241      *               ...                        *       t255      *
  |    *                 *                 *                                          *                 *
  |    *                 *                 *                                          *                 *
  ⬇️   ********08****************08***********************************************************08*********





 .                      **************************************BN*****************************************
 .                      *                                                                               *
 .                      BK                                                                              *
 .                      *                                                                               *
 .                      *********************************************************************************


        ****BK****      ***************************************BN****************************************
        *        *      *        *                                                                      *
        *        *      *   t0  TM                                                                      *
        *        *      *        *                                                                      *
    strideA      *      ****TN****                                                                      *
        *        *      *                                                                               *
        *        *      *                                                                               *
        * ****** *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               BM
        *        *      *                                                                               *
        *        *      *                                                                               *
        * ****** *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
       BM        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        * ****** *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        *        *      *                                                                               *
        **********      *********************************************************************************

 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void blocktile_2d_thread(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	/*which is the view of C matrix, layout 2*/
	const long cRow = blockIdx.y;
	const long cColumn = blockIdx.x;

	const uint totalResultsBlocktile = BM * BN;
	const uint totalthreadperblock = totalResultsBlocktile / (TM * TN);
	assert(totalthreadperblock == blockDim.x );

	/*used to locate thread in block*/
	const long threadRow = threadIdx.x / (BN / TN);
	const long threadColumn = threadIdx.x % (BN / TN);

	/*used to locate the coordinates (this thread use)in shared memory of A matrix*/
	const uint innerRowA = threadIdx.x / BK;
	const uint innerColA = threadIdx.x % BK;
	const uint strideA = totalthreadperblock / BK; //unit : row
	/*used to locate the coordinates (this thread use)in shared memory of B matrix*/
	const uint innerRowB = threadIdx.x / BN;
	const uint innerColB = threadIdx.x % BN;
	const uint strideB = totalthreadperblock / BN;

	float threadresult[TM * TN] = {0.0f};
	float regM[TM] = {0.0f};//cache for As
	float regN[TN] = {0.0f};

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	/*input matrix should located at the block area*/
	A += cRow * BM * K;
	B += cColumn * BN;
	C += cRow * BM * N + cColumn * BN;

	/*block moved along k*/
	for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {
		/*use cuda core load data, every time*/
		for (uint iterAs = 0; iterAs < BM; iterAs += strideA) {
			As[(innerRowA + iterAs) * BK + innerColA] = A[(innerRowA + iterAs) * K + innerColA];
		}
		for (uint iterBs = 0; iterBs < BK; iterBs += strideB) {
			Bs[(innerRowB + iterBs) * BN + innerColB] = B[(innerRowB + iterBs) * K + innerColB];
		}
		__syncthreads();

		A += BK;
		B += BK * N;

		/*outer product*/
		for (uint indexdot = 0; indexdot < BK; indexdot++) {
			/*load As one column to reg, which column will be used in outer product*/
			for (uint i = 0; i < TM; i++)
				regM[i] = As[(threadRow * TM + i) * BK + indexdot];
			/*load Bs one row to reg, which row will be used in outer product*/
			for (uint j = 0; j < TN; j++)
				regN[j] = Bs[indexdot * BN + threadColumn * TN + j];

			for (uint indexregM = 0; indexregM < TM; indexregM++)
				for (uint indexregN = 0; indexregN < TN; indexregN++)
					threadresult[indexregM * TN + indexregN] += regM[indexregM] * regN[indexregN];

		}
		__syncthreads();
	}
	for (uint indexcrow = 0; indexcrow < TM; indexcrow++)
		for (uint indexccolumn = 0; indexccolumn < TN; indexccolumn++)
			C[(threadRow * TM + indexcrow) * N + threadColumn * TN + indexccolumn] =
				alpha * threadresult[indexcrow * TN + indexccolumn] + beta * C[(threadRow * TM + indexcrow) * N + threadColumn * TN + indexccolumn];
}
