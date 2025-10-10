#pragma once
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

2> warptile

  here show one block
 ---------------------------BN--------------------------➡️
 | *******************************************************
 | *                          *                          *
 | TM          warp0          *           warp1          *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp2           *           warp3          *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp4           *           warp5          *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp6           *           warp7          *
 | *                          *                          *
 BM*******************************************************
 | *                          *                          *
 | *          warp8           *           warp9          *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp10          *           warp11         *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp12          *           warp13         *
 | *                          *                          *
 | *******************************************************
 | *                          *                          *
 | *          warp14          *           warp15         *
 | *                          *                          *
⬇️ *******************************************************
3> threadtile

from the C matrix's point of view

 *************************************************************************************************************************************************************************************************
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 * t0  * t1  * t2  * t3  * t4  * t5  * t6  * t7  * t8  * t9  * t10 * t11 * t12 * t13 * t14 * t15 * t16 * t17 * t18 * t19 * t20 * t21 * t22 * t23 * t24 * t25 * t26 * t27 * t28 * t29 * t30 * t31 *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
 *************************************************************************************************************************************************************************************************

 */
template <const int BM, const int BN, const int BK, const int TM>
__global__ void blocktile_1d_thread(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
	/*which is the view of C matrix, layout 2*/
	const long cRow = blockIdx.y;
	const long cColumn = blockIdx.x;

	/*used to locate thread in block*/
	const long threadRow = threadIdx.x / BN;
	const long threadColumn = threadIdx.x % BN;

	/*used to locate the coordinates (this thread use)in shared memory of A matrix*/
	const uint innerRowA = threadIdx.x / BK;
	const uint innerColA = threadIdx.x % BK;
	/*used to locate the coordinates (this thread use)in shared memory of B matrix*/
	const uint innerRowB = threadIdx.x / BN;
	const uint innerColB = threadIdx.x % BN;
	float threadresult[TM] = {0.0f};
	float tmpB = 0.0f;

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	/*input matrix should located at the block area*/
	A += cRow * BM * K;
	B += cColumn * BN;
	C += cRow * BM * N + cColumn * BN;

	/*block moved along k*/
	for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {
		/*use cuda core load data, every time*/
		As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
		Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
		__syncthreads();

		A += BK;
		B += BK * N;

		/*outer product*/
		for (uint indexbs = 0; indexbs < BK; indexbs++) {
			tmpB = Bs[indexbs * BN + innerColB];
			for (uint indexAs = 0; indexAs < TM; indexAs++)
				threadresult[indexAs] += As[(threadRow * TM + indexAs) * BK + indexbs] * tmpB;
		}
		__syncthreads();
	}
	for (uint indexresult = 0; indexresult < TM; indexresult++)
		C[(threadRow * TM  + indexresult) * N + threadColumn] = alpha * threadresult[indexresult] + beta * C[(threadRow * TM  + indexresult) * N + threadColumn];
}
