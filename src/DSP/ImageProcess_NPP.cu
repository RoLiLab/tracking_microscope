//#include "KBase/common/common.h"
//#include "KBase/Util/Path.h"
//#include "KBase/Util/Time.h"

#include "DSP/ImageProcess_NPP.h"
//#include "device_launch_parameters.h"
//#include <cuda_runtime.h>


void findFishPositionFromDivImg_cuda(
	Npp32f* src, int srchSize1, int srchSize2, Npp32f* dst) {

}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void findLocalMaxima_cuda(
	Npp32f* src,
	Npp32f* src_grad_x,
	Npp32f* src_grad_y,
	Npp8u* dst,
	NppiSize imgSize) {

	// maximum number of blocks = 1024
	// maximum number of threads = 1024
	int ThreadNo = 640;
	int BlockSize = 480;
	dim3 blockDims(BlockSize,1,1); // block
	dim3 threadDims(ThreadNo, 1, 1); // thread
//	Npp32f* src_shifted = src +
    //findLocalMaxima_cudaGPU<<<blockDims, threadDims>>>(src, src_grad_x, src_grad_y, dst, imgSize, threshold);
	//add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
}




__global__ void convolution_1D_basic_kernel(float *N, float *P, float *M, int Mask_Width, int Width) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ float N_ds[512];
  N_ds[threadIdx.x] = N[i];
  __syncthreads();
  int This_tile_start_point = blockIdx.x * blockDim.x;
  int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
  int N_start_point = i - (Mask_Width/2);
  float Pvalue = 0;
  for (int j = 0; j < Mask_Width; j ++) {
    int N_index = N_start_point + j;
    if (N_index >= 0 && N_index < Width) {
      if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point)) {
        Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j];
      } else {
        Pvalue += N[N_index] * M[j];
      }
    }
  }
  P[i] = Pvalue;
}

__global__ void findLocalMaxima_cudaGPU(Npp32f* src, Npp32f* src_grad_x, Npp32f* src_grad_y, Npp8u* dst, NppiSize imgSize, int threshold)
{
	int RADIUS = 30;
	int BLOCKSIZE = 659;
	//__shared__ int temp[494];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x;
	/*
	temp[lindex] = src_grad[gindex]; //????
	// Synchronize (ensure all the data is available)
	__syncthreads();*/

	// code
	bool _isSatisfied = false;
	 if (src[gindex] < (-threshold)) { // check int is greather than threshold
		 //! x-axis and its left
		 for (int offset = -1 ;  ; offset--) { // from center to the left
			if (src_grad_x[gindex + offset] != 0) { // if it is less than 0, continue
				if (src_grad_x[gindex + offset] > 0)
					_isSatisfied = true;
			}
			break;
		}
		 //! x-axis and its right
		if (_isSatisfied ) {
			_isSatisfied = false;
			for (int offset = 1 ; ; offset++) { // from center to the right
				if (src_grad_x[gindex + offset] != 0) { // if it is less than 0, continue
					if (src_grad_x[gindex + offset] < 0)
						_isSatisfied = true;
					break;
				}
			}
		}

		//! y-axis and its left
		if (_isSatisfied ) {
			_isSatisfied = false;
			for (int offset = -1 ;  ; offset--) { // from center to the left
				if (src_grad_y[gindex + offset*BLOCKSIZE] != 0) { // if it is less than 0, continue
					if (src_grad_y[gindex + offset*BLOCKSIZE] > 0)
						_isSatisfied = true;
				}
				break;
			}
		}
		//! y-axis and its right
		if (_isSatisfied ) {
			_isSatisfied = false;
			for (int offset = 1 ; ; offset++) { // from center to the right
				if (src_grad_y[gindex + offset*BLOCKSIZE] != 0) { // if it is less than 0, continue
					if (src_grad_y[gindex + offset*BLOCKSIZE] < 0)
						_isSatisfied = true;
					break;
				}
			}
		}
	 }

	 // Store the result
	if (_isSatisfied)
		dst[gindex] = 255;
	else
		dst[gindex] = 0;
}
