#ifndef _ImgReg_KERNEL_H_
#define _ImgReg_KERNEL_H_

#define IMGROI 1800
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <thrust/device_vector.h>


cudaArray * CUDA_createcuArray_f32(float * src, int width, int height) {
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Copy to device memory some data located at address h_data
	// in host memory
	cudaMemcpyToArray(cuArray, 0, 0, src, width*height*sizeof(float), cudaMemcpyHostToDevice);
	return cuArray;
};

cudaTextureObject_t CUDA_texture_init(cudaArray * cuArray) {

	cudaTextureObject_t texObj = 0;

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	//cudaTextureObject_t texObj = 0;
	cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	return texObj;
};

void CUDA_texture_close(cudaTextureObject_t texObject, cudaArray * cuArray) {
	// Destroy texture object
	cudaDestroyTextureObject(texObject);
	// Free device memory
	cudaFreeArray(cuArray);
};


extern "C" __global__ void CUDA_cor(float * cor, int * mask, float * src,
	float * src2, float * dst, float * dst2, float * srcdst)
{
	int count = mask[0] + mask[1] + mask[2] + mask[3];
	if (count > 0) {
		float src_mean = (src[0] + src[1] + src[2] + src[3]) / count;
		float src2_mean = (src2[0] + src2[1] + src2[2] + src2[3]) / count;
		float dst_mean = (dst[0] + dst[1] + dst[2] + dst[3]) / count;
		float dst2_mean = (dst2[0] / count + dst2[1] / count + dst2[2] / count + dst2[3] / count);
		float srcdst_mean = (srcdst[0] + srcdst[1] + srcdst[2] + srcdst[3]) / count;

		float num = srcdst_mean - src_mean*dst_mean;
		float den = sqrt(src2_mean - src_mean*src_mean)*sqrt(dst2_mean - dst_mean * dst_mean);
		cor[0] = num/den;
	}
	else
		cor[0] = 0.0;

}

extern "C" __global__ void CUDA_rotate(float * src, float * transpose,
	int * mask, float * dst, float * dst2, float * src2, float * srcdst)
{
	// src (1800 x 1800 Float32)
	// transpose [xshift_px, yshift_px, cos(th), sin(th)] (1x4, Float32)
	// mask (1800 x 1800 int)
	// sum_dummy, sqsum_summy (1800 x 1800 Float32)

 // Thread indexes

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int roi = 1800;
	int n = roi * roi;

	if (id < n) {
		float idx = (id % roi);
		float idy = floorf(id / roi);

		float x = (idx - transpose[0]) - (roi - 1) / 2;
		float y = (idy - transpose[1]) - (roi - 1) / 2;
		float xt = transpose[2] * x - transpose[3] * y + (roi - 1) / 2;
		float yt = transpose[3] * x + transpose[2] * y + (roi - 1) / 2;
		if (((0 <= xt) && (xt <= roi - 1)) && ((0 <= yt) && (yt <= roi - 1))) {
			mask[id] = 1; // on
			float x0 = floor(xt); float y0 = floor(yt);
			int x_cord[4]; x_cord[0] = x0 - 1; x_cord[1] = x0; x_cord[2] = x0 + 1; x_cord[2] = x0 + 2;
			int y_cord[4]; y_cord[0] = y0 - 1; y_cord[1] = y0; y_cord[2] = y0 + 1; y_cord[2] = y0 + 2;
			for (int it_i = 0; it_i < 4; it_i++) {
				if (x_cord[it_i] < 0) x_cord[it_i] = 0;
				if (y_cord[it_i] < 0) y_cord[it_i] = 0;
				if (x_cord[it_i] > (roi - 1)) x_cord[it_i] = (roi - 1);
				if (y_cord[it_i] > (roi - 1)) y_cord[it_i] = (roi - 1);
			}
			int gid[4][4];
			for (int it_i = 0; it_i < 4; it_i++) {
				for (int it_j = 0; it_j < 4; it_j++) {
					gid[it_i][it_j] = x_cord[it_i] + y_cord[it_i] * roi;
				}
			}
			if (1) { // bilinear method
				float dx = xt - x0;
				float dy = yt - y0;
				float _v1 = (1.0 - dx)*src[gid[1][1]] + dx*src[gid[1][2]];// tex2D(refTex, xt, yt);
				float _v2 = (1.0 - dx)*src[gid[2][1]] + dx*src[gid[2][2]];// tex2D(refTex, xt, yt);
				dst[id] = (1.0 - dy)*_v1 + dy*_v2;// tex2D(refTex, xt, yt);
			}
			else {
				//sum_dummy[id] = 1.0;// tex2D(refTex, xt, yt);
			}
		}
		else {
			mask[id] = 0;
			dst[id] = 0.0;
			dst2[id] = 0.0;
		}
		src2[id] = src[id] * src[id];
		srcdst[id] = src[id] * dst[id];
		dst2[id] = dst[id] * dst[id];
	}

	__syncthreads();
}

extern "C" __global__ void reduce6_float(float *g_idata, float *g_odata, unsigned int n)
{
	int blockSize = 512;
	__shared__ float sdata[1024];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();
#endif

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

extern "C" __global__ void reduce6_int(int *g_idata, int *g_odata, unsigned int n)
{
	int blockSize = 512;
	__shared__ int sdata[1024];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	int mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();
#endif

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

extern "C" __global__ void reduce6_float_mask(float *g_idata, float *g_odata, int *mask, unsigned int n)
{
	int blockSize = 512;
	__shared__ float sdata[1024];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (mask[i] > 0)
			mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + blockSize < n)
			if (mask[i + blockSize] > 0)
				mySum += g_idata[i + blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid <  8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid <  4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid <  2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid <  1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();
#endif

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

extern "C" __global__ void CUDA_rotate_texture(float * src, float * transpose, float * dst, float * mask)
{
	// src (1800 x 1800 Float32)
	// transpose [xshift_px, yshift_px, cos(th), sin(th)] (1x4, Float32)
	// mask (1800 x 1800 float)

	// Thread indexes

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int roi = 1800;
	int n = roi * roi;

	if (id < n) {
		float idx = (id % roi);
		float idy = floorf(id / roi);

		float x = (idx - transpose[0]) - (roi - 1) / 2;
		float y = (idy - transpose[1]) - (roi - 1) / 2;
		float xt = transpose[2] * x - transpose[3] * y + (roi - 1) / 2;
		float yt = transpose[3] * x + transpose[2] * y + (roi - 1) / 2;
		if (((0 <= xt) && (xt <= roi - 1)) && ((0 <= yt) && (yt <= roi - 1))) {
			mask[id] = 1.0; // on
			float x0 = floor(xt); float y0 = floor(yt);
			int x_cord[4]; x_cord[0] = x0 - 1; x_cord[1] = x0; x_cord[2] = x0 + 1; x_cord[2] = x0 + 2;
			int y_cord[4]; y_cord[0] = y0 - 1; y_cord[1] = y0; y_cord[2] = y0 + 1; y_cord[2] = y0 + 2;
			for (int it_i = 0; it_i < 4; it_i++) {
				if (x_cord[it_i] < 0) x_cord[it_i] = 0;
				if (y_cord[it_i] < 0) y_cord[it_i] = 0;
				if (x_cord[it_i] > (roi - 1)) x_cord[it_i] = (roi - 1);
				if (y_cord[it_i] > (roi - 1)) y_cord[it_i] = (roi - 1);
			}
			int gid[4][4];
			for (int it_i = 0; it_i < 4; it_i++) {
				for (int it_j = 0; it_j < 4; it_j++) {
					gid[it_i][it_j] = x_cord[it_i] + y_cord[it_i] * roi;
				}
			}
			if (1) { // bilinear method
				float dx = xt - x0;
				float dy = yt - y0;
				float _v1 = (1.0 - dx)*src[gid[1][1]] + dx*src[gid[1][2]];// tex2D(refTex, xt, yt);
				float _v2 = (1.0 - dx)*src[gid[2][1]] + dx*src[gid[2][2]];// tex2D(refTex, xt, yt);
				dst[id] = (1.0 - dy)*_v1 + dy*_v2;// tex2D(refTex, xt, yt);
			}
			else {
				dst[id] = 1.0;// tex2D(refTex, xt, yt);
			}
		}
		else {
			mask[id] = 0.0;
			dst[id] = 0.0;
		}
	}

	__syncthreads();
}
#endif
