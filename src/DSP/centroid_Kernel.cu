#ifndef _CENTROID_KERNEL_H_
#define _CENTROID_KERNEL_H_

#include "CUDA_Kernels.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "vector_types.h"
#include "device_functions.h"
#include "device_atomic_functions.h"
//#include "device_atomic_functions.hpp"
#include "math_functions.h"
#include "sm_30_intrinsics.h"


__device__ void AtomicMax(float * const address, const float value)
{
	if (*address >= value) return;
	int * const address_as_i = (int *)address;
	int old = *address_as_i, assumed;

	do {
		assumed = old;
		if (__int_as_float(assumed) >= value) break;
		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}

__device__ float2 warp_reduce_max(float2 val) {
	for (int offset = 16; offset > 0; offset /= 2) {
		float temp = __shfl_down(val.x, offset);
		float temp_loc = __shfl_down(val.y, offset);
		if (temp > val.x) {
			val.x = temp;
			val.y = temp_loc;
		}
	}
	return val;
}

__global__ void max_kernel(float* __restrict__ output, const float* __restrict__ input, const int2 i_size)
{
	__shared__ float smem[2];

	int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread ID within the block
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (tid < 2) {
		smem[tid] = 0.0f;
	}
	__syncthreads();

	int id = idx_y*i_size.x + idx_x;
	float2 f = { (float)input[id], (float)id };

	for (int y = idx_y; y < i_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < i_size.x; x += blockDim.x * gridDim.x) {
			int gid = y*i_size.x + x;
			if (f.x < input[gid]) {
				f.x = input[gid];
				f.y = gid;
			}
		}
	}

	const float2 f_max = warp_reduce_max(f);
	if ((tid & 31) == 0) { // once per warp
		AtomicMax(&smem[0], f_max.x);
		if (f_max.x == smem[0]) smem[1] = f_max.y;
	}

	__syncthreads();
	if (tid == 0) { // lowest 9 threads of each block
		AtomicMax(&output[1], smem[0]);
		if (output[1] == smem[0]) output[0] = smem[1];
	}

	__syncthreads();
	for (int y = idx_y; y < i_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < i_size.x; x += blockDim.x * gridDim.x) {
			int gid = y*i_size.x + x;
			if (output[1] == input[gid]) {
				output[0] = gid;
			}
		}
	}

};




// Device code
extern "C" void findmax_GPU(
	float* output,
	const float* input,
	int width, int height, int GPUNo
)
{
    // Launch the Vector Add CUDA Kernel
	dim3 _threads(32, 32);
	dim3 _blocks(10, 10);
	int2 i_size = { width, height };
	//cudaSetDevice(GPUNo);
	max_kernel << <_blocks, _threads >> >(output, input, i_size);
}
 
__inline__ __device__ float3 warp_reduce_sum_triple(float3 val) {
	for (int offset = 16; offset > 0; offset /= 2) {
		val.x += __shfl_xor(val.x, offset);
		val.y += __shfl_xor(val.y, offset);
		val.z += __shfl_xor(val.z, offset);
	}
	return val;
}

extern "C" __global__ void ImageMoment_Max_Radius_binarization_kernel(float* __restrict__ output,
	const float* __restrict__ input, const int2 input_size,
	const float* __restrict__ i_max, const float threshold, const float radius_max2)
{
	__shared__ float smem[6];
	int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread ID within the block
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;


	if (tid < 6) {
		smem[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	float3 g = { 0 };
	float3 h = { 0 };

	float yc = floor(i_max[0] / input_size.x);
	float xc = (i_max[0] - yc*input_size.x);

	for (int y = idx_y; y < input_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < input_size.x; x += blockDim.x * gridDim.x) {
			if (input[y*input_size.x + x] > threshold) {
				if (((x - xc)*(x - xc) + (y - yc)*(y - yc)) < radius_max2) {
					g.x += x;
					g.y += y;
					g.z += 1;
					h.x += x*x;
					h.y += y*y;
					h.z += x*y;
				}
			}
		}
	}

	const float3 g_sum = warp_reduce_sum_triple(g);
	const float3 h_sum = warp_reduce_sum_triple(h);
	if ((tid & 31) == 0) { // once per warp
		atomicAdd(&smem[0], g_sum.x);
		atomicAdd(&smem[1], g_sum.y);
		atomicAdd(&smem[2], g_sum.z);
		atomicAdd(&smem[3], h_sum.x);
		atomicAdd(&smem[4], h_sum.y);
		atomicAdd(&smem[5], h_sum.z);
	}

	__syncthreads();
	if (tid < 6) // lowest 9 threads of each block
		atomicAdd(&output[threadIdx.x], smem[threadIdx.x]);
}

extern "C" __global__ void imgmoment2centroid_kernel(float* __restrict__ d_Dst, const float* __restrict__ d_Src) {
	d_Dst[0] = d_Src[0] / d_Src[2];
	d_Dst[1] = d_Src[1] / d_Src[2];
};

extern "C" void getImageMoment_GPU(
	float* output,
	const float* input,
	const float* max_pt,
	int width, int height,
	float threshold,
	float radius_max
	) {
	// Launch the Vector Add CUDA Kernel
	dim3 _threads(16, 16);
	dim3 _blocks(10, 10);
	int2 i_size = { width, height };
	//cudaSetDevice(GPU0);
	ImageMoment_Max_Radius_binarization_kernel << <_blocks, _threads >> >(output, input, i_size, max_pt, threshold, radius_max*radius_max);
}
#endif