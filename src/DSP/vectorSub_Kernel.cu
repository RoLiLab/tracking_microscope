#ifndef _VectorSubW12_KERNEL_H_
#define _VectorSubW12_KERNEL_H_

#include "CUDA_Kernels.h"

__global__ void vectorSub08_kernel(unsigned char *d_src, float *d_src_f, float *d_BG, float *d_sub,
	int x0, int y0, int imageW, int imageH, int bg_w, int bg_h)
{

	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			int gid_local = y*imageW + x;
			if (gid_local > 20)
				d_src_f[gid_local] = (float)d_src[gid_local];
			else
				d_src_f[gid_local] = 0;

			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = _y*bg_w + _x;				
				d_sub[gid_local] = d_src_f[gid_local] - d_BG[gid];
				if (d_sub[gid_local] < 0) d_sub[gid_local] = 0;
			}
		}
	}
}


// Device code
extern "C" void vectorSub08_kernelGPU(
	unsigned char *d_src,
	float *d_src_f,
	float *d_BG,
	float *d_sub,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h
	)
{
	// Launch vectorSub12the Vector Add CUDA Kernel
	dim3 blocks(32, 32);
	dim3 threads(32, 32);
	vectorSub08_kernel << <blocks, threads >> >(d_src, d_src_f, d_BG, d_sub, x0, y0, w, h, bg_w, bg_h);
}

__global__ void casting_u12_f32_Kernel(float *d_Dst, unsigned short *d_Src, int imageW, int imageH)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			if (x % 2 == 0) {
				int gid = y*imageW + x;
				int gid2 = (int)(gid * 3 / 2);
				unsigned char * pSource = (unsigned char*)d_Src + gid2;
				unsigned char * b0 = pSource;
				unsigned char * b1 = pSource + 1;
				unsigned char * b2 = pSource + 2;
				d_Dst[gid] = (float)(((unsigned short)(*b0) << 4) | (*b1 & 0x0F));
				d_Dst[gid + 1] = (float)(((unsigned short)(*b2) << 4) | (*b1 & 0xF0));
			}
		}
	}
}

__global__ void vectorSub_kernel(float *d_src_f, float *d_BG, float *d_sub,
	int x0, int y0, int imageW, int imageH, int bg_w, int bg_h)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			int gid_local = y*imageW + x;

			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = _y*bg_w + _x;
				d_sub[gid_local] = d_src_f[gid_local] - (float)d_BG[gid];
				if (d_sub[gid_local] < 0) d_sub[gid_local] = 0;
			}
		}
	}
}

__global__ void vectorSub16_kernel(unsigned short * d_src, float *d_src_f, float *d_BG, float *d_sub,
	int x0, int y0, int imageW, int imageH, int bg_w, int bg_h, float max_intensity_set, float *d_save)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = y;
			int _y = imageW - 1 - x;
			int gid_local = y*imageW + x;
			int gid_local2 = _y*imageH + _x; // conversion			
			d_src_f[gid_local2] = (float)(d_src[gid_local] >> 4);
			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = (_y + y0)*bg_w + (_x + x0);
				d_sub[gid_local2] = d_src_f[gid_local2] - d_BG[gid];
				if (d_sub[gid_local2] > max_intensity_set) d_sub[gid_local2] = max_intensity_set;
				if (d_sub[gid_local2] < 0) d_sub[gid_local2] = 0;
				d_save[gid_local2] = d_BG[gid];
			}
			else
				d_save[gid_local2] = 0;
		}
	}
}


// Device code
extern "C" void vectorSub16_kernelGPU(
	unsigned short *d_src,
	float *d_src_f,
	float *d_BG,
	float *d_sub,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h,
	float max_intensity_set,
	float *d_save
	)
{
	// Launch vectorSub12the Vector Add CUDA Kernel
	dim3 blocks(20, 20);
	dim3 threads(32, 32);
	vectorSub16_kernel << <blocks, threads >> >(d_src, d_src_f, d_BG, d_sub, x0, y0, w, h, bg_w, bg_h, max_intensity_set, d_save);
	//casting_u12_f32_Kernel << <blocks, threads >> >(d_src_f, d_src, w, h);
	//vectorSub_kernel << <blocks, threads >> >(d_src_f, d_BG, d_sub, x0, y0, w, h, bg_w, bg_h);
}

#endif