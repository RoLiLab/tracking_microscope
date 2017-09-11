#ifndef _BlendImage_KERNEL_H_
#define _BlendImage_KERNEL_H_

#include "CUDA_Kernels.h"
//!--------------------------------------------------------------------
__global__ void copyBG_kernel(float *d_BG, float *d_Dst,
	int x0, int y0, int imageW, int imageH, int bg_w, int bg_h)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = _y*bg_w + _x;
				int gid_local = y*imageW + x;
				d_Dst[gid_local] = d_BG[gid];
			}
		}
	}
};

// Device codeextern "C" void
extern "C" void CopyGlobalMapImg_GPU(
	float *d_BG,
	float *d_Dst,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h
	)
{
	dim3 blocks(32, 32);
	dim3 threads(32, 32);
	copyBG_kernel << <blocks, threads >> >(d_BG, d_Dst, x0, y0, w, h, bg_w, bg_h);
};

//!--------------------------------------------------------------------
__global__ void blendImg_kernel(float *d_BG, float *d_Src, unsigned char *d_bBG, int x0, int y0, int imageW, int imageH, int bg_w, int bg_h, float updateRatio)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = _y*bg_w + _x;
				int gid_local = y*imageW + x;

				if (d_bBG[gid] == 0) { // not explored
					d_bBG[gid] = 255;
					d_BG[gid] = d_Src[gid_local];
				}
				else { // explorded
					d_BG[gid] = updateRatio*(float)d_Src[gid_local] + (1 - updateRatio)*(float)d_BG[gid];
				}

			}
		}
	}
};

// Device code
extern "C" void blendImg_GPU(
	float *d_BG,
	float *d_Src,
	unsigned char *d_bBG,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h,
	double updateRatio
)
{
	dim3 blocks(32, 32);
	dim3 threads(32, 32);
	float updateRatio_float = (float)updateRatio;
	blendImg_kernel << <blocks, threads >> >(d_BG, d_Src, d_bBG, x0, y0, w, h, bg_w, bg_h, updateRatio_float);
};

//!--------------------------------------------------------------------

// Device code
__global__ void blendImg_updateUnexpore_kernel(float *d_BG, float *d_Src, unsigned char *d_bBG, int x0, int y0, int imageW, int imageH, int bg_w, int bg_h)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			if ((_x < bg_w) && (_y < bg_h)) {
				int gid = _y*bg_w + _x;
				int gid_local = y*imageW + x;
				if (d_bBG[gid] == 0) { // not explored
					d_bBG[gid] = 255;
					d_BG[gid] = d_Src[gid_local];
				}
			}
		}
	}
};


__global__ void blendImg_fishExclude_kernel(float *d_BG, float *d_Src, unsigned char *d_bBG, int x0, int y0, int imageW, int imageH, int bg_w, int bg_h,
	float cx, float cy, float th, int mask_dy, int mask_dx_F, int mask_dx_B, float updateRatio, float * d_sub)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	float cth = cos(th);
	float sth = sin(th);

	// transform
	for (int y = idx_y; y < imageH; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < imageW; x += blockDim.x * gridDim.x) {
			int _x = x0 + x;
			int _y = y0 + y;
			float tx = cth*((float)x - cx) + sth*((float)y - cy);
			float ty = -sth*((float)x - cx) + cth*((float)y - cy);
			bool flag = ((fabs(ty) < mask_dy) && (tx < mask_dx_F) && (tx > -mask_dx_B));
			int gid = _y*bg_w + _x;
			int gid_local = y*imageW + x;
			if ((_x < bg_w) && (_y < bg_h) && (!flag)) {
				//d_sub[gid_local] = 100;
				if (d_bBG[gid] == 0) { // not explored
					d_bBG[gid] = 255;
					d_BG[gid] = d_Src[gid_local];
				}
				else { // explorded
					d_BG[gid] = updateRatio*(float)d_Src[gid_local] + (1 - updateRatio)*(float)d_BG[gid];
				}
			}
			//else
			//	d_sub[gid_local] = d_Src[gid_local];
			d_sub[gid_local] = d_BG[gid];
		}
	}
}

// Device codeextern "C" void
extern "C" void blendImg_GPU_fishExclude(
	float *d_BG,
	float *d_Src,
	unsigned char *d_bBG,
	int x0, int y0,
	int w, int h, int bg_w, int bg_h,
	double updateRatio,
	float cx, float cy, float th,
	int mask_dy, int mask_dx_F, int mask_dx_B,
	bool isfishdetected, float * d_sub
	)
{
	dim3 blocks(32, 32);
	dim3 threads(32, 32);
	float updateRatio_float = (float)updateRatio;

	if (isfishdetected)
		blendImg_fishExclude_kernel << <blocks, threads >> >(d_BG, d_Src, d_bBG, x0, y0, w, h, bg_w, bg_h, cx, cy, th, mask_dy, mask_dx_F, mask_dx_B, updateRatio_float, d_sub);
	else
		blendImg_updateUnexpore_kernel << <blocks, threads >> >(d_BG, d_Src, d_bBG, x0, y0, w, h, bg_w, bg_h);
}
#endif
