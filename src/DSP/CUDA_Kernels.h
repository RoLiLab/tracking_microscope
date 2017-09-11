/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CUDAKERNELS_COMMON_H
#define CUDAKERNELS_COMMON_H

#define GPU0 1


#define threadsPerBlock 1024


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	float *c_Kernel,
	int KERNEL_RADIUS
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	float *c_Kernel,
	int KERNEL_RADIUS
);
extern "C" void convolutionRowsGPU_8U(
    float *d_Dst,
    unsigned char *d_Src,
    int imageW,
    int imageH,
	float *c_Kernel,
	int KERNEL_RADIUS
);

extern "C" void convolutionColumnsGPU_8U(
    float *d_Dst,
    unsigned char *d_Src,
    int imageW,
    int imageH,
	float *c_Kernel,
	int KERNEL_RADIUS
);
extern "C" void VecAdd_kernelGPU(
    float *d_Src1,
    float *d_Src2,
	float *d_Dst,
    int numElements
);

extern "C" void reductionMaxIdxGPU_NIR(
	float *d_Src, 
	float *d_Dst,	int *item_Dst,
	int width, int height,
	int RstCount, int * RstIdx, int * RstRadius, 
	float *d_Dummy, 	int *item_Dummy,// 1024 vectors
	const int * RefInd_pow2, const int * RefInd_y
);

extern "C" void reductionMaxIdxGPU_NIR_NOGain(
	float *d_Src,
	float *d_Dst,	int *item_Dst,
	int width, int height,
	int RstCount, int * RstIdx, int * RstRadius, 
	float *d_Dummy, 	int *item_Dummy, // 1024 vectors
	const int * RefInd_pow2, const int * RefInd_y
);

extern "C" void LinearIdx2SubInd( int * Idx, int * Idx_sub, int width, int Count);
extern "C" void thresholdimg_GPU(float * src, int * idx2d, int width, float * xpos, float * ypos, float * labelmap);
// Device code

extern "C" void vectorSub08_kernelGPU(
	unsigned char *d_src,
	float *d_src_f,
	float *d_BG,
	float *d_sub,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h
);

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
);


// Device code
extern "C" void blendImg_GPU(
	float *d_BG,
	float *d_Src,
	unsigned char *d_bBG,
	int x0,	int y0,
	int width, int height,
	int bg_width, int bg_height,
	double updateRatio
);
// Device code

extern "C" void blendImg_GPU_fishExclude(
	float *d_BG,
	float *d_Src,
	unsigned char *d_bBG,
	int x0, int y0,
	int width, int height,
	int bg_width, int bg_height,
	double updateRatio,
	float cx, float cy, float th, 
	int mask_dy, int mask_dx_F, int mask_dx_B,
	bool isfishdetected, float * d_sub
	);

extern "C" void CopyGlobalMapImg_GPU(
    float *d_BG,
	float *d_Dst,
	int x0, int y0,
	int w, int h,
	int bg_w, int bg_h
);

extern "C" void findmax_GPU(
	float* output,
	const float* input,
	int width, int height, int GPUNo
);

extern "C" void getImageMoment_GPU(
	float* output,
	const float* input,
	const float* max_pt,
	int width, int height,
	float threshold,
	float radius_max
	);
#endif