#pragma once

typedef unsigned short ushort;
#include "CUDA_Kernels.h"
#include <stdint.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math_constants.h>
#include "npp.h"
#include "nppdefs.h"
#include "ipp.h"

#define GPU1 0
class ZMatcher {
public:
	int layerskip = 3;
	int img_height = 0;
	const int n_ref = 50;
	const int W = 1024;
	const int H = 1024;
	const int W2 = 128;
	const int H2 = 128;
	const int W3 = 65;
	const float sigma = 1.0f;
	const int kernel_n = 5;
	const int W2_f32 = sizeof(float) * W2;
	int W_img_height_u16 = sizeof(uint16_t) * W * img_height;
	const int W_H_u16 = sizeof(uint16_t) * W * H;
	const int W2_H2_f32 = sizeof(float) * W2 * H2;
	const int W3_H2_f32 = sizeof(float) * W3 * H2;
	const int W3_H2_c32 = sizeof(cufftComplex) * W3 * H2;
	const int N_W2_H2_f32 = sizeof(float) * n_ref * W2 * H2;
	const int N_W3_H2_c32 = sizeof(cufftComplex) * n_ref * W3 * H2;
	int2 img_size;// = 
	int2 img_size_roi;
	int fft_size[2];// = { H2, W2 };

	cufftHandle plan_R2C;
	cufftHandle plan_C2R_batch;
	uint16_t* d_moving_u16;
	float* d_moving;
	float* d_moving_temp;
	float* d_moving_LP;
	float* d_kernel;
	cufftComplex* d_moving_F;
	float* d_moving_F_mag;
	float* d_moving_F_mag_polar;
	cufftComplex* d_moving_F_mag_polar_F;
	cufftComplex* d_ref_F_mag_polar_F;
	cufftComplex* d_correlation_F;
	float* d_correlation;
	float* d_correlation_max;
	int* d_theta_ind_max;
	float* correlation_max;
	int* theta_ind_max;
	double* d_mean_std;
	int n_ignore_lower = 0;
	int n_ignore_upper= n_ref;

	uint16_t ** buffer_simulation;
	int n_simulation;
	bool enable_simulation;

	ZMatcher();
	void initialization(int img_height);
	void set_ref_sweep(uint16_t** pp_ref);
	int match_z(uint16_t* p_moving, float * cormax);
	void img_process(uint16_t* p_moving);
	~ZMatcher();
	void test();
};
