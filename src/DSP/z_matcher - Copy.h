#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math_constants.h>
#include <npp.h>
#include <nppdefs.h>

#define GPU1 0
typedef unsigned short ushort;
class ZMatcher {
public:
	int error_code = 0;
	int error_line = 0;
	int img_height;
	const int n_batch = 50;
	const int W = 1024;
	const int H = 1024;
	const int W2 = 128;
	const int H2 = 128;
	const int W3 = 65;
	const int W2_f32 = sizeof(float) * W2;
	int W_img_height_u16;
	const int W_H_u16 = sizeof(uint16_t) * W * H;
	const int W2_H2_f32 = sizeof(float) * W2 * H2;
	const int W3_H2_f32 = sizeof(float) * W3 * H2;
	const int W3_H2_c32 = sizeof(cufftComplex) * W3 * H2;
	const int N_W2_H2_f32 = sizeof(float) * n_batch * W2 * H2;
	const int N_W3_H2_c32 = sizeof(cufftComplex) * n_batch * W3 * H2;
	int fft_size[2];// = { H2, W2 };
	NppiSize nppi_size;
	NppiSize nppi_size_W2_H2;// = { W2, H2 };
	NppiPoint nppi_offset;// = { 0, 0 };
	cufftHandle plan_R2C;
	cufftHandle plan_C2R_batch;

	uint16_t* d_moving_u16;
	float* d_moving;
	float* d_moving_temp;
	float* d_moving_LP;
	cufftComplex* d_moving_F;
	float* d_moving_F_mag;
	float* d_moving_F_mag_polar;
	cufftComplex* d_moving_F_mag_polar_F;
	cufftComplex* d_ref_F_mag_polar_F;
	cufftComplex* d_correlation_F;
	float* d_correlation;
	float* d_correlation_max;
	Npp8u* d_npp_buffer;
	float correlation_max[50];
	float* d_max;
	int* d_idx_max;
	double* d_mean_std;

	ZMatcher(void);
	void set_ref_sweep(uint16_t** pp_ref);
	int match_z(uint16_t* p_moving);
	~ZMatcher();
	int test(uint16_t* p_moving);


	const float sigma = 1.0f;
	const int kernel_n = 5;
	float* d_kernel = NULL;
	int2 img_size_roi;

	uint16_t ** buffer_simulation;
	int n_simulation;
	bool enable_simulation;
	void initialization(int img_height);
};
