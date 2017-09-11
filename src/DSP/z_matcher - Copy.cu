#include "z_matcher.h"

struct __align__(16) ushort8{
	ushort a0, a1, a2, a3, a4, a5, a6, a7;
};

__device__ inline ushort sum_ushort8(ushort8 v) {
	return v.a0 + v.a1 + v.a2 + v.a3 + v.a4 + v.a5 + v.a6 + v.a7;
}

struct __align__(16) cufftComplex2{
	cufftComplex x, y;
};

extern "C" __global__ void downsample_8x8(
	const ushort * const __restrict__ input,
	float * const __restrict__ output
	) {
	int idx = 4 * threadIdx.x + 1024 * blockIdx.x;
	float4 px_output = { 0 };
	for (int i = 0; i < 8; i++) {
		px_output.x += sum_ushort8(((ushort8*)input)[idx++]);
		px_output.y += sum_ushort8(((ushort8*)input)[idx++]);
		px_output.z += sum_ushort8(((ushort8*)input)[idx++]);
		px_output.w += sum_ushort8(((ushort8*)input)[idx]);
		idx += 125;
	}
	((float4*)output)[threadIdx.x + 32 * blockIdx.x] = px_output;
}

__device__ inline void set_float4(float* p_lhs, float4 rhs) {
	p_lhs[0] = rhs.x;
	p_lhs[1] = rhs.y;
	p_lhs[2] = rhs.z;
	p_lhs[3] = rhs.w;
}

extern "C" __global__ void complex_mag(
	const cufftComplex * const __restrict__ input,
	float * const __restrict__ output
	) {
	int idx = 2 * threadIdx.x + 64 * blockIdx.x;
	cufftComplex2 p01 = ((cufftComplex2*)input)[idx];
	cufftComplex2 p23 = ((cufftComplex2*)input)[idx + 1];
	((float4*)output)[threadIdx.x + 32 * blockIdx.x] = { cuCabsf(p01.x), cuCabsf(p01.y), cuCabsf(p23.x), cuCabsf(p23.y) };
}

extern "C" __global__ void to_polar(
	const float * const __restrict__ input,
	float * const __restrict__ output
	) {
	float theta = CUDART_PI_F * (-0.5f + blockIdx.x / 128.0f);
	float cos_theta, sin_theta;
	sincosf(theta, &sin_theta, &cos_theta);
	float r = 1.0f + (0.1f * 4) * threadIdx.x;
	float p0 = input[__float2int_rz(r*cos_theta) + 65 * ((__float2int_rz(r*sin_theta) + 128) % 128)];
	r += 0.1f;
	float p1 = input[__float2int_rz(r*cos_theta) + 65 * ((__float2int_rz(r*sin_theta) + 128) % 128)];
	r += 0.1f;
	float p2 = input[__float2int_rz(r*cos_theta) + 65 * ((__float2int_rz(r*sin_theta) + 128) % 128)];
	r += 0.1f;
	float p3 = input[__float2int_rz(r*cos_theta) + 65 * ((__float2int_rz(r*sin_theta) + 128) % 128)];
	((float4*)output)[threadIdx.x + 32 * blockIdx.x] = { p0, p1, p2, p3 };
}

extern "C" __global__ void conj_batch(
	cufftComplex * const __restrict__ inout
	) {
	int idx = threadIdx.x + 64 * blockIdx.x;
	cufftComplex2 p01 = ((cufftComplex2*)inout)[idx];
	((cufftComplex2*)inout)[idx] = { cuConjf(p01.x), cuConjf(p01.y) };
}

extern "C" __global__ void conv_batch(
	const cufftComplex * const __restrict__ moving,
	const cufftComplex * const __restrict__ ref_batch,
	cufftComplex * const __restrict__ correlation_F_batch
	) {
	int idx = threadIdx.x + 64 * blockIdx.x;
	cufftComplex2 p01a = ((cufftComplex2*)moving)[idx % (65 * 64)];
	cufftComplex2 p01b = ((cufftComplex2*)ref_batch)[idx];
	((cufftComplex2*)correlation_F_batch)[idx] = { cuCmulf(p01a.x, p01b.x), cuCmulf(p01a.y, p01b.y) };
}

__inline__ __device__ float warp_reduce_max(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = fmaxf(val, __shfl_down(val, offset));
	return val;
}

__inline__ __device__ float4 warp_reduce_max4(float4 val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val.x = fmaxf(val.x, __shfl_down(val.x, offset));
		val.y = fmaxf(val.y, __shfl_down(val.y, offset));
		val.z = fmaxf(val.z, __shfl_down(val.z, offset));
		val.w = fmaxf(val.w, __shfl_down(val.w, offset));
	}
	return val;
}

__inline__ __device__ float max4(float4 val) {
	return fmaxf(fmaxf(fmaxf(val.x, val.y), val.z), val.w);
}

extern "C" __global__ void max_batch(
	const float * const __restrict__ correlation,
	float * const __restrict__ correlation_max
	) {
	static __shared__ float shared[32];
	int lane = threadIdx.x % 32;
	int wid = threadIdx.x / 32;

	float4 val = warp_reduce_max4(((float4*)correlation)[threadIdx.x]);
	val = warp_reduce_max4(((float4*)correlation)[threadIdx.x + 1024]);
	if (threadIdx.x < 32)
		val = warp_reduce_max4(((float4*)correlation)[threadIdx.x + 2048]);
	if (lane == 0)
		shared[wid] = max4(val);
	__syncthreads();
	if (wid == 0)
		correlation_max[blockIdx.x] = warp_reduce_max(shared[lane]);
}


extern "C" __global__ void filterRow_kernel(
	float* __restrict__ output, const int2 o_size,
	const float* __restrict__ input,
	const float* __restrict__ kernel, const int k_sz_half) // k_sz = 2*k_sz_half +1 
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int k_sz = 2 * k_sz_half + 1;
	for (int y = idx_y; y < o_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < o_size.x; x += blockDim.x * gridDim.x) {
			output[y*o_size.x + x] = 0;
			for (int i = 0; i < k_sz; i++) {
				int id = x + i - k_sz_half;
				if (id < 0) id = 0;
				if (id >= o_size.x) id = o_size.x - 1;
				int v = input[y*o_size.x + id];
				output[y*o_size.x + x] += v*kernel[i];
			}
		}
	}
}

extern "C" __global__ void filterColumn_kernel(
	float* __restrict__ output, const int2 o_size,
	const float* __restrict__ input,
	const float* __restrict__ kernel, const int k_sz_half) // k_sz = 2*k_sz_half +1 
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int k_sz = 2 * k_sz_half + 1;
	for (int y = idx_y; y < o_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < o_size.x; x += blockDim.x * gridDim.x) {
			output[y*o_size.x + x] = 0;
			for (int i = 0; i < k_sz; i++) {
				int id = y + i - k_sz_half;
				if (id < 0) id = 0;
				if (id >= o_size.y) id = o_size.y - 1;
				int v = input[id*o_size.x + x];
				output[y*o_size.x + x] += v*kernel[i];
			}
		}
	}
}

ZMatcher::ZMatcher(){
}

void ZMatcher::set_ref_sweep(uint16_t** pp_ref) {
	double mean_std[2];
	for (int i = 0; i<50; i++) {
		error_code = cudaMemcpy(d_moving_u16, pp_ref[i], W_img_height_u16, cudaMemcpyHostToDevice);
		if (error_code != 0) { error_line = 101; return; }
		downsample_8x8 << <128, 32 >> >(d_moving_u16, d_moving);
		error_code = nppiFilterGaussBorder_32f_C1R(d_moving, W2_f32, nppi_size, nppi_offset, d_moving_LP, W2_f32, nppi_size, NPP_MASK_SIZE_11_X_11, NPP_BORDER_REPLICATE);
		if (error_code != 0) { error_line = 105; return; }
		error_code = nppiSub_32f_C1IR(d_moving_LP, W2_f32, d_moving, W2_f32, nppi_size);
		if (error_code != 0) { error_line = 107; return; }
		error_code = cufftExecR2C(plan_R2C, d_moving, d_moving_F);
		if (error_code != 0) { error_line = 109; return; }
		complex_mag << <65, 32 >> >(d_moving_F, d_moving_F_mag);
		to_polar << <128, 32 >> >(d_moving_F_mag, d_moving_F_mag_polar);
		nppiMean_StdDev_32f_C1R(d_moving_F_mag_polar, W2_f32, nppi_size_W2_H2, d_npp_buffer, d_mean_std, d_mean_std + 1);
		cudaMemcpy(&mean_std, d_mean_std, sizeof(double) * 2, cudaMemcpyDeviceToHost);
		nppiSubC_32f_C1IR(mean_std[0], d_moving_F_mag_polar, W2_f32, nppi_size_W2_H2);
		nppiDivC_32f_C1IR(mean_std[1], d_moving_F_mag_polar, W2_f32, nppi_size_W2_H2);
		error_code = cufftExecR2C(plan_R2C, d_moving_F_mag_polar, d_ref_F_mag_polar_F + i * 128 * 65);
		if (error_code != 0) { error_line = 113; return; }
	}
	conj_batch << <65 * 50, 64 >> >(d_ref_F_mag_polar_F);
	error_code = cudaDeviceSynchronize();
	if (error_code != 0) { error_line = 117; return; }
}


void ZMatcher::initialization(int _img_height) {
	if (this->img_height > 0)
		return;
	img_height = (8 * (_img_height / 8));
	W_img_height_u16 = (sizeof(uint16_t) * W * 8 * (img_height / 8));
	nppi_size = { W2, img_height / 8 };

	this->img_height = img_height;
	fft_size[0] = H2;// = { H2, W2 };
	fft_size[1] = W2;
	img_size_roi.x = W2;
	img_size_roi.y = 96;

	nppi_size_W2_H2 = { W2, H2 };
	nppi_offset = { 0, 0 };

	cudaSetDevice(GPU1);

	const float sigma2 = 2 * sigma * sigma;
	const int kernel_n2 = 2 * kernel_n + 1;
	const int kernel_size = sizeof(float) * kernel_n2;
	float* p_kernel = (float*)malloc(kernel_size);
	for (int i = -kernel_n; i <= kernel_n; i++)
		p_kernel[i + kernel_n] = 1 / sqrt(sigma2 * CUDART_PI_F) * exp(-i*i / sigma2);
	cudaMalloc(&d_kernel, kernel_size);
	cudaMemcpy(d_kernel, p_kernel, kernel_size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_moving_u16, W_H_u16);
	cudaMemset(d_moving_u16, 0, W_H_u16);
	cudaMalloc(&d_moving, W2_H2_f32);
	cudaMalloc(&d_moving_temp, W2_H2_f32);
	cudaMalloc(&d_moving_LP, W2_H2_f32);
	cudaMalloc(&d_moving_F, W3_H2_c32);
	cudaMalloc(&d_moving_F_mag, W3_H2_f32);
	cudaMalloc(&d_moving_F_mag_polar, W2_H2_f32);
	cudaMalloc(&d_moving_F_mag_polar_F, W3_H2_c32);
	cudaMalloc(&d_ref_F_mag_polar_F, N_W3_H2_c32);
	cudaMalloc(&d_correlation_F, N_W3_H2_c32);
	cudaMalloc(&d_correlation, N_W2_H2_f32);
	cudaMalloc(&d_correlation_max, sizeof(float) * 50);
	cudaMalloc(&d_max, sizeof(float));
	cudaMalloc(&d_idx_max, sizeof(float));
	cudaMalloc(&d_mean_std, sizeof(double) * 2);
	int buffer_size, buffer_size2;
	nppiMaxGetBufferHostSize_32f_C1R(nppi_size, &buffer_size);
	nppsMaxIndxGetBufferSize_32f(50, &buffer_size2);
	if (buffer_size2 > buffer_size) buffer_size = buffer_size2;
	nppiMeanStdDevGetBufferHostSize_32f_C1R(nppi_size_W2_H2, &buffer_size2);
	if (buffer_size2 > buffer_size) buffer_size = buffer_size2;
	cudaMalloc(&d_npp_buffer, buffer_size);
	cufftPlan2d(&plan_R2C, W2, H2, CUFFT_R2C);
	cufftPlanMany(&plan_C2R_batch, 2, fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, n_batch);
	//correlation_max = new float[50];

}


int ZMatcher::match_z(uint16_t* p_moving) {
	cudaMemcpy(d_moving_u16, p_moving, W_img_height_u16, cudaMemcpyHostToDevice);
	downsample_8x8 << <128, 32 >> >(d_moving_u16, d_moving);
	nppiFilterGaussBorder_32f_C1R(d_moving, W2_f32, nppi_size, nppi_offset, d_moving_LP, W2_f32, nppi_size, NPP_MASK_SIZE_11_X_11, NPP_BORDER_REPLICATE);
	filterRow_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving_temp, img_size_roi, d_moving, d_kernel, kernel_n);
	filterColumn_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving_LP, img_size_roi, d_moving_temp, d_kernel, kernel_n);
	nppiSub_32f_C1IR(d_moving_LP, W2_f32, d_moving, W2_f32, nppi_size);
	cufftExecR2C(plan_R2C, d_moving, d_moving_F);
	complex_mag << <65, 32 >> >(d_moving_F, d_moving_F_mag);
	to_polar << <128, 32 >> >(d_moving_F_mag, d_moving_F_mag_polar);
	cufftExecR2C(plan_R2C, d_moving_F_mag_polar, d_moving_F_mag_polar_F);
	conv_batch << <65 * 50, 64 >> >(d_moving_F_mag_polar_F, d_ref_F_mag_polar_F, d_correlation_F);
	cufftExecC2R(plan_C2R_batch, d_correlation_F, d_correlation);
	max_batch << <50, 1024 >> >(d_correlation, d_correlation_max);
	cudaMemcpy(&correlation_max, d_correlation_max, sizeof(float) * 50, cudaMemcpyDeviceToHost);
	float max = correlation_max[0];
	int idx_max = 0;
	for (int i = 1; i < 50; i++) {
		float val = correlation_max[i];
		if (val > max) {
			max = val;
			idx_max = i;
		}
	}
	return idx_max;
}

ZMatcher::~ZMatcher() {
	cudaFree(d_moving_u16);
	cudaFree(d_moving);
	cudaFree(d_moving_LP);
	cudaFree(d_moving);
	cudaFree(d_moving_F);
	cudaFree(d_moving_F_mag);
	cudaFree(d_moving_F_mag_polar);
	cudaFree(d_moving_F_mag_polar_F);
	cudaFree(d_ref_F_mag_polar_F);
	cudaFree(d_correlation_F);
	cudaFree(d_correlation);
	cudaFree(d_correlation_max);
	cudaFree(d_npp_buffer);
	cudaFree(d_max);
	cudaFree(d_idx_max);
	cudaFree(d_mean_std);
	delete correlation_max;
}
