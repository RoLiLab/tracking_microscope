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
	)  {
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

extern "C" __global__ void norm_kernel(
	float* __restrict__ srcdst, const int2 o_size, float _mean, float _std)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < o_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < o_size.x; x += blockDim.x * gridDim.x) {
			int id = y*o_size.x + x;
			srcdst[id] = (srcdst[id] - _mean) / _std;
		}
	}
}


extern "C" __global__ void elememtsub_kernel(
	float* __restrict__ output, const int2 o_size,
	const float* __restrict__ input1, const float* __restrict__ input2)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int y = idx_y; y < o_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < o_size.x; x += blockDim.x * gridDim.x) {
			int id = y*o_size.x + x;
			output[id] = input1[id] - input2[id];
			if (y >= 96)  output[id] = 0;
		}
	}
}


__device__ void AtomicMax_kernel(float * const address, const float value)
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

__device__ float2 warp_reduce_max_deviceKernel(float2 val) {
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

__global__ void findmax_kernel(float* __restrict__ output, const float* __restrict__ input, const int3 i_size, const int layer_skip, const int prevId)
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
	int z_stride = i_size.x*i_size.y;
	float2 f = { 0, 0};

	for (int y = idx_y; y < i_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < i_size.x; x += blockDim.x * gridDim.x) {
			for (int z = layer_skip; z < i_size.z; z++) {
				float gain = 1.0f;// -fabsf(z - prevId) / 50;
				int gid = z*z_stride + y*i_size.x + x;
				if (f.x < input[gid] * gain) {
					f.x = input[gid] * gain;
					f.y = gid;
				}
			}
		}
	}

	const float2 f_max = warp_reduce_max_deviceKernel(f);

	__syncthreads();
	if ((tid & 31) == 0) { // once per warp
		AtomicMax_kernel(&smem[0], f_max.x);
		if (f_max.x == smem[0]) smem[1] = f_max.y;
	}

	__syncthreads();
	if (tid == 0) { // lowest 9 threads of each block
		AtomicMax_kernel(&output[1], smem[0]);
		if (output[1] == smem[0]) output[0] = smem[1];
	}

	__syncthreads();
	for (int y = idx_y; y < i_size.y; y += blockDim.y * gridDim.y) {
		for (int x = idx_x; x < i_size.x; x += blockDim.x * gridDim.x) {
			for (int z = layer_skip; z < i_size.z; z++) {
				int gid = z*z_stride + y*i_size.x + x;
				if (output[1] == input[gid]) {
					output[0] = gid;
				}
			}
		}
	}

};

ZMatcher::ZMatcher(){
	buffer_simulation = NULL;
	n_simulation = 0;
	enable_simulation = false;
}
void ZMatcher::initialization(int img_height) {
	if (this->img_height > 0)
		return;
	this->img_height = img_height;
	W_img_height_u16 = (2 * 1024 * 8 * (img_height / 8));
	fft_size[0] = H2;// = { H2, W2 };
	fft_size[1] = W2;
	img_size.x = W2;
	img_size.y = H2;
	img_size_roi.x = W2;
	img_size_roi.y = 96;

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
	cufftPlan2d(&plan_R2C, W2, H2, CUFFT_R2C);
	cufftPlanMany(&plan_C2R_batch, 2, fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, n_batch);
	cudaMalloc(&d_mean_std, sizeof(double) * 2);
}

void ZMatcher::set_ref_sweep(uint16_t** pp_ref) {
	cudaSetDevice(GPU1);
	float * a = (float *)malloc(W2_H2_f32);
	for (int i = 0; i<50; i++) {
		img_process(pp_ref[i]);
		// nomalization of the image
		cudaMemcpy(a, d_moving_F_mag_polar, W2_H2_f32, cudaMemcpyDeviceToHost);
		IppiSize _sz = { W2, H2 };
		Ipp32f mean_std[2] = { 0 };
		ippsMeanStdDev_32f(a, W2*H2, &mean_std[0], &mean_std[1], ippAlgHintAccurate);
		norm_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving_F_mag_polar, img_size, mean_std[0], mean_std[1]);
		cufftExecR2C(plan_R2C, d_moving_F_mag_polar, d_ref_F_mag_polar_F + i * 128 * 65);
	}
	free(a);
	conj_batch << <65 * 50, 64 >> >(d_ref_F_mag_polar_F);
}
void ZMatcher::img_process(uint16_t* p_moving) {
	cudaMemcpy(d_moving_u16, p_moving, W_img_height_u16, cudaMemcpyHostToDevice);
	downsample_8x8 << <128, 32 >> >(d_moving_u16, d_moving);
	filterRow_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving_temp, img_size_roi, d_moving, d_kernel, kernel_n);
	filterColumn_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving_LP, img_size_roi, d_moving_temp, d_kernel, kernel_n);
	elememtsub_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_moving, img_size, d_moving, d_moving_LP); // d_moving = d_moving - d_moving_LP

	cufftExecR2C(plan_R2C, d_moving, d_moving_F);
	complex_mag << <65, 32 >> >(d_moving_F, d_moving_F_mag);
	to_polar << <128, 32 >> >(d_moving_F_mag, d_moving_F_mag_polar);
}
int ZMatcher::match_z(uint16_t* p_moving, int id_prev) {
	cudaSetDevice(GPU1);
	img_process(p_moving);
	cufftExecR2C(plan_R2C, d_moving_F_mag_polar, d_moving_F_mag_polar_F);
	conv_batch << <65 * 50, 64 >> >(d_moving_F_mag_polar_F, d_ref_F_mag_polar_F, d_correlation_F);
	cufftExecC2R(plan_C2R_batch, d_correlation_F, d_correlation);

	int3 _sz = { W2, H2, n_batch};
	cudaMemset(d_correlation_max, 0, 8);
	findmax_kernel << <dim3(10, 10), dim3(32, 32) >> >(d_correlation_max, d_correlation, _sz, layerskip, id_prev);
	float maxInfo[2] = { 0 };
	cudaError err = cudaMemcpy(&maxInfo[0], d_correlation_max, 8, cudaMemcpyDeviceToHost);
	return ((int)maxInfo[0]) / (W2 * H2);
}

ZMatcher::~ZMatcher() {
	cudaSetDevice(GPU1);
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
	cudaFree(d_mean_std);
}

void ZMatcher::test() {
	cudaSetDevice(GPU1);
	to_polar << <128, 32 >> >(d_moving_F_mag, d_moving_F_mag_polar);
}
