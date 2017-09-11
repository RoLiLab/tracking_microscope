#include "Base/base.h"
#include "DSP/GlobalMap.h"
#include "DSP/CUDA_Kernels.h"
#include "HDF5/hdf5imagewriter.h"



GlobalMap::GlobalMap(void)
{
	h_src = NULL;
	d_src = NULL;
	d_BG = NULL;
	d_bBG = NULL;
	d_sub = NULL;
	d_src_float = NULL;
	d_save1 = NULL;
	d_save2 = NULL;
	d_save3 = NULL;
	BG_x0 = 0;
	BG_y0 = 0;
	//! Variables
	bgEnable = false; // background GPU buffer enable
	UpdateEnabled = false; // update enable (excluding the fish area)
	UpdateAllEnabled = false; // update all area (no exclusion)
	subEnabled = false; // BG subtraction enable
	//! setting
	PxDist_mmppx = 14.8 / 1000; // pixel distance (mm/px)
	theta = 0.0; // radian
	cth = 1.0;
	sth = 0.0; // cos(theta), sin(theta)
	width = 768;
	height = 576;
	ADC = 16;
	byte_per_image = (int)((width*height*ADC)/8);
	bg_width = 1024 * 8;
	bg_height = 1024 * 8;
	bg_width_half = bg_width / 2;
	bg_height_half = bg_height / 2;
	fish_mask_dy_mm = 0.4;
	fish_mask_dx_F_mm = 0.7;
	fish_mask_dx_B_mm = 4.0;
	updateFishMask();
	temp_count = 0;
	max_intensity_set = 1500;
}

GlobalMap::~GlobalMap(void)
{
	gpuFree();
}

void GlobalMap::updateParameters(double _pxDist, double _theta, int W, int H, int _ADCbit)
{
	if (bgEnable == false) {
		PxDist_mmppx = _pxDist;
		theta = _theta;
		cth = cos(theta);
		sth = sin(theta);
		width = W;
		height = H;
		ADC = _ADCbit;
		byte_per_image = (int)((width*height*ADC) / 8);
		bgEnable = true;
		updateFishMask();
	}
}



void GlobalMap::updateFishMask() {
	fish_mask_dy = fish_mask_dy_mm / PxDist_mmppx;
	fish_mask_dx_F = fish_mask_dx_F_mm / PxDist_mmppx;
	fish_mask_dx_B = fish_mask_dx_B_mm / PxDist_mmppx;

}
void GlobalMap::updateFishMask(double rx_pos, double rx_neg, double ry) {
	fish_mask_dy_mm = ry;
	fish_mask_dx_F_mm = rx_pos;
	fish_mask_dx_B_mm = rx_neg;
	updateFishMask();
}

void GlobalMap::gpuMalloc(void) {
	cudaSetDevice(GPU0);
	gpuFree();
	gpuErrchk(cudaMalloc((void **)(&d_BG), bg_width*bg_height*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_bBG), bg_width*bg_height*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_sub), height*width*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_src_float), height*width*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_save1), height*width*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_save2), height*width*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_save3), height*width*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&d_src), (size_t)byte_per_image));

}

void GlobalMap::gpuFree(void) {
	cudaSetDevice(GPU0);
	if (d_BG != NULL) gpuErrchk(cudaFree(d_BG)); d_BG = NULL;
	if (d_bBG != NULL) gpuErrchk(cudaFree(d_bBG)); d_bBG = NULL;
	if (d_sub != NULL) gpuErrchk(cudaFree(d_sub)); d_sub = NULL;
	if (d_src_float != NULL) gpuErrchk(cudaFree(d_src_float)); d_src_float = NULL;
	if (d_save1 != NULL) gpuErrchk(cudaFree(d_save1)); d_save1 = NULL;
	if (d_save2 != NULL) gpuErrchk(cudaFree(d_save2)); d_save2 = NULL;
	if (d_save3 != NULL) gpuErrchk(cudaFree(d_save3)); d_save3 = NULL;
	if (d_src != NULL) gpuErrchk(cudaFree(d_src)); d_src = NULL;
}

void GlobalMap::gpuInit(void) {
	cudaSetDevice(GPU0);
	cudaMemset(d_BG, 0xff, bg_width*bg_height*sizeof(float));
	cudaMemset(d_bBG, 0, bg_width*bg_height*sizeof(unsigned char));
}

Point2i GlobalMap::StagePos2GMapPxPos(Point2f _ref) {
	if (bgEnable) {
		BG_x0 = (int)((cth*(-_ref.x) + sth*(-_ref.y)) / PxDist_mmppx + bg_width_half);
		BG_y0 = (int)((-sth*(-_ref.x) + cth*(-_ref.y)) / PxDist_mmppx + bg_height_half);
	}
	else
	{
		BG_x0 = -1;
		BG_y0 = -1;
	}
	return Point2i(BG_x0, BG_y0);
}

Point2i GlobalMap::StagePos2GMapPxPos(float x, float y) {
	return StagePos2GMapPxPos(Point2f(x, y));
}

bool GlobalMap::cudaBGsub(uint16 * _src, double * _pos_double) {
	cudaError_t error_temp = cudaMemcpy(d_src, _src, (size_t)byte_per_image, cudaMemcpyHostToDevice);//
	if (cudaSuccess != error_temp)
		return false;
	else
		temp_count++;
	if (subEnabled && _pos_double) { // position available
		StagePos2GMapPxPos(_pos_double[0], _pos_double[1]);
		vectorSub16_kernelGPU(d_src, d_src_float, d_BG, d_sub, BG_x0, BG_y0, width, height, bg_width, bg_height, max_intensity_set, d_save1);
		gpuErrchk( cudaPeekAtLastError() );
		return true;
	}
	return false;
}



void GlobalMap::cudacopyBG(float * d_dst, double * _pos_double) {
	StagePos2GMapPxPos(_pos_double[0], _pos_double[1]);
	CopyGlobalMapImg_GPU(d_BG, d_dst, BG_x0, BG_y0, width, height, bg_width, bg_height);
	gpuErrchk(cudaPeekAtLastError());
}


void GlobalMap::cudasaveBG(void)
{
	if (true) {//!bgEnable
		float * GMap = (float*)malloc((size_t)(bg_width * bg_height)*sizeof(float));
		cudaMemcpy((unsigned char*)GMap, (const unsigned char*)d_BG, (size_t)(bg_width * bg_height)*sizeof(float), cudaMemcpyDeviceToHost);

		time_t t = time(0); // get time now;
		struct tm * now = localtime(&t);

		char NAME[1024];
		sprintf(NAME, "d:\\TrackingMicroscopeData\\_Snapshot\\BG_%d_%d_%dbyte_%04d%02d%02d_%02d%02d%02d_.dat",
			bg_width, bg_height, sizeof(float),
			now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
		pwriter_binary->write((char*)GMap, (size_t)(bg_width * bg_height)*sizeof(float));
		pwriter_binary->close();
		delete(pwriter_binary);
		free(GMap);
	}
}



void GlobalMap::cudaupdateBG(float cx, float cy, float th_deg, bool isFishFound) {
	if (UpdateEnabled) {
		double updateRatio = 0.1;
		//double sizeRatio_Allowed = 0.2;
		//bool isFishFound = false;
		//if (FishSize_Ref*(1-sizeRatio_Allowed) < FishSize && FishSize < FishSize_Ref*(1+sizeRatio_Allowed))
		//	isFishFound = true;
		float th_rad = th_deg*PI / 180;
		if (UpdateAllEnabled) { // update map anyway
			// update global map from image (Whole image)
			blendImg_GPU(d_BG, d_src_float, d_bBG, BG_x0, BG_y0, height, width, bg_width, bg_height, updateRatio);
		}
		else { // when the fish is found update only (~ 1200)
			// update global map from image (Whole image and exclude the fish area)
			blendImg_GPU_fishExclude(d_BG, d_src_float, d_bBG, BG_x0, BG_y0, height, width, bg_width, bg_height, updateRatio,
				cx, cy, th_rad, fish_mask_dy, fish_mask_dx_F, fish_mask_dx_B, isFishFound, d_save2);
			gpuErrchk( cudaPeekAtLastError() );
		}
	}
}
