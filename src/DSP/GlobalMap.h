#pragma once


#include "ipp.h"
// CUDA runtime
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//char gpuErrorMsg[1024];
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
	//fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

	char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\gpuErrorMsg.txt";
	FILE  * ofp = fopen(outputFilename, "w");
	fprintf(ofp,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	fclose(ofp);
    //if (abort) exit(code);
   }
}



class GlobalMap
{
public:
	GlobalMap(void);
	~GlobalMap(void);
	//! CPU variable
	uint16 * h_src; // source image in CPU
	//! GPU variable
	uint16 * d_src; // source image in GPU
	float * d_src_float; // BG subtracted image from the source image in GPU
	float * d_sub; // BG subtracted image from the source image in GPU
	float * d_BG; // Background (BG) image

	float * d_save1; // BG images - before update
	float * d_save2; // BG images - after update
	float * d_save3; //

	unsigned char * d_bBG; // Background exploring map (0: not explored, 1: explored)
	int BG_x0;
	int BG_y0;

	//! Variables
	bool bgEnable; // background GPU buffer enable
	bool UpdateEnabled; // update enable (excluding the fish area)
	bool UpdateAllEnabled; // update all area (no exclusion)
	bool subEnabled; // BG subtraction enable
	//! setting
	double PxDist_mmppx; // pixel distance (mm/px)
	double theta; // radian
	double cth, sth; // cos(theta), sin(theta)
	int width;
	int height;
	int byte_per_image;
	int ADC; // bit
	int bg_width;
	int bg_height;
	int bg_width_half;
	int bg_height_half;
	double fish_mask_dy_mm;
	double fish_mask_dx_F_mm;
	double fish_mask_dx_B_mm;
	int fish_mask_dy;
	int fish_mask_dx_F;
	int fish_mask_dx_B;
	float max_intensity_set;

	int temp_count;

	// functions
	void updateParameters(double _pxDist, double _theta, int W, int H, int _ADCbit);
	void updateFishMask(double rx_pos, double rx_neg, double ry);
	void updateFishMask();
	void gpuMalloc(void);
	void gpuFree(void);
	void gpuInit(void);
	Point2i StagePos2GMapPxPos(Point2f _ref);
	Point2i StagePos2GMapPxPos(float x, float y);
	bool cudaBGsub(uint16 * _src, double * _pos);
	void cudacopyBG(float * dst, double * pos_double);
	void cudasaveBG(void);
	void cudaupdateBG(float cx, float cy, float th_rad, bool isFishFound);
};
