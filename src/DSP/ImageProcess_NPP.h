#pragma once


// CUDA runtime
#include <cuda_runtime.h>
#include "DSP/GlobalMap.h"

class DivGradDetector_GPU
{
public:
	DivGradDetector_GPU(void);
	DivGradDetector_GPU(int _Kernelradius, int _threshold);
	~DivGradDetector_GPU(void);
	//! Variables for image processing
	int Kernelradius;
	int Sigma;
	int SrchRadius;
	Point2d offset;

	// GPU memory - images
	uint8 * p_Src_8U;
	uint8 * p_Src_8U_Dummy;
	// GPU memory - images/data
	uint8 * p_NPPSrc_8U;
	float * p_NPPSrc;
	float * p_NPPimg_Gaussian_x;
	float * p_NPPimg_Gaussian;
	float * p_NPPLN_x1;
	float * p_NPPLN_x;
	float * p_NPPLN_y1;
	float * p_NPPLN_y;
	float * p_NPPGRD_x;
	float * p_NPPGRD_y;
	float * p_NPPimDivF;
	float * p_NPPim_xpos;
	float * p_NPPim_ypos;
	float * p_NPPim_labelmap;
	float * p_NPPimDivF_gard_X;
	float * p_NPPimDivF_gard_Y;
	uint8 * p_NPPpeak;

	// GPU memory - Kernels (filters)
	float * p_NPPKernel_Gaussian_C;
	float * p_NPPKernel_Gaussian_R;
	float * p_NPPKernel_divX_C;
	float * p_NPPKernel_divX_R;
	float * p_NPPKernel_divY_C;
	float * p_NPPKernel_divY_R;
	float * p_NPPKernel_GradX;
	float * p_NPPKernel_GradY;

	imSize imgSize;
	imSize imgROISize_Kernel_GradX;
	imSize imgROISize_Kernel_GradY;
	imSize imgROISize_Kernel_row;
	imSize imgROISize_Kernel_col;

	// ROI update
	float * p_NppMask_ROI;
	imSize ROISize_Mask;
	int ROIRadius_Mask;
	int n_ROILength_Mask;

	// buffers & Image min/max value Pointer (may not need)
	uint8 * p_NPPDeviceBuffer_Min;
	float * p_NPPMinValue;
	int * p_pt;
	int * p_pt_xy;
	float * p_pt_xy_f;
	int * RefInd_pow2;
	int * RefInd_y;
	int numPts;


	float * Reduction_SrcAtm;
	float * Reduction_SrcBuf;
	int * Reduction_IdxAtm;
	int * Reduction_IdxBuf;
	int * Reduction_SrchRadius;

	// functions
	void updateParameter(int _Kernelradius, int _Sigma, int _SrchRadius, int _heigth, int _width);


	void detect_CUDA(float * src);
	void detect_CUDA_gaussianOnly(float * src);
	std::vector<Point2d> findFishPositionFromDivImg_CUDA(int * area, bool FishDetectionSuccess_PrevFrms);

	//! global map
	//uint8 * p_NPPGlobalMap;
	//imSize imgGlobalROISize;
	//double PxDist_mmppx;
	//double theta; // radian
	//double cth, sth;

	//! variables for Display (take ~ 10ms - bad)
	enum {OrignalImg, GaussianImg, DivFImg};
	int ImageShowIdx;
	void updateDispImage(void);

	bool enableROISearch;
	int SearchROISize;
	int SearchROISize_half;


	double fishRefSize;
	double fishRefSize_variation;
	double fitnessRef_heading; // cos(phi) - phi is the angle between yolk-brain and left-rightEyes

	float centroid_BW_threshold;
	float centroid_srch_radius;
	double centroid_distC2Y;
	double centroid_distC2B;
	double centroid_distB2E;

private:
	void clearCudaMem(void);
	void mallocCudaMem(void);
	void setZeroCudaMem(void);

	void createGPUKernelfromOpenCV(void);
	int KernelSize;
	int n_NPPLength;

	cudaError_t error;

};

void findFishPositionFromDivImg_cuda(float* src, int srchSize1, int srchSize2, float* dst);
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n);

__global__ void convolution_1D_basic_kernel(float *N, float *P,  float *M, int Mask_Width, int Width);


void findLocalMaxima_cuda(float* src, float* src_grad_x, float* src_grad_y, uint8* dst, imSize imgSize, int threshold);
__global__ void findLocalMaxima_cudaGPU(float* src, float* src_grad_x, float* src_grad_y, uint8* dst, imSize imgSize, int threshold);
//void NPP_Process(cv::Mat &src, cv::Mat &src2, cv::Mat &dst);
