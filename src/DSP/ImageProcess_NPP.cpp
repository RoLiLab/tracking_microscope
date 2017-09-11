#include "Base/base.h"


#include "DSP/ImageProcess_NPP.h"
//#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "CUDA_Kernels.h"
#include <cufft.h>


DivGradDetector_GPU::DivGradDetector_GPU(void) {


	//cufftHandle test;
	//cufftResult err = cufftPlan2d(&test, 100, 100, CUFFT_R2C);

	// CPU memory
	p_Src_8U  = NULL;
	p_Src_8U_Dummy = NULL;
	// GPU memory - images
	p_NPPSrc_8U = NULL;
	p_NPPSrc = NULL;
	p_NPPimg_Gaussian_x = NULL;
	p_NPPimg_Gaussian = NULL;
	p_NPPLN_x1 = NULL;
	p_NPPLN_x = NULL;
	p_NPPLN_y1 = NULL;
	p_NPPLN_y = NULL;
	p_NPPGRD_x = NULL;
	p_NPPGRD_y = NULL;
	p_NPPimDivF = NULL;
	p_NPPimDivF_gard_X = NULL;
	p_NPPimDivF_gard_Y = NULL;
	p_NPPim_xpos = NULL;
	p_NPPim_ypos = NULL;
	p_NPPim_labelmap = NULL;
	p_NPPpeak = NULL;

	// GPU memory - Kernels
	p_NPPKernel_Gaussian_C = NULL;
	p_NPPKernel_Gaussian_R = NULL;
	p_NPPKernel_divX_C = NULL;
	p_NPPKernel_divX_R = NULL;
	p_NPPKernel_divY_C = NULL;
	p_NPPKernel_divY_R = NULL;
	p_NPPKernel_GradX = NULL;
	p_NPPKernel_GradY = NULL;
	p_NppMask_ROI = NULL;

	numPts = 3;
	p_NPPMinValue = NULL;
	p_pt = NULL;
	p_pt_xy = NULL;
	p_pt_xy_f = NULL;

	Reduction_SrcAtm = NULL;
	Reduction_SrcBuf = NULL;
	Reduction_IdxAtm = NULL;
	Reduction_IdxBuf = NULL;
	Reduction_SrchRadius = NULL;


	RefInd_pow2 = NULL;
	RefInd_y = NULL;

	enableROISearch = false;

	KernelSize = 0;
	SrchRadius = 70; // pixels
	Sigma = 6;
	Kernelradius = 11;
	imgSize.height = 480;
	imgSize.width = 640;

	ROIRadius_Mask = 50;
	n_ROILength_Mask = 0;

	int devNum;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&devNum);
	cudaStatus = cudaSetDevice(GPU0);
	cudaStatus = cudaDeviceReset();

	int version = 0;
	cudaDriverGetVersion(&version);
	cudaRuntimeGetVersion(&version);

	gpuErrchk(cudaFree(0));

	updateParameter(Kernelradius, Sigma, SrchRadius, imgSize.height, imgSize.width);
	ImageShowIdx = 0;

	fishRefSize = 700; //px
	fishRefSize_variation = 0.2;
	fitnessRef_heading = 0.5;

	centroid_BW_threshold = 0.2f;
	centroid_srch_radius = 70.0f;
	centroid_distC2Y = 25.0;
	centroid_distC2B = 25.0;
	centroid_distB2E = 12.0;
};
DivGradDetector_GPU::~DivGradDetector_GPU(void) {
	clearCudaMem();
};
void DivGradDetector_GPU::updateParameter(int _Kernelradius, int _Sigma, int _SrchRadius, int _heigth, int _width) {
	SrchRadius = _SrchRadius; // pixels
	Sigma = _Sigma;
	Kernelradius = _Kernelradius;
	imgSize.height = _heigth;
	imgSize.width = _width;
	KernelSize = 2 * Kernelradius + 1;
	//offset = Point2d(-_Kernelradius, -_Kernelradius);
	offset = Point2d(1,1);
	n_NPPLength = imgSize.height*imgSize.width;

	SearchROISize_half = 50;
	SearchROISize = SearchROISize_half * 2 + 1;

	ROISize_Mask.height = 2*ROIRadius_Mask + 1; ROISize_Mask.width = ROISize_Mask.height;
	n_ROILength_Mask = ROISize_Mask.width * ROISize_Mask.height;



	// GPU memory allocation - images
	clearCudaMem();
	mallocCudaMem();
	createGPUKernelfromOpenCV();
	// free runs for CUDA code for initialization

}
void DivGradDetector_GPU::mallocCudaMem(void) {

	// GPU memory - images
	//PINNEDMomoryAllocation((float*)p_Src_8U, (float*)p_NPPSrc_8U, (size_t)n_NPPLength*sizeof(Npp8u), (float*)p_Src_8U_Dummy);
	gpuErrchk(cudaMalloc((void **)(&p_NPPSrc_8U), (size_t)n_NPPLength*sizeof(unsigned char)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPSrc), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPimg_Gaussian_x), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPimg_Gaussian), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPLN_x1), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPLN_x), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPLN_y1), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPLN_y), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPGRD_x), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPGRD_y), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPimDivF), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPimDivF_gard_X), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPimDivF_gard_Y), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPim_xpos), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPim_ypos), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPim_labelmap), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPpeak), (size_t)n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&RefInd_pow2), (size_t)n_NPPLength*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)(&RefInd_y), (size_t)n_NPPLength*sizeof(int)));

	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_Gaussian_C), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_Gaussian_R), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_divX_C), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_divX_R), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_divY_C), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_divY_R), (size_t)KernelSize*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_GradX), (size_t)2*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_NPPKernel_GradY), (size_t)2*sizeof(float)));
	// pointer for the locations
	gpuErrchk(cudaMalloc((void **)(&p_NPPMinValue), numPts*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&p_pt), 1024*numPts*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)(&p_pt_xy), 1024 * numPts*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)(&p_pt_xy_f), 1024 * numPts*sizeof(float)));

	gpuErrchk(cudaMalloc((void **)(&p_NppMask_ROI), n_ROILength_Mask*sizeof(float)));


	gpuErrchk(cudaMalloc((void **)(&Reduction_SrcAtm), threadsPerBlock*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&Reduction_SrcBuf), threadsPerBlock*sizeof(float)));
	gpuErrchk(cudaMalloc((void **)(&Reduction_IdxAtm), threadsPerBlock*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)(&Reduction_IdxBuf), threadsPerBlock*sizeof(int)));
	gpuErrchk(cudaMemset(Reduction_IdxAtm, 0, threadsPerBlock*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)(&Reduction_SrchRadius), 3*sizeof(int)));
	setZeroCudaMem();
}
void DivGradDetector_GPU::createGPUKernelfromOpenCV(void) {
	// create filters using opencv functions (local variable)
	float * Kernel_Gaussian = (float *)malloc(KernelSize*sizeof(float));
	createGaussianFilter((float)Sigma, KernelSize, Kernel_Gaussian);


	float * Kernel_div = (float *)malloc(KernelSize*sizeof(float));
	float * Kernel_div_1 = (float *)malloc(KernelSize*sizeof(float));
	for (int i = -Kernelradius; i <= Kernelradius; i++) {
		Kernel_div[i + Kernelradius] = (float)i;
		Kernel_div_1[i + Kernelradius] = (float)1;
	}

	float Kernel_Grad[2] = { -1, 1 };

	// Memory allocation for image processing (CPU - Scale 1)
	//cv::Mat blankImg32f = cv::Mat::zeros(imgSize.height, imgSize.width, CV_32F);
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_Gaussian_C, (const float*)Kernel_Gaussian, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_Gaussian_R, (const float*)Kernel_Gaussian, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_divX_C, (const float*)Kernel_div_1, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_divX_R, (const float*)Kernel_div, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_divY_C, (const float*)Kernel_div, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_divY_R, (const float*)Kernel_div_1, (size_t)KernelSize*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_GradX, (const float*)Kernel_Grad, (size_t)2 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((float*)p_NPPKernel_GradY, (const float*)Kernel_Grad, (size_t)2 * sizeof(float), cudaMemcpyHostToDevice));

	int ReductionRadius[3];
	ReductionRadius[0] = (int)(50 * 50);
	ReductionRadius[1] = (int)(20 * 20);
	ReductionRadius[2] = (int)(50 * 50);
	gpuErrchk(cudaMemcpy((int*)Reduction_SrchRadius, (const int*)ReductionRadius, (size_t)3 * sizeof(int), cudaMemcpyHostToDevice));

	free(Kernel_Gaussian);
	free(Kernel_div);
	free(Kernel_div_1);

	int * ind_x = (int *)malloc(n_NPPLength*sizeof(int));
	int * ind_y = (int *)malloc(n_NPPLength*sizeof(int));
	for (int i = 0; i < imgSize.height; i++) {
		for (int j = 0; j < imgSize.width; j++) {
			ind_x[i*imgSize.width + j] = 0;
			ind_y[i*imgSize.width + j] = i;
		}
	}

	int * ind_pow = (int *)malloc(1000 * sizeof(int));
	for (int i = 0; i < 1000; i++) {
		ind_pow[i] = i*i;
	}
	//error = cudaMemcpy((int*)RefInd_x, (const int*)ind_x, n_NPPLength*sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk(cudaMemcpy((int*)RefInd_pow2, (const int*)ind_pow, 1000 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy((int*)RefInd_y, (const int*)ind_y, n_NPPLength*sizeof(float), cudaMemcpyHostToDevice));
	free(ind_x); free(ind_y); free(ind_pow);

	int Radius2 = ROIRadius_Mask*ROIRadius_Mask;
	float ** ROIImgCV = (float **)malloc(ROISize_Mask.width*sizeof(float*));
	for (int i = 0; i < ROISize_Mask.width; i++)
		ROIImgCV[i] = (float *)malloc(ROISize_Mask.height*sizeof(float));

	//cv::Mat ROIImgCV = cv::Mat::zeros(ROISize_Mask.height, ROISize_Mask.width, CV_32F);
	for (int i = 0; i < ROISize_Mask.height; i++) {
		for (int j = 0; j < ROISize_Mask.height; j++) {
			int dx = j - ROIRadius_Mask;
			int dy = i - ROIRadius_Mask;
			if (dx*dx + dy*dy < Radius2)
				ROIImgCV[i][j] = (float)2;
			else
				ROIImgCV[i][j] = (float)1;
		}
	}
 	//gpuErrchk(cudaMemcpy((float*)p_NppMask_ROI, (const float*)ROIImgCV, (size_t)n_ROILength_Mask*sizeof(float), cudaMemcpyHostToDevice));

	for (int i = 0; i < ROISize_Mask.width; i++) free(ROIImgCV[i]);
	free(ROIImgCV);
}

void DivGradDetector_GPU::clearCudaMem(void) {
	if (p_NPPSrc_8U != NULL) gpuErrchk(cudaFree(p_NPPSrc_8U));p_NPPSrc_8U = NULL;
	if (p_NPPSrc != NULL) gpuErrchk(cudaFree(p_NPPSrc));p_NPPSrc = NULL;
	if (p_NPPimg_Gaussian_x != NULL) gpuErrchk(cudaFree(p_NPPimg_Gaussian_x));p_NPPimg_Gaussian_x = NULL;
	if (p_NPPimg_Gaussian != NULL) gpuErrchk(cudaFree(p_NPPimg_Gaussian));p_NPPimg_Gaussian = NULL;
	if (p_NPPLN_x1 != NULL) gpuErrchk(cudaFree(p_NPPLN_x1));p_NPPLN_x1 = NULL;
	if (p_NPPLN_x != NULL) gpuErrchk(cudaFree(p_NPPLN_x));p_NPPLN_x = NULL;
	if (p_NPPLN_y1 != NULL) gpuErrchk(cudaFree(p_NPPLN_y1));p_NPPLN_y1 = NULL;
	if (p_NPPLN_y != NULL) gpuErrchk(cudaFree(p_NPPLN_y));p_NPPLN_y = NULL;
	if (p_NPPGRD_x != NULL) gpuErrchk(cudaFree(p_NPPGRD_x));p_NPPGRD_x = NULL;
	if (p_NPPGRD_y != NULL) gpuErrchk(cudaFree(p_NPPGRD_y));p_NPPGRD_y = NULL;
	if (p_NPPimDivF != NULL) gpuErrchk(cudaFree(p_NPPimDivF)); p_NPPimDivF = NULL;
	if (p_NPPimDivF_gard_X != NULL) gpuErrchk(cudaFree(p_NPPimDivF_gard_X));p_NPPimDivF_gard_X = NULL;
	if (p_NPPimDivF_gard_Y != NULL) gpuErrchk(cudaFree(p_NPPimDivF_gard_Y)); p_NPPimDivF_gard_Y = NULL;
	if (p_NPPim_xpos != NULL) gpuErrchk(cudaFree(p_NPPim_xpos)); p_NPPim_xpos = NULL;
	if (p_NPPim_ypos != NULL) gpuErrchk(cudaFree(p_NPPim_ypos)); p_NPPim_ypos = NULL;
	if (p_NPPim_labelmap != NULL) gpuErrchk(cudaFree(p_NPPim_labelmap)); p_NPPim_labelmap = NULL;
	if (p_NPPpeak != NULL) gpuErrchk(cudaFree(p_NPPpeak));p_NPPpeak = NULL;

	if (p_NPPKernel_Gaussian_C != NULL) gpuErrchk(cudaFree(p_NPPKernel_Gaussian_C));p_NPPKernel_Gaussian_C = NULL;
	if (p_NPPKernel_Gaussian_R != NULL) gpuErrchk(cudaFree(p_NPPKernel_Gaussian_R));p_NPPKernel_Gaussian_R = NULL;
	if (p_NPPKernel_divX_C != NULL) gpuErrchk(cudaFree(p_NPPKernel_divX_C));p_NPPKernel_divX_C = NULL;
	if (p_NPPKernel_divX_R != NULL) gpuErrchk(cudaFree(p_NPPKernel_divX_R));p_NPPKernel_divX_R = NULL;
	if (p_NPPKernel_divY_C != NULL) gpuErrchk(cudaFree(p_NPPKernel_divY_C));p_NPPKernel_divY_C = NULL;
	if (p_NPPKernel_divY_R != NULL) gpuErrchk(cudaFree(p_NPPKernel_divY_R));p_NPPKernel_divY_R = NULL;
	if (p_NPPKernel_GradX != NULL) gpuErrchk(cudaFree(p_NPPKernel_GradX));p_NPPKernel_GradX = NULL;
	if (p_NPPKernel_GradY != NULL) gpuErrchk(cudaFree(p_NPPKernel_GradY));p_NPPKernel_GradY = NULL;

	if (p_NppMask_ROI != NULL) gpuErrchk(cudaFree(p_NppMask_ROI));p_NppMask_ROI = NULL;

	if (p_NPPMinValue != NULL) gpuErrchk(cudaFree(p_NPPMinValue));p_NPPMinValue = NULL;
	if (p_pt != NULL) gpuErrchk(cudaFree(p_pt));p_pt = NULL;
	if (p_pt_xy != NULL) gpuErrchk(cudaFree(p_pt_xy)); p_pt_xy = NULL;
	if (p_pt_xy_f != NULL) gpuErrchk(cudaFree(p_pt_xy_f)); p_pt_xy_f = NULL;
	if (RefInd_pow2 != NULL) gpuErrchk(cudaFree(RefInd_pow2));RefInd_pow2 = NULL;
	if (RefInd_y != NULL) gpuErrchk(cudaFree(RefInd_y));RefInd_y = NULL;

	if (Reduction_SrcAtm != NULL) gpuErrchk(cudaFree(Reduction_SrcAtm));Reduction_SrcAtm = NULL;
	if (Reduction_SrcBuf != NULL) gpuErrchk(cudaFree(Reduction_SrcBuf));Reduction_SrcBuf = NULL;
	if (Reduction_IdxAtm != NULL) gpuErrchk(cudaFree(Reduction_IdxAtm));Reduction_IdxAtm = NULL;
	if (Reduction_IdxBuf != NULL) gpuErrchk(cudaFree(Reduction_IdxBuf));Reduction_IdxBuf = NULL;
	if (Reduction_SrchRadius != NULL) gpuErrchk(cudaFree(Reduction_SrchRadius));Reduction_SrchRadius = NULL;


}





void DivGradDetector_GPU::setZeroCudaMem(void) {
	// 0. clean
	gpuErrchk(cudaMemset(p_NPPimg_Gaussian_x, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPimg_Gaussian, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPLN_x1, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPLN_x, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPLN_y1, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPLN_y, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPGRD_x, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPGRD_y, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPimDivF_gard_X, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPimDivF_gard_Y, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPim_xpos, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPim_ypos, 0, n_NPPLength*sizeof(float)));
	gpuErrchk(cudaMemset(p_NPPim_labelmap, 0, n_NPPLength*sizeof(float)));

}
void DivGradDetector_GPU::detect_CUDA(float * d_src) {
	//convolutionRowsGPU_8U(p_NPPimg_Gaussian_x, src, imgSize.width, imgSize.height, p_NPPKernel_Gaussian_R, Kernelradius);
	convolutionRowsGPU(p_NPPimg_Gaussian_x, d_src, imgSize.height, imgSize.width, p_NPPKernel_Gaussian_R, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );

	convolutionColumnsGPU(p_NPPimg_Gaussian, p_NPPimg_Gaussian_x, imgSize.height, imgSize.width, p_NPPKernel_Gaussian_C, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );

	// 4. Apply linear filters (Comprehensive) for x
	convolutionRowsGPU(p_NPPLN_x1, p_NPPimg_Gaussian, imgSize.height, imgSize.width, p_NPPKernel_divX_R, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );
	convolutionColumnsGPU(p_NPPLN_x, p_NPPLN_x1, imgSize.height, imgSize.width, p_NPPKernel_divX_C, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );
	// 5. Apply linear filters (Comprehensive) for y
	convolutionRowsGPU(p_NPPLN_y1, p_NPPimg_Gaussian, imgSize.height, imgSize.width, p_NPPKernel_divY_R, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );
	convolutionColumnsGPU(p_NPPLN_y, p_NPPLN_y1, imgSize.height, imgSize.width, p_NPPKernel_divY_C, Kernelradius);
	gpuErrchk( cudaPeekAtLastError() );
	// 6. calculate Gradient
	convolutionRowsGPU(p_NPPGRD_x, p_NPPLN_x, imgSize.height, imgSize.width, p_NPPKernel_GradX, 2);
	gpuErrchk( cudaPeekAtLastError() );
	convolutionColumnsGPU(p_NPPGRD_y, p_NPPLN_y, imgSize.height, imgSize.width, p_NPPKernel_GradY, 2);
	gpuErrchk( cudaPeekAtLastError() );
	// 7. calculate DivMap 	cv::add(-GRD_x, -GRD_y, imDivF);
	VecAdd_kernelGPU(p_NPPGRD_x, p_NPPGRD_y, p_NPPimDivF, imgSize.height*imgSize.width);
	gpuErrchk( cudaPeekAtLastError() );
}
void DivGradDetector_GPU::detect_CUDA_gaussianOnly(float * d_src) {
	//convolutionRowsGPU_8U(p_NPPimg_Gaussian_x, src, imgSize.width, imgSize.height, p_NPPKernel_Gaussian_R, Kernelradius);
	convolutionRowsGPU(p_NPPimg_Gaussian_x, d_src, imgSize.height, imgSize.width, p_NPPKernel_Gaussian_R, Kernelradius);
	gpuErrchk(cudaPeekAtLastError());

	convolutionColumnsGPU(p_NPPimg_Gaussian, p_NPPimg_Gaussian_x, imgSize.height, imgSize.width, p_NPPKernel_Gaussian_C, Kernelradius);
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemset(p_pt_xy_f, 0, 8*sizeof(float)));
	findmax_GPU(p_pt_xy_f + 6, p_NPPimg_Gaussian, imgSize.height, imgSize.width, GPU0);

	getImageMoment_GPU(p_pt_xy_f, p_NPPimg_Gaussian, p_pt_xy_f + 6, imgSize.height, imgSize.width, centroid_BW_threshold, centroid_srch_radius);
	gpuErrchk(cudaPeekAtLastError());
}
vector<Point2d> DivGradDetector_GPU::findFishPositionFromDivImg_CUDA(int * area, bool FishDetectionSuccess_PrevFrms) {
	vector<Point2d> pts;
	Point2d yolk, eye1, eye2;
	imSize zeroLT; imSize zeroROI;
	// -- GPU Kernel
	if (FishDetectionSuccess_PrevFrms) {
		reductionMaxIdxGPU_NIR(p_NPPimDivF,
			Reduction_SrcAtm, Reduction_IdxAtm,
			imgSize.height, imgSize.width,
			0, p_pt_xy, Reduction_SrchRadius,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk(cudaPeekAtLastError());
	}
	else
	{
		reductionMaxIdxGPU_NIR_NOGain(p_NPPimDivF,
			Reduction_SrcAtm, Reduction_IdxAtm,
			imgSize.height, imgSize.width,
			0, NULL, NULL,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk(cudaPeekAtLastError());
	}

	LinearIdx2SubInd( Reduction_IdxAtm, p_pt_xy, imgSize.width, 4); // #0 yolk, #3 previous pos
	gpuErrchk( cudaPeekAtLastError() );

	if (FishDetectionSuccess_PrevFrms) {
		reductionMaxIdxGPU_NIR(p_NPPimDivF,
			Reduction_SrcAtm + 1, Reduction_IdxAtm + 1,
			imgSize.height, imgSize.width,
			1, p_pt_xy, Reduction_SrchRadius,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk( cudaPeekAtLastError() );
	}
	else {
		reductionMaxIdxGPU_NIR_NOGain(p_NPPimDivF,
			Reduction_SrcAtm + 1, Reduction_IdxAtm + 1,
			imgSize.height, imgSize.width,
			1, p_pt_xy, Reduction_SrchRadius,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk( cudaPeekAtLastError() );
	}


	LinearIdx2SubInd( Reduction_IdxAtm + 1 , p_pt_xy + 2, imgSize.width, 1); // #1 eye1
	gpuErrchk( cudaPeekAtLastError() );

	if (FishDetectionSuccess_PrevFrms) {
		reductionMaxIdxGPU_NIR(p_NPPimDivF,
			Reduction_SrcAtm + 2, Reduction_IdxAtm + 2,
			imgSize.height, imgSize.width,
			2, p_pt_xy, Reduction_SrchRadius,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk( cudaPeekAtLastError() );
	}
	else {
		reductionMaxIdxGPU_NIR_NOGain(p_NPPimDivF,
			Reduction_SrcAtm + 2, Reduction_IdxAtm + 2,
			imgSize.height, imgSize.width,
			2, p_pt_xy, Reduction_SrchRadius,
			Reduction_SrcBuf, Reduction_IdxBuf,
			(const int *)RefInd_pow2, (const int *)RefInd_y);
		gpuErrchk( cudaPeekAtLastError() );
	}


	LinearIdx2SubInd(Reduction_IdxAtm, p_pt_xy, imgSize.height, 5); // all points (yolk, eye1, eye2, center(b/t eyes), center(yolk and center)
	gpuErrchk( cudaPeekAtLastError() );


	// 3 points detected from the maximum value
	// get center of area from the given position
	// threshold is the 90% of the maximum value)
	// 1. location : p_pt_xy[0]: yolk, p_pt_xy[4]: brain
	// 2. Reduction_SrcAtm[0]*0.8 & (Reduction_SrcAtm[1] + Reduction_SrcAtm[2])*0.8*0.5 : threhold
	// 3. p_NPPimDivF : source image
	// 4. SrcRadius = 50 (from yolk & from brain)
	//gpuErrchk(cudaMemset(p_NPPim_ypos, 0, n_NPPLength*sizeof(float)));
	//gpuErrchk(cudaMemset(p_NPPim_ypos, 0, n_NPPLength*sizeof(float)));
	//gpuErrchk(cudaMemset(p_NPPim_ypos, 0, n_NPPLength*sizeof(float)));

	//thresholdimg_GPU(p_NPPimDivF, p_pt_xy, imgSize.width, p_NPPim_xpos, p_NPPim_ypos, p_NPPim_labelmap);
	//getCentroidGPU(p_NPPimDivF, p_pt_xy+3, Reduction_SrcAtm+1, p_pt_xy_f+2);
	//



	int pt_idx[4] = { 0 };
	//cudaError t = cudaMemcpy(pt_idx, p_pt_xy, sizeof(int) * 4, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaMemcpy( pt_idx, Reduction_IdxAtm, sizeof(int) * 3, cudaMemcpyDeviceToHost));

	// 11. Yolk, Left, Right eyes in order

	yolk = Point2d(pt_idx[0] % imgSize.height, pt_idx[0] / imgSize.height);
	eye1 = Point2d(pt_idx[1] % imgSize.height, pt_idx[1] / imgSize.height);
	eye2 = Point2d(pt_idx[2] % imgSize.height, pt_idx[2] / imgSize.height);
	area[0] = yolk.cross(eye1) + eye1.cross(eye2) + eye2.cross(yolk);
	pts.push_back(yolk + offset);
	if (area > 0) {
		pts.push_back(eye1+ offset);
		pts.push_back(eye2+ offset);
	}
	else {
		pts.push_back(eye2+ offset);
		pts.push_back(eye1+ offset);
	}
	area[0] = abs(area[0]/2);
	return pts;
}
