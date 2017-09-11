#pragma once

#include "ipp.h"
#include "ippi.h"
enum {Lower, Middle, Upper};
enum {Upward, Downward};

struct FLRImageData {
	FLRImageData() : FrmNo(0), filled(0), intensity_mean(0), intensity_max(0){ piezo_z[0] = 0; piezo_z[1] = 0; }
	int FrmNo;
	double filled; // percentatge of filled area
	double intensity_mean; // intensity mean with a mask (above the threshold (=300))
	uint16 intensity_max; // intensity mean with a mask (above the threshold (=300))
	double piezo_z[2]; // set & real value
};


class AutoFocusVolumeScan{
	public:
	AutoFocusVolumeScan(void);
	~AutoFocusVolumeScan(void);
	void updateparameters(double _dzset, uint16 _threshold, double _mu_low, double _mu_high);
	void updateparameters_fixed(double _dzset, uint16 _threshold, double _mu_low, double _mu_high);
	void updateparameters_imgSetup(int _w, int _h);
	enum{ Adaptive_continue, Adaptive_sawtooth, Fixed_sawtooth};
	int mode;
	FLRImageData * imgdata;
	FLRImageData * imgdata_cur;
	FLRImageData * getimgdatafromIdx(int _FrmNo);
	double imgMeanIntensity_DIFF[2];
	double imgMeanIntensity_DIFF_ratio;
	vector<double> z; // image data
	vector<double> mu; // image data
	vector<double> mu_mva; // image data
	vector<double> z_real; // image data
	int histlen;
	double lockdown_minchange;
	double lockdown_minchangeset;
	int slidewinsize;
	double zmax;
	double zmin;
	double dzset;
	uint16 threshold;
	double mu_low;
	double mu_high;
	double mu_max;
	double mu_min;
	double mu_cur;
	double mu_threshold;
	IppiSize imgsize;

	double dz;
	double z_top;
	double z_bottom;

	double z_init;
	double vmean_cur;
	double vmean_max;
	double vmean_max_last;
	double vmean_min;
	double vmean_min_last;
	double z_vmean_max;
	int z_vmean_max_index;
	double z_vmean_max_offset;
	double z_vmean_max_last;
	int lockdown_steps;
	int lockdown_stepsset;
	double update_step(void);
	double update_step_sweep(void);
	double update_step_sweep_fixed(void);
	double update_step_sweep_1D(void);
	double update_step_sweep_1D_fixed(void);
	int samestep_size;
	double * sawthooth_buffer;

	unsigned char * mask;
	bool isfocused_cur;
	bool isfocused_prev;
	double piezo_inputVolts;
	double compute_mu(uint16 * pImg, int _FrmNo);
	double compute_mean(uint16 * pImg, int _FrmNo);
	double compute_mean_GS3(uint16 * pImg, int _FrmNo);
	double compute_mean_GS3u10(uint16 * pImg, int _FrmNo);
	double compute_mean_old(uint16 * pImg);
	double update_steps(void);
	double update_step_minchange(void);
	double update_old(void);
	double down(void);
	double up(void);


};
/*
bool sortByFocalValue(const AutoFocusVolumeScan &lhs, const AutoFocusVolumeScan &rhs);
vector<AutoFocusVolumeScan> createScanTable(double scanCenter, double scanRange, int scanStep, double piezodir, int scanImgsPerLayer);

class AutoFocusVolumeScan
{
public:
	AutoFocusVolumeScan(void);
	~AutoFocusVolumeScan(void);
	void Initialization(void); // initialization
	void Initialization(double zPos); // initialize all temproal values.
	bool isCompleteVolumeScan(void); // return whether the volume scan is completed or not
	double getNextScanLayerPosition(void); // return next scan layer Zpos
	double getNextVolumeScanCenterLayer(void); // return next Volume scan center layer
	void setFocusValue(cv::Mat image); // calculate a focus value from image
	void setFocusValue(double FocusValue); // add focus value
	double getFocusValue(int layer); // return focus value

	void setCurrentScanLayer(int layer); // setCurrent plane layer
	void updateCurrentScanLayer(void); // setCurrent plane layer
	int getCurrentScanLayer(void); // get the current plane layer

	double getPlaneZpos(int layer); // get the plane position
	double getUpperPlaneZpos(void); // get the upper plane position
	double getMiddlePlaneZpos(void); // get the middle plane position
	double getLowerPlaneZpos(void); // get the lower plane position

	void setZMax(double zMax); // set the maximum z-position value
	void setZMin(double zMin); // set the minimum z-position value
	void setdz(double dz); // set the z-position gap
	void setdzMax(double dzMax); // set the z-position gap Max
	void setdzMin(double dzMin); // set the z-position gap Min
	void setScanSequence(int * Scan); // set the scan sequence (layers)
	double getZMax(void); // get the maximum z-position value
	double getZMin(void); // get the minimum z-position value
	double getdz(void); // get the z-position gap
	double getdzMax(void); // get the z-position gap Max
	double getdzMin(void); // get the z-position gap Min
	int * getScanSequence(void); // get the scan sequence (layer)
	void setROI(cv::Rect ROI); // set a region of Interest for an image
	cv::Rect getROI(void); // get a region of Interest for an image
private:
	double m_zMax; // zPos maximum
	double m_zMin; // zPos minimum
	double m_dz; // zPos gap
	double m_dzMax; // zPos gap maximum
	double m_dzMin; // zPos gap minimum
	cv::Rect m_ROI; // Region of Interest for an image
	bool m_VolumeScanCompleteFlag; // volume scan complete flag
	int m_ScanSequence[3]; // 0: Lower plane, 1: Middle plane, 2: Upper plane
	double m_zPos[3]; // position fo the zPos (Lower, Middle, Upper)
	bool m_ImageCompleteFlag[3]; // true when the focus value of the plane is calculated (Lower, Middle, Upper)
	double m_FocusValue[3]; // focus value (Lower, Middle, Upper)
	int m_CurrentPlane; // current plane
	int m_NextFocalPlane; // next plane (next step)
}; // end struct AutoFocusVolumeScan*/
