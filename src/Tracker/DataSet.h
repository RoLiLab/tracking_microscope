#ifndef _DATASET_H
#define _DATASET_H

#include "KBase/common/common.h"
#include "KBase/OpenCV/OpenCV.h"
#include "HDF5/hdf5imagewriter.h"
#include "HDF5/hdf5imagereader.h"
#include "HDF5/hdf5datawriter.h"
#include "HDF5/hdf5datareader.h"
#include "XPSQ8/XPS_Q8.h"							// stage control
#include "Tracker\TrackingInfo.h"

class TrkData {
public:
	// basic functions
	TrkData(void);
	~TrkData(void);
	unsigned int frameNo; // clock when data is collected
	XPSGatheringInfo stagePos;
	cv::Mat * srcNIR; // image data
	cv::Mat * srcFLR; // image data
	zebrafishInfo NIRfishPos;
	CntrData m_CntrData;
};

class TrkDataSet {
public:
	// basic functions
	TrkDataSet(void);
	~TrkDataSet(void);
		unsigned int RecBuffer_FLR_PreTrigFrmCountMax;
	void PostTriggerRec();
	void PostTriggerRec_StartStop(void);
	bool EnableRecording;
private:
	vector<TrkData> * EntireData;
	vector<TrkData> * EntireDataRecOnly;
	HDF5ImageWriter * pNIR_writer;
	HDF5ImageWriter * pFLR_writer;
	HDF5DataWriter * pData_writer;
};

#endif
