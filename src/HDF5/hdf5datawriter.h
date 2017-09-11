#ifndef HDF5DATAWRITER_H
#define HDF5DATAWRITER_H

#include <string>
#include <H5Cpp.h>

using namespace std;
using namespace H5;

#include "HDF5\hdf5_TrackingMicroscope.h"



class HDF5DataWriter
{
public:
    HDF5DataWriter(const string hdf5_filename);
	void flush(void);
    ~HDF5DataWriter();
    H5File* _file;
    vector<HDF5SingleDataSet *> SingleData;
	vector<HDF5PointDataSet *> PointData;
	vector<HDF5PointDataSet *> parameters;
};

class HDFWriter {
public:
	HDFWriter(const string hdf5_filename);
	~HDFWriter();
	void write(char * Name, int SingleData_Type, void * v);
	void write(char * NAME, int SingleData_Type, void * v, int n);
	H5File* _file;
	bool err;
};

class HDFSettingWriter {
public:
    HDFSettingWriter(const string hdf5_filename);
    ~HDFSettingWriter();
    H5File* _file;
    vector<HDF5SingleDataSet *> parameters;
	vector<HDF5PointDataSet *> parameters_xy;
};


class parameterSet {
public:
	parameterSet(void);
	~parameterSet(void);
	void updateOnline(void);
	void updateAll(void);

	// recording variables
	double StageManualVel;
	double StageDeadband;
	double ClbrNIIter;
	double ClbrNIStepPerCycle;
	double ClbrNIMaxVolt;
	double ClbrNIResolutionVolt;
	double ClbrMPCIter;
	double ClbrMPCStepPerCycle;
	double ClbrMPCMaxVel;
	double ClbrMPCResolutionVel;
	double NIRCameraExpTime_us; // us //!!//
	double NIRCameraPxDist; //!!//
	double NIRCameraAlignAngle; //!!//
	double NIRImgProcDivKernelSize; //!!//
	double NIRImgProcDivSrchSize; //!!//
	double EPICameraExpTime_us; // us
	double EPIPxDist; // us
	double ExtTrgNIRCycle; // ms
	double ExtTrgEPICycle; // ms
	double ExtTrgStageCycle; // ms
	double TrackingMode; //!!//
	double TrackingTarget;; //!!//
	double TrackingMPCVelMax;; //!!//
	double TrackingAreaPxSet; //!!//
	double StageAVTVelMax;
	double StageAVTAccMax;
	double StageAVTInputMaxVolt;

	double FLRCameraExpTime; // [ms]
	double FLRCameraReadoutTime; // [ms]
	double FLRCameraFrameRate; // [fps]
	double FLRCameraExpTime_Max; // [ms] 

	Point2d StageAVTScale1;
	Point2d StageAVTScale2;
	Point2d StageAVTOffset;	
	Point2d TrackingPIDKp; //!!//
	Point2d TrackingPIDKd; //!!//
	Point2d TrackingPIDKi; //!!//
	Point2d TrackingCenterPx; //!!//
	vector<Point2d> ClbrMPCParameter_s;
	vector<Point2d> ClbrMPCParameter_ds;
	
	// non recording variables
};
enum SettingParameter1D{
	StageManualVel
	,StageDeadband
	,StageAVTVelMax
	,StageAVTAccMax
	,StageAVTInputMaxVolt
	,ClbrNIIter
	,ClbrNIStepPerCycle
	,ClbrNIMaxVel
	,ClbrNIResolutionVel
	,ClbrMPCIter
	,ClbrMPCStepPerCycle
	,ClbrMPCMaxVel
	,ClbrMPCResolutionVel
	,NIRCameraExpTime_us // us
	,NIRCameraPxDist
	,NIRCameraAlignAngle
	,NIRImgProcDivKernelSize
	,NIRImgProcDivSrchSize
	,EPICameraExpTime_us // us
	,EPIPxDist // us
	,ExtTrgNIRCycle // ms
	,ExtTrgEPICycle // ms
	,ExtTrgStageCycle // ms
	,TrackingMode
	,TrackingTarget
	,TrackingMPCVelMax	
	,TrackingAreaPxSet
};

enum SettingParameter2D{
	StageAVTScale1
	,StageAVTScale2
	,StageAVTOffset
	,TrackingPIDKp
	,TrackingPIDKd
	,TrackingPIDKi
	,TrackingCenterPx
	,ClbrMPCParameter_s
	,ClbrMPCParameter_ds
};



#endif // HDF5IMAGEWRITER_H
