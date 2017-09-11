#include "Base/base.h"
#include "hdf5imagewriter.h"

#include "HDF5\hdf5datawriter.h"

HDF5DataWriter::HDF5DataWriter(const string hdf5_filename)
{
	try {
	 _file = new H5File(hdf5_filename.c_str(), H5F_ACC_TRUNC);
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5DataWriterError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5DataWriter function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error

	// single data type definition
	SingleData.push_back(new HDF5SingleDataSet(_file, "FrameNo", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isTracking", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isPositionReadingSuccess", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isFishPosDetection", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isReadyMPCInput", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isReadyOnTime", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isFrameDropped", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "DroppedFrameCount", H5_INT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FrameSimulation", H5_UINT64));
	SingleData.push_back(new HDF5SingleDataSet(_file, "GlobalMapIndex", H5_UINT64));
	SingleData.push_back(new HDF5SingleDataSet(_file, "isFishMoving", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "StageZPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "StageZPos_Volt", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "StageZSetPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "StageZSetPos_Volt", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FishPos_Orientation_Deg", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FishPos_Heading_DegMean", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FishPos_MovingDirection_Deg", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "PredictionWeight", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FLR_FrameNo", H5_INT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FLR_intmean", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FLR_intmax", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FLR_intfilled", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "ThermalControlInput", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "ThermalControlInput_Mode", H5_INT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "FishIBICount", H5_INT));

	SingleData.push_back(new HDF5SingleDataSet(_file, "PiezoPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "BrainPosZ_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "BrainPosdZ_um", H5_double));


	SingleData.push_back(new HDF5SingleDataSet(_file, "reserved1", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "reserved2", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "reserved3", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "reserved4", H5_double));

	// point data type definition
	PointData.push_back(new HDF5PointDataSet(_file, "StagePos", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_Target_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_Leye_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_Reye_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_Yolk_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_Brain_px", H5_double));

	PointData.push_back(new HDF5PointDataSet(_file, "StageInputVolt", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "StageSetInputVel", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "StageSetVoltRead", H5_double));

	PointData.push_back(new HDF5PointDataSet(_file, "TargetPos", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "RefPos", H5_double));

	PointData.push_back(new HDF5PointDataSet(_file, "FishPos", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPosProj", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPosPred", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "Err", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "Err_px", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishVelPrlPpd", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishVelWeighted", H5_double));

	PointData.push_back(new HDF5PointDataSet(_file, "StagePosMPC", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "StageSetInputVelMPC", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "PID_DesiredPos", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "PID_CurrentPos", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "StageSetInputVelPID", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "StageI2T", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "L2_Weight", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "fish_fittness", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "Piezo_Z", H5_double));

	PointData.push_back(new HDF5PointDataSet(_file, "StageAcc_setpoint", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "teensyIdx", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "FishPos_EyeAngleDeg", H5_double));
}


HDF5DataWriter::~HDF5DataWriter()
{
	for (int i = 0; i < SingleData.size(); i++)
		delete SingleData[i];
	for (int i = 0; i < PointData.size(); i++)
		delete PointData[i];
	for (int i = 0; i < parameters.size(); i++)
		delete parameters[i];
	if (_file)
		delete _file;
}

void HDF5DataWriter::flush(void)
{
	if (_file) {
		_file->flush(H5F_SCOPE_GLOBAL);
	}
}

HDFWriter::HDFWriter(const string hdf5_filename)
{
	err = false;
	try {
		_file = new H5File(hdf5_filename.c_str(), H5F_ACC_TRUNC);
	}
	catch (H5::Exception  & e) {
		char pathFile_data[255];
		sprintf(pathFile_data, e.getCDetailMsg());
		err = true;
		_file = NULL;
		return;
	}
}

void HDFWriter::write(char * NAME, int SingleData_Type, void * v) {
	HDF5SingleDataSet temp(_file, NAME, SingleData_Type);
	temp.write(v);
}

void HDFWriter::write(char * NAME, int SingleData_Type, void * v, int n) {
	HDF5SingleDataSet temp(_file, NAME, SingleData_Type);
	for (int i = 0; i < n; i++) {
		switch (SingleData_Type) { // switch: get the key (1-9 : number pad)
		case H5_UINT:
			temp.write((unsigned int*)v + i);
			break;
		case H5_double:
			temp.write((double*)v + i);
			break;
		case H5_float:
			temp.write((float*)v + i);
			break;
		case H5_UINT64:
			temp.write((uint64*)v + i);
			break;
		case H5_INT:
			temp.write((int*)v + i);
			break;
		} // end switch
	}
}

HDFWriter::~HDFWriter()
{
	if (_file)
		delete _file;
}

HDFSettingWriter::HDFSettingWriter(const string hdf5_filename)
{
	try {
	 _file = new H5File(hdf5_filename.c_str(), H5F_ACC_TRUNC);
	} catch(H5::Exception  & e) {
		char pathFile_data [255];
		sprintf (pathFile_data, e.getCDetailMsg());
		return;
	} // end catch error
	/*
	// single data type definition
	parameters.push_back(new HDF5SingleDataSet(_file, "StageManualVel", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "StageDeadband", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "StageAVTVelMax", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "StageAVTAccMax", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "StageAVTInputMaxVolt", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrNIIter", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrNIStepPerCycle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrNIMaxVel", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrNIResolutionVel", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrMPCIter", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrMPCStepPerCycle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrMPCMaxVel", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ClbrMPCResolutionVel", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "NIRCameraExpTime_us", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "NIRCameraPxDist", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "NIRCameraAlignAngle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "NIRImgProcDivKernelSize", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "NIRImgProcDivSrchSize", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "EPICameraExpTime_us", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "EPIPxDist", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ExtTrgNIRCycle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ExtTrgEPICycle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "ExtTrgStageCycle", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "TrackingMode", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "TrackingTarget", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "TrackingMPCVelMax", H5_double));
	parameters.push_back(new HDF5SingleDataSet(_file, "TrackingAreaPxSet", H5_double));

	// point data type definition
	parameters_xy.push_back(new HDF5PointDataSet(_file, "StageAVTScale1", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "StageAVTScale2", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "StageAVTOffset", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "TrackingPIDKp", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "TrackingPIDKd", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "TrackingPIDKi", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "TrackingCenterPx", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "ClbrMPCParameter_s", H5_double));
	parameters_xy.push_back(new HDF5PointDataSet(_file, "ClbrMPCParameter_ds", H5_double));
	*/
}


HDFSettingWriter::~HDFSettingWriter()
{
	for (int i = 0; i < parameters.size(); i++)
		delete parameters[i];
	for (int i = 0; i < parameters_xy.size(); i++)
		delete parameters_xy[i];
	if (_file)
		delete _file;
}

parameterSet::parameterSet(void)
{
	StageManualVel = 10;
	StageDeadband = 0.112;
	ClbrNIIter = 100;
	ClbrNIStepPerCycle = 15;
	ClbrNIMaxVolt = 9.5;
	ClbrNIResolutionVolt = 0.1;
	ClbrMPCIter = 10;
	ClbrMPCStepPerCycle = 15;
	ClbrMPCMaxVel = 50;
	ClbrMPCResolutionVel = 5;
	NIRCameraExpTime_us = 60; // us
	NIRCameraPxDist = 14.2816;
	NIRCameraAlignAngle = 0.0326020;
	NIRImgProcDivKernelSize = 15;
	NIRImgProcDivSrchSize = 70;
	EPICameraExpTime_us = 2030; // us
	EPIPxDist = 0.2440; // us
	ExtTrgNIRCycle = 83; // ms
	ExtTrgEPICycle = 83; // ms
	ExtTrgStageCycle = 747; // ms
	TrackingMode = 0;
	TrackingTarget = 0;
	TrackingMPCVelMax = 50;
	StageAVTVelMax = 285;
	StageAVTAccMax = 4000;
	StageAVTInputMaxVolt = 0;
	TrackingAreaPxSet = 550;

	FLRCameraFrameRate = 100;
	FLRCameraExpTime = 0.1; // [ms]
	FLRCameraReadoutTime = 8.4; // [ms]
	FLRCameraExpTime_Max = (1000/FLRCameraFrameRate) - FLRCameraReadoutTime;
	if (FLRCameraExpTime_Max < 0) FLRCameraExpTime_Max = 0.0;


	StageAVTScale1 = Point2d(30,30);
	StageAVTScale2 = Point2d(0.9946,0.9998);
	StageAVTOffset = Point2d(0.012338,0.005799);
	TrackingCenterPx = Point2d(374,256);
	TrackingPIDKp = Point2d(0,0);
	TrackingPIDKd = Point2d(0,0);
	TrackingPIDKi = Point2d(0,0);

	double temp_Mx[14] = {0.0011085250,0.0036431720,0.0019771253,0.0008768403,0.0002990879,0.0000794769,0.0000332976,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000};
	double temp_My[14] = {0.0010918617,0.0037464493,0.0016940932,0.0008053296,0.0004632465,0.0001900920,0.0000598710,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000};
	for (int i = 0 ; i < 14; i++)
		ClbrMPCParameter_ds.push_back(Point2d(temp_Mx[i],temp_My[i]));
	ClbrMPCParameter_s.push_back(ClbrMPCParameter_ds[0]);
	for (int i = 1 ; i < 14; i++)
		ClbrMPCParameter_s.push_back(ClbrMPCParameter_s[i-1] + ClbrMPCParameter_ds[i]);
}

parameterSet::~parameterSet(void) {
}

PredType convertDatatype2H5PredType(int SingleData_Type){
	switch(SingleData_Type) {
		case H5_UINT:
			return PredType::NATIVE_UINT;
		case H5_double:
			return PredType::NATIVE_DOUBLE;
		case H5_float:
			return PredType::NATIVE_FLOAT;
		case H5_UINT64:
			return PredType::NATIVE_UINT64;
		case H5_INT:
			return PredType::NATIVE_INT;
		case H5_CHAR:
			return PredType::NATIVE_CHAR;
		} // end switch
	return PredType::NATIVE_DOUBLE;
}
