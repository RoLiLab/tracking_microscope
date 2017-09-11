#include "Base/base.h"
#include "hdf5datareader.h"

HDF5DataReader::HDF5DataReader(const char * hdf5_filename)
//    : _read_offset{}
{
	_file = NULL;
	try {
		_file = new H5File (hdf5_filename, H5F_ACC_RDONLY);
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5DataReaderError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5DataReader function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error


	SingleData.push_back(new HDF5SingleDataSetReader(_file, "FrameNo", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "TimeStamp", H5_UINT64));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "StageZPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "StageZPos_Volt", H5_double));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "StageZSetPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "StageZSetPos_Volt", H5_double));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "FishPos_Orientation_Deg", H5_double));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isTracking", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isPositionReadingSuccess", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isFishPosDetection", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isReadyMPCInput", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isReadyOnTime", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isFrameDropped", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "DroppedFrameCount", H5_INT));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "FrameSimulation", H5_UINT64));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "GlobalMapIndex", H5_UINT64));
	SingleData.push_back(new HDF5SingleDataSetReader(_file, "isFishMoving", H5_UINT));


	// point data type definition
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePos", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageVel", H5_double));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageAcc", H5_double));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageSetPos", H5_double));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageInputVolt", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageSetInputVel", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageSetVoltRead", H5_double));


	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN2", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN1", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN0", H5_double));

	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN2_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN1_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosN0_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StagePosP0_MPC", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "StageVelN0_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageVelP1_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageVelP2_MPC", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "StageVelP3_MPC", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_Target_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_local_px", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_mmN2", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_mmN1", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_mmN0", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_mmP1", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "ErrPos_px", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "ErrPos_mmN2", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "ErrPos_mmN1", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "ErrPos_mmN0", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "ErrPos_mmP1", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_Leye_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_Reye_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_Yolk_px", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "FishPos_Brain_px", H5_doubleCV));

	PointData.push_back(new HDF5PointDataSetReader(_file, "PID_DesiredPos", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "PID_CurrentPos", H5_doubleCV));
	PointData.push_back(new HDF5PointDataSetReader(_file, "PID_InputVel", H5_doubleCV));

}

HDF5DataReader::~HDF5DataReader()
{
	if (_file)
		delete _file;
}
