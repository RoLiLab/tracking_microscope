#ifndef HDF5TRACKINGMICROSCOPE_H
#define HDF5TRACKINGMICROSCOPE_H

#include <string>
#include <H5Cpp.h>

using namespace std;
using namespace H5;
// how to add field
// 1. add the item here (RecSingleData or RecPointData)
// 2. add (using push_back fn) SingleData or PointData at the contructor of HDF5DataWriter (HDF5DataWriter::HDF5DataWriter)
// *** (#1 and #2 should have same order)
// 3. record !
// e.g.
// HDF5DataWriter * pData_writer;	
// pData_writer = new HDF5DataWriter("a.h5");
// pData_writer->SingleData[R1_FrameNo]->write(&msg.NIRmsg.SyncPos.index);
// pData_writer->PointData[R2_StagePos]->write(msg.NIRmsg.SyncPos.CurrentPosition);
						
enum RecSingleData{
	R1_FrameNo // X 
	,R1_isTracking // X
	,R1_isPositionReadingSuccess  // X
	,R1_isFishPosDetection  // X
	,R1_isReadyMPCInput // X
	,R1_isReadyOnTime // X
	,R1_isFrameDropped // X
	,R1_DroppedFrameCount // X
	,R1_FrameSimulation // X
	,R1_GlobalMapIndex // X
	,R1_isFishMoving // X
	,R1_StageZPos_um
	,R1_StageZPos_Volt
	,R1_StageZSetPos_um	
	,R1_StageZSetPos_Volt
	,R1_FishPos_Orientation_Deg //X fish heading angle measured using the center of eyes and yolk in history
	,R1_FishPos_Heading_DegMean //X fish heading direction - Medium value of the angles b/t the center of eyes and yolk in history
	,R1_FishPos_MovingDirection_Deg //X moving direction of the projection positions
	,R1_PredictionWeight //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_FLR_FrmNo // fluorescent image number
	, R1_FLR_intmean // fluorescent image number
	, R1_FLR_intmax// fluorescent image number
	, R1_FLR_intfilled // fluorescent image number
	, R1_thermalcntrinput //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_thermalControlInput_Mode //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_FishIBICount //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_PiezoPos_um //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_BrainZ_um //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_BraindZ_um //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_reserved1 //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_reserved2 //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_reserved3 //X Prediction weight (How much the moving direction and heading are aligned)
	, R1_reserved4 //X Prediction weight (How much the moving direction and heading are aligned)
};

enum RecPointData{
	R2_StagePos // X
	,R2_FishPos_px  // X
	,R2_FishPos_Leye_px // X
	,R2_FishPos_Reye_px // X
	,R2_FishPos_Yolk_px // X
	,R2_FishPos_Brain_px // X
	,R2_StageInputVolt //X overall controller input voltage set
	,R2_StageSetInputVel //X overall controller input velocity set
	,R2_StageSetVoltRead //X input voltage set value reading
	, R2_TargetPos //X target position in mm (Global position)
	, R2_RefPos //X ref position in mm (Global position)
	,R2_FishPos //X fish position in mm (Global position)
	,R2_FishPosProj //X fish projection position (centered line)
	,R2_FishPosPred //X fish prediction position (x10 position each)
	,R2_Err //X fish position error
	,R2_Err_px	//X fish position error in px
	,R2_FishVelPrlPpd //X fish velocity in parallel and perpendicular axis
	,R2_FishVelWeighted //X fish predicted weighed velocity and its gradient (mm/s)/(frame)
	,R2_StagePosMPC //X stage position in MPC controller (open-loop position)
	,R2_StageSetInputVelMPC //X MPC controller input velocity set
	,R2_PID_DesiredPos //X PID controller desired position
	,R2_PID_CurrentPos //X PID controller target position
	,R2_StageSetInputVelPID //X PID controller input velocity set
	,R2_StageI2T //X Stage I2T value (approximated)
	,R2_Stage_L2Weight //X Stage I2T value (approximated)
	,R2_fish_fittness // fish size and fitness
	, R2_Piezo_Z // set and real value
	, R2_StageAcc_setpoint // stage acceleration x and y
	, R2_teensyIdx //X Prediction weight (How much the moving direction and heading are aligned)
	, R2_FishPos_EyeAngleDeg // X
};
enum RecParameters{
	RP_PID_Kp 
	,RP_PID_Ki
	,RP_PID_Kd
	,RP_PID_method
};

enum RecDataType{ H5_UINT, H5_double, H5_float, H5_UINT64, H5_INT, H5_CHAR, H5_doubleCV, H5_intCV, H5_floatCV };

PredType convertDatatype2H5PredType(int SingleData_Type);

class HDF5PointDataSet
{
public:
    HDF5PointDataSet(H5File* _file, char * Name, int SingleData_Type);
    void write(void * point);
    ~HDF5PointDataSet();
private:
    DataSet* _dataset;
    DataSpace* _memory_dataspace;
    hsize_t _write_size[2];
    hsize_t _write_offset[2];
    hsize_t _total_size[2];
	int _DataType;
};


class HDF5SingleDataSet
{
public:
    HDF5SingleDataSet(H5File* _file, char * Name, int SingleData_Type);
    void write(void * data);
    ~HDF5SingleDataSet();
private:
    DataSet* _dataset;
    DataSpace* _memory_dataspace;
    hsize_t _write_size;
    hsize_t _write_offset;
    hsize_t _total_size;	
	int _DataType;
};
 
class HDF5PointDataSetReader
{
public:
    HDF5PointDataSetReader(H5File* _file, char * Name, int SingleData_Type);
    void read(int64_t image_index, void * point);
	int Data_Length(void);
    ~HDF5PointDataSetReader();
private:
    DataSet* _dataset;
	DataSpace* _file_dataspace;
    DataSpace* _memory_dataspace;
    hsize_t _read_size[2];
    hsize_t _read_offset[2];
    hsize_t _total_size[2];
	int _DataType;
};


class HDF5SingleDataSetReader
{
public:
    HDF5SingleDataSetReader(H5File* _file, char * Name, int SingleData_Type);
    void read(int64_t index, void * point);
	int Data_Length(void);
    ~HDF5SingleDataSetReader();
private:
    DataSet* _dataset;
    DataSpace* _file_dataspace;
    DataSpace* _memory_dataspace;
    hsize_t _read_size;
    hsize_t _read_offset;
    hsize_t _total_size;	
	int _DataType;
};
 

#endif // HDF5IMAGEREADER_H
