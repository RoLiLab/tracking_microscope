#pragma once

#define SIZE_SMALL 1024
#define SIZE_NOMINAL 1024
#define SIZE_BIG 2048
#define SIZE_HUGE 65536

#include "XPSQ8/XPS_Q8_drivers.h"
#include "DAQmx/DAQmx_2P.h"
#include "XPSQ8/XPS_I2T.h"

struct XPSGatheringInfo {
	XPSGatheringInfo() : isNIRSynced(false), isEPISynced(false), isStageSynced(false), GatheringCompleted(false), DOToggle(-1) {
		CurrentPosition_Chamber[0] = 0; CurrentPosition_Chamber[1] = 0; 
		SetpointPosition[0] = 0; SetpointPosition[1] = 0; 
		CurrentPosition[0] = 0; CurrentPosition[1] = 0; 
		SetpointVelocity[0] = 0; SetpointVelocity[1] = 0; 
		CurrentVelocity[0] = 0; CurrentVelocity[1] = 0; 
		SetpointAcceleration[0] = 0; SetpointAcceleration[1] = 0; 
		CurrentAcceleration[0] = 0; CurrentAcceleration[1] = 0; 
		InputVoltage[0] = 0; InputVoltage[1] = 0; 
		Pos_Z = 0; Pos_Z_set = 0, index = 0;
		thermalcntrinput = 0;
		temperature_circle_rev = 0;
	}
	double CurrentPosition_Chamber[2];
	double SetpointPosition[2]; // setpoint position (two positioner)
	double CurrentPosition[2]; // current position (two positioner)
	double SetpointVelocity[2]; // setpoint Velocity (two positioner)
	double CurrentVelocity[2]; // current Velocity (two positioner)
	double SetpointAcceleration[2]; // setpoint Acceleration (two positioner)
	double CurrentAcceleration[2]; // current Acceleration (two positioner)
	double InputVoltage[2]; // current voltage (two positioner)
	double Pos_Z;
	double Pos_Z_set;
	bool isNIRSynced;
	bool isEPISynced;
	bool isStageSynced;
	bool GatheringCompleted;
	int DOToggle;
	double thermalcntrinput;
	unsigned int index;
	int temperature_circle_rev;
};

class DConvImpRsp
{
public:
	DConvImpRsp(void);
	~DConvImpRsp(void);
	double getOutput(void);
	double updateNewInput(double _input);

	void init(void);
	void setImpRsp(int n, double * _h);
	void setImpRsp(vector<double> _h);


private:
	int predh_size;
	double * h; // impulse response
	vector<double> input;
	double output;
	void convInputImpRsp(void);
};


struct XPSmsg
{
	XPSmsg() : frmNo(0), b_vaild(true) {};
	XPSGatheringInfo stagepos;
	bool b_vaild;
	unsigned int frmNo;
};

class XPS_Q8
{
public:
	// basic functions
	XPS_Q8(void);
	~XPS_Q8(void);
	unsigned int frmNo;
	double x_center;
	double y_center;
	double PosLimit;
	void DisplayErrorAndClose(int error, char* APIName);
	// ------------- Baic variables ------------- //
	// Connection variables
	char * pIPAddress; // IP address
	int nPort;
	double dTimeOut; // Connection timeout time
	int error; // error code
	int ControllerStatus; // status code
	char * ControllerStatus_buf; // < status comment
	int SocketID; // socket ID
	int eventID; 
	int PositionReadingDelay;
	// ------------- Group variables ------------- //
	// Group & Position IDs
	char* pGroup; // Group name
	char* pPositioner_X; // Positioner name
	char* pPositioner_Y; // Positioner name
	int nPositioners; // Number of positioners in the group
	XPSGatheringInfo CurrentStatus;
	XPSGatheringInfo * CurrentStatusBuffer;
	// ------------- Manual control variables ------------- //
	double ManualControlVeloicty_x;
	double ManualControlVeloicty_y;

	//------------- NewDesign2015 -------------- //
	// Range Checking function
	bool RangeChecking(void);
	volatile bool bRangeError;
	volatile bool b_emergencystop;
	vector<XPSmsg> msg_positionreading;
	volatile bool b_start;
	volatile bool b_stop;
	bool isHighLow;

	//------------- ---------------------------- //

	// ----- analog tracking parameters ----- //
	int AnalogTrackingMode; // 0: Velocity tracking, 1:Position tracking
	enum{AnalogVelocityTracking, AnnalogPositionTracking};
	bool bAnalogContronEnable;
	double AVT_scale_x; // scale (gain)
	double AVT_scale_x2; // scale (gain)
	double AVT_offset_x; // offset [mm]
	double AVT_scale_y;
	double AVT_scale_y2;
	double AVT_offset_y;
	double AVT_order; // order 1: linear, 2: polynomial
	double AVT_Velocity_max; // Maximum velocity for mapping
	double AVT_acceleration; // Accerlation (setting)
	double AVT_deadbandPosition; // Deadband position range (+- mm)
	double AVT_MMPSperMMerror; // Velocity setting per mVelocity2Volt_xm error [velocity / mm-error] [(mm/s)/(mm)] 
	double AVT_InputVoltage_max; // AVT input voltage maximum	
	double APT_scale_x; // scale (gain)
	double APT_offset_x; // offset [mm]
	double APT_scale_y;
	double APT_offset_y;
	double APT_Velocity_max; // Maximum velocity for mapping
	double APT_acceleration; // Accerlation (setting)
	
	// ------------- Functions ------------- //
	// Advanced
	bool InitializeStageAtBeginning(void);
	bool InitializeStageWReferencing(void);
	// Basic functions
	bool KillAll_NonInitiateStatus(void); // from any status to non-initiated status
	bool Initialization(void);  // from non-initiated status to non-refernced status
	bool ReferencingReady(void); // from non-refernced status to Ready (w/o home search - stay on the current position)
	bool HommingReady(void); // from non-refernced status to Ready (w home search - go to home(0,0))


	bool EnableAnalogTracking(void); // from Ready to AnalogTracking
	bool DisableAnalogTracking(void); // from AnalogTracking to Ready

	// Group stats functions
	int MyXYGroupControllerStatusRead(void);
	bool MyXYGroupCurrentLocation(void); // using network (using command)
	bool MyXyGroupGathering(void); // current position update (using always event)
	XPSGatheringInfo MyXyGroupGathering_old(void);
	bool isXpsDataInRange(void);
	bool isXpsDatavalid(XPSGatheringInfo t);

	// Analog control functions
	bool MyXyAnalogPositionTrakcingEnable(void);
	bool MyXyAnalogVelocityTrakcingEnable(void);
	bool MyXyAnalogTrakcingDisable(void);	
	void MyXyAnalogTrakcingInternalValueUpdate(void);
	double VoltsPerMMPS(void); // read only
	double DeadBandThresholdVolts(void); // read only

	// Group Jog motion
	void XYJogging(void);
	bool MyXYGroupJogModeEnable(void);
	bool MyXYGroupJogModeDisable(void);
	bool MyXYGroupJogParametersSet(double* pVelocity, double* pAcceleration);
	bool MyXYGroupMoveAbsolute(double* TargetPosition);
	
	bool MyXYGroupMoveEnable(void);
	bool MyXYGroupMoveDisable(void);


	// Analog voltage control / Internal values
	double AVT_Deadbandthreshold;
	double AVT_Voltpermmps;
	// Initialization - private?
	bool MyXYGroupKill(void);
	bool MyXYGroupInitialize(void);
	bool MyXYGroupHomeSearch(void);
	bool MyXYGroupDACSet(void); // setting the event 
	// Connection functions
	bool TCPConnection(void);
	bool TCPDisconnection(void);
	bool VersionGet(void);
	
	double Velocity2Volt_x(double vel);
	double Velocity2Volt_y(double vel);
	double Volt2Velocity_x(double volt);
	double Volt2Velocity_y(double volt);
	// motion observation only (testing a stage spec) - before running, deactivate the gathering event in "MyXYGroupDACSet" and "Tracking Voltage applied." section in real-time control thread
	void MotionObservation_AnalogTracking(double * Volts, int mode); // double Voltage[2] (the real value applied in x, y; Mode 0: Voltage Control, 1: Position Control

	// --------------- I2T ----------//
	XPS_I2T I2T_X;
	XPS_I2T I2T_Y;
private:
	// motion observation only (testing a stage spec) - before running, deactivate the gathering event in "MyXYGroupDACSet"
	int MotionObservation_GatheringExport(char * pathFile_data); // export the current saved multiple gathering datat (return number of samples)
	DAQmx_2P * m_NIBoard;
	double Velocity2Volt(double vel, double scale, double db, double offset, double scale2);
	double Volt2Velocity(double volts, double scale, double db, double offset, double scale2);
	DConvImpRsp ChamberPosX;
	DConvImpRsp ChamberPosY;	
	void updateChamberPos(void);

};

