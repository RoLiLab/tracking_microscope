#include "Base/base.h"
#include "XPS_Q8.h"
#include <Windows.h>


void XPS_Q8::DisplayErrorAndClose(int error, char* APIName)
{
	int error2;
	char strError[250];

	// Timeout error
	if (-2 == error) {printf("TCP timeout\n"); TCP_CloseSocket(SocketID); return;	}
	// The TCP/IP connection was closed by an administrator
	if (-108 == error) { printf("The TCP/IP connection was closed by an administrator\n"); return; }

	// Error => Get error description
	error2 = ErrorStringGet(SocketID, error, strError);
	// If error occurred with the API ErrorStringGet
	if (0 != error2) {
		sprintf(strError,"%s: %s",APIName, strError);
		//ErrorDisplayMessage(strError);
	}
	// Close TCP socket
	//TCP_CloseSocket(SocketID);
	return;
}

XPS_Q8::XPS_Q8(void)
{
	frmNo = 0;
	x_center = -13.65;
	y_center = 12.0;
	PosLimit = 2.8 * 25.4 / 2;
	isHighLow = false;
	b_emergencystop = false;
	b_start = false;
	b_stop = false;
	// setting the Stage Calibration from Stage position to Chamber position
	double h_x[] = {0.6675354, 1.1302682, -1.2386104, 0.1927997, 0.7702495, -0.7303005, 0.0117694, 0.5225657, -0.4141705, -0.0572995, 0.3453889, -0.2268252, -0.0744807, 0.2226126, -0.1189502, -0.0691915, 0.1401605, -0.0587030, -0.0561492, 0.0862730, -0.0262869, -0.0422153, 0.0519028, -0.0096891, -0.0301585, 0.0304782, -0.0017909, -0.0207401, 0.0174218, 0.0015186, -0.0138315, 0.0096485, 0.0025441, -0.0089839, 0.0051358, 0.0025317, -0.0056974, 0.0025902, 0.0021239, -0.0035318, 0.0012030, 0.0016295, -0.0021402, 0.0004803, 0.0011808, -0.0012666, 0.0001271};
	//double h_x[] = {0.651726, 1.251644, -1.421699, 0.408952, 0.525097, -0.678493, 0.238902, 0.214709, -0.320829, 0.133299, 0.084840, -0.150322, 0.071998, 0.031921, -0.069783, 0.037941, 0.011111, -0.032088, 0.019606, 0.003337, -0.014608, 0.009967, 0.000664, -0.006579, 0.004997, -0.000118, -0.002929, 0.002475, -0.000258, -0.001287, 0.001212, -0.000216, -0.000557, 0.000587, -0.000145};
	double h_y[] = {0.6666396127, 0.8219503582, -0.5670991269, -0.0448283787, 0.2114848562, -0.0723104488, -0.0420075868, 0.0419920532, -0.0030951333, -0.0130040352, 0.0064369984, 0.0017588111, -0.0029180371, 0.0006098123, 0.0007402233, -0.0005139293, -0.0000390390, 0.0001909998, -0.0000657985, -0.0000377348, 0.0000380077};
	int RespNx = 47;
	int RespNy = 21;
	ChamberPosX.setImpRsp(RespNx, h_x);	 ChamberPosX.init();
	ChamberPosY.setImpRsp(RespNy, h_y);	ChamberPosY.init();
	bRangeError = false;
	// Connecting variables
	pIPAddress = "192.168.254.254";
	double dTimeOut = 60;
	nPort = 5001;
	// Error initialization
	error = 0;
	ControllerStatus_buf = "\0";
	// Group & Position IDs
	SocketID = -1;
	eventID = -1;
	pGroup = "MyXYGroup";
	pPositioner_X = "MyXYGroup.ILS200LM-X";
	pPositioner_Y = "MyXYGroup.ILS200LM-Y";
	nPositioners = 2; // Number of positioners in the group
	ManualControlVeloicty_x = 10;
	ManualControlVeloicty_y = 10;
	bAnalogContronEnable = 0;
	PositionReadingDelay = 10; // 0.1ms
	// ----- analog tracking parameters ----- //
	AVT_scale_x = 20; // scaling
	AVT_scale_x2 = 1; // scaling
	AVT_offset_x = 0.005; // offset voltage
	AVT_scale_y = 20; // scaling
	AVT_scale_y2 = 1; // scaling
	AVT_offset_y = 0.0; // offset voltage
	AVT_order = 1; // Order 1st or 2nd
	AVT_InputVoltage_max = 10; // Volts
	AVT_Velocity_max = AVT_scale_x*AVT_InputVoltage_max*0.95; // mm/s
	AVT_acceleration = 1000; // mm/s^2
	AVT_Deadbandthreshold = 0;
	AVT_deadbandPosition = 0; // mm (for setting deadbandthreshold)
	AVT_MMPSperMMerror = 50; // "x" mm/s per 1 mm gap
	AnalogTrackingMode = 0; // 0: Velocity tracking, 1:Position tracking
	APT_scale_x = 1; // scale (gain)
	APT_offset_x = 0; // offset [mm]
	APT_scale_y = 1;
	APT_offset_y = 0;
	APT_Velocity_max = 200; // Maximum velocity for mapping
	APT_acceleration = 1000; // Accerlation (setting)
	MyXyAnalogTrakcingInternalValueUpdate();
	CurrentStatusBuffer = (XPSGatheringInfo *)malloc(1000 * sizeof(XPSGatheringInfo));




}

XPS_Q8::~XPS_Q8(void)
{
	free(CurrentStatusBuffer);
	TCPDisconnection();
}

// ------------------------------------------------------- //
bool XPS_Q8::RangeChecking(void) {
	double x = 10.0;
	double y = -15.0;
	double r = 27.0;
	if (SocketID == -1) {
		double dx2 = (CurrentStatus.CurrentPosition[1] - x)*(CurrentStatus.CurrentPosition[1] - x);
		double dy2 = (CurrentStatus.CurrentPosition[2] - y)*(CurrentStatus.CurrentPosition[2] - y);
		double ds = dx2 + dy2;
		if (ds > r*r)
			bRangeError = true;
	}
	return bRangeError;
}
// ------------------------------------------------------- //

bool XPS_Q8::InitializeStageAtBeginning(void) {
	bAnalogContronEnable = false;
	if (SocketID == -1) {	// Initialize stage
		TCPConnection();
		//MyXYGroupKill(); if (0 != error) return false; // Kill Group
		//if(!Initialization()) return false;
		MyXYGroupDACSet();
		Sleep(200);
		//if (AnalogTrackingMode == AnalogVelocityTracking)
		//	if(!HommingReady()) return false;
		//else if (AnalogTrackingMode == AnnalogPositionTracking)
		//	if(!ReferencingReady()) return false;
		HommingReady();
		ReferencingReady();
		Sleep(200);
		if(!EnableAnalogTracking()) return false;
		bAnalogContronEnable = true;
	}
	return bAnalogContronEnable;
}

bool XPS_Q8::InitializeStageWReferencing(void){
	if (SocketID == -1)
		return false;
	// If not success, connected already
	//if(!Initialization()) return false;
	//Sleep(100);
	//if (AnalogTrackingMode == AnalogVelocityTracking)
	//	if(!HommingReady()) return false;
	//else if (AnalogTrackingMode == AnnalogPositionTracking)
	if(!HommingReady()) return false;
	//if(!ReferencingReady()) return false;
	Sleep(100);
	if(!EnableAnalogTracking()) return false;
	Sleep(100);
	MyXYGroupControllerStatusRead();
	return true;
}

bool XPS_Q8::KillAll_NonInitiateStatus(void){ // from any status to non-initiated status
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	if (SocketID != -1) {
		do {
			MyXYGroupKill();
			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
		} while(ControllerStatus < 0 || ControllerStatus >= 10); // continue until it is non-Initiated(0~9)
	}
	return true;
}
bool XPS_Q8::Initialization(void){ // from non-initiated status to non-refernced status
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	MyXYGroupControllerStatusRead();
	if (ControllerStatus >= 0 && ControllerStatus < 10) { // 7: NOTINIT state due to a KillAll command
		while(ControllerStatus != 42) { // Not referenced state
			MyXYGroupInitialize();
			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
		};
	}
	else
		return false; // it is not initiated stutus
	return true;
}
bool XPS_Q8::ReferencingReady(void) { // from non-refernced status to Ready (w/o home search - stay on the current position)
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	MyXYGroupControllerStatusRead();
	if (ControllerStatus == 42) {// non-refernced status
		while(ControllerStatus != 11) { // Ready status from referencing
			//Sleep(10);
			if (ControllerStatus != 64) // referencing status
				//GroupReferencingStart(SocketID, pGroup); // referencing start
				GroupHomeSearch(SocketID, pGroup); // referencing start
			else
				GroupReferencingStop(SocketID, pGroup); // referencing stop
			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
		}
		bRangeError = false;
		return true;
	}
	else
		return false;
}
bool XPS_Q8::HommingReady(void) { // from non-refernced status to Ready (w home search - go to home(0,0))
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	MyXYGroupControllerStatusRead();
	if (ControllerStatus == 42) {// non-refernced status
		while(ControllerStatus <= 10 || ControllerStatus > 19) { // Ready status from referencing
			MyXYGroupHomeSearch();//SleepThread(10);
			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
		}
		return true;
	}
	else
		return false;
}
bool XPS_Q8::EnableAnalogTracking(void) { // from non-refernced status to Ready (w home search - go to home(0,0))
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	MyXYGroupControllerStatusRead();
	if (ControllerStatus >= 10 && ControllerStatus <= 19) { // 10~19: Ready Status
		bAnalogContronEnable = false;
		while(ControllerStatus != 48) { // Analog tracking state due to a TrackingEnable command
			if (AnalogTrackingMode == AnalogVelocityTracking)
				MyXyAnalogVelocityTrakcingEnable(); // < Endable velocity tracking control
			else if (AnalogTrackingMode == AnnalogPositionTracking)
				MyXyAnalogPositionTrakcingEnable(); // < Endable velocity tracking control

			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
			//SleepThread(10);
		}
		bAnalogContronEnable  = true;
		return true;
	}
	else
		return false;
}
bool XPS_Q8::DisableAnalogTracking(void) { // from non-refernced status to Ready (w home search - go to home(0,0))
	if (SocketID == -1)
		return false; // connect first
	int MaxTrial = 1000;
	int TrialNo = 0;
	MyXYGroupControllerStatusRead();
	if (ControllerStatus == 48) {// TrackingEnable status
		while(!(ControllerStatus >= 10 && ControllerStatus <= 19)) { // until become ready status
			MyXyAnalogTrakcingDisable(); // Disable tracking mode
			MyXYGroupControllerStatusRead();
			if (TrialNo++ > MaxTrial) return false;
		}
		return true;
	}
	else
		return false;
}

// ------------------------------------------------------- //
// basic functions
bool XPS_Q8::TCPConnection(void)
{
	/////////////////////////////////////////////////
	// TCP / IP connection
	/////////////////////////////////////////////////
	SocketID = TCP_ConnectToServer(pIPAddress, nPort, 60); // Open a socket
	int error = Login(SocketID, "Administrator", "Administrator");
	error = CloseAllOtherSockets(SocketID);
	//TCP_SetTimeout(SocketID, 0.0015);
	if (-1 == SocketID)	{ printf("Connection failed\n"); return false; }
	return true;
}
bool XPS_Q8::TCPDisconnection(void)
{
	/////////////////////////////////////////////////
	// TCP / IP disconnection
	/////////////////////////////////////////////////
	if (SocketID == -1) { printf ("Already disconnected\n"); return false; }
	else { TCP_CloseSocket(SocketID); SocketID = -1; return true; } // Close Socket
}
bool XPS_Q8::MyXYGroupKill(void) // Kill group
{
	//error = GroupKill (SocketID, pGroup);
	error = KillAll(SocketID);
	if (0 != error) {DisplayErrorAndClose(error,  "GroupKill");return false;}
	return true;
}
bool XPS_Q8::MyXYGroupInitialize(void) // Initialize group
{
	do {
		error = GroupInitialize (SocketID, pGroup);
	} while( error == -1);
	if (0 != error) {DisplayErrorAndClose(error,  "GroupInitialize"); return false;}
	return true;
}
bool XPS_Q8::MyXYGroupHomeSearch(void) // Search home group
{
	double zeroDisplacement[2] = {0, 0};
	error = GroupHomeSearch (SocketID, pGroup);
	if (0 != error) {DisplayErrorAndClose(error,  "GroupHomeSearch"); return false; }
	return true;
}
bool XPS_Q8::VersionGet(void)
{
	char FirmwareVersion[SIZE_SMALL] = "\0";
	error = FirmwareVersionGet (SocketID, FirmwareVersion); // Get controller version
	if (0 != error)	{ DisplayErrorAndClose(error, "Fail to get FirmwareVersion"); return false; }
	else { sprintf(ControllerStatus_buf, "XPS Firmware Version: %s\n", FirmwareVersion); return true; }
}
int XPS_Q8::MyXYGroupControllerStatusRead(void) {
	int ControllerStatus_old = ControllerStatus;
	int error = GroupStatusGet(SocketID, pGroup, &ControllerStatus);
	if (error != 0)
		ControllerStatus = ControllerStatus_old;
	return ControllerStatus;
}


bool XPS_Q8::MyXYGroupCurrentLocation(void)
{
	error = GroupPositionCurrentGet(SocketID, pGroup, nPositioners, CurrentStatus.CurrentPosition);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupPositionCurrentGet"); return false; }
	return true;
}



double XPS_Q8::VoltsPerMMPS(void){// read only
	return AVT_Voltpermmps;
}

double XPS_Q8::DeadBandThresholdVolts(void) { // read only
	return AVT_Deadbandthreshold;
}

// ------------------------------------------------------- //

bool XPS_Q8::MyXYGroupDACSet(void) {
	bool flag = true;
	int error = 0;

	char* pEvent; // Event
	char * pZero; pZero = "0";// Null parameter
	char* pAction; // Action triggered on the event

	//char gain_X[256]; sprintf(gain_X, "%f", 1.89/14.53); //"0.1300757054370268"; // 1V = 10mm (0-100mm) std:+-0.01V
	//char gain_Y[256]; sprintf(gain_Y, "%f", 1.89/14.53); //"0.1300757054370268"; // 1V = 10mm (0-100mm) std:+-0.01V
	//char gain_vel[256]; sprintf(gain_vel, "%f", 1.89/14.53); //gain_vel = "0.1300757054370268";	 // 1V = 10mm/s(0-100mm/s) std:+-0.01V
	//char * offset_X; offset_X = "-0.068"; // std:+-0.01V
	//char * offset_Y; offset_Y = "-0.0665";// std:+-0.01V
	//char * offset_vel_X; offset_vel_X = "-0.0715";	// std:+-0.02V -0.0715
	//char * offset_vel_Y; offset_vel_Y = "-0.0765";	// std:+-0.02V

	//char gain_X[256]; sprintf(gain_X, "%f", 1.0); // 20.0/65.536 20V = 65.536 mm | 65536 count (1um / 1count) | +- 6 count error = +- 6um error
	//char gain_Y[256]; sprintf(gain_Y, "%f", 1.0); // 20V = 65.536 mm | 65536 count (1um / 1count) | +- 6 count error = +- 6um error
	//char gain_vel[256]; sprintf(gain_vel, "%f", 1.0); // 10/131.072 10V = 131.072 mm/s | 65536 count (4 um/s / 1count) | +- 6 count error = +- 24um error
	//char * offset_X; offset_X = "0.0681"; // std:+-0.01V
	//char * offset_Y; offset_Y = "0.0665";// std:+-0.01V
	//char * offset_vel_X; offset_vel_X = "0.0770";	// std:+-0.02V -0.0715
	//char * offset_vel_Y; offset_vel_Y = "0.0763";	// std:+-0.02V
	//char * offset_X; offset_X = "0.0"; // std:+-0.01V
	//char * offset_Y; offset_Y = "0.0";// std:+-0.01V
	//char * offset_vel_X; offset_vel_X = "0.0";	// std:+-0.02V -0.0715
	//char * offset_vel_Y; offset_vel_Y = "0.0";	// std:+-0.02V


	int nbEvent = 1; // number of events
	if (SocketID >= 0) {
		for(int i = 0; i < 10; i++) { // clear all event
			do {
				error = EventExtendedRemove(SocketID, i);
			} while(error == -1);
		}

		//error = EventExtendedConfigurationTriggerSet(SocketID, 1, "Always", "0", "0", "0", "0");
		do {
			if (!isHighLow)
				error = EventExtendedConfigurationTriggerSet(SocketID, 2, "Always GPIO3.DI.DILowHigh","0 0", "0 0","0 0","0 0"); // falling edge --> original code
			else
				error = EventExtendedConfigurationTriggerSet(SocketID, 2, "Always GPIO3.DI.DIHighLow", "0 0", "0 0", "0 0", "0 0"); // falling edge --> try?
		} while(error == -1);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationTriggerSet"); return false; }
		//if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationTriggerSet"); return false; }
		/*
		// Configure ADCHighLimit event: Action - gathering current positions and velocities of two positioners
		pAction = "GPIO2.DAC1.DACSet.CurrentPosition";
		error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction, pPositioner_X, gain_X, offset_X, pZero);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		error = EventExtendedStart (SocketID, &eventID); if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; }

		pAction = "GPIO2.DAC2.DACSet.CurrentPosition";
		error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction, pPositioner_Y, gain_Y, offset_Y, pZero);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		error = EventExtendedStart (SocketID, &eventID); if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; }

		pAction = "GPIO2.DAC3.DACSet.CurrentVelocity";
		error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction, pPositioner_X, gain_vel, offset_vel_X, pZero);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		error = EventExtendedStart (SocketID, &eventID); if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; }

		pAction = "GPIO2.DAC4.DACSet.CurrentVelocity";
		error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction, pPositioner_Y, gain_vel, offset_vel_Y, pZero);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		error = EventExtendedStart (SocketID, &eventID); if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; }
		*/
		// Event for the DO Toggle (Stage Reading Copy Check)

		pAction = "GatheringRun";
		//int nTypes = 17; char* pDataTypeList = "MyXYGroup.ILS200LM-X.SetpointPosition MyXYGroup.ILS200LM-Y.SetpointPosition MyXYGroup.ILS200LM-X.CurrentPosition MyXYGroup.ILS200LM-Y.CurrentPosition MyXYGroup.ILS200LM-X.SetpointVelocity MyXYGroup.ILS200LM-Y.SetpointVelocity MyXYGroup.ILS200LM-X.CurrentVelocity MyXYGroup.ILS200LM-Y.CurrentVelocity MyXYGroup.ILS200LM-X.SetpointAcceleration MyXYGroup.ILS200LM-Y.SetpointAcceleration MyXYGroup.ILS200LM-X.CurrentAcceleration MyXYGroup.ILS200LM-Y.CurrentAcceleration GPIO2.ADC2 GPIO2.ADC1 GPIO2.ADC3 GPIO2.ADC4 GPIO2.DI";
		int nTypes = 7; char* pDataTypeList = "MyXYGroup.ILS200LM-X.CurrentPosition MyXYGroup.ILS200LM-Y.CurrentPosition MyXYGroup.ILS200LM-X.CurrentAcceleration MyXYGroup.ILS200LM-Y.CurrentAcceleration MyXYGroup.ILS200LM-X.SetpointAcceleration MyXYGroup.ILS200LM-Y.SetpointAcceleration GPIO2.DI";
		//int nTypes = 7; char* pDataTypeList = "MyXYGroup.ILS200LM-X.CurrentPosition MyXYGroup.ILS200LM-Y.CurrentPosition GPIO2.ADC2 GPIO2.ADC1 GPIO2.ADC3 GPIO2.ADC4 GPIO2.DI";
		char* pNbPoints = "1";
		char* pDiv = "1";
		do {
			error = GatheringConfigurationSet (SocketID, nTypes, pDataTypeList);
		} while(error == -1);
		if (0 != error) { DisplayErrorAndClose(error,  "GatheringConfigurationSet"); return false; }
		do {
			error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction, pNbPoints,pDiv, pZero, pZero);
		} while(error == -1);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		do {
			error = EventExtendedStart (SocketID, &eventID);
		} while(error == -1);
		if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; } // deactivated because of the stage calibration - active later.
		/*

		pAction = "ExternalGatheringRun";
		int nTypes_ext = 4;
		char* pDataTypeList_ext = "GPIO2.DAC1 GPIO2.DAC2 GPIO2.DAC3 GPIO2.DAC4";
		char* pNbPoints_ext = "1";
		char* pDiv_ext = "1";
		error = GatheringExternalConfigurationSet (SocketID, nTypes_ext, pDataTypeList_ext);
		if (0 != error) { DisplayErrorAndClose(error,  "GatheringExternalConfigurationSet "); return false; }
		error = EventExtendedConfigurationActionSet (SocketID, nbEvent, pAction,pNbPoints_ext,pDiv_ext, pZero, pZero);
		if (0 != error) { DisplayErrorAndClose(error,  "EventExtendedConfigurationActionSet"); return false; }
		error = EventExtendedStart (SocketID, &eventID); if (0 != error) { DisplayErrorAndClose(error, "EventExtendedStart"); return false; }*/
	}
	return flag;
}


bool XPS_Q8::MyXyAnalogPositionTrakcingEnable(void) {
	/* Initialization */
	char* pType; pType = "Position";
	//double dOffset = 0; // Offset of the positioner moves during tracking
	//double dScale = 1; // Scale of the positioner moves during tracking

	// Disable tracking mode
	error = GroupAnalogTrackingModeDisable (SocketID, pGroup);
	if (SocketID >= 0) {
		// Set tracking parameters
		error = PositionerAnalogTrackingPositionParametersSet(SocketID, pPositioner_X, "GPIO2.ADC1", APT_offset_x, APT_scale_x, APT_Velocity_max, APT_acceleration);
		error = PositionerAnalogTrackingPositionParametersSet(SocketID, pPositioner_Y, "GPIO2.ADC2", APT_offset_y, APT_scale_y, APT_Velocity_max, APT_acceleration);
		if (0 != error) { DisplayErrorAndClose(error, "PositionerAnalogTrackingPositionParametersSet"); return false; }
		// Enable tracking mode
		error = GroupAnalogTrackingModeEnable (SocketID, pGroup, pType);
		if (0 != error) { DisplayErrorAndClose(error,  "GroupAnalogTrackingModeEnable"); return false; }
	}
	return true;
}


bool XPS_Q8::MyXyAnalogVelocityTrakcingEnable(void) {
	/* Initialization */
	char* pType; pType = "Velocity";
	// internal value update before enable
	MyXyAnalogTrakcingInternalValueUpdate();

	// Enable tracking mode
	if (SocketID >= 0) {

		error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_X, "GPIO2.ADC1", AVT_offset_x, AVT_scale_x, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);
		error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_Y, "GPIO2.ADC2", AVT_offset_y, AVT_scale_y, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);
		//error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_X, "GPIO2.ADC1", AVT_offset_x, AVT_scale_x, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);
		//error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_Y, "GPIO2.ADC2", AVT_offset_y, AVT_scale_y, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);
		//error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_X, "GPIO2.ADC1", AVT_offset_x, AVT_scale_x, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);
		//error = PositionerAnalogTrackingVelocityParametersSet(SocketID, pPositioner_Y, "GPIO2.ADC2", AVT_offset_y, AVT_scale_y, AVT_Deadbandthreshold, AVT_order, AVT_Velocity_max, AVT_acceleration);

		if (0 != error) { return false; }
		// Enable tracking mode
		error = GroupAnalogTrackingModeEnable (SocketID, pGroup, pType);
		if (0 != error) {  return false; }
	}
	return true;
}

bool XPS_Q8::MyXyAnalogTrakcingDisable(void) {
	// Disable tracking mode
	error = GroupAnalogTrackingModeDisable (SocketID, pGroup);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupAnalogTrackingModeDisable"); return false; }
	return true;
}

void XPS_Q8::MyXyAnalogTrakcingInternalValueUpdate(void) {
	AVT_Velocity_max = AVT_scale_x*AVT_InputVoltage_max*0.95; // mm/s
}


void XPS_Q8::updateChamberPos(void) {
}

bool XPS_Q8::isXpsDataInRange(void) {
	if (CurrentStatus.GatheringCompleted) {
		double ds = sqrt(CurrentStatus.CurrentPosition[0] * CurrentStatus.CurrentPosition[0] + CurrentStatus.CurrentPosition[1] * CurrentStatus.CurrentPosition[1]);
		if (ds > PosLimit)
			bRangeError = true;
		else
			bRangeError = false;
	}
	return bRangeError;
}

bool XPS_Q8::isXpsDatavalid(XPSGatheringInfo t) {
	double limit = 200.0;
	if (fabs(t.CurrentPosition[0]) > limit)
		return false;
	if (fabs(t.CurrentPosition[1]) > limit)
		return false;
	return true;
}

bool XPS_Q8::MyXyGroupGathering(void) {
	XPSGatheringInfo t;
	if (SocketID == -1)
		return false;

	if (ControllerStatus < 10)
		return false;

	char output[SIZE_NOMINAL];
	char data1[SIZE_NOMINAL];
	char data2[SIZE_NOMINAL];
	char data3[SIZE_NOMINAL];
	char data4[SIZE_NOMINAL];
	char data5[SIZE_NOMINAL];
	char data6[SIZE_NOMINAL];
	char data7[SIZE_NOMINAL];
	char data8[SIZE_NOMINAL];

	error = GatheringDataGet(SocketID, 0, output);
	if (0 != error) { return false; }


	int n = 0; int m = 0;  char * pch;

	pch = strchr(output, ';'); if (!pch) return false; m = pch - output; strncpy(data1, output, m); n = m;
	t.CurrentPosition[0] = atof(data1) - x_center;
	pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data2, output + n + 1, m - n - 1); n = m;
	t.CurrentPosition[1] = atof(data2) - y_center;

	pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data3, output + n + 1, m - n - 1); n = m;
	t.CurrentAcceleration[0] = atof(data3);
	pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data4, output + n + 1, m - n - 1); n = m;
	t.CurrentAcceleration[1] = atof(data4);

	//pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data5, output + n + 1, m - n - 1); n = m;
	//t.CurrentVelocity[0] = atof(data5);
	//if (abs(t.CurrentVelocity[0]) > 500.0) t.CurrentVelocity[0] = CurrentStatus.CurrentVelocity[0];
	//pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data6, output + n + 1, m - n - 1); n = m;
	//t.CurrentVelocity[1] = atof(data6);
	//if (abs(t.CurrentVelocity[1]) > 500.0) t.CurrentVelocity[1] = CurrentStatus.CurrentVelocity[1];


	pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data7, output + n + 1, m - n - 1); n = m;
	t.SetpointAcceleration[0] = atof(data7);
	pch = strchr(pch + 1, ';');	if (!pch) return false; m = pch - output; strncpy(data8, output + n + 1, m - n - 1); n = m;
	t.SetpointAcceleration[1] = atof(data8);

	//pch=strchr(pch+1,';');	if (!pch) return false; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.DOToggle = atof(pch + 1);
	if (CurrentStatus.DOToggle != t.DOToggle) { // if the DOToggle is same, it assumed that it is the same signal
		t.GatheringCompleted = true;
	}
	else {
		t.GatheringCompleted = false;
	}
	if (isXpsDatavalid(t))
		CurrentStatus = t;
	else {
		CurrentStatus.GatheringCompleted = false;

		char NAME[1024];
		sprintf(NAME, "d:\\TrackingMicroscopeData\\_report\\stageError_invalidregion.txt");
		FILE  * ofp = fopen(NAME, "w");
		fprintf(ofp, "x: %f, y: %f\n", t.CurrentPosition[0], t.CurrentPosition[1]);
		fprintf(ofp, "%s\n", output);
		fclose(ofp);
	}
	return true;
}

XPSGatheringInfo XPS_Q8::MyXyGroupGathering_old(void) {
	XPSGatheringInfo t;
	if (SocketID == -1)
		return t;

	if (ControllerStatus < 10)
		return t;

	char output[SIZE_NOMINAL];
	char data[SIZE_NOMINAL];

	error = GatheringDataGet(SocketID, 0, output);
	if (0 != error) { return t; }


	int n = 0; int m = 0;  char * pch;

	// Setpoint Position
	pch = strchr(output,';'); if (!pch) return t; m = pch - output; strncpy(data, output, n);
	t.SetpointPosition[0] = atof(data) - x_center;;
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.SetpointPosition[1] = atof(data) - y_center;;

	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentPosition[0] = atof(data) - x_center;
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentPosition[1] = atof(data) - y_center;
	if (fabs(CurrentStatus.CurrentPosition[1] - t.CurrentPosition[1]) > 100 && CurrentStatus.GatheringCompleted)
		t.CurrentPosition[1] = CurrentStatus.CurrentPosition[1];

	t.CurrentPosition_Chamber[0] = ChamberPosX.updateNewInput(t.CurrentPosition[0]);
	t.CurrentPosition_Chamber[1] = ChamberPosY.updateNewInput(t.CurrentPosition[1]);

	// Setpoint Velocity
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.SetpointVelocity[0] = atof(data);
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.SetpointVelocity[1] = atof(data);
	// current Velocity
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentVelocity[0] = atof(data);
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentVelocity[1] = atof(data);
	// Setpoint Acceleration
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.SetpointAcceleration[0] = atof(data);
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.SetpointAcceleration[1] = atof(data);
	// current Acceleration
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentAcceleration[0] = atof(data);
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.CurrentAcceleration[1] = atof(data);

	// Voltage Input_ADC1
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.InputVoltage[0] = atof(data);
	// Voltage Input_ADC2
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.InputVoltage[1] = atof(data);
	// Voltage Input_DAC
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.Pos_Z_set = atof(data);
	// Voltage Input_DAC2
	pch=strchr(pch+1,';');	if (!pch) return t; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.Pos_Z = atof(data);

	//pch=strchr(pch+1,';');	if (!pch) return false; m = pch - output; strncpy(data, output+n+1, m-n-1); n = m;
	t.DOToggle = atof(pch+1);


	if (CurrentStatus.DOToggle != t.DOToggle) { // if the DOToggle is same, it assumed that it is the same signal
		t.GatheringCompleted = true;
	}
	else {
		t.GatheringCompleted = false;
	}
	CurrentStatus = t;
	return t;
}


// ------------------------------------------------------- //

bool XPS_Q8::MyXYGroupJogModeEnable(void)
{
	/* Enable jog mode */
	error = GroupJogModeEnable (SocketID, pGroup);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupJogModeEnable"); return false; }
	return true;
}

bool XPS_Q8::MyXYGroupJogModeDisable(void)
{
	/* Disable Jog mode (constant velocity must be null on all positioners from group) */
	error = GroupJogModeDisable (SocketID, pGroup);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupJogModeDisable"); return false; }
	return true;
}

bool XPS_Q8::MyXYGroupJogParametersSet(double* pVelocity, double* pAcceleration)
{
	/* Set Jog parameters to stop a positioner => constant velocity is null */
	error = GroupJogParametersSet (SocketID, pGroup, nPositioners, pVelocity, pAcceleration);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupJogParametersSet"); return false; }
	return true;
}

void XPS_Q8::XYJogging(void)
{
	MyXYGroupKill(); if (0 != error) return; // Kill Group
	MyXYGroupInitialize(); if (0 != error) return; // Initialize Group
	MyXYGroupHomeSearch(); if (0 != error) return; // Search home group
	MyXYGroupJogModeEnable(); if (0 != error) return; // Enable jog mode

	/* Initialization */
	Sleep(3000);  // Wait 3 seconds

	// Group motion velocity parameters
	double pVelocity[2];
	double pAcceleration[2];

	MyXYGroupCurrentLocation(); // set current location

	pVelocity[0] = -30; // Velocity of the X positioner demanded during second jog op
	pVelocity[1] = -30; // Velocity of the Y positioner demanded during second jog op
	pAcceleration[0] = 80; // Acceleration of the X positioner demanded during jog op
	pAcceleration[1] = 80; // Acceleration of the Y positioner demanded during jog op

	MyXYGroupJogParametersSet(pVelocity, pAcceleration); if (0 != error) return;
	while((CurrentStatus.CurrentPosition[0] > -10)&&(CurrentStatus.CurrentPosition[1] > -10)) {
		MyXYGroupCurrentLocation();
		printf ("Position = [%.3lf: %.3lf]\n", CurrentStatus.CurrentPosition[0], CurrentStatus.CurrentPosition[1]);
	}


	/* Set Jog parameters to move the positioner in the	reverse sense => constant velocity is not null*/
	pVelocity[0] = 10; // Velocity of the X positioner demanded during second jog op
	pVelocity[1] = 10; // Velocity of the Y positioner demanded during second jog op
	MyXYGroupJogParametersSet(pVelocity, pAcceleration); if (0 != error) return;
	while((CurrentStatus.CurrentPosition[0] < 10)&&(CurrentStatus.CurrentPosition[1] < 10)) {
		MyXYGroupCurrentLocation();
		printf ("Position = [%.3lf: %.3lf]\n", CurrentStatus.CurrentPosition[0], CurrentStatus.CurrentPosition[1]);
	}

	/* Set Jog parameters to stop a positioner => constant velocity is null */
	pVelocity[0] = 0; // Velocity of the X positioner demanded during second jog op
	pVelocity[1] = 0; // Velocity of the Y positioner demanded during second jog op
	MyXYGroupJogParametersSet(pVelocity, pAcceleration); if (0 != error) return;

	MyXYGroupJogModeDisable(); if (0 != error) return; // Disable Jog mode
	// (constant velocity must be null on all positioners from group)

}

bool XPS_Q8::MyXYGroupMoveEnable(void)
{
	/* Enable jog mode */
	error = GroupMotionEnable (SocketID, pGroup);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupMotionEnable"); return false; }
	return true;
}

bool XPS_Q8::MyXYGroupMoveDisable(void)
{
	/* Disable Jog mode (constant velocity must be null on all positioners from group) */
	error = GroupMotionDisable (SocketID, pGroup);
	if (0 != error) { DisplayErrorAndClose(error,  "GroupMotionDisable"); return false; }
	return true;
}

double XPS_Q8::Velocity2Volt_x(double vel)
{
	if (fabs(vel) < 0.1)
		return 0;
	else
		return Velocity2Volt(vel, AVT_scale_x, AVT_Deadbandthreshold, AVT_offset_x, AVT_scale_x2);
}
double XPS_Q8::Velocity2Volt_y(double vel)
{
	if (fabs(vel) < 0.1)
		return 0;
	else
		return Velocity2Volt(vel, AVT_scale_y, AVT_Deadbandthreshold, AVT_offset_y, AVT_scale_y2);
}

double XPS_Q8::Velocity2Volt(double vel, double scale, double db, double offset, double scale2) {
	double inputValue = (vel)/scale;
	if (inputValue > 0)
		inputValue  = inputValue + db;
	else if (inputValue < 0 )
		inputValue  = inputValue - db;

	//double Volts = (inputValue - offset)/scale2;
	double Volts = (inputValue)/scale2;
	return Volts;
}

double XPS_Q8::Volt2Velocity(double volts, double scale, double db, double offset, double scale2) {
	double inputValue = scale2*volts;
	//double inputValue = scale2*volts - offset;
	if (inputValue > 0) {
		inputValue  = inputValue - db;
		if (inputValue < 0)
			inputValue = 0;
	}
	else if (inputValue < 0 ) {
		inputValue  = inputValue + db;
		if (inputValue > 0)
			inputValue = 0;
	}
	double vel = inputValue*scale;
	return vel;
}

double XPS_Q8::Volt2Velocity_x(double volts)
{
	return Volt2Velocity(volts, APT_scale_x, AVT_Deadbandthreshold, AVT_offset_x, AVT_scale_x2);
}

double XPS_Q8::Volt2Velocity_y(double volts)
{
	return Volt2Velocity(volts, APT_scale_y, AVT_Deadbandthreshold, AVT_offset_y, AVT_scale_y2);
}

bool XPS_Q8::MyXYGroupMoveAbsolute(double* TargetPosition) {
	/* Set Move Abolute */
	MyXYGroupControllerStatusRead(); // check status of stage
	if (ControllerStatus >= 10 && ControllerStatus < 20) { // if the stage is READY (10-20), move the stage to the target pose
		error = GroupMoveAbsolute(SocketID, pGroup, nPositioners, TargetPosition);
		if (0 != error) { DisplayErrorAndClose(error,  "GroupMoveAbsolute"); return false; }
		return true;
	} // endif (if stage is ready )
	return false;
}


int XPS_Q8::MotionObservation_GatheringExport(char * pathFile_data) // export the current saved multiple gathering datat (return number of samples)
{
	int cur = 0;
	int maxNum = 0;
	GatheringCurrentNumberGet(SocketID, &cur, &maxNum);
	if (cur > 0) {
		char Muloutput[SIZE_NOMINAL];
		int i = 0;
		int chunkSize = 1;
		// record file
		FILE * fp; // NIR frame number , timestamp, Target(maggot) position and error
		fp = fopen(pathFile_data, "w");

		for (i = 0; i+chunkSize < cur; i = i + chunkSize) {
			error = GatheringDataMultipleLinesGet(SocketID, i, chunkSize, Muloutput);
			fprintf(fp, "%s", Muloutput);
		}
		error = GatheringDataMultipleLinesGet(SocketID, i, cur - i, Muloutput);
		fprintf(fp, "%s", Muloutput);
		fclose(fp);
	}
	return cur;
}
