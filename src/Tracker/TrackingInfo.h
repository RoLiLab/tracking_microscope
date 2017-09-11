#ifndef _TRACKINGINFO_H
#define _TRACKINGINFO_H

#include<windows.h>
#include "Base/base.h"

#include "HDF5/hdf5imagewriter.h"
#include "HDF5/hdf5imagereader.h"
#include "HDF5/hdf5datawriter.h"
#include "HDF5/hdf5datareader.h"
#include "XPSQ8/XPS_Q8.h"							// stage control
#include "mkl_lapacke.h"
#include "mkl_cblas.h"
#include "MPC.h"
#include "AutoFocusVolumeScan.h"



class JitterFilter {
public:
	// basic functions
	JitterFilter(void);
	~JitterFilter(void);
	Point2d update(Point2d _p, bool _isfishdetected);
	// members
	double threshold; // [mm]
	Point2d CurrentValue; // [mm/s]
};

enum{eyeLeft, eyeRight, yolk};
enum{Left_Right, Right_Yolk, Yolk_Left};
enum{Right_Left, Yolk_Right, Left_Yolk};

class spotCompareResult;
class SpotInfo
{
public:
	SpotInfo(void);
	SpotInfo(Point2d _x);
	~SpotInfo(void);
	// members
	Point2d center;
	double width;
	double height;
	double angle; // deg
	double area;
	double eccentricity;
	void updateValue(void);
	spotCompareResult compare(SpotInfo nextSpot);
};

class spotCompareResult{
public:
	spotCompareResult(void);
	~spotCompareResult(void);
	// member
	double fitness; // fitness value (with weight) - the samller the similar
	double distTravel; // abs distance to the next spot (p_given - p)
	double areaDiff; // abs area normalized difference with the given spot ((area_given-area)/area)
	double eccDiff; // abs eccentricity difference with the given spot [0-1]
	double angleDiff; // abs angle difference [0-180]/180
	SpotInfo SpotRef;
	SpotInfo SpotGiven;
	struct paraSpotInfoCompare {
		paraSpotInfoCompare() : w_distTravel(1), w_areaDiff(0), w_eccDiff(0), w_angleDiff(0) {}
		double w_distTravel; // abs distance to the next spot (p_given - p)
		double w_areaDiff; // abs area normalized difference with the given spot ((area_given-area)/area)
		double w_eccDiff; // abs eccentricity difference with the given spot [0-1]
		double w_angleDiff; // abs angle difference [0-180]/180
	};
	paraSpotInfoCompare para;
	// function
	void computeFitness(double * weight);
	void computeFitness(void);
};

class zebrafishInfo
{
public:
	zebrafishInfo(void);
	zebrafishInfo(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk);
	zebrafishInfo(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk, double distance);
	zebrafishInfo(vector<Point2d> fish_pt, double fish_size);
	~zebrafishInfo(void);
	// members
	SpotInfo m_eyeLeft;
	SpotInfo m_eyeRight;
	SpotInfo m_yolk;
	Point2d centerEyes; // reference point (center point between eyes)
	Point2d centerFish; // reference point (center point between eyes)
	Point2i target_px; // reference point (center point between eyes)
	double orienation; // angle from yolk to the center of eyes
	double AreaSize; // area size enclodsed by the three center points.
	double CosAngle_LYR; /// cosine of angle Left-Yolk-Right
	double fitness_heading; // cos(phi) value, phi is the angle between brain-yolk and left-rightEyes.
	void getVaildation(int FishSize);
	bool DetectionFailed;

	void ReplacePosition(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk);
};
vector<Point2d> estimateFuturePos(vector<double> orientationHist, vector<Point2d> fishPosHist);
vector<Point2d> estimateFuturePos_linear(vector<double> orientationHist, vector<Point2d> fishPosHist);
vector<Point2d> estimateFuturePos_Front(vector<double> orientationHist, vector<Point2d> fishPosHist, double gain);

class zebrafishInfoReference
{
public:
	zebrafishInfoReference(void);
	~zebrafishInfoReference(void);
	// members
	unsigned int _SizeHist;
	double mean_AngleLYR;
	double mean_dEyes;
	double mean_dEyesYolk;
	double RangePercent_AngleLYR;
	double RangePercent_dEyes;
	double RangePercent_dEyesYolk;
	bool getVaildation(zebrafishInfo * ref);
private:
	void getMeanValue(void);
	void updateRefFish(double dLR, double dRY, double dLY, double AngLYR);
	double computeAngLYR(double dRY, double dLY, double dRL) ;
	bool CheckInRange(double v, double vRef, double ratio) ;
	vector<double> AngleLYR;
	vector<double> dEyes;
	vector<double> dEyesYolk;

};


class MovingAverage_double
{
public:
	MovingAverage_double(void);
	MovingAverage_double(int _steps);
	MovingAverage_double(int _steps, int _stepHist);
	~MovingAverage_double(void);
	double update(double NewValue);
	vector<double> MVA_Hist;
	vector<double> RawData;
	double MVA;
	double _MAX;
	double _stddev;
	int stepMax;
	int stepMax_Hist;
	double MVA_Hist_FDD(void);
	void clear(void);
private:
	void updateRawData(double newValue);
		double updateMVA_Hist(int steps);
};

class MovingAverage_cvPoint2d
{
public:
	MovingAverage_cvPoint2d(void);
	MovingAverage_cvPoint2d(int _steps);
	MovingAverage_cvPoint2d(int _steps, int _stepHist);
	~MovingAverage_cvPoint2d(void);
	Point2d update(Point2d NewValue);
	vector<Point2d> MVA_Hist;
	vector<Point2d> RawData;
	Point2d MVA;
	int stepMax;
	int stepMax_Hist;
	Point2d MVA_Hist_FDD(void);
	void clear(void);
	void updateRawData(Point2d newValue);
private:
	Point2d updateMVA_Hist(int steps);
};


class zebrafishPositionEstimate
{
public:
	zebrafishPositionEstimate(void);
	~zebrafishPositionEstimate(void);

	int steps;

	MovingAverage_cvPoint2d MVA_TargetPosition; // Fish position (RAW DATA)
	MovingAverage_double MVA_Heading; // Fish Heading (RAW DATA)
	MovingAverage_double MVA_Heading_filtered;

	MovingAverage_double MVA_AxisHeading; // FishHeading (Filtered)
	MovingAverage_cvPoint2d MVA_TargetPosition_Heading; // Fish Moving Direction


	MovingAverage_double MVA_TargetSpeed;
	MovingAverage_double MVA_TargetSpeed_5Step;
	int WindowSize_AngMTh;


	vector<Point2d> estimateFuturePosition(zebrafishInfo CurInfo);
	vector<Point2d> estimateFuturePosition_option3(Point2d TargetPos, double _HeadingAngle);
	vector<Point2d> estimateFuturePosition_MeanVelProjPt(Point2d TargetPos, double _HeadingAngle);
	void estimateFuturePosition_MeanVelProjPt_double(Point2d TargetPos, double _HeadingAngle);
	void estimateFuturePosition_Ver1(Point2d TargetPos, double _HeadingAngle);
	double * predX;
	double * predY;
	vector<Point2d> estimateFuturePosition(Point2d TargetPos, double _HeadingAngle);

	void clear(void);
	void init(double cycle_sec, int _steps);
	double VelocityUpdateCycle_ms;
	double TargetPositionUpdateCycle_ms;
	double AxisHeadingUpdateCycle_ms;
	double HeadingUpdateCycle_ms;
	double angularMomentum_DegPerSec_UPPER;
	double angularMomentum_DegPerSec_LOWER;
	double Cycle_dt;
	double Cycle_dt_ms;
	double deActivationTime_ms;
	double AngularMomentumWindow_ms;
//private:
	Point2d TrgtPt;
	Point2d BasePt;
	double ForwardVelocity;
	double AxisHeadingAngle;

	Point2d ProjPt;
	Point2d AxisHeadingVector;
	double cosH;
	double sinH;

	double angularMomentum;
	double angMomentum_maxThrehold;
	double angMomentum_minThrehold;
	double angMomentum_Threhold;
	int angMomentum_FrameCountAfterMoving;
	bool FrameCountStopEnabled;
	bool isFishMoving;
	vector<double> weight_angMomentum;
	double weight;
	double stopFishThreshold; // when the last n-step avg velocity is lower than this value, the weight is set to 0.

	void getWeight(void);
	void getWeight_FishMotion(void);
	void computeAxisHeading_fitting(void); // update AxisHeading, AxisHeadingAngle, cosH, sinH
	void computeAxisHeading_fitting_SeparateHeadingPos(void); // update AxisHeading, AxisHeadingAngle, cosH, sinH
	void computeAxisHeading_dth(void);
	void computeProjPt(void);
	void computeAngularMomentum(void);
	void computeAxisHeading_MeanTheta(void);
	void computeAxisHeading_MINMAX(void);
};





double NormDiff_1D(double a, double ref);
double NormDiff_cvPoint(Point2d * a, Point2d * ref);
double NormDiff_cvPoint(Point2i * a, Point2i * ref);
double Diff_1D(double a, double ref);
double Diff_2D(double * a, double * ref);
double Diff_cvPoint(Point2d a, Point2d ref);
double Diff_cvPoint(Point2i a, Point2i ref);
double Diff_Ang(double a, double ref);
double Var_Ang(double a, double b, double c);
double adjustAngRange(double a); // +-180 // degree input
double Area_3Points(Point2i a, Point2i b, Point2i c);
double Area_3Points(Point2d a, Point2d b, Point2d c);
double atan2_twoPoints(Point2i a, Point2i b);
double atan2_twoPoints(Point2d a, Point2d b);
double atan2_twoPoints_Right(Point2i a, Point2i b);
double atan2_twoPoints_Right(Point2d a, Point2d b);
double Var_Dist(double a, double b, double c);
vector<Point2i> GetZebraFishROI_exact(Point2d p, double Orientation_deg, imSize imgSize);
Point2d MulVectorPoint2d(vector<Point2d> A, vector<Point2d> B, int ai, int bi, int n);

enum CntrDataIndex{
	Xfish, Yfish, //X fish position in mm (Global position)
	XerrPx, YerrPx, //X fish position error in px
	Xerr, Yerr, //X fish position error
	HeadingFish, //X fish heading direction - An angles b/t the center of eyes and yolk in history
	Xfishproj, Yfishproj, //X fish projection position (centered line)
	HeadingFishMean, //X fish heading direction - Medium value of the angles b/t the center of eyes and yolk in history
	MovingDirFish, //X moving direction of the projection positions
	weight, //X Prediction weight (How much the moving direction and heading are aligned)
	VfishPrl, VfishPpd, //X fish velocity in parallel and perpendicular axis
	VfishPrlWeighted, dVfishPrlWeighted, //X fish predicted weighed velocity and its gradient (mm/s)/(frame)
	Xfishpred0, Yfishpred0, //X fish prediction position at n+1 step
	xfishpred1, yfishpred1, //X fish prediction position at n+2 step
	xfishpred2, yfishpred2, //X fish prediction position at n+3 step
	xfishpred3, yfishpred3, //X fish prediction position at n+4 step
	xfishpred4, yfishpred4, //X fish prediction position at n+5 step
	xfishpred5, yfishpred5, //X fish prediction position at n+6 step
	xfishpred6, yfishpred6, //X fish prediction position at n+7 step
	xfishpred7, yfishpred7, //X fish prediction position at n+8 step
	xfishpred8, yfishpred8, //X fish prediction position at n+9 step
	xfishpred9, yfishpred9, //X fish prediction position at n+10 step
	XstgMPC, YstgMPC, //X stage position in MPC controller (open-loop position)
	XstgInputVelMPC, YstgInputVelMPC, //X MPC controller input velocity set
	XstgI2T, YstgI2T, //X Stage I2T value (approximated)
	XstgPIDdesired, YstgPIDdesired,  //X PID controller desired position
	XstgPIDtarget, YstgPIDtarget,  //X PID controller target position
	XstgPIDInputVel, YstgPIDInputVel, //X PID controller input velocity set
	XstgInputVolt, YstgInputVolt,//X overall controller input voltage set
	XstgInputVel, YstgInputVel, //X overall controller input velocity set
	Xstg_L2weight, Ystg_L2weight, //X overall controller input velocity set
	Z_set, Z_real,
	x_target_mm, y_target_mm,
	x_ref_px, y_ref_px,
	CntrDataIndex_MAX
};
class CntrData
{
public:
	// basic functions
	CntrData(void);
	~CntrData(void);
	unsigned int isTracking;
	unsigned int isPositionReadingSuccess;
	unsigned int isFishPosDetection;
	unsigned int isReadyMPCInput;
	unsigned int isReadyOnTime;
	unsigned int isReadyTeensyIndex;
	unsigned int isReadyTeensyIndex_input;
	unsigned int isFrameDropped;
	unsigned int isFishMoving;
	unsigned int DroppedFrameCount;
	unsigned int FrameSimulation;
	unsigned int GlobalMapIdx;
	unsigned int MovementIBICount;
	int n;
	double data[60];
};

class TrackingMessage {
public:
	// basic functions
	TrackingMessage(void);
	~TrackingMessage(void);
	unsigned int frameNo; // clock when data is collected
	XPSGatheringInfo stagePos;
	img_uint16 * srcNIR; // image data
	img_uint16 * srcFLR; // image data
	img_uint16 * srcNIRproc; // image data
	zebrafishInfo NIRfishPos; // px
	FLRImageData FLRdata;
	// From MPC and PID
	CntrData m_CntrData;
};

class TrackingMessageBuffer {
public:
	// basic functions
	TrackingMessageBuffer(void);
	~TrackingMessageBuffer(void);
	volatile bool b_recstart;
	volatile bool b_recstop;
	volatile bool b_recording;
	uint64 recFrms_Start;
	uint64 recFrms_End;
	int32 recFrms_left;
	uint64 recFrms;
	uint64 FrmNo;
	time_t t_now;// = time(0); // get time now;
	char DATESTR[128];// = time(0); // get time now;

	//-----------------------
	void allocateBuffer(unsigned int _MaxBufferSize);
	void freeBuffer(void);
	void updateFramNo(unsigned int _frameNo);
	void updateStagePos(unsigned int _frameNo, XPSGatheringInfo pos);
	void updateNIRImage(unsigned int _frameNo, img_uint16 * src);
	void updateFLRImage(unsigned int _frameNo, img_uint16 * src);
	void updateNIRImageProc(unsigned int _frameNo, img_uint16* src);
	void updateNIRFishPos(unsigned int _frameNo, zebrafishInfo pos);
	void updateCntrData(unsigned int _frameNo, CntrData _cntrData);
	void updateCntrData_lateinputreport(unsigned int _frameNo);

	void updateAllNIR(unsigned int _frameNo,
		XPSGatheringInfo pos,
		zebrafishInfo fishPos,
		CntrData _cntrData,
		uint16 * src, int rows, int cols);
	void updateAllNIRcpy(unsigned int _frameNo,
		XPSGatheringInfo pos,
		zebrafishInfo fishPos,
		CntrData _cntrData,
		uint16 * src, int rows, int cols,
		FLRImageData _FLRdata);
	zebrafishInfo * getframeZebrafishInfo(unsigned int _frameNo);
	double * getFramePos(unsigned int _frameNo);
	double * getFramePosChamber(unsigned int _frameNo);
	Point2d getFramePos_cv(unsigned int _frameNo);
	Point2d getFramePoschamber_cv(unsigned int _frameNo);
	TrackingMessage * getDispFrameRec(void);
	TrackingMessage * getFrameData(unsigned int _frameNo);
	uint16 * getNIRImgPtrFrameNo(unsigned int _frameNo);

	unsigned int getPreTriggerFrmCount(int mode);
	enum{MinCount, NIRPOSFrameCount, PosFrameCount, NIRFrameCount, FLRFrameCount};

	void setPostTrigSet(int mode , unsigned int _frame,  unsigned int _PreTrgRecFrms);
	void SetStopRec(void);
	unsigned int getRecEndIdx(void);
	unsigned int getRecStartIdx(void);
	unsigned int getRecBlockIdx(void);

	void InitializeRecording(const char * drive, bool enableNIRhdf5, bool enableFLRhdf5, int binaryDriveCount);
	void ReleaseRecording(void);
	void ReleaseRecording_NIR(void);
	void ReleaseRecording_FLR(void);
	void RecordFrame(unsigned int _Frame);
	bool RecordFrame_FLR(uint16_t * src);
	unsigned int getRecCount(void);
	unsigned int getRecCount_NIR(void);
	unsigned int getRecCount_FLR(void);


	bool NIRRecEnabled;
	bool NIRRecProcEnabled;
	bool FLRRecEnabled;
	bool DataRecEnabled;

	bool VelocityCalibration(double * offset);

	imSize FLRSize;
	imSize NIRSize;
	HDF5ImageWriter * pFLR_writer;
	HANDLE pFLR_writer_binary[harddrivecount_FLRsave];
	HANDLE pNIR_writer_binary[harddrivecount_NIRsave];
	int32 ImageSizeByte_NIR;
	unsigned int preRecImageCount;
	unsigned int MaxBufferSize;
	HDF5ImageWriter * pNIR_writer;
	HDF5DataWriter * pData_writer;
private:

	std::thread m_pThreadNIRRecording;

	unsigned int RecDoneCounter; // for display only
	unsigned int RecDoneCounter_NIR; // for display only
	unsigned int RecDoneCounter_FLR; // for display only

	TrackingMessage * Data;
	unsigned int DispIdx;
	unsigned int RecActiveWindowSize;
	unsigned int MaxBufferSizeBeforeTrigger;
	unsigned int getIndexFromFrameNo(unsigned int _framNo);
	TrackingMessage * getDataPtrFrameNo(unsigned int _framNo);
	unsigned int NIRCount;
	unsigned int FLRCount;
	unsigned int POSCount;

	bool isRecording;
	unsigned int RecBlockIdx;
	unsigned int RecStartIdx;
	unsigned int RecEndIdx;
	void RecordDataVector(int index, vector<Point2d> * _data);


};

#endif
