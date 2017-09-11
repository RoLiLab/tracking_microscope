#include "KBase/common/common.h"
#include "Tracker/TrackingInfo.h"
#include "ipp.h"
#include <numeric>

TrackingMessage::TrackingMessage(void)
{
	frameNo = 0;
	srcNIR = NULL;
	srcFLR = NULL;
	srcNIRproc = NULL;
}

TrackingMessage::~TrackingMessage(void)
{
	if (srcNIR) delete(srcNIR);
	if (srcFLR) delete(srcFLR);
	if (srcFLR) delete(srcNIRproc);
}



TrackingMessageBuffer::TrackingMessageBuffer(void)
{

	FLRRecEnabled = false;
	NIRRecEnabled = false;
	SyncImgRecEnabled = false;
	isRecording = false;
	RecStartIdx = 0;
	RecActiveWindowSize = 10;
	RecBlockIdx = 0;
	NIRCount = 0;
	FLRCount = 0;
	POSCount = 0;
	DispIdx = 0;
	MaxBufferSizeBeforeTrigger = 0;
	MaxBufferSize = 0;
	Data = NULL;

	pNIR_writer = NULL;
	pFLR_writer = NULL;
	pData_writer = NULL;
	RecDoneCounter = 0;
	RecDoneCounter_NIR = 0;
	RecDoneCounter_FLR = 0;
}

TrackingMessageBuffer::~TrackingMessageBuffer(void)
{
	freeBuffer();
}


void TrackingMessageBuffer::allocateBuffer(unsigned int _MaxBufferSize) {
	MaxBufferSizeBeforeTrigger = _MaxBufferSize;
	MaxBufferSize = MaxBufferSizeBeforeTrigger + 2*RecActiveWindowSize;
	DispIdx = 0;
	NIRCount = 0;
	FLRCount = 0;
	POSCount = 0;
	TrackingMessage * _Data = (TrackingMessage *) malloc (MaxBufferSize * sizeof(TrackingMessage));
	for (int i = 0; i < MaxBufferSize ; i++) {
		_Data[i].frameNo = 0;
		_Data[i].srcNIR = NULL;
		_Data[i].srcFLR = NULL;
		_Data[i].srcNIRproc = NULL;
	}
	TrackingMessage * old = Data;
	Data = _Data;
	free(old);
}
void TrackingMessageBuffer::freeBuffer(void) {
	if (Data) {
		free(Data);
		Data = NULL;
	}
}

unsigned int TrackingMessageBuffer::getIndexFromFrameNo(unsigned int _frameNo) {
	return _frameNo%MaxBufferSize;
}

TrackingMessage * TrackingMessageBuffer::getDataPtrFrameNo(unsigned int _frameNo) {
	unsigned int idx = getIndexFromFrameNo(_frameNo);
	return &(Data[idx]);
}
uchar * TrackingMessageBuffer::getNIRImgPtrFrameNo(unsigned int _frameNo) {
	unsigned int idx = getIndexFromFrameNo(_frameNo);
	return (Data[idx].srcNIR->data);
}

void TrackingMessageBuffer::updateFramNo(unsigned int _frameNo) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->frameNo = _frameNo;
		DispIdx = getIndexFromFrameNo(_frameNo - 2);
		RecBlockIdx = max(_frameNo - RecActiveWindowSize, (unsigned int) 0);
	}
}

void TrackingMessageBuffer::updateStagePos(unsigned int _frameNo, XPSGatheringInfo pos) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->stagePos = pos;
		POSCount++;
		if (POSCount > MaxBufferSizeBeforeTrigger  && !isRecording)
			POSCount = MaxBufferSizeBeforeTrigger;
	}
}

void TrackingMessageBuffer::updateNIRFishPos(unsigned int _frameNo, zebrafishInfo pos) {
	// Fish pos update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->NIRfishPos = pos;
	}
}

void TrackingMessageBuffer::updateCntrData(unsigned int _frameNo, CntrData _cntrData ) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->m_CntrData = _cntrData;
	}
}

void TrackingMessageBuffer::updateNIRImage(unsigned int _frameNo, cv::Mat *  src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if(_data->srcNIR) delete(_data->srcNIR);
		_data->srcNIR = src;
		NIRSize = src->size();
		NIRCount++;
		if (NIRCount > MaxBufferSizeBeforeTrigger && !isRecording)
			NIRCount = MaxBufferSizeBeforeTrigger;
	}
}

void TrackingMessageBuffer::updateNIRImageProc(unsigned int _frameNo, cv::Mat *  src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if(_data->srcNIRproc) delete(_data->srcNIRproc);
		_data->srcNIRproc = src;
	}
}


void TrackingMessageBuffer::updateFLRImage(unsigned int _frameNo, cv::Mat * src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if(_data->srcFLR) delete(_data->srcFLR);
		_data->srcFLR = src;
		FLRSize = src->size();
	}
}

double * TrackingMessageBuffer::getFramePos(unsigned int _frameNo) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if (_data->stagePos.GatheringCompleted)
			return _data->stagePos.CurrentPosition;
	}
	return 0;
}

double * TrackingMessageBuffer::getFramePosChamber(unsigned int _frameNo) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if (_data->stagePos.GatheringCompleted)
			return _data->stagePos.CurrentPosition_Chamber;
	}
	return 0;
}

zebrafishInfo * TrackingMessageBuffer::getframeZebrafishInfo(unsigned int _frameNo) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		return &(_data->NIRfishPos);
	}
	return 0;
}

cv::Point2d TrackingMessageBuffer::getFramePos_cv(unsigned int _frameNo) {
	double * pos = getFramePos(_frameNo);
	if (pos)
		return cv::Point2d(pos[0], pos[1]);
	else
		return cv::Point2d(-1,-1);
}

cv::Point2d TrackingMessageBuffer::getFramePoschamber_cv(unsigned int _frameNo) {
	double * pos = getFramePosChamber(_frameNo);
	if (pos)
		return cv::Point2d(pos[0], pos[1]);
	else
		return cv::Point2d(-1,-1);
}

TrackingMessage * TrackingMessageBuffer::getDispFrameRec(void) {
	return &(Data[DispIdx]);
}

TrackingMessage * TrackingMessageBuffer::getFrameData(unsigned int _frameNo) {
	unsigned int idx = getIndexFromFrameNo(_frameNo);
	return &(Data[idx]);
}


unsigned int TrackingMessageBuffer::getPreTriggerFrmCount(int mode) {

	int minCount = 0;
	switch(mode) {
	case MinCount: //
		minCount = NIRCount;
		if (minCount > POSCount) minCount = POSCount;
		if (minCount > FLRCount) minCount = FLRCount;
		break;
	case NIRPOSFrameCount: //
		minCount = NIRCount;
		if (minCount > POSCount) minCount = POSCount;
		break;
	case PosFrameCount: //
		minCount = POSCount;
		break;
	case NIRFrameCount: //
		minCount = NIRCount;
		break;
	case FLRFrameCount: //
		minCount = FLRCount;
		break;
	default:
		minCount = NIRCount;
		if (minCount > POSCount) minCount = POSCount;
		break;
	}; // end switch
	return minCount;
}

void TrackingMessageBuffer::setPostTrigSet(int mode, unsigned int _Frame, unsigned int _PreTrgRecFrms) {
	unsigned int FrmCount = getPreTriggerFrmCount(mode);
	if (_PreTrgRecFrms < FrmCount)
		RecStartIdx = _Frame - _PreTrgRecFrms;
	else
		RecStartIdx = _Frame - FrmCount + RecActiveWindowSize;
	RecEndIdx = 0;
	isRecording = true;
}

unsigned int TrackingMessageBuffer::getRecEndIdx(void) {
	return RecEndIdx;
}

unsigned int TrackingMessageBuffer::getRecStartIdx(void) {
	return RecStartIdx;
}

unsigned int TrackingMessageBuffer::getRecBlockIdx(void) {
	return RecBlockIdx;
}



void TrackingMessageBuffer::SetStopRec(void) {
	RecEndIdx = Data[DispIdx].frameNo;
	NIRCount = 0;
	FLRCount = 0;
	POSCount = 0;
	isRecording = false;
}

bool TrackingMessageBuffer::IsRecDone(void) {
	if (pNIR_writer == NULL && pFLR_writer == NULL && pData_writer == NULL)
		return true;
	return false;
}
unsigned int TrackingMessageBuffer::getRecCount_NIR(void) {
	return RecDoneCounter_NIR;
}unsigned int TrackingMessageBuffer::getRecCount_FLR(void) {
	return RecDoneCounter_FLR;
}
unsigned int TrackingMessageBuffer::getRecCount(void) {
	return RecDoneCounter;
}

void TrackingMessageBuffer::InitializeRecording(void) {
	char NAME[128];
	RecDoneCounter = 0;
	RecDoneCounter_NIR = 0;
	RecDoneCounter_FLR = 0;
	time_t t = time(0); // get time now;
	struct tm * now = localtime(&t);
	sprintf(NAME,"F:\\Calibration\\%04d%02d%02d_%02d%02d%02d\\", now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
	std::string FilePath = std::string(NAME);
	if (CreateDirectoryA(FilePath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		sprintf(NAME,"F:/Calibration/%04d%02d%02d_%02d%02d%02d/HDF5_%04d%02d%02d_%02d%02d%02d_NIR.h5", now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		if (NIRSize.width > 0 && NIRSize.height > 0)
			pNIR_writer = new HDF5ImageWriter(NAME, NIRSize.width, NIRSize.height);
		else
			pNIR_writer = new HDF5ImageWriter(NAME, 640, 480);

		sprintf(NAME,"f:/Calibration/%04d%02d%02d_%02d%02d%02d/HDF5_%04d%02d%02d_%02d%02d%02d_EPI.h5", now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		if (FLRSize.width > 0 && FLRSize.height > 0)
			pFLR_writer = new HDF5ImageWriter(NAME, FLRSize.width, FLRSize.height);
		else
			pFLR_writer = new HDF5ImageWriter(NAME, 1328, 1048);

		sprintf(NAME,"f:/Calibration/%04d%02d%02d_%02d%02d%02d/HDF5_%04d%02d%02d_%02d%02d%02d_DATA.h5", now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		pData_writer = new HDF5DataWriter(NAME);
	} // End of File/Folder setting for recording
}


void TrackingMessageBuffer::ReleaseRecording(void) {
	// close files
	delete pData_writer; pData_writer = NULL;
}
void TrackingMessageBuffer::ReleaseRecording_NIR(void) {
	// close files
	delete pNIR_writer; pNIR_writer = NULL;
}
void TrackingMessageBuffer::ReleaseRecording_FLR(void) {
	// close files
	delete pFLR_writer; pFLR_writer = NULL;
}
void TrackingMessageBuffer::RecordDataVector(int index, vector<cv::Point2d> * _data) {
	 vector<cv::Point2d>::iterator it; // declare an iterator to a vector of strings
	 int i = 0;
	 for(it=_data->begin() ; it < _data->end(); it++) {
		 pData_writer->PointData[index + i++]->write(&it);
    }
}

void TrackingMessageBuffer::RecordFrame_NIR(unsigned int _Frame) {
	TrackingMessage * t = getFrameData(_Frame);
	// NIR image recording
	if (t->srcNIR && NIRRecEnabled) {
		pNIR_writer->write(t->srcNIR->data);
		RecDoneCounter_NIR++;
	}

}

bool TrackingMessageBuffer::RecordFrame_FLR(uchar * src) {

	if (FLRRecEnabled) {
		if (src && pFLR_writer) {
			pFLR_writer->write(src);
			RecDoneCounter_FLR++;
			return true;
		}
		return false;
	}
	return true;
}

void TrackingMessageBuffer::RecordFrame(unsigned int _Frame) {
	TrackingMessage * t = getFrameData(_Frame);

	// NIR image recording
	if (t->srcNIR && NIRRecEnabled) {
		//cv::Mat temp;
		//t->srcNIR->copyTo(temp, t->NIRfishPos.getRotatedRectMask(t->NIRfishPos.AreaSize, 0.5, t->srcNIR->size()));
		//pNIR_writer->write(temp.data);
		//pNIR_writer->write(t->srcNIR->data);
		if (t->srcNIRproc != NULL) {
			//if (t->srcNIRproc->data)
			//	pNIR_writer->write(t->srcNIRproc->data);
			//else
				pNIR_writer->write(t->srcNIR->data);
		}
		else
			pNIR_writer->write(t->srcNIR->data);
		RecDoneCounter_NIR++;
	}


	// FLR image recording
	if (t->srcFLR && FLRRecEnabled) {
		pFLR_writer->write(t->srcFLR->data);
		RecDoneCounter_FLR++;
	}

	// Data Recording
	pData_writer->SingleData[R1_FrameNo]->write(&(t->frameNo));

	// Fish Data
	pData_writer->PointData[R2_FishPos_Leye_px]->write(&t->NIRfishPos.m_eyeLeft.center);
	pData_writer->PointData[R2_FishPos_Reye_px]->write(&t->NIRfishPos.m_eyeRight.center);
	pData_writer->PointData[R2_FishPos_Yolk_px]->write(&t->NIRfishPos.m_yolk.center);
	pData_writer->PointData[R2_FishPos_Brain_px]->write(&t->NIRfishPos.centerEyes);
	pData_writer->SingleData[R1_FishPos_Orientation_Deg]->write(&(t->NIRfishPos.orienation));

	// Stage Pos Data
	pData_writer->PointData[R2_StagePos]->write(t->stagePos.CurrentPosition);//
	pData_writer->PointData[R2_StageVel]->write(t->stagePos.SetpointVelocity);//
	pData_writer->PointData[R2_StageAcc]->write(t->stagePos.SetpointAcceleration);//
	pData_writer->PointData[R2_StageSetVoltRead]->write(t->stagePos.InputVoltage);//
	pData_writer->PointData[R2_StagePosN0]->write(t->stagePos.CurrentPosition_Chamber);//

	pData_writer->PointData[R2_StagePosP1_MPC]->write(&t->m_CntrData.StgPosMPC_mmP1);//

	pData_writer->PointData[R2_FishPos_mmN2]->write(&t->m_CntrData.FishPos_mmN2);//
	pData_writer->PointData[R2_FishPos_mmN1]->write(&t->m_CntrData.FishPos_mmN1);//
	pData_writer->PointData[R2_FishPos_mmN0]->write(&t->m_CntrData.FishPos_mmN0);//
	pData_writer->PointData[R2_FishPos_mmP1]->write(&t->m_CntrData.FishPos_mmP1);//


	pData_writer->PointData[R2_FishPos_Target_px]->write(&t->m_CntrData.Cur_Target_px);//
	pData_writer->PointData[R2_ErrPos_px]->write(&t->m_CntrData.ErrPos_Px);//
	pData_writer->PointData[R2_ErrPos_mmN2]->write(&t->m_CntrData.ErrPos_mmN2);//
	pData_writer->PointData[R2_PID_DesiredPos]->write(&t->m_CntrData.CntrPID_DesiredPos);
	pData_writer->PointData[R2_PID_CurrentPos]->write(&t->m_CntrData.CntrPID_CurrentPos);
	pData_writer->PointData[R2_PID_InputVel]->write(&t->m_CntrData.CntrPID_InputVel);
	pData_writer->PointData[R2_StageSetInputVel]->write(&t->m_CntrData.Cntr_VelInput);
	pData_writer->PointData[R2_StageInputVolt]->write(&t->m_CntrData.Cntr_VoltInput);

	RecDoneCounter++;
}


JitterFilter::JitterFilter(void)
{
	threshold = 0.015;
	CurrentValue = cv::Point2d(0,0);
}

JitterFilter::~JitterFilter(void)
{
}

cv::Point2d JitterFilter::update(cv::Point2d _p)
{
	if (fabs(_p.x - CurrentValue.x) > threshold)
		CurrentValue.x = _p.x;
	if (fabs(_p.y - CurrentValue.y) > threshold)
		CurrentValue.y = _p.y;
	return CurrentValue;
}

//------------------------------------------------------------------------------
// PoseInfo2d class
//------------------------------------------------------------------------------
PoseInfo2d::PoseInfo2d(void)
{
	Pos = cv::Point2d(0,0);
	Vel = cv::Point2d(0,0);
	Acc = cv::Point2d(0,0);
}

PoseInfo2d::~PoseInfo2d(void)
{
}

//------------------------------------------------------------------------------
// PoseInfo class
//------------------------------------------------------------------------------

PoseInfo::PoseInfo(void)
{
	Pos = cv::Point2i(0,0);
	Vel = cv::Point2i(0,0);
	Acc = cv::Point2i(0,0);
}

PoseInfo::~PoseInfo(void)
{
}

//------------------------------------------------------------------------------
// MaggotInfo class
//------------------------------------------------------------------------------

MaggotInfo::MaggotInfo(void)
{
}

MaggotInfo::~MaggotInfo(void)
{
}

//------------------------------------------------------------------------------
// TrackingInfo class
//------------------------------------------------------------------------------
TrackingInfo::TrackingInfo(void)
{
	voltageInput[0] = 0;
	voltageInput[1] = 0;
	TimeStamp = 0;
}

TrackingInfo::~TrackingInfo(void)
{
}

void TrackingInfo::clearAll(void)
{
}


void TrackingInfo::ExportData(void)
{
}

/*

TrackingInfo::TrackingInfo() {
	stage_pos_mm = cv::Point2f(0,0);
	stage_vel_mmps = cv::Point2f(0,0);
	target_dpos_px = cv::Point(0,0);
	target_dpos_mm = cv::Point2f(0,0);
	target_vel_mmps = cv::Point2f(0,0);
	dt = 0;
	size = 0;
	TimeStamp_NIR = 0;
	img = NULL;
}

TrackingInfo::TrackingInfo(
	double _stage_pos_x_mm, double _stage_pos_y_mm, double _stage_vel_x_mmps, double _stage_vel_y_mmps,
	double _target_pos_dx_px, double _target_pos_dy_px, double _target_pos_dx_mm, double _target_pos_dy_mm, uint64 _TimeStamp_NIR, int _size) {
	stage_pos_mm = cv::Point2f(_stage_pos_x_mm, _stage_pos_y_mm);
	stage_vel_mmps = cv::Point2f(_stage_vel_x_mmps, _stage_vel_y_mmps);
	target_dpos_px = cv::Point(_target_pos_dx_px, _target_pos_dy_px);
	target_dpos_mm = cv::Point2f(_target_pos_dx_mm,_target_pos_dy_mm);
	target_vel_mmps = cv::Point2f(0,0);
	size = _size;
	dt = 0;
	TimeStamp_NIR = _TimeStamp_NIR;
}

TrackingInfo::TrackingInfo(
	double _stage_pos_x_mm, double _stage_pos_y_mm, double _stage_vel_x_mmps, double _stage_vel_y_mmps,
	double _target_pos_dx_px, double _target_pos_dy_px, double _target_pos_dx_mm, double _target_pos_dy_mm, uint64 _TimeStamp_NIR, int _size, cv::Mat _img) {
	stage_pos_mm = cv::Point2f(_stage_pos_x_mm, _stage_pos_y_mm);
	stage_vel_mmps = cv::Point2f(_stage_vel_x_mmps, _stage_vel_y_mmps);
	target_dpos_px = cv::Point(_target_pos_dx_px, _target_pos_dy_px);
	target_dpos_mm = cv::Point2f(_target_pos_dx_mm,_target_pos_dy_mm);
	target_vel_mmps = cv::Point2f(0,0);
	size = _size;
	dt = 0;
	//dt = (double)(_TimeStamp_NIR - TimeStamp_NIR) * 8 / 1000 / 10000; // [ms]
	TimeStamp_NIR = _TimeStamp_NIR;
	img.create(_img.size(), _img.type());
	_img.copyTo(img);
}*/


double NormDiff_1D(double a, double ref) {
	return fabs((a-ref))/min(a, ref);
};
double NormDiff_cvPoint(cv::Point2d * a, cv::Point2d * ref) {
	double d1 = Diff_cvPoint(a[0], ref[0]);
	double d2 = Diff_cvPoint(a[1], ref[1]);
	double Ndiff = 1000;
	Ndiff = NormDiff_1D(d1, d2);
	return Ndiff;
};
double NormDiff_cvPoint(cv::Point * a, cv::Point * ref) {
	double d1 = Diff_cvPoint(a[0], ref[0]);
	double d2 = Diff_cvPoint(a[1], ref[1]);
	double Ndiff = 1000;
	Ndiff = NormDiff_1D(d1, d2);
	return Ndiff;
};
double Diff_1D(double a, double ref) {
	return a - ref;
};
double Diff_2D(double * a, double * ref){
	double a1 = a[0];
	double a2 = a[1];
	double ref1 = ref[0];
	double ref2 = ref[1];
	double d1 = NormDiff_1D(a1, ref1);
	double d2 = NormDiff_1D(a2, ref2);
	return sqrt(d1*d1 + d2*d2);
};
double Diff_cvPoint(cv::Point2d a, cv::Point2d ref){
	return cv::norm(a - ref);
};
double Diff_cvPoint(cv::Point a, cv::Point ref){
	return Diff_cvPoint(cv::Point2d(a), cv::Point2d(ref));
};
double Diff_Ang(double a, double ref){
	double d = (a - ref);
	d = fabs(adjustAngRange(d));
	return d;
};
double Var_Ang(double a, double b, double c){
	double d1 = Diff_Ang(a, b);
	double d2 = Diff_Ang(b, c);
	double d3 = Diff_Ang(c, a);
	return Var_Dist(d1, d2, d3);
};
double Var_Dist(double a, double b, double c){
	double meand = (a + b + c) / 3;
	return (sqrt((a - meand)*(a - meand) + (b - meand)*(b - meand) + (c - meand)*(c - meand)))/3;
};
double adjustAngRange(double ang) {
	while (ang >= 180)
		ang = ang - 360;
	while (ang < -180)
		ang = ang + 360;
	return ang;
};


double Area_3Points(cv::Point a, cv::Point b, cv::Point c){
	return abs(0.5*(a.cross(b) + b.cross(c) +c.cross(a)));
};
double Area_3Points(cv::Point2d a, cv::Point2d b, cv::Point2d c){
	return fabs(0.5*(a.cross(b) + b.cross(c) +c.cross(a)));
};

double atan2_twoPoints(cv::Point a, cv::Point b){
	return atan2_twoPoints((cv::Point2d)a, (cv::Point2d) b);
};
double atan2_twoPoints(cv::Point2d a, cv::Point2d b){
	double angle = atan2(a.y - b.y, a.x - b.x);
	return angle*180/PI;
};
double atan2_twoPoints_Right(cv::Point c, cv::Point LEye){
	return atan2_twoPoints_Right((cv::Point2d)c, (cv::Point2d) LEye);
};
double atan2_twoPoints_Right(cv::Point2d c, cv::Point2d LEye){

	double angle = atan2(LEye.y - c.y, LEye.x - c.x) + PI/2;
	if (angle < -PI) angle = angle + 2*PI;
	if (angle > PI) angle = angle - 2*PI;
	return angle*180/PI;
};

bool sortByFitness(const zebrafishCompareResult &lhs, const zebrafishCompareResult &rhs) { return lhs.fitness < rhs.fitness; }
//-------------------------------------------------
SpotInfo::SpotInfo(void) {
	center = cv::Point2d();
	width = 1;
	height = 1;
	angle = 0; // deg
	updateValue();
};
SpotInfo::SpotInfo(cv::RotatedRect _x)  {
	updateValue(_x);
};
SpotInfo::SpotInfo(cv::Point2d _x)  {
	center = _x;
	width = 1;
	height = 1;
	angle = 0; // deg
	updateValue();
};
SpotInfo::~SpotInfo(void)  {
};
void SpotInfo::updateValue(cv::RotatedRect _x) {
	center = cv::Point2d(_x.center);
	width = _x.size.width;
	height = _x.size.height;
	angle = _x.angle; // deg
	if (height >= width) { // make a width longer than height
		swap(width, height);
		angle = adjustAngRange(angle + 90); // +-180 // degree input
	}
	updateValue();
};
void SpotInfo::updateValue(void) {
	area = 3.141592*width*height/4;
	double a = width/2;
	double b = height/2;
	eccentricity = sqrt(a*a - b*b)/a;
};
spotCompareResult SpotInfo::compare(SpotInfo nextSpot) {
	spotCompareResult cmp;

	// ---- Travel Distance ---- //
	double maximum_travelDistance_Px = 14; // 14 px * 14.5 um/px = 203 um
	cmp.distTravel = Diff_cvPoint(nextSpot.center, center);
	if (cmp.distTravel > maximum_travelDistance_Px)
		cmp.distTravel = 1;
	else
		cmp.distTravel = cmp.distTravel/maximum_travelDistance_Px;

	// ---- Area Difference ---- //
	double maximum_AreaDiff= 0.1; // allowance for area change
	cmp.areaDiff = fabs(nextSpot.area - area)/area; // 100% change of the area is maximum
	if (cmp.areaDiff > maximum_AreaDiff)
		cmp.areaDiff = 1;
	else
		cmp.areaDiff = cmp.areaDiff / maximum_AreaDiff;


	// ---- eccentricity differnce ---- //
	double maximum_EccDiff= 0.1; // allowance for area change
	cmp.eccDiff  = fabs(nextSpot.eccentricity - eccentricity);
	if (cmp.eccDiff > maximum_EccDiff)
		cmp.eccDiff = 1;
	else
		cmp.eccDiff = cmp.eccDiff / maximum_EccDiff;

	// ---- Angle difference ---- //
	double maximum_AngDiff= 0.1; // allowance for area change
	cmp.angleDiff= Diff_Ang(nextSpot.angle, angle); // 30 degree is the maximum
	if (cmp.angleDiff > maximum_AngDiff)
		cmp.angleDiff = 1;
	else
		cmp.angleDiff = cmp.angleDiff / maximum_AngDiff;

	// ---- copy the references ---- //
	cmp.SpotRef = *this;
	cmp.SpotGiven = nextSpot;
	return cmp;
};


//-------------------------------------------------
spotCompareResult::spotCompareResult(void){
	fitness = 0; // fitness value (with weight) - the samller the similar
	distTravel = 0; // abs distance to the next spot (p_given - p)
	areaDiff = 0; // abs area normalized difference with the given spot ((area_given-area)/area)
	eccDiff = 0; // abs eccentricity difference with the given spot [0-1]
	angleDiff = 0; // abs angle difference [0-180]/180
};
spotCompareResult::~spotCompareResult(void){
};
void spotCompareResult::computeFitness(double weight[4]) {
	fitness
		= weight[0] * distTravel
		+ weight[1] * areaDiff
		+ weight[2] * eccDiff
		+ weight[3] * angleDiff;
};
void spotCompareResult::computeFitness(void) {
	fitness
		= para.w_distTravel * distTravel
		+ para.w_areaDiff * areaDiff
		+ para.w_eccDiff * eccDiff
		+ para.w_angleDiff * angleDiff;
};


//-------------------------------------------------
zebrafishInfo::zebrafishInfo(void){
	DetectionFailed = false;
	cv::RotatedRect left;
	cv::RotatedRect right;
	cv::RotatedRect yolk;
	left.center = cv::Point2f(300,220);
	left.angle = 0;
	left.size = cv::Size(10,10);

	right.center = cv::Point2f(340,220);
	right.angle = 0;
	right.size = cv::Size(10,10);

	yolk.center = cv::Point2f(320,260);
	yolk.angle = 90;
	yolk.size = cv::Size(10,20);
	m_eyeLeft = SpotInfo(left);
	m_eyeRight = SpotInfo(right);
	m_yolk = SpotInfo(yolk);

	centerEyes = (m_eyeLeft.center + m_eyeRight.center)*0.5;
	orienation = atan2_twoPoints(centerEyes, m_yolk.center);
	AreaSize = Area_3Points(m_eyeLeft.center, m_eyeRight.center, m_yolk.center);
	CosAngle_LYR = 0.866;
};
zebrafishInfo::zebrafishInfo(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk){
	ReplacePosition(_eyeLeft, _eyeRight, _yolk);
};

void zebrafishInfo::ReplacePosition(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk) {
	m_eyeLeft = _eyeLeft;
	m_eyeRight = _eyeRight;
	m_yolk = _yolk;
	centerEyes = (m_eyeLeft.center + m_eyeRight.center)*0.5;
	orienation = atan2_twoPoints(centerEyes, m_yolk.center);
	//orienation = atan2_twoPoints_Right(centerEyes, m_eyeLeft.center);
	AreaSize = Area_3Points(m_eyeLeft.center, m_eyeRight.center, m_yolk.center);

	//double distLY = Diff_cvPoint(_eyeLeft.center, _yolk.center);
	//double distRY = Diff_cvPoint(_eyeRight.center, _yolk.center);
	//double distLR = Diff_cvPoint(_eyeRight.center, _eyeLeft.center);
	//CosAngle_LYR = (distRY*distRY + distLY*distLY - distLR*distLR) / (2 * distRY * distLY);
}
zebrafishInfo::~zebrafishInfo(void){
};

void zebrafishInfo::getVaildation(zebrafishInfo * refFish) {
	if (refFish->DetectionFailed) {
		DetectionFailed = false;
		return;
	}
	// 1. Assumption 1 : A fish CANNOT changes its orientation 90 degree in one frame
	double dTh = orienation - refFish->orienation;
	while (dTh >  180) dTh = dTh - 360;
	while (dTh <= -180) dTh = dTh + 360;
	if (dTh > 90 || dTh <-90) {
		DetectionFailed = true;
		ReplacePosition(refFish->m_eyeLeft, refFish->m_eyeRight, refFish->m_yolk);
		return;
	}
	// 2. Assumption 2 : if the position changes in eyes should be simillar (std < 140um)
	cv::Point2d dL = m_eyeLeft.center - refFish->m_eyeLeft.center;
	cv::Point2d dR = m_eyeRight.center - refFish->m_eyeRight.center;
	cv::Point2d dmean = cv::Point2d((dL.x + dR.x)/2, (dL.x + dR.x)/2);
	cv::Point2d varL = dL - dmean;
	cv::Point2d varR = dR - dmean;
	double xvar[2] = {varL.x, varR.x}; double xmean = varL.x + varR.x;
	double yvar[2] = {varL.y, varR.y}; double ymean = varL.y + varR.y;
	double xstd = sqrt((xvar[0]-xmean)*(xvar[0]-xmean) + (xvar[1]-xmean)*(xvar[1]-xmean));
	double ystd = sqrt((xvar[0]-xmean)*(xvar[0]-xmean) + (xvar[1]-xmean)*(xvar[1]-xmean));
	double stdmax = max(xstd, ystd);
	if (stdmax > 10) {
		DetectionFailed = true;
		ReplacePosition(refFish->m_eyeLeft, refFish->m_eyeRight, refFish->m_yolk);
		return;
	}
	DetectionFailed = false;
}

cv::Mat zebrafishInfo::getRotatedRectMask(double AreaMean, double range, cv::Size imgSize) {
	double Area_range = 0.2;
	if ((AreaMean*(1-range) < AreaSize) && (AreaSize < AreaMean*(1+range))) { // size checking
		// 1. find the center
		cv::Point2d c = (m_yolk.center + centerEyes)*0.5;
		// 2. create ellipse mask
		double height = 2*cv::norm(m_yolk.center - centerEyes);
		double width = 2.4*cv::norm(m_eyeLeft.center - m_eyeRight.center);
		double _orientation = orienation;
		if (_orientation <=  0) _orientation += 360;
		if (_orientation > 360) _orientation -= 360;
		cv::RotatedRect maskEps(c, cv::Size2f(height, width), (float)orienation );
		cv::Point2f pts[4];
		cv::Point pts_i[4];
		maskEps.points(pts);
		for (int i = 0; i < 4; i++)
			pts_i[i] = (cv::Point)pts[i];

		cv::Mat mask = cv::Mat(imgSize, CV_8UC1);
		mask.setTo(cv::Scalar(0));
		cv::fillConvexPoly(mask, pts_i, 4, cv::Scalar(255), 8);
		return mask;
	}
	return cv::Mat();
}

vector<cv::Point2d> estimateFuturePos(vector<double> orientationHist, vector<cv::Point2d> fishPosHist) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<cv::Point2d> estPos;
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPosHist[0]);

	// 1. get peaks from history
	double peaks[2];
	cv::Point2d peaks_pos[2];
	int peaks_index[2];
	double dir = 0; double dir_c = 0;
	int peakfound = 0;
	for (int i = 1; i < orientationHist.size(); i++) {
		dir_c = orientationHist[i] - orientationHist[i-1];
		if (dir*dir_c < 0) peakfound++;
		if (dir_c != 0) dir = dir_c;
		if (peakfound == 2) break;
		peaks[peakfound] = orientationHist[i];
		peaks_pos[peakfound] = fishPosHist[i];
		peaks_index[peakfound] = i;
	}
	if (peakfound < 2)
		return estPos;

	// 2. estimate parameters for orientation function
	// theta = a*sin(w*(t-t0)) + th0
	double a = (peaks[1] - peaks[0])/2;
	double w = (2*PI) / (2 * (peaks_index[1] - peaks_index[0]));
	double di = (double)(peaks[1] + peaks[0]);
	double th0 = di/2.0;
	double asin_value = (peaks[0] - th0)/a;	if (fabs(asin_value) > 1) asin_value = asin_value/fabs(asin_value);
	double phi = asin(asin_value);
	double t0 = -phi/w;

	// 3. estimate parameters for position functions (from origin)
	// theta = alpha*sin(omega*(x-d0*dx))
	cv::Point2d dS = peaks_pos[1] - peaks_pos[0];
	double costh = cos(th0*PI/180);
	double sinth = sin(th0*PI/180);
	double dx = fabs((costh*dS.x + sinth*dS.y)/di);
	double dy = fabs(-sinth*dS.x + costh*dS.y);
	double alpha = a/fabs(a)*dy/2;
	double omega = w/dx;
	phi = t0*dx;
	vector<cv::Point2d> p_org(EstSteps);
	for (int i = 0; i < EstSteps; i++) {
		double x_temp = i*dx;
		double y_temp = alpha*sin(omega*(x_temp-phi));
		p_org[i] = cv::Point2d(x_temp, y_temp);
	}
	// 4. rotataion & translation
	vector<cv::Point2d> p_rot(EstSteps);
	cv::Point2d p_offset;
	for (int i = 0; i < EstSteps; i++) {
		double x_temp = costh*p_org[i].x - sinth*p_org[i].y;
		double y_temp = sinth*p_org[i].x + costh*p_org[i].y;
		if (i == 0)
			p_offset = cv::Point2d(x_temp, y_temp) + fishPosHist[0];
		p_rot[i] = cv::Point2d(x_temp, y_temp) - p_offset;
	}
	return estPos;
}


vector<cv::Point2d> estimateFuturePos_linear(vector<double> orientationHist, vector<cv::Point2d> fishPosHist) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<cv::Point2d> estPos;
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPosHist[0]);
	//int pastSteps = orientationHist.size();
	int pastSteps = 5;

	// 1. get peaks from history
	double peaks[2];
	cv::Point2d peaks_pos[2];
	int peaks_index[2];
	double dir = 0; double dir_c = 0;
	int peakfound = 0;
	for (int i = 1; i < orientationHist.size(); i++) {
		dir_c = orientationHist[i] - orientationHist[i-1];
		if (dir*dir_c < 0) peakfound++;
		if (dir_c != 0) dir = dir_c;
		if (peakfound == 2) break;
		peaks[peakfound] = orientationHist[i];
		peaks_pos[peakfound] = fishPosHist[i];
		peaks_index[peakfound] = i-1; // peak for position is one step before
	}
	if (peakfound < 2)
		return estPos;

	// 2. estimate parameters for orientation function
	// theta = a*sin(w*(t-t0)) + th0
	double di = (double)(peaks[1] + peaks[0]);
	double th0 = di/2.0;
	double costh = cos(th0*PI/180);
	double sinth = sin(th0*PI/180);

	// 3. move to origin
	cv::Point2d p_org = (peaks_pos[1] + peaks_pos[0])*0.5;
	vector<cv::Point2d> p_orgB(pastSteps);
	for (int i = 0; i < pastSteps; i++) {
		cv::Point2d p_trans = fishPosHist[i] - p_org;
		double x_temp =  costh*p_trans.x + sinth*p_trans.y;
		double y_temp = -sinth*p_trans.x + costh*p_trans.y;
		p_orgB[i] = cv::Point2d(x_temp, y_temp);
	}
	// 4. estimate dx
	double diffx_sum = 0;
	double threhold = 0.01;
	int sampleCount = 0;
	for (int i = 1; i < pastSteps; i++) {
		double temp = fabs(p_orgB[i].x - p_orgB[i-1].x);
		if (temp > threhold) {
			diffx_sum += temp;
			sampleCount++;
		}
	}
	double dx = 0;
	if (sampleCount > 3)
		dx = diffx_sum / (sampleCount - 1 );
	else
		return estPos;


	// 5. estimate x position
	vector<double> p_orgF_x(EstSteps);
	for (int i = 0; i < EstSteps; i++) {
		p_orgF_x[i] = dx*(i+1);
	}
	// 6. move back to the real-space
	for (int i = 0; i < EstSteps; i++) {
		double x_temp = costh*p_orgF_x[i] + p_org.x;
		double y_temp = sinth*p_orgF_x[i] + p_org.y;
		estPos[i].x = x_temp; estPos[i].y = y_temp;
	}
	return estPos;
}

vector<cv::Point2d> estimateFuturePos_Front(vector<double> orientationHist, vector<cv::Point2d> fishPosHist, double gain) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<cv::Point2d> estPos;

	// 1. get peaks from history
	//double gain = 0.5*0.014;
	double OriDiff_sum=0;
	double HistCount = 6;

	//for i = 1:1:orientationHist.size();
	for (int i = 1; i<HistCount; i++) {
		double A0 = cv::norm(fishPosHist[i] - fishPosHist[i-1]);
		//double A0 = (orientationHist[i] - orientationHist[i-1]);
		//if (A0 > 180) A0 = A0 - 360;
		//if (A0 <= -180) A0 = A0 + 360;
		OriDiff_sum += (abs(A0));
	}

	double A0 = OriDiff_sum/(HistCount-1);

	cv::Point2d fishPos_adjust = fishPosHist[0];
	if (A0 > 0.01) {
		double costh = cos(orientationHist[0]*PI/180);
		double sinth = sin(orientationHist[0]*PI/180);
		cv::Point2d offset = cv::Point2d(gain*costh*A0, gain*sinth*A0);
		fishPos_adjust = fishPosHist[0]+offset;
	}
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPos_adjust);

	return estPos;
}

zebrafishCompareResult zebrafishInfo::compare(spotCompareResult _eyeLeft, spotCompareResult _eyeRight, spotCompareResult _yolk){
	zebrafishCompareResult cmp;

	double distLY = Diff_cvPoint(_eyeLeft.SpotGiven.center, _yolk.SpotGiven.center);
	double distRY = Diff_cvPoint(_eyeRight.SpotGiven.center, _yolk.SpotGiven.center);
	double distLR = Diff_cvPoint(_eyeRight.SpotGiven.center, _eyeLeft.SpotGiven.center);
	cv::Point2d centerLREyes = (_eyeLeft.SpotGiven.center + _eyeRight.SpotGiven.center)*0.5;


	// -------------------------------//
	// ---- features w/o history ---- //

	// --- isosceles triangle (meansure angle between the (yolk-lefteye-center) and (yolk-righteye-center)
	double Maximum_CosDiff = 0.3; // maximum value for (cos(theta_YLC) - cos(theta_YRC)
	double CosLEye = (distLY*distLY + distLR*distLR - distRY*distRY) / (2 * distLY * distLR);
	double CosREye = (distRY*distRY + distLR*distLR - distLY*distLY) / (2 * distRY * distLR);
	cmp.angleEyesYolkSymmetry = fabs(CosLEye - CosREye); // 0 (isosceles) - 2 (not)
	if (cmp.angleEyesYolkSymmetry > Maximum_CosDiff)
		cmp.angleEyesYolkSymmetry = 1;
	else
		cmp.angleEyesYolkSymmetry = cmp.angleEyesYolkSymmetry/Maximum_CosDiff;

	// --- Area Difference between Left-eye and Right-Eye
	double Maximum_AreaDiff_Eyes = 0.3; // maximum value
	cmp.areaLREyeNDiff= NormDiff_1D(_eyeLeft.SpotGiven.area, _eyeRight.SpotGiven.area);
	if (cmp.areaLREyeNDiff > Maximum_AreaDiff_Eyes)
		cmp.areaLREyeNDiff = 1;
	else
		cmp.areaLREyeNDiff = cmp.areaLREyeNDiff/Maximum_AreaDiff_Eyes;

	// --- area_yolk is Greater Than Left-Eye
	double minimum_ratioAreaDiff = 0.2;
	if ((_yolk.SpotGiven.area > _eyeLeft.SpotGiven.area * (1+minimum_ratioAreaDiff)))
		cmp.areayolkGreaterThanLeftEyes = 0;
	else
		cmp.areayolkGreaterThanLeftEyes = 1;

	// --- area_yolk is Greater Than Right-Eye
	if ((_yolk.SpotGiven.area > _eyeRight.SpotGiven.area * (1+minimum_ratioAreaDiff)))
		cmp.areayolkGreaterThanRightEyes = 0;
	else
		cmp.areayolkGreaterThanRightEyes  = 1;

	// --- Dist_LY > Dist_LR
	if (distLY > distLR*1.2)
		cmp.Dist_LY_Greater_Dist_LR = 0;
	else
		cmp.Dist_LY_Greater_Dist_LR = 1;

		// --- Dist_LY > Dist_LR
	if (distRY > distLR*1.2)
		cmp.Dist_RY_Greater_Dist_LR = 0;
	else
		cmp.Dist_RY_Greater_Dist_LR = 1;

	// --- Dist_LY > Left.height + yolk.height
	if (distLY > (_eyeLeft.SpotGiven.height + _yolk.SpotGiven.height)/2)
		cmp.Overlap_LY = 0;
	else
		cmp.Overlap_LY = 1;

	// --- Dist_RY > Right.height + yolk.height
	if (distRY > (_eyeRight.SpotGiven.height + _yolk.SpotGiven.height)/2)
		cmp.Overlap_RY = 0;
	else
		cmp.Overlap_RY = 1;

	// --- Dist_RL > Right.height + Left.height
	if (distLR > (_eyeRight.SpotGiven.height + _eyeLeft.SpotGiven.height)/2)
		cmp.Overlap_LR = 0;
	else
		cmp.Overlap_LR = 1;

	// if there is any overlap - the distLR and distRL set to one
	if (cmp.Overlap_LY || cmp.Overlap_RY || cmp.Overlap_LR) {
		cmp.Overlap_RY = 1;
		cmp.Overlap_LY = 1;
	}


	// ---- END::features w/o history ---- //
	// ------------------------------------//



	// -------------------------------//
	// ---- features w/ history ---- //

	// --- Travel distance change should be same
	double Maximum_Variance_TravelDist = 3; // px^2
	cmp.TravdistNVar= Var_Dist(_eyeLeft.distTravel, _eyeRight.distTravel, _yolk.distTravel);
	if (cmp.TravdistNVar > Maximum_Variance_TravelDist)
		cmp.TravdistNVar = 1;
	else
		cmp.TravdistNVar = cmp.TravdistNVar/Maximum_Variance_TravelDist;

/*
	double distLY_p = Diff_cvPoint(m_eyeLeft.center, m_yolk.center);
	double distRY_p = Diff_cvPoint(m_eyeRight.center, m_yolk.center);
	double distLR_p = Diff_cvPoint(m_eyeLeft.center, m_eyeRight.center);

	// --- the orientation change with the previous frame should be small. [0 - 1]
	cmp.OrientationDiff = Diff_Ang(atan2_twoPoints(centerLREyes, _yolk.SpotGiven.center), orienation)/30;
	if (cmp.OrientationDiff > 1) // 30 degree maximum
		cmp.OrientationDiff = 1;

	// LengthChangeVar
	cmp.LengthChangeVar = (NormDiff_1D(distLY, distLY_p) + NormDiff_1D(distRY, distRY_p) + NormDiff_1D(distLR, distLR_p));
	if (cmp.LengthChangeVar > 1)
		cmp.LengthChangeVar = 1;


	// --- area inside of three points should be same // 0 (same), 1(double), ...
	cmp.bodyAreaDiff = fabs(Area_3Points(_eyeLeft.SpotGiven.center, _eyeRight.SpotGiven.center, _yolk.SpotGiven.center) - AreaSize) / AreaSize;
	if (cmp.bodyAreaDiff>1) // maximum 100% change
		cmp.bodyAreaDiff = 1;

	//--- area ratio between the lefteye-yolk and righteye-yolk should be same
	double areaLY = fabs(_eyeLeft.SpotGiven.area - _yolk.SpotGiven.area);
	double areaRY = fabs(_eyeRight.SpotGiven.area - _yolk.SpotGiven.area);
	cmp.areaLREyeYolkNDiff = NormDiff_1D(areaLY, areaRY); // 0 (same), 1(double), ...
	if (cmp.distLREyeYolkNDiff > 1)
		cmp.distLREyeYolkNDiff  = 1;

	// ---- END::features w history ---- //
	// ----------------------------------//
	*/


	//--- copy the reference data ---//
	cmp.m_eyeLeft = _eyeLeft;
	cmp.m_eyeRight = _eyeRight;
	cmp.m_yolk = _yolk;
	cmp.zebrafishCompareResult_ref = this;
	return cmp;
};


//-------------------------------------------------

zebrafishCompareResult::zebrafishCompareResult(void) {
	fitness = 0;
	// feature parameters w/o history
	angleEyesYolkSymmetry = 0;
	areaLREyeNDiff = 0;
	areayolkGreaterThanLeftEyes = 0;
	areayolkGreaterThanRightEyes = 0;
	Dist_LY_Greater_Dist_LR = 0;
	Dist_RY_Greater_Dist_LR = 0;
	Overlap_LY = 0;
	Overlap_RY = 0;
	Overlap_LR = 0;
	// feature parameters w history (not activated yet)
	LengthChangeVar = 0;
	TravdistNVar = 0;
	bodyAreaDiff= 0;
	OrientationDiff = 0;
	distLREyeYolkNDiff = 0;
}

zebrafishCompareResult::~zebrafishCompareResult(void) {
};

void zebrafishCompareResult::computeFitness(double * weight) {
	fitness =
		weight[0] * angleEyesYolkSymmetry
		+ weight[1] * areaLREyeNDiff
		+ weight[2] * areayolkGreaterThanLeftEyes
		+ weight[3] * areayolkGreaterThanRightEyes
		+ weight[4] * Dist_LY_Greater_Dist_LR
		+ weight[5] * Dist_RY_Greater_Dist_LR
		+ weight[6] * Overlap_LY
		+ weight[7] * Overlap_RY
		+ weight[8] * Overlap_LR
		+ weight[9] * LengthChangeVar
		+ weight[10] * TravdistNVar
		+ weight[11] * bodyAreaDiff
		+ weight[12] * OrientationDiff
		+ weight[13] * distLREyeYolkNDiff
		+ m_eyeLeft.fitness
		+ m_eyeRight.fitness
		+ m_yolk.fitness;
};


void zebrafishCompareResult::computeFitness(void) {
	fitness =
		para.w_angleEyesYolkSymmetry * angleEyesYolkSymmetry
		+ para.w_areaLREyeNDiff * areaLREyeNDiff
		+ para.w_areayolkGreaterThanLeftEyes * areayolkGreaterThanLeftEyes
		+ para.w_areayolkGreaterThanRightEyes * areayolkGreaterThanRightEyes
		+ para.w_Dist_LY_Greater_Dist_LR * Dist_LY_Greater_Dist_LR
		+ para.w_Dist_RY_Greater_Dist_LR * Dist_RY_Greater_Dist_LR
		+ para.w_Overlap_LY * Overlap_LY
		+ para.w_Overlap_RY * Overlap_RY
		+ para.w_Overlap_LR * Overlap_LR
		+ para.w_LengthChangeVar * LengthChangeVar
		+ para.w_TravdistNVar * TravdistNVar
		+ para.w_bodyAreaDiff * bodyAreaDiff
		+ para.w_OrientationDiff * OrientationDiff
		+ para.w_distLREyeYolkNDiff * distLREyeYolkNDiff
		+ m_eyeLeft.fitness
		+ m_eyeRight.fitness
		+ m_yolk.fitness;
};

void zebrafishCompareResult::DetermineLeftRightEyes(void) {
	double Angle_LY = atan2_twoPoints(m_eyeLeft.SpotGiven.center, m_yolk.SpotGiven.center);
	double Angle_RY = atan2_twoPoints(m_eyeRight.SpotGiven.center, m_yolk.SpotGiven.center);
	double delta_angle = (Angle_LY - Angle_RY);
	if (delta_angle < -180)
		delta_angle = delta_angle + 360;
	if (delta_angle > 0) { // because the y-axis is inverted
		// left and right eyes are switched
		swap(m_eyeLeft, m_eyeRight);
	}
}

void zebrafishCompareResult::estimateAnotherEye(spotCompareResult exsitingEye, const cv::Mat &img_org) {
	// yolk and one eye are found. One eye is still missing.
	double c = zebrafishCompareResult_ref->CosAngle_LYR;
	double s = sqrt(1 - c*c);

	cv::Point2d cand_R_Org(c*(exsitingEye.SpotGiven.center.x - m_yolk.SpotGiven.center.x) - s*(exsitingEye.SpotGiven.center.y - m_yolk.SpotGiven.center.y),
		s*(exsitingEye.SpotGiven.center.x - m_yolk.SpotGiven.center.x) + c*(exsitingEye.SpotGiven.center.y - m_yolk.SpotGiven.center.y));
	cv::Point2d cand_L_Org(c*(exsitingEye.SpotGiven.center.x - m_yolk.SpotGiven.center.x) + s*(exsitingEye.SpotGiven.center.y - m_yolk.SpotGiven.center.y),
		(-s)*(exsitingEye.SpotGiven.center.x - m_yolk.SpotGiven.center.x) + c*(exsitingEye.SpotGiven.center.y - m_yolk.SpotGiven.center.y));

	cv::Rect ROI_cand_R(cand_R_Org + m_yolk.SpotGiven.center - cv::Point2d(8,8), cv::Size(16, 16));
	adjustROI_ValidRegion(ROI_cand_R, img_org.size());
	cv::Rect ROI_cand_L(cand_L_Org + m_yolk.SpotGiven.center - cv::Point2d(8,8), cv::Size(16, 16));
	adjustROI_ValidRegion(ROI_cand_L, img_org.size());

	CvScalar meanR = cv::mean(img_org(ROI_cand_R));
	CvScalar meanL = cv::mean(img_org(ROI_cand_L));

	if (meanL.val[0] > meanR.val[0]) { // mean value
		// existing eye -> right eye, and new eye (left)
		m_eyeRight = exsitingEye;
		m_eyeLeft.SpotGiven.center = cand_L_Org + m_yolk.SpotGiven.center;
		m_eyeLeft.SpotGiven.area = -1;
	}
	else {
		m_eyeLeft = exsitingEye;
		m_eyeRight.SpotGiven.center = cand_R_Org + m_yolk.SpotGiven.center;
		m_eyeRight.SpotGiven.area = -1;
	}
}

void zebrafishCompareResult::estimateYolk(const cv::Mat &img_org) {
	double angle_LR = atan2_twoPoints(m_eyeLeft.SpotGiven.center, m_eyeRight.SpotGiven.center) + 90;
	if (angle_LR > 180) angle_LR = angle_LR - 360;
	cv::Point2d center_temp = (m_eyeLeft.SpotGiven.center + m_eyeRight.SpotGiven.center)*0.5;
	double Dist_YolkCenter = 50.0;

	double c = cos(angle_LR*PI/180);
	double s = sin(angle_LR*PI/180);

	cv::Point2d cand_Y1(c*Dist_YolkCenter, s*Dist_YolkCenter);
	cv::Point2d cand_Y2 = -cand_Y1;

	cv::Rect ROI_cand1(cand_Y1 + center_temp - cv::Point2d(10,10), cv::Size(20, 20));
	adjustROI_ValidRegion(ROI_cand1, img_org.size());
	cv::Rect ROI_cand2(cand_Y2 + center_temp - cv::Point2d(10,10), cv::Size(20, 20));
	adjustROI_ValidRegion(ROI_cand2, img_org.size());

	CvScalar mean1 = cv::mean(img_org(ROI_cand1));
	CvScalar mean2 = cv::mean(img_org(ROI_cand2));

	if (mean1.val[0] > mean2.val[0]) { // mean value
		// update yolk position
		m_yolk.SpotGiven.center = cand_Y1 + center_temp;
		m_yolk.SpotGiven.area = -1;
	}
	else {
		m_yolk.SpotGiven.center = cand_Y2 + center_temp;
		m_yolk.SpotGiven.area = -1;
	}
}

vector<cv::Point> GetZebraFishROI_exact(cv::Point2d p, double Orientation_deg, cv::Size imgSize) {
	vector<cv::Point2d> v(12);
	v[0].x = -65; v[0].y = -20; //px
	v[1].x = -145; v[1].y = 35; //px
	v[2].x = -145; v[2].y = 90; //px
	v[3].x = -128; v[3].y = 122; //px
	v[4].x = -85; v[4].y = 164; //px
	v[5].x = -30; v[5].y = 180; //px
	v[6].x = 35; v[6].y = 180; //px
	v[7].x = 85; v[7].y = 164; //px
	v[8].x = 128; v[8].y = 122; //px
	v[9].x = 145; v[9].y = 95; //px
	v[10].x = 145; v[10].y = 35; //px
	v[11].x = 65; v[11].y = -20; //px

	double cos_o = cos((-Orientation_deg+90)*PI/180);
	double sin_o = -sin((-Orientation_deg+90)*PI/180);

	vector<cv::Point> v_new;
	for (size_t i = 0; i<v.size(); i++) {
		v_new.push_back(cv::Point(cos_o*v[i].x -sin_o*v[i].y +p.x, sin_o*v[i].x + cos_o*v[i].y+p.y));
	}

	return v_new;
}

cv::Rect GetZebraFishROI(cv::Point2d p, double Orientation_deg, cv::Size imgSize) {
	vector<cv::Point2d> v(12);
	v[0].x = -45; v[0].y = -20; //px
	v[1].x = -145; v[1].y = 35; //px
	v[2].x = -145; v[1].y = 90; //px
	v[3].x = -128; v[1].y = 122; //px
	v[4].x = -85; v[1].y = 164; //px
	v[5].x = -30; v[1].y = 180; //px
	v[6].x = 35; v[1].y = 180; //px
	v[7].x = 85; v[1].y = 164; //px
	v[8].x = 128; v[1].y = 122; //px
	v[9].x = 145; v[1].y = 95; //px
	v[10].x = 145; v[1].y = 35; //px
	v[11].x = -45; v[1].y = -20; //px

	double cos_o = cos((Orientation_deg+90)*PI/180);
	double sin_o = -sin((Orientation_deg+90)*PI/180);

	vector<cv::Point> v_new;
	for (size_t i = 0; i<v.size(); i++) {
		v_new.push_back(cv::Point(cos_o*v[i].x -sin_o*v[i].y +p.x, sin_o*v[i].x + cos_o*v[i].y+p.y));
	}
	cv::Rect new_ROI;
	new_ROI = cv::boundingRect(cv::Mat(v_new));
	adjustROI_ValidRegion(new_ROI, imgSize);

	return new_ROI;
};

void adjustROI_ValidRegion(cv::Rect &R, cv::Size imgSize) {
	if (R.x < 0) {
		R.width = R.width + R.x;
		R.x = 0;
	}
	if (R.y < 0) {
		R.height = R.height + R.y;
		R.y = 0;
	}
	if (R.x + 1 > imgSize.width) R.x = imgSize.width;
	if (R.y + 1 > imgSize.height) R.x = imgSize.height;

	if (R.x + R.width > imgSize.width)
		R.width = imgSize.width - R.x;

	if (R.y + R.height > imgSize.height)
		R.height = imgSize.height - R.y;

}

cv::Mat createROI_Mask(const cv::Mat &src, cv::Rect roi_rect) {
	if ((roi_rect.width!=0) && (roi_rect.height!=0)) {
		cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);  // type of mask is CV_8U
		cv::Mat roi(mask, roi_rect);
		roi = cv::Scalar(255, 255, 255);
		return mask;
	}
	return cv::Mat();
}

cv::Mat createROI_Mask_Shaped(const cv::Mat &src, vector<cv::Point> ROI_contour) {
	if (ROI_contour.size() > 2) {
		cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);  // type of mask is CV_8U
		vector<vector<cv::Point> > contourVec;
		contourVec.push_back(ROI_contour);
		drawContours(mask, contourVec, 0, cv::Scalar(255, 0, 0),CV_FILLED);

		//cv::Mat dest(src.size(), CV_8UC3);
		//src.copyTo(dest, mask);
		//cv::namedWindow( "mask", cv::WINDOW_AUTOSIZE );// Create a window for display.
		//cv::imshow( "Original Image", dest );

		return mask;
	}
	return cv::Mat();
}

// ---------------------------------------------------------------------
// ------------- Moving Average class ----------------------------------
// ---------------------------------------------------------------------
MovingAverage_double::MovingAverage_double(void)  {
	stepMax = 7;
	stepMax_Hist = 7;
};
MovingAverage_double::MovingAverage_double(int _steps)  {
	stepMax = _steps;
	stepMax_Hist = _steps;
};
MovingAverage_double::MovingAverage_double(int _steps, int _stepHist)  {
	stepMax = _steps;
	stepMax_Hist = _stepHist;
};
MovingAverage_double::~MovingAverage_double(void)  {
};

void MovingAverage_double::clear(void)  {
	RawData.clear();
	MVA_Hist.clear();
};
double MovingAverage_double::update(double NewValue) {
	updateRawData(NewValue);
	MVA = updateMVA_Hist(stepMax);
	return MVA;
}

void MovingAverage_double::updateRawData(double NewValue) {
	if (RawData.size() == 0)
		RawData.push_back(NewValue);
	else
		RawData.insert(RawData.begin(), NewValue);
	while (RawData.size() > stepMax_Hist)
		RawData.pop_back();
}

double MovingAverage_double::updateMVA_Hist(int steps) {
	double _MVA = 0;
	steps = min(steps, (int)(RawData.size()));
	for (int i = 0; i<steps; i++) {
		_MVA += RawData[i];
	}
	_MVA = _MVA / (double)(steps);

	if (MVA_Hist.size() == 0)
		MVA_Hist.push_back(_MVA);
	else
		MVA_Hist.insert(MVA_Hist.begin(), _MVA);
	while (MVA_Hist.size() > stepMax_Hist)
		MVA_Hist.pop_back();

	return _MVA;
}

double MovingAverage_double::MVA_Hist_FDD(void) {
	double fdev = 0;
	vector<double> diff;
	for (int i = 1; i < MVA_Hist.size(); i++)
		diff.push_back(MVA_Hist[i-1] - MVA_Hist[i]);
	switch (diff.size()) {
		case 0:
			fdev = 0;
			break;
		case 1:
			fdev = 0;
			break;
		case 2:
			fdev = -diff[1] + diff[0];
			break;
		case 3:
			fdev = -3/2*diff[2] + 2*diff[1] -1/2*diff[0];
			break;
		case 4:
			fdev = -11/6*diff[3] +3*diff[2] - 3/2*diff[1] - 1/3*diff[0];
			break;
		default:
			fdev = 0;
	}
	return fdev;
}

// ---------------------------------------------------------------------
MovingAverage_cvPoint2d::MovingAverage_cvPoint2d(void)  {
	stepMax = 7;
	stepMax_Hist = 7;
};
MovingAverage_cvPoint2d::MovingAverage_cvPoint2d(int _steps)  {
	stepMax = _steps;
	stepMax_Hist = _steps;
};
MovingAverage_cvPoint2d::MovingAverage_cvPoint2d(int _steps, int _stepHist)  {
	stepMax = _steps;
	stepMax_Hist = _stepHist;
}
MovingAverage_cvPoint2d::~MovingAverage_cvPoint2d(void)  {
};

void MovingAverage_cvPoint2d::clear(void)  {
	RawData.clear();
	MVA_Hist.clear();
};
cv::Point2d MovingAverage_cvPoint2d::update(cv::Point2d NewValue) {
	updateRawData(NewValue);
	MVA = updateMVA_Hist(stepMax);
	return MVA;
}

void MovingAverage_cvPoint2d::updateRawData(cv::Point2d NewValue) {
	if (RawData.size() == 0)
		RawData.push_back(NewValue);
	else
		RawData.insert(RawData.begin(), NewValue);
	while (RawData.size() > stepMax)
		RawData.pop_back();
}

cv::Point2d MovingAverage_cvPoint2d::updateMVA_Hist(int steps) {
	cv::Point2d _MVA = 0;
	steps = min(steps, (int)(RawData.size()));
	for (int i = 0; i<steps; i++) {
		_MVA += RawData[i];
	}
	_MVA.x = _MVA.x / (double)(steps);
	_MVA.y = _MVA.y / (double)(steps);

	if (MVA_Hist.size() == 0)
		MVA_Hist.push_back(_MVA);
	else
		MVA_Hist.insert(MVA_Hist.begin(), _MVA);
	while (MVA_Hist.size() > stepMax)
		MVA_Hist.pop_back();

	return _MVA;
}
cv::Point2d MovingAverage_cvPoint2d::MVA_Hist_FDD(void) {
	cv::Point2d fdev(0,0);
	vector<cv::Point2d> diff;
	for (int i = 1; i < MVA_Hist.size(); i++)
		diff.push_back(MVA_Hist[i-1] - MVA_Hist[i]);

	switch (diff.size()) {
		case 0:
			break;
		case 1:
			break;
		case 2:
			fdev = -diff[1] + diff[0];
			break;
		case 3:
			fdev = -3/2*diff[2] + 2*diff[1] -1/2*diff[0];
			break;
		case 4:
			fdev = -11/6*diff[3] +3*diff[2] - 3/2*diff[1] - 1/3*diff[0];
			break;
		default:
			fdev = -11/6*diff[3] +3*diff[2] - 3/2*diff[1] - 1/3*diff[0];
			break;
	}
	return fdev;
}

// ---------------------------------------------------------------------
// ------------- Fish Position Estimation class ------------------------
// ---------------------------------------------------------------------
zebrafishPositionEstimate::zebrafishPositionEstimate(void) :
	MVA_TargetSpeed(5),
	MVA_TargetPosition(9),
	MVA_AxisHeading(9),
	MVA_Heading(20),
	ProjPt(0,0),
	TrgtPt(0,0),
	BasePt(0,0)
{
	angMomentum_maxThrehold = 100;
	angMomentum_minThrehold = 10;
	angMomentum_Threhold = angMomentum_maxThrehold;
	for (int i = 0; i < 9; i++)
		weight_angMomentum.push_back(0);
	for (int i = 0; i < 8; i++)
		weight_angMomentum.push_back(log10(i+1));
	weight_angMomentum.push_back(1);
	WindowSize_AngMTh = 2; //[steps]

	angMomentum_FrameCountAfterMoving = weight_angMomentum.size()-1;


	AxisHeadingAngle = 0;
	ForwardVelocity = 0;
	cosH = 1;
	sinH = 0;
	FrameCountStopEnabled = true;
};
zebrafishPositionEstimate::~zebrafishPositionEstimate(void)  {
};

void zebrafishPositionEstimate::clear(void) {
	angMomentum_maxThrehold = 100;
	angMomentum_minThrehold = 10;
	angMomentum_Threhold = angMomentum_maxThrehold;
	angMomentum_FrameCountAfterMoving = 0;

	AxisHeadingAngle = 0;
	ForwardVelocity = 0;
	cosH = 1;
	sinH = 0;
	FrameCountStopEnabled = true;

	MVA_TargetSpeed.clear();
	MVA_TargetPosition.clear();
	MVA_AxisHeading.clear();
	MVA_Heading.clear();
}

vector<cv::Point2d> zebrafishPositionEstimate::estimateFuturePosition(zebrafishInfo * CurInfo, int steps) {
	vector<cv::Point2d> estPts = estimateFuturePosition(CurInfo->centerEyes, CurInfo->orienation, steps);
	return estPts;
};

vector<cv::Point2d> zebrafishPositionEstimate::estimateFuturePosition(cv::Point2d TargetPos, double _HeadingAngle, int steps) {
	// update target position and orientation
	double dt = 0.004;
	TrgtPt = TargetPos;
	BasePt = MVA_TargetPosition.update(TargetPos);
	MVA_Heading.update(_HeadingAngle);
	// update velocity
	//cv::Point2d ForwardVelocity_v = MVA_TargetPosition.MVA_Hist_FDD();
	cv::Point2d ForwardVelocity_v(0,0);
	if (MVA_TargetPosition.RawData.size() > 1)
		ForwardVelocity_v = MVA_TargetPosition.RawData[0] - MVA_TargetPosition.RawData[1];
	ForwardVelocity = MVA_TargetSpeed.update(cv::norm(ForwardVelocity_v)/dt);

	// option 0. (fitting) Orientation (angle) - not updated to the history  & Projection of Point & correct the heading vector and headingangle.
	//computeAxisHeading_fitting(); // update to the axis heading angle
	//computeProjPt(); // compute ProjPt

	// option 2. New developped
	computeAxisHeading_dth();
	ProjPt = MVA_TargetPosition.RawData[0];
	AxisHeadingVector = cv::Point2d(-cosH, -sinH);

	// Angular momentum
	computeAngularMomentum();
	getWeight();

	vector<cv::Point2d> estPts(steps);
	for (int i = 0; i < steps; i++) {
		estPts[i] = (i+1)*AxisHeadingVector*weight*ForwardVelocity*dt + ProjPt;
	}
	return estPts;
};
void zebrafishPositionEstimate::computeAxisHeading_dth(void)  {
	AxisHeadingAngle = 0;

	int n = MVA_Heading.RawData.size();
	double dtheta_sum = 0;
	for (int i = 1; i < n; i++) {
		// get angle difference b/t [-180, 180)
		double dth = MVA_Heading.RawData[i] - MVA_Heading.RawData[0];
		if (dth < -180) dth += 360;
		if (dth >= 180) dth -+ 360;
		dtheta_sum += dth;
	}
	double dth_addition;
	if (n > 1)
		dth_addition = dtheta_sum / (n);
	else
		dth_addition = 0;

	AxisHeadingAngle = MVA_Heading.RawData[0] + dth_addition;
	cosH = cos(AxisHeadingAngle*PI/180);
	sinH = sin(AxisHeadingAngle*PI/180);
	MVA_AxisHeading.update(AxisHeadingAngle);
}


void zebrafishPositionEstimate::computeAxisHeading_fitting(void)  {
	AxisHeadingAngle = 0;
	if (MVA_TargetPosition.MVA_Hist.size() >= 3) {
		int n_item = MVA_TargetPosition.MVA_Hist.size();
		// allocate memory
		double * X = (double *) malloc (2 * n_item * sizeof(double));
		// --- fitting ---  y = a0*x + a1 =[a0 a1][x;1]
		for (int i = 0; i < n_item; i++) {
			X[2*i] = MVA_TargetPosition.MVA_Hist[i].x;
			X[2*i + 1] = 1;
		};
		double * Y = (double *) malloc (n_item * sizeof(double));
		for (int i = 0; i < n_item; i++) {
			Y[i] = MVA_TargetPosition.MVA_Hist[i].y;
		};
		MKL_INT m = n_item, n = 2, nrhs = 1, lda = n, ldb = 1, info;
		info = LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, X, lda, Y, ldb );
		AxisHeadingAngle = atan(Y[0]);
		free(Y);
		free(X);
	}
	cosH = cos(AxisHeadingAngle);
	sinH = sin(AxisHeadingAngle);
	MVA_AxisHeading.update(AxisHeadingAngle);
	return;
};

void zebrafishPositionEstimate::computeProjPt(void) {
	// 1. Org - base
	cv::Point2d pt_o = TrgtPt - BasePt;
	// 2. rotate (-theta)
	cv::Point2d pt_o_rot(0, 0);
	pt_o_rot.x = cosH * pt_o.x + sinH * pt_o.y;
	// 3. rotate (theta) and get back to original position
	cv::Point2d pt_o_proj;
	ProjPt.x = cosH * pt_o_rot.x + BasePt.x;
	ProjPt.y = sinH * pt_o_rot.x + BasePt.y;

	AxisHeadingVector = ProjPt - BasePt;
	double dist =cv::norm(AxisHeadingVector);
	if (dist != 0)
		AxisHeadingVector = AxisHeadingVector * (1/dist);
};



void zebrafishPositionEstimate::computeAngularMomentum(void) {
	// compute moving average of difference in angular momentum
	int n = min((int)MVA_Heading.RawData.size(), WindowSize_AngMTh+1);
	double dtheta_sum = 0;
	for (int i = 1; i < n; i++) {
		// get angle difference b/t [-180, 180)
		double dth = MVA_Heading.RawData[i] - MVA_Heading.RawData[i - 1];
		if (dth < -180) dth += 360;
		if (dth >= 180) dth -= 360;
		dtheta_sum += dth*dth;
	}
	if (n > 1)
		angularMomentum = dtheta_sum / (n - 1);
	else
		angularMomentum = 0;
};

void zebrafishPositionEstimate::getWeight(void) {
	angMomentum_FrameCountAfterMoving = min(angMomentum_FrameCountAfterMoving + 1, (int)weight_angMomentum.size()-1);
    if (FrameCountStopEnabled) {
        if (angularMomentum > angMomentum_Threhold) {
            angMomentum_Threhold = angMomentum_minThrehold;
            angMomentum_FrameCountAfterMoving = 0;
            FrameCountStopEnabled = false;
		}
	}
	else {
        if (angularMomentum < angMomentum_Threhold) {
            angMomentum_Threhold = angMomentum_maxThrehold;
            FrameCountStopEnabled = true;
		}
	}
    weight = weight_angMomentum[angMomentum_FrameCountAfterMoving];
}
// -----------------PID ----------------------------------------------------
StagePID_SingleStage::StagePID_SingleStage(void)  {
	Pos = 0;
	Pos_desired = 0;
	Pos_desired_set = false;
	kp = 0;
	ki = 0;
	kd = 0;
	k1 = 0;
	k2 = 0;
	k3 = 0;
	err.push_back(0);
	err.push_back(0);
	err.push_back(0);
	Enable = false;
};
StagePID_SingleStage::~StagePID_SingleStage(void)  {
};

void StagePID_SingleStage::clear(void) {
	for (int i = 0; i < err.size(); i++)
		err[i] = 0;
	Pos_desired_set = false;
};
void StagePID_SingleStage::updatePIDparameter(double _kp, double _ki, double _kd) {
	kp = _kp;
	ki = _ki;
	kd = _kd;
	if (_kd == 0) {
		k1 = kp;
		k2 = 0;
		k3 = 0;
	}
	else {
		k1 = kp + ki + kd;
		k2 = -kp - 2*ki;
		k3 = kd;
	}
	if (kp == 0)
		Enable = false;

}
void StagePID_SingleStage::setDesiredPos(double _Pos_desired) {
	Pos_desired_set = true;
	Pos_desired = _Pos_desired;
}

double StagePID_SingleStage::getDesiredPos(void) {
	return Pos_desired;
}
void StagePID_SingleStage::setCurrentPos(double _pos) {
	Pos = _pos;
}

double StagePID_SingleStage::getCurrentPos(void) {
	return Pos;
}

double StagePID_SingleStage::computeNewInputPID(void) {
	if (Pos_desired_set) {
		err.insert(err.begin(), Pos_desired - Pos); err.pop_back();
		return k1*err[0] + k2*err[1]+ k3*err[2];
	}
	return 0;
}

double StagePID_SingleStage::computeNewInputPID_Round(void) {
	double pxDist = 0.015*2;
	if (Pos_desired_set) {
		double err_cur = Pos_desired - Pos;
		if (fabs(err_cur) <= pxDist) err_cur = 0;
		err.insert(err.begin(), err_cur); err.pop_back();
		return k1*err[0] + k2*err[1]+ k3*err[2];
	}
	return 0;
}

// ---------------------------------------------------------------------
// ------------- Stage Position Estimation class ------------------------
// ---------------------------------------------------------------------
StageMPC_SingleStage::StageMPC_SingleStage(void) :
	EstStep_Max(10),
	HstStep_Max(14),
	SetVelocityHist(14)
{
	w = NULL;
	int _contH = 3;
	int _predH = 6;
	updateIndexVariables(_predH, _contH);
	HstStep = HstStep_Max;
	EstStep = HstStep_Max;
	InitMat();
	for (int i = 0; i < HstStep_Max; i++)
		updateSetVelocity(0);
};
StageMPC_SingleStage::~StageMPC_SingleStage(void)  {
	if (w != NULL)
		free(w);
};

void StageMPC_SingleStage::clear(void) {
	for (int i = 0; i < SetVelocityHist.size(); i++)
		SetVelocityHist[i] = 0;
};


void StageMPC_SingleStage::InitMat(void)  {
	for (int i = 0; i <EstStep_Max; i++) {
		for (int j = 0;	j < EstStep_Max+HstStep_Max; j++) {
			M[i][j] = 0;
			A[j][i] = 0;
		}
		AX[i] = 0;
		dP[i] = 0;
		BY[i] = 0;
		X[i] = 0;
		dPmBY[i] = 0;
		dPacc[i] = 0;
		BYacc[i] = 0;
		dPmBYacc[i] = 0;
	}

	for (int j = 0;	j < HstStep_Max; j++) {
		Y[j] = 0;
		paraAcc[j] = 0;
	}
	Xmax = 50;
};

void StageMPC_SingleStage::updateIndexVariables(int _predH, int _contH) {
	contH = _contH;
	predH = _predH;
	if (w != NULL)
		free(w);
	w = (double *) malloc (predH * predH * sizeof(double));
	for (int i = 0; i < predH; i++) {
		for (int j = 0; j < predH; j++) {
			if (i == j) {
				w[i*predH+j] = 1;
				//w[i*predH+j] = 0.01;
				if (i == predH-2 || i == predH-3)
					w[i*predH+j] = 1;
			}
			else
				w[i*predH+j] = 0;
		}
	}
}

void StageMPC_SingleStage::updateSetVelocity(double setVel) {
	if(SetVelocityHist.size() == 0)
		SetVelocityHist.push_back(setVel);
	else
		SetVelocityHist.insert(SetVelocityHist.begin(), setVel);
	if (SetVelocityHist.size() > HstStep_Max)
		SetVelocityHist.pop_back();
}


vector<double> StageMPC_SingleStage::getMPCsetVel(vector<double> _dpacc, double _Xmax) {
	Xmax = _Xmax;
	vector<double> estsetVel(predH);
	// -- ||A*X - (dpacc - BYacc)||

	for (int i = 0; i < predH; i++)
		dPacc[i] = _dpacc[i];

	refreshY();
	computeBYAcc_MPC();

	// dp - BY
	for (int i = 0; i < predH; i++) {
		dPmBYacc[i] = dPacc[i] - BYacc[i];
	}

	// compute X (MPC)
	computeX();

	// copy variable for return;
	vector<double> estVelSet(contH);
	for (int i = 0; i < contH; i++)
		estVelSet[i] = X[i];

	return estVelSet;
}


void StageMPC_SingleStage::computeX(void) {
	int A_indX = (EstStep_Max) - (predH-1);
	int A_indY = (EstStep_Max + HstStep_Max - 1) - (predH);

	// allocate memory
	double * _A = (double *) malloc (contH * predH * sizeof(double));
	double * _Aw = (double *) malloc (contH * predH * sizeof(double));

	// copy variables
	for (int i = 0; i < predH; i++)
		for (int j = 0; j < contH; j++) {
			//_A[i*contH+j] = A[A_indY+i][A_indX+j];
			int ind = predH - contH - i + j;
			if (ind < 0)
				_A[i*contH+j] = 0;
			else
				_A[i*contH+j] = paraAcc[predH - contH - i + j];
			_Aw[i*contH+j] = 0;
		}
	// apply weight
	MKL_INT m = predH, n = contH, nrhs = 1, lda = n, ldb = 1, info;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, 1.0, w, m, _A, n, 0, _Aw, n);

	// allocate memory
	double * _B = (double *) malloc (predH * sizeof(double));
	for (int i = 0; i < predH; i++)
		_B[i] = dPmBYacc[i]*w[i*(predH+1)];
		//_B[i] = dPmBYacc[i];

	LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, _Aw, lda, _B, ldb );

	for (int i = 0; i < contH; i++)
		X[i] = _B[i];

	// free memory
	free(_A); free(_B); free(_Aw);
}


vector<double> StageMPC_SingleStage::estimateFuturePositions(double ref, vector<double> newSetVel) {
	EstStep = newSetVel.size();
	// A*X + B*Y = dp
	for (int i = 0; i < EstStep; i++)
		X[i] = newSetVel[i];
	computedP();
	vector<double> estPos(EstStep);
	for (int i = 0; i < EstStep; i++) {
		estPos[i] = ref + dPacc[i];
	}
	return estPos;;
}

double StageMPC_SingleStage::predictCurrentPositions(double ref_prev) {
	// AX + BY
	double AXpBY = 0;
	for (int i = 0; i < HstStep; i++)
		AXpBY += M[0][i]*SetVelocityHist[i];
	double CurPosN2 = ref_prev + AXpBY;
	return  CurPosN2;
}

void StageMPC_SingleStage::computedP(void) {
	computeAX();
	refreshY();
	computeBY();
	// AX + BY
	for (int i = 0; i < EstStep; i++) {
		dP[i] = AX[i] + BY[i];
	}
	// accumulation
	double acc = 0;
	for (int i = EstStep-1; i >= 0; i--) {
		dPacc[i] = dP[i] + acc;
		acc += dP[i];
	}
};


void StageMPC_SingleStage::computeAX(void) {
	for (int i = 0; i < EstStep; i++) {
		double output = 0;
		for (int j = 0; j < EstStep; j++) {
			output += M[i][j]*X[j]; // A*X
		}
		AX[i] = output;
	}
}

void StageMPC_SingleStage::computeBY(void) {
	for (int i = 0; i < EstStep; i++) {
		double output = 0;
		for (int j = 0; j < HstStep; j++) {
			output += M[i][EstStep+j]*Y[j]; // B*Y
		}
		BY[i] = output;
	}
	// accumulation
	double acc = 0;
	for (int i = EstStep-1; i >= 0; i--) {
		BYacc[i] = BY[i] + acc;
		acc += BY[i];
	}
}

void StageMPC_SingleStage::computeBYAcc_MPC(void) {
	for (int i = 0; i < EstStep; i++) {
		double output = 0;
		for (int j = 0; j < HstStep; j++) {
			output += M[i][predH+j]*Y[j]; // B*Y
		}
		BY[i] = output;
	}
	// accumulation
	double acc = 0;
	for (int i = 0; i < 10; i++) {
		BYacc[i] = 0;
	}
	for (int i = predH-1; i >= 0; i--) {
		BYacc[i] = BY[i] + acc;
		acc += BY[i];
	}
}




void StageMPC_SingleStage::refreshY(void) {
	for (int i = 0; i < HstStep; i++)
		Y[i] = SetVelocityHist[i];
}

void StageMPC_SingleStage::fillUpperMatrix(double * vM) {

	for (int i = 0; i <EstStep_Max; i++) {
		for (int j = 0;	j < HstStep_Max; j++) {
			M[i][i+j] = vM[j];
		}
	}
	double acc = 0;
	for (int i = 0; i <HstStep_Max; i++) {
		paraAcc[i] = vM[i] + acc;
		acc += vM[i];
	}
	for (int i = 0; i <EstStep_Max; i++) {
		for (int j = i;	j < HstStep_Max; j++) {
			A[j][i] = paraAcc[HstStep_Max-(j+1)];
		}
	}

}
// ------------------------------------------------

StageMPC::StageMPC(void) :
	EstStep_Max(10),
	HstStep_Max(14),
	PastPos(3)
{
	int _contH = 5;
	int _predH = 9;
	EstStep = EstStep_Max;
	HstStep = HstStep_Max;
	updateMPCparameters(EstStep_Max, HstStep_Max, 50, _predH, _contH);
	//M[10][23]
	// for 250Hz, Stage MPC (Working)
	//double temp_Mx[14] = {0.0000830866, 0.0008058807, 0.0009864153, 0.0007939951, 0.0004253637, 0.0003398310, 0.0002566965, 0.0001581667, 0.0000286151, 0.0000507616, 0.0000694375, -0.0000183974, 0.0000000000, 0.0000000000};
	//double temp_My[14] = {0.0000724406, 0.0007797038, 0.0008801793, 0.0008891519, 0.0005352894, 0.0002246686, 0.0001400520, 0.0001560481, 0.0001311033, 0.0000560353, 0.0000776077, 0.0000254470, 0.0000134825, 0.0000241594};
	// _contH=4;_predH=5;_MaxStageSpeed=75;dp = 0.1; weight = [1;1;1;1;1]
	//-75	37.1697	39.5285	-49.679		34.3637	-3.9486	-17.7006	10.7911	3.3875	-0.8663	-24.5608	38.5386	-14.8702	-23.7448	43.7102	-31.55650	1.70170	17.1402	-13.7727	3.3409
	//-75	24.7065	68.0932	-52.5305	-17.2109	38.3808	5.9061	-27.8392	-2.2455	27.8415	-25.6206	2.475	23.9456	-26.7027	1.508	19.93460	-8.83410	-13.9771	9.6953	14.8599
	// _contH=2;_predH=5;_MaxStageSpeed=75;dp = 0.1; weight = [1;1;1;1;1]
	//-75.00	47.77	6.19	-5.70	7.14	-5.75	-2.43	3.91	0.09	-1.76	-0.71	1.60	0.65	-1.92	1.41	-0.68	-0.17	0.68	-0.34	-0.16
	//-75.00	46.99	15.81	-23.44	6.81	9.22	-5.74	-0.43	1.61	0.24	-1.75	0.87	-4.57	7.34	-3.39	-1.99	3.11	-0.31	-1.69	1.18


	// for 250Hz, Image MPC
	//double temp_Mx[14] = {0, 0, 0.00000713020, 0.00062032740, 0.00161855540, 0.00067736900, 0.00009269260, 0.00043494220, 0.00052050460, -0.00010695300, -0.00000713020, 0.00021390600, 0.00006417180, -0.00019251540};
	//double temp_My[14] = {0, 0, 0.00004991140, 0.00051337440, 0.00124065480, 0.00101248840, 0.00038503080, 0.00027807780, 0.00019251540, 0.00008556240, 0.00019251540, 0.00007843220, -0.00002139060, -0.00001426040};
	// _contH=2;_predH=5;_MaxStageSpeed=75;dp = 0.1; weight = [1;1;1;1;1] (not working) ???
	//-63.61	39.89	-16.05	23.74	-2.27	-9.86	2.93	2.23	-1.90	-1.57	2.79	0.32	0.74	0.11	-4.90	3.81	-0.63	0.35	0.01	-0.72
	//-75.00	62.09	-18.33	7.31	-4.22	5.65	1.66	-4.91	-0.45	2.94	-4.90	1.13	8.60	-12.14	8.97	-5.14	3.52	-2.72	0.55	1.10
	// _contH=3;_predH=5;_MaxStageSpeed=75;dp = 0.1; weight = [1;1;1;1;1]
	//-74.05	70.06	-50.84	44.58	-6.44	-19.94	22.59	-18.08	14.11	-14.05	14.70	-14.89	12.64	-2.62	-9.58	16.04	-16.87	12.98	-6.18	0.60
	//-75.00	69.44	-37.12	27.73	-14.44	-1.19	21.80	-29.95	25.38	-23.13	22.27	-17.80	10.44	-4.47	0.77	3.23	-7.58	10.27	-10.84	10.37
	// _contH=4;_predH=9;_MaxStageSpeed=75;dp = 0.1; weight = [1;1;1;1;1]
	//-68.06	59.20	-44.07	42.11	-8.00	-14.45	15.55	-12.30	9.28	-6.04	-1.70	0.80	0.87	4.13	-4.40	2.82	-3.39	1.98	0.72	-1.89
	//-75.00	66.20	-29.91	24.01	-18.86	9.61	7.21	-12.11	3.42	1.26	-6.83	15.06	-20.60	21.40	-17.94	13.46	-8.39	2.75	1.12	-2.28

	// for 120Hz, Stage MPC
	//double temp_Mx[14] = {0.0011085250,0.0036431720,0.0019771253,0.0008768403,0.0002990879,0.0000794769,0.0000332976,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000};
	//double temp_My[14] = {0.0010918617,0.0037464493,0.0016940932,0.0008053296,0.0004632465,0.0001900920,0.0000598710,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000};

	double temp_Mx[14] = {0.000002576, -0.000001507, 0.000068454, 0.000788942, 0.001049860, 0.000782650, 0.000339859, 0.000396923, 0.000267465, 0.000146165, -0.000005602, 0.000084592, 0.000064182, -0.000039931};
	double temp_My[14] = {-0.000002670, -0.000002615, 0.000084095, 0.000789149, 0.000946882, 0.000934889, 0.000447619, 0.000213409, 0.000142344, 0.000169982, 0.000100179, 0.000079036, 0.000057867, 0.000010451};


	StageX.fillUpperMatrix(temp_Mx);
	StageY.fillUpperMatrix(temp_My);
	MPCEnabled = false;

};
StageMPC::~StageMPC(void)  {
};

void StageMPC::clear(void) {
	StageX.clear();
	StageY.clear();
	PastPos.clear();
};

void StageMPC::updateMPCparameters(int _EstStep, int _HstStep, int _Xmax, int _predH, int _contH) {
	if (_EstStep > EstStep_Max) _EstStep = EstStep_Max;
	if (_HstStep > HstStep_Max) _HstStep = HstStep_Max;

	predH = _predH;
	contH = _contH;
	EstStep = _EstStep;
	HstStep = _HstStep;
	Xmax = _Xmax;

	StageX.EstStep = EstStep;
	StageX.HstStep = HstStep;
	StageX.Xmax = Xmax;
	StageX.updateIndexVariables(_predH, _contH);

	StageY.EstStep = EstStep;
	StageY.HstStep = HstStep;
	StageY.Xmax = Xmax;
	StageY.updateIndexVariables(_predH, _contH);
}


void StageMPC::updateSetVelocity(cv::Point2d setVel) {
	StageX.updateSetVelocity(setVel.x);
	StageY.updateSetVelocity(setVel.y);
}
vector<cv::Point2d> StageMPC::estimateFuturePositions(cv::Point2d ref, vector<cv::Point2d> newSetVel){
	EstStep = newSetVel.size();
	vector<double> newSetVel_x(EstStep);
	vector<double> newSetVel_y(EstStep);
	for (int i =0; i<EstStep; i++) {
		newSetVel_x[i] = newSetVel[i].x;
		newSetVel_y[i] = newSetVel[i].y;
	}

	vector<double> estPosX = StageX.estimateFuturePositions(ref.x, newSetVel_x);
	vector<double> estPosY = StageY.estimateFuturePositions(ref.y, newSetVel_y);

	vector<cv::Point2d> estPos(EstStep);
	for (int i =0; i<EstStep; i++) {
		estPos[i].x = estPosX[i];
		estPos[i].y = estPosY[i];
	}
	return estPos;
}
cv::Point2d StageMPC::predictCurrentPositions(cv::Point2d ref_prev) {
	double Xcur = StageX.predictCurrentPositions(ref_prev.x);
	double Ycur = StageY.predictCurrentPositions(ref_prev.y);
	return cv::Point2d(Xcur, Ycur);
}
vector<cv::Point2d> StageMPC::getMPCsetVel(vector<cv::Point2d> _dpacc, double _Vmax){
	Xmax = _Vmax;
	vector<double> dp_x(predH);
	vector<double> dp_y(predH);
	for (int i =0; i<predH; i++) {
		dp_x[i] = _dpacc[i].x;
		dp_y[i] = _dpacc[i].y;
	}

	vector<double> setVelX = StageX.getMPCsetVel(dp_x, _Vmax);
	vector<double> setVelY = StageY.getMPCsetVel(dp_y, _Vmax);

	vector<cv::Point2d> setVel(contH);
	for (int i =0; i<contH; i++) {
		setVel[i].x = setVelX[i];
		setVel[i].y = setVelY[i];
	}
	return setVel;
}

cv::Point2d StageMPC::getMPCsetVel_Next(vector<cv::Point2d> _MPCVelSet, double _Xmax) {
	if (_MPCVelSet.size() == 0)
		return cv::Point2d(0,0);

	int i = _MPCVelSet.size()-1;
	cv::Point2d output = _MPCVelSet[i];
	if (output.x > _Xmax)	output.x = _Xmax;
	if (output.x < -_Xmax)	output.x = -_Xmax;
	if (output.y > _Xmax)	output.y = _Xmax;
	if (output.y < -_Xmax)	output.y = -_Xmax;
	return output;
}

vector<cv::Point2d> StageMPC::getMPCsetVel_FromPos(cv::Point2d ref, vector<cv::Point2d> _p, double _Xmax) {
	vector<cv::Point2d> setVel;
	if (_p.size() > 0) {
		vector<cv::Point2d> _dpacc(_p.size());
		for (int i = 0; i < _p.size(); i++)
			_dpacc[i] = _p[i] -  ref;
		setVel = getMPCsetVel(_dpacc, _Xmax);
	}
	return setVel;
}

cv::Point2d StageMPC::computePIDCorrectionVelocity(void) {

	cv::Point2d CorrVel;
	CorrVel.x = StageX.PIDCntr.computeNewInputPID();
	CorrVel.y = StageY.PIDCntr.computeNewInputPID();
	return CorrVel;
}

cv::Point2d StageMPC::computePIDCorrectionVelocity(cv::Point2d CurrentPos, cv::Point2d DesiredPos) {
	StageX.PIDCntr.setCurrentPos(CurrentPos.x);
	StageY.PIDCntr.setCurrentPos(CurrentPos.y);
	StageX.PIDCntr.setDesiredPos(DesiredPos.x);
	StageY.PIDCntr.setDesiredPos(DesiredPos.y);
	return computePIDCorrectionVelocity();
}


cv::Point2d StageMPC::computePIDCorrectionVelocity_Round(void) {
	cv::Point2d CorrVel;
	CorrVel.x = StageX.PIDCntr.computeNewInputPID_Round();
	CorrVel.y = StageY.PIDCntr.computeNewInputPID_Round();
	return CorrVel;
}

cv::Point2d StageMPC::computePIDCorrectionVelocity_Round(cv::Point2d CurrentPos, cv::Point2d DesiredPos) {
	StageX.PIDCntr.setCurrentPos(CurrentPos.x);
	StageY.PIDCntr.setCurrentPos(CurrentPos.y);
	StageX.PIDCntr.setDesiredPos(DesiredPos.x);
	StageY.PIDCntr.setDesiredPos(DesiredPos.y);
	return computePIDCorrectionVelocity_Round();
}
// --------------------------------------------------------------------------------
MPCwImpRsp::MPCwImpRsp(void)
{
	w = NULL;
	Aw = NULL;
	B = NULL;
	u = NULL;
	ImpRsp = NULL;
	CntrHrz = 0;
	PredHrz = 0;
	PredHrz_Max = 0;
	stepRsp = NULL;
	A = NULL;
	AtP = NULL;
	AtPApGtG = NULL;
	gamma_TR = NULL;
};
MPCwImpRsp::~MPCwImpRsp(void)  {
	if (w) free(w);
	if (Aw) free(Aw);
	if (A) free(A);
	if (B) free(B);
	if (u) free(u);
	if (ImpRsp) free(ImpRsp);
	if (stepRsp) free(stepRsp);
	if (AtP) free(AtP);
	if (AtPApGtG) free(AtPApGtG);
	if (gamma_TR) free(gamma_TR);
};
// 1. update step response (impulse response)
// 2. set weight, the control, predict horizon
// 4. prepare matrix for MPC
// 5. MPC control

void MPCwImpRsp::updateImpulseResponse(unsigned int n, double * _ImpRsp) {
	if (n <= 0)
		return;
	PredHrz_Max = n;

	// update impulse response
	if (ImpRsp) free(ImpRsp);
	ImpRsp = (double *) malloc (PredHrz_Max * sizeof(double));
	for (int i = 0; i < n ; i++)
		ImpRsp[i] = _ImpRsp[i];

	// step impulse response
	if (stepRsp) free(stepRsp);
	stepRsp = (double *) malloc (n * sizeof(double));
	stepRsp[0] = ImpRsp[0];
	for (int i = 1; i < n ; i++)
		stepRsp[i] = ImpRsp[i] + stepRsp[i-1];
}

void MPCwImpRsp::updateControlParameters(unsigned int _CntrHrz, unsigned int _PredHrz,  double * _w, double * _gamma) {
	if (!stepRsp)
		return;
	if (PredHrz_Max < _CntrHrz || PredHrz_Max < _PredHrz || _PredHrz < _CntrHrz )
		return;

	if (CntrHrz != _CntrHrz || PredHrz != _PredHrz) {
		CntrHrz = _CntrHrz;
		PredHrz = _PredHrz;
		setWeight(PredHrz, _w);
		setgamma(CntrHrz, _gamma);
		generateMat();
	}
}
void MPCwImpRsp::setWeight(unsigned int n, double * _w) {
	if (w) free(w);
	w = (double *) malloc (n * sizeof(double));
	if (!_w) {
		for (int i = 0; i < n ; i++) w[i] = 1;
	}
	else {
		for (int i = 0; i < n ; i++) w[i] = _w[i];
	}
}

void MPCwImpRsp::setgamma(unsigned int n, double * _gamma) {
	if (gamma_TR) free(gamma_TR);
	gamma_TR = (double *) malloc (n * sizeof(double));
	if (!_gamma) {
		for (int i = 0; i < n ; i++) gamma_TR[i] = 0.001;
	}
	else {
		for (int i = 0; i < n ; i++) gamma_TR[i] = _gamma[i];
	}
}

void MPCwImpRsp::generateMat(void) {
	MKL_INT m, n, k, nrhs, lda, ldb, ldc, info;
	double alpha, beta;
	// prepare weight

	if (!w) return;
	if (!gamma_TR) return;
	double * wdiag = (double *) malloc (PredHrz * PredHrz * sizeof(double));
	double * P = (double *) malloc (PredHrz * PredHrz * sizeof(double)); // wdiag * wdiag
	for (int i = 0; i < PredHrz; i++) {
		for (int j = 0; j < PredHrz; j++) {
			if (i == j) {
				wdiag[i*PredHrz+j] = w[i];
				P[i*PredHrz+j] = w[i]*w[i];
			}
			else {
				wdiag[i*PredHrz+j] = 0;
				P[i*PredHrz+j] = 0;
			}
		}
	}

	// Prepare A (weighted)
	if (A) free(A);
	if (Aw) free(Aw);
	A = (double *) malloc (CntrHrz * PredHrz * sizeof(double));
	Aw = (double *) malloc (CntrHrz * PredHrz * sizeof(double));

	// copy variables
	for (int j = 0; j < CntrHrz; j++) {
		for (int i = 0; i < PredHrz; i++) {
			int ind = i - j;
			if (ind < 0)
				A[i*CntrHrz+j] = 0;
			else
				A[i*CntrHrz+j] = stepRsp[ind];
			Aw[i*CntrHrz+j] = 0;
		}
	}
	// get w x A
	m = PredHrz; k = PredHrz; n = CntrHrz; lda = k; ldb = n; ldc = n; alpha = 1.0; beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, wdiag, lda, A, ldb, beta, Aw, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C

	// getAtP
	if (AtP) free(AtP);
	AtP = (double *) malloc (CntrHrz * PredHrz * sizeof(double));
	m = CntrHrz; k = PredHrz; n = PredHrz; nrhs = 1; lda = m; ldb = n; ldc = n; alpha = 1.0; beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, A, lda, P, ldb, beta, AtP, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C


	// Gamma & GtG (== gamma2)
	double * gdiag = (double *) malloc (CntrHrz * CntrHrz * sizeof(double));
	if (AtPApGtG) free(AtPApGtG); // temperally save GtG
	AtPApGtG = (double *) malloc (CntrHrz * CntrHrz * sizeof(double));
	for (int i = 0; i < CntrHrz; i++) {
		for (int j = 0; j < CntrHrz; j++) {
			if (i == j) {
				gdiag[i*CntrHrz+j] = gamma_TR[i];
				AtPApGtG[i*CntrHrz+j] = gamma_TR[i]*gamma_TR[i];
			}
			else {
				gdiag[i*CntrHrz+j] = 0;
				AtPApGtG[i*CntrHrz+j] = 0;
			}
		}
	}

	// AtPApGtG
	m = CntrHrz; k = PredHrz; n = CntrHrz; nrhs = 1; lda = k; ldb = n; ldc = n; alpha = 1.0; beta = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, AtP, lda, A, ldb, beta, AtPApGtG, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C


	// Prepare B & accumulation of B
	if (B) free(B);
	B = (double *) malloc ((PredHrz_Max - 1) * PredHrz * sizeof(double));
	for (int i = 0; i < PredHrz; i++) {
		for (int j = 0; j < (PredHrz_Max - 1); j++) {
			int id1 = min(PredHrz_Max-1-j+i, PredHrz_Max-1);
			int id2 = (PredHrz_Max - 2 - j);
			B[i*(PredHrz_Max-1)+j] = stepRsp[id1] - stepRsp[id2];
		}
	}


	// prepare u & uNext
	if (u) free(u);
	u = (double *) malloc (PredHrz_Max* sizeof(double));
	for (int i = 0; i < PredHrz_Max; i++) {
		u[i] = 0;
	}

	free(gdiag);
	free(wdiag);
	free(P);
}

void MPCwImpRsp::MPC(double * dpacc) {
	// dpacc - B*u
	MKL_INT m, n, k, nrhs, lda, ldb, ldc, info; double alpha, beta;
	m = PredHrz; k = PredHrz_Max-1; n = 1; lda = k; ldb = n; ldc = n; alpha = -1.0; beta = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, B, lda, u, ldb, beta, dpacc, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C

	// apply weight
	for (int i = 0; i < PredHrz; i++)
		dpacc[i] = dpacc[i]*w[i];

	// get U
	double * tempBuf = (double *) malloc (CntrHrz * PredHrz * sizeof(double));
	memcpy(tempBuf, AtPApGtG, (CntrHrz * PredHrz)*sizeof(double));
	m = CntrHrz; n = PredHrz; nrhs = 1; lda = n; ldb = nrhs;
	LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, Aw, lda, dpacc, ldb );
	// (m x n) / (m x nrhs), lda = n, ldb = nrhs;
	free(tempBuf);

	return;
}

void MPCwImpRsp::MPC_Rg(double * dpacc) {
	MKL_INT m, n, k, nrhs, lda, ldb, ldc, info; double alpha, beta;

	// dpacc - B*u
	m = PredHrz; k = PredHrz_Max-1; n = 1; lda = k; ldb = n; ldc = n; alpha = -1.0; beta = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, B, lda, u, ldb, beta, dpacc, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C

	// AtP x (dpacc - B*u)
	m = CntrHrz; k =PredHrz; n = 1; lda = k; ldb = n; ldc = n; alpha = 1.0; beta = 0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, AtP, lda, dpacc, ldb, beta, dpacc, ldc);
	//alpha*(m x k) x (k x n) + beta*(m x n) --> C

	// get U
	double * tempBuf = (double *) malloc (CntrHrz * PredHrz * sizeof(double));
	memcpy(tempBuf, AtPApGtG, (CntrHrz * PredHrz)*sizeof(double));
	m = CntrHrz; n = CntrHrz; nrhs = 1; lda = n; ldb = nrhs;
	LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, tempBuf, lda, dpacc, ldb );
	free(tempBuf);
	// (m x n) / (m x nrhs), lda = n, ldb = nrhs;

	return;
}

double MPCwImpRsp::PredictNextPos(double _OrgP, double _input) {

	double _dpsum = ImpRsp[0] * _input;
	for (int i = 1; i < PredHrz_Max - 1; i++)
		_dpsum  += ImpRsp[i] * u[PredHrz_Max - 1 - i];

	return _OrgP + _dpsum ;
}

double MPCwImpRsp::PredictCurPos(double _OrgP) {
	if (u == NULL)
		return _OrgP;
	double _dpsum = 0;
	for (int i = 0; i < PredHrz_Max - 2; i++)
		_dpsum  += ImpRsp[i] * u[PredHrz_Max - 2 - i];
	return _OrgP + _dpsum ;
}

double MPCwImpRsp::UpdateNextPos(double _OrgP, double _input) {
	// shift input
	UpdateNextInput(_input);
	return PredictCurPos(_OrgP) ;
}
double  MPCwImpRsp::ReplaceNextPos(double _OrgP, double _input){
	// shift input
	ReplaceNextInput(_input);
	return PredictCurPos(_OrgP);
}

void MPCwImpRsp::UpdateNextInput(double _input) {
	// shift input
	double * utemp = (double *) malloc ( (PredHrz_Max)* sizeof(double));
	memcpy(utemp, &u[1], (PredHrz_Max-2)*sizeof(double));
	utemp[PredHrz_Max - 2] = _input;
	free(u); u = utemp;
}

void MPCwImpRsp::ReplaceNextInput(double _input) {
	// shift input
	u[PredHrz_Max - 2] = _input;
}
vector<double> MPCwImpRsp::PredictFuturePos(double _OrgP, vector<double> input) {
	int _steps = input.size();
	double * output = (double *) malloc ( (2*PredHrz_Max - 1)* sizeof(double));
	double * utemp = (double *) malloc ( PredHrz_Max* sizeof(double));
	for (int i = 0; i < PredHrz_Max - _steps; i++)
		utemp[i] = u[i+_steps];

	for (int i = 0; i < _steps; i++)
		utemp[PredHrz_Max - _steps + i] = input[i]; // [0] n, n+1, n+2 , ...

	ippsConv_32f((Ipp32f *)ImpRsp, PredHrz_Max, (Ipp32f *)utemp, PredHrz_Max, (Ipp32f *)output);

	vector<double> FuturePos(_steps);
	for (int i = 0; i < _steps; i++) {
		_OrgP += output[PredHrz_Max + i]; // [0] n, n+1, n+2 , ...
		FuturePos[i] = _OrgP;
	}
	free(output); free(utemp); free(output);
	return FuturePos;
}
void MPCwImpRsp::clear(void) {
	for (int i = 0; i < PredHrz_Max - 1; i++)
		u[i] = 0;
}


// --------------------------------------------------------------------------------
MPCwImpRspSet::MPCwImpRspSet(void)
{
	MaxVel = 75;
	MPCEnabled = false;
	// 250 Hz, Chamber position Response function
	//double h_x[] = {0, 0, 0.000008024950465, 0.000611679397619, 0.001640294321462, 0.000583638793391, 0.000018272218970, 0.000608266053076, 0.000551293257897, -0.000232306301117, -0.000059417783715, 0.000309225687867, 0.000098649261892, -0.000182742796501, -0.000038074657090, 0.000122824587006, 0.000057453440774, -0.000070337309294, -0.000120384241972, 0.000095259578425, 0.000154604301177, -0.000135509559939, -0.000129637417150, 0.000091060374716, 0.000127007878089, -0.000014076462844, -0.000208739169545, 0.000017533161735, 0.000244626730510, -0.000049437490116, -0.000211278209622, 0.000031089287157, 0.000192727096783, -0.000026717448620, -0.000200594282920, 0.000046876255418, 0.000196464877831, -0.000095115417392, -0.000164870927843, 0.000104812579495, 0.000151437390497, -0.000124797103485, -0.000124010125846, 0.000127696297653, 0.000076869729839, -0.000102599828849, -0.000054793065178, 0.000062444232838, 0.000048688788449, -0.000047179731837, -0.000039239305465, 0.000031733343361, 0.000016270291076, -0.000018666416689, -0.000006559175197, -0.000010917604083, 0.000031309135667, 0.000011151171447, -0.000065602672004, -0.000001880954104, 0.000077977965487, -0.000019628211540, -0.000077566652827, 0.000030594531413, 0.000062573970972, -0.000046427913088, -0.000037275498406, 0.000049715949293, 0.000008223516444, -0.000049105771044, 0.000005922022710, 0.000054185311014, -0.000023341784747, -0.000076681220200, 0.000047398556139, 0.000059584762285, -0.000069289582333, -0.000026035188231, 0.000046477203311, -0.000000958273758, -0.000019733775011, 0.000009859449663, -0.000000619902471, -0.000015210295986, 0.000009626519775, 0.000006192718266, -0.000012348756724, -0.000003413244152, 0.000003719464768, -0.000010919437100, -0.000006430290163, 0.000018327940149, 0.000007432051846, -0.000044116797988, -0.000000544806156, 0.000057786816329, -0.000031147578261, -0.000046219002342, 0.000046454827262, 0.000031504648113, -0.000061099550263, -0.000005412537302};
	//double h_y[] = {0, 0, 0.000031986070896, 0.000605198650519, 0.001266174547842, 0.000921984049687, 0.000381165587521, 0.000216906762170, 0.000220710144235, 0.000161249037867, 0.000147272675770, 0.000108336714366, 0.000004693987535, -0.000010251336836, -0.000029086821265, -0.000027632868875, 0.000000335756714, -0.000015107450375, -0.000025735335677, 0.000016087281999, 0.000040715217887, 0.000038868202669, 0.000010550146690, 0.000009767776354, 0.000023721964659, -0.000002777172534, -0.000044900253732, -0.000026777686526, -0.000002475772550, -0.000024402900523, -0.000030304149032, -0.000000453610613, 0.000035437608699, 0.000030143462811, 0.000003042296616, 0.000017027806387, 0.000022513670718, 0.000020761623667, -0.000004970800635, -0.000020218072603, -0.000023993087242, -0.000021499939315, -0.000014310606574, -0.000008803937894, 0.000004457639114, -0.000000919365075, 0.000007366181458, 0.000034335269405, 0.000038237546248, 0.000011755270208, -0.000011914357289, 0.000004113989009, 0.000004685480503, -0.000006374035963, -0.000031221041780, -0.000016880371563, -0.000007650402103, 0.000008132660384, 0.000003850443739, 0.000009821775279, 0.000010530631136, 0.000019056826797, 0.000016301814749, 0.000017022641072, -0.000002166210089, -0.000009181756500, -0.000003513560702, -0.000007456791917, -0.000010083706581, -0.000005183983137, -0.000001137650227, -0.000000507161765, 0.000012233126240, 0.000016286579812, 0.000006135908042, 0.000012244981280, 0.000007510030523, -0.000005822099383, -0.000000992988695, 0.000005340655045, -0.000005063289211, -0.000015577981540, -0.000021228575907, 0.000005345764499, 0.000014867955545, -0.000002708924991, 0.000004941751292, 0.000008402042422, 0.000013984031533, 0.000007163567762, 0.000009908653972, -0.000005633240952, -0.000013372172955, -0.000000800114828, 0.000000523095573, -0.000002205526848, -0.000009436088537, -0.000006684662093, 0.000018298419478, 0.000012662674314, -0.000000464813694, 0.000004298578086};
	// 250Hz, Chamber position (from model)
	//double h_x[] = {0.000000000000000, 0.000000000000000, 0.000611209364359, 0.001637524167179, 0.000587160512301, 0.000029259907089, 0.000612802829218, 0.000531150515698, -0.000247706486066, -0.000064055476807, 0.000330241370175, 0.000132656212924, -0.000198891566156, -0.000065138659511, 0.000129051595732, 0.000090514923636, -0.000081926891055, -0.000109966437647, 0.000040325821237, 0.000134264656347, -0.000029538118951, -0.000158029443757, 0.000021312639093, 0.000174109371052, -0.000028127926381, -0.000185238002470, 0.000036924770275, 0.000188479039330, -0.000050105202251, -0.000186075892212, 0.000063030377281, 0.000177744343277, -0.000074832888666, -0.000165673941164, 0.000084495232250, 0.000150246449300, -0.000090816865669, -0.000133672240073, 0.000094421711984, 0.000116126031718, -0.000094477977754, -0.000099574173342, 0.000092375669163, 0.000083702974977, -0.000087566639676, -0.000070073768995, 0.000081713090372, 0.000057883570162, -0.000074311709344, -0.000048308332092, 0.000066999553008, 0.000040185573601, -0.000059140945181, -0.000034433002940, 0.000052201286454, 0.000029696623268, -0.000045341314223, -0.000026785655131, 0.000039823211993, 0.000024301072000, -0.000034618258967, -0.000023061750586, 0.000030824808014, 0.000021719404359, -0.000027284820666, -0.000021168595272, 0.000025000684625, 0.000020153107675, -0.000022755171512, -0.000019660730177, 0.000021519693522, 0.000018526568044, -0.000020074464915, -0.000017816490528, 0.000019405241255, 0.000016432495571, -0.000018323112236, -0.000015488817138, 0.000017849793528, 0.000013923136245, -0.000016838328746, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000};

	// 250Hz, Chamber position (New Stage Cable Assembly)
	double h_x[] = {0.000000000000000,0.000000000000000,0.000110473932014,0.001443428352914,0.001796400355429,0.000702766171322,0.000191001422077,-0.000138078040320,-0.000128508430406,-0.000051318348194,-0.000042774207425,0.000054710198275,-0.000018291682691,0.000060811970377,-0.000043362429537,0.000014073301506,-0.000066509777494,0.000015492435888,-0.000019170878164,0.000034021865486,-0.000011846298998,-0.000024901328528,-0.000001972422537,0.000019695900129,0.000008483913038,0.000000054914342,0.000009397536354,-0.000016420500903,0.000025854665388,-0.000018123016144,0.000019266144120,-0.000036087455220,0.000008853078764,-0.000028422125091,-0.000014408845011,-0.000034912211439,-0.000016017985580,0.000003846231899,-0.000020545606965,-0.000019324931689,0.000003146834474,0.000010959203257,-0.000012849182421,-0.000007147857525,-0.000022594456934,-0.000026270764938,-0.000019143820140,-0.000003634230166,0.000003117701450,-0.000006271432060,0.000001001411312,-0.000017125958107,0.000025490971265,0.000000000000000,0.000000000000000,0.000000000000000,0.000000000000000,0.000000000000000};
	double h_y[] = {0.000000000000000,0.000000000000000,0.000077656723327,0.001194047254689,0.001653801718327,0.000738258167821,0.000298647255297,0.000111131850771,-0.000058672954238,0.000037317292837,-0.000056806929556,0.000019945222660,-0.000019004250743,-0.000053259189011,-0.000042493700311,-0.000083716597640,-0.000051254119186,-0.000076215525649,-0.000074536821318,0.000006701245364,-0.000027306456890,-0.000030606253600,0.000034460536865,-0.000039873919374,0.000052009875808,-0.000063238396327,0.000028255327199,-0.000050958443169,-0.000071584082939,-0.000025276117079,-0.000050198630149,0.000038545384412,-0.000054256361414,0.000009677205943,-0.000003831371524,-0.000016395937212,-0.000002655892133,-0.000020359067571,-0.000035642659809,-0.000025121971109,-0.000028452151966,-0.000003991992157,-0.000021002844815,0.000001418014601,0.000017954081159,-0.000021673143591,0.000007185407701,-0.000046828294587,0.000012214872995,-0.000009923284467,-0.000027576401847,0.000044803002542,-0.000024405696279,0.000000000000000,0.000000000000000,0.000000000000000,0.000000000000000,0.000000000000000};


	unsigned int n = 102;
	unsigned int H_pred=7, C_pred = 4;
	double w[] = {0.001, 0.001, 0.001, 1, 1, 1, 2};
	double g[] = {0.003, 0.003, 0.003, 0.003};
	X.updateImpulseResponse(n, h_x);
	X.updateControlParameters(C_pred, H_pred, &w[0], &g[0]);
	Y.updateImpulseResponse(n, h_y);
	Y.updateControlParameters(C_pred, H_pred, &w[0], &g[0]);
};
MPCwImpRspSet::~MPCwImpRspSet(void)
{
};

cv::Point2d MPCwImpRspSet::MPC(vector<cv::Point2d> pt, cv::Point2d p) {
	double * dpx = (double *) malloc ( pt.size()* sizeof(double));
	double * dpy = (double *) malloc ( pt.size()* sizeof(double));
	computedp(pt, p, dpx, dpy);
	X.MPC(dpx);
	Y.MPC(dpy);
	if (dpx[0] > MaxVel) dpx[0] = MaxVel;	if (dpx[0] < -MaxVel) dpx[0] = -MaxVel;
	if (dpy[0] > MaxVel) dpy[0] = MaxVel;	if (dpy[0] < -MaxVel) dpy[0] = -MaxVel;
	cv::Point2d setVel(dpx[0], dpy[0]);
	free(dpx); free(dpy);
	return setVel;
}

cv::Point2d MPCwImpRspSet::MPC_Rg(vector<cv::Point2d> pt, cv::Point2d p) {
	double * dpx = (double *) malloc ( pt.size()* sizeof(double));
	double * dpy = (double *) malloc ( pt.size()* sizeof(double));
	computedp(pt, p, dpx, dpy);
	X.MPC_Rg(dpx);
	Y.MPC_Rg(dpy);
	if (dpx[0] > MaxVel) dpx[0] = MaxVel;	if (dpx[0] < -MaxVel) dpx[0] = -MaxVel;
	if (dpy[0] > MaxVel) dpy[0] = MaxVel;	if (dpy[0] < -MaxVel) dpy[0] = -MaxVel;
	cv::Point2d setVel(dpx[0], dpy[0]);
	free(dpx); free(dpy);
	return setVel;
}

cv::Point2d MPCwImpRspSet::MPC(vector<cv::Point2d> pt, cv::Point2d p, double _MaxVel) {
	MaxVel = _MaxVel;
	return MPC(pt, p);
}
cv::Point2d MPCwImpRspSet::MPC(vector<cv::Point2d> dp, double _MaxVel) {
	MaxVel = _MaxVel;
	return MPC(dp, cv::Point(0,0));
}
cv::Point2d MPCwImpRspSet::MPC(vector<cv::Point2d> dp) {
	return MPC(dp, cv::Point(0,0));
}

cv::Point2d MPCwImpRspSet::MPC_Rg(vector<cv::Point2d> pt, cv::Point2d p, double _MaxVel) {
	MaxVel = _MaxVel;
	return MPC_Rg(pt, p);
}
cv::Point2d MPCwImpRspSet::MPC_Rg(vector<cv::Point2d> dp, double _MaxVel) {
	MaxVel = _MaxVel;
	return MPC_Rg(dp, cv::Point(0,0));
}

cv::Point2d MPCwImpRspSet::MPC_Rg(vector<cv::Point2d> dp) {
	return MPC_Rg(dp, cv::Point(0,0));
}

cv::Point2d MPCwImpRspSet::PredictNextPos(cv::Point2d _OrgP, cv::Point2d _input) {
	return cv::Point2d(X.PredictNextPos(_OrgP.x, _input.x), Y.PredictNextPos(_OrgP.y, _input.y));
}

cv::Point2d MPCwImpRspSet::UpdateNextPos(cv::Point2d _OrgP, cv::Point2d _input) {
	return cv::Point2d(X.UpdateNextPos(_OrgP.x, _input.x), Y.UpdateNextPos(_OrgP.y, _input.y));
}

cv::Point2d MPCwImpRspSet::ReplaceNextPos(cv::Point2d _OrgP, cv::Point2d _input) {
	return cv::Point2d(X.ReplaceNextPos(_OrgP.x, _input.x), Y.ReplaceNextPos(_OrgP.y, _input.y));
}

vector<cv::Point2d> MPCwImpRspSet::PredictFuturePos(cv::Point2d _OrgP, vector<cv::Point2d> input) {
	vector<double> inputX;
	vector<double> inputY;
	for (int i = 0; i < input.size(); i++) {
		inputX.push_back(input[i].x);
		inputY.push_back(input[i].y);
	}
	vector<double> px = X.PredictFuturePos(_OrgP.x, inputX);
	vector<double> py = X.PredictFuturePos(_OrgP.y, inputY);
	vector<cv::Point2d> p;
	for (int i = 0; i < input.size(); i++)
		p.push_back(cv::Point2d(px[i], py[i]));
	return p;
}


void MPCwImpRspSet::computedp(vector<cv::Point2d> pt, cv::Point2d p, double * dpx, double * dpy) {
	for (int i = 0; i < pt.size(); i++) {
		dpx[i] = pt[i].x - p.x;
		dpy[i] = pt[i].y - p.y;
	}
}


cv::Point2d MPCwImpRspSet::PredictCurPos(cv::Point2d _OrgP) {
	return cv::Point2d(X.PredictCurPos(_OrgP.x), Y.PredictCurPos(_OrgP.y));
}


void MPCwImpRspSet::clear(void) {
	X.clear();
	Y.clear();
	MPCEnabled = true;
}

// --------------------------------------------------------------------------------
DConvImpRsp::DConvImpRsp(void)
{
	predh_size = 0;
	h = NULL;
	input = vector<double>(0);
	output = 0;
};
DConvImpRsp::~DConvImpRsp(void)  {
	if (h)
		free(h);
};


double DConvImpRsp::getOutput(void) {
	return output;
}


double DConvImpRsp::updateNewInput(double _input) {
	input.insert(input.begin(), _input);
	input.pop_back();
	convInputImpRsp();
	return output;
}


void DConvImpRsp::init(void) {
	if (h) {
		input.clear();
		input = vector<double>(predh_size);
		for (int i = 0; i< predh_size; i++)
			input[i] = 0;
	}
}


void DConvImpRsp::setImpRsp(int n, double * _h) {
	if (h)
		free(h);
	predh_size = n;
	h = (double *) malloc (predh_size * sizeof(double));
	for (int i = 0; i < predh_size; i++) {
		h[i] = _h[i];
	}
}


void DConvImpRsp::setImpRsp(vector<double> _h) {
	if (h)
		free(h);
	predh_size = _h.size();
	h = (double *) malloc (predh_size * sizeof(double));
	for (int i = 0; i < predh_size; i++) {
		h[i] = _h[i];
	}
}

void DConvImpRsp::convInputImpRsp(void) {
	double _sum = 0;
	if (h) {
		for (int i = 0; i < predh_size; i++)
			_sum += h[i]*input[i];
	}
	output = _sum;
}
// ------- Zebrafish Referenece
zebrafishInfoReference::zebrafishInfoReference(void){
	double PxDist = 14.03; // / 14.03 px/mm
	_SizeHist = 50;
	mean_AngleLYR = 26*PI/180; // 26 degree default
	mean_dEyes =  1150/PxDist; // 1.14mm defualt
	mean_dEyesYolk = 2350/PxDist;	 // 2.35mm
	RangePercent_AngleLYR = 0.35;
	RangePercent_dEyes = 0.33;
	RangePercent_dEyesYolk = 0.25;

	for (int i = 0; i < _SizeHist; i++) {
		AngleLYR.push_back(mean_AngleLYR);
		dEyes.push_back(mean_dEyes);
		dEyesYolk.push_back(mean_dEyesYolk);
	}

};

zebrafishInfoReference::~zebrafishInfoReference(){
};

void zebrafishInfoReference::getMeanValue(void) {
	double Sum_Angle_LYR =accumulate(AngleLYR.begin(),AngleLYR.end(),0);
	double Sum_d_Eyes =accumulate(dEyes.begin(),dEyes.end(),0);
	double Sum_d_EyesYolk =accumulate(dEyesYolk.begin(),dEyesYolk.end(),0);
	mean_AngleLYR = Sum_Angle_LYR/AngleLYR.size();
	mean_dEyes = Sum_d_Eyes/dEyes.size();
	mean_dEyesYolk = Sum_d_EyesYolk/dEyesYolk.size();
}


double zebrafishInfoReference::computeAngLYR(double dRY, double dLY, double dLR) {
	return acos((dLY*dLY + dRY*dRY - dLR*dLR)/(2*dRY*dLY));
}

void zebrafishInfoReference::updateRefFish(double dLR, double dRY, double dLY, double AngLYR) {
	AngleLYR.push_back(AngLYR);
	dEyes.push_back(dLR);
	dEyesYolk.push_back((dRY+dLY)/2);
	AngleLYR.erase(AngleLYR.begin());
	dEyes.erase(dEyes.begin());
	dEyesYolk.erase(dEyesYolk.begin());
}

bool zebrafishInfoReference::CheckInRange(double v, double vRef, double ratioRef) {
	double dv = fabs(vRef - v);
	double ratio = dv/vRef;
	if (ratio > ratioRef)
		return false;
	return true;
}


bool zebrafishInfoReference::getVaildation(zebrafishInfo * ref) {

	double dLR = cv::norm(ref->m_eyeLeft.center - ref->m_eyeRight.center);
	double dRY = cv::norm(ref->m_eyeLeft.center - ref->m_yolk.center);
	double dLY = cv::norm(ref->m_eyeRight.center - ref->m_yolk.center);
	double AngLYR = computeAngLYR(dRY, dLY, dLR);
	getMeanValue();

	bool b_dLR = CheckInRange(dLR, mean_dEyes, RangePercent_dEyes);
	bool b_dLEyeYolk = CheckInRange(dLY, mean_dEyesYolk, RangePercent_dEyesYolk);
	bool b_dREyeYolk = CheckInRange(dRY, mean_dEyesYolk, RangePercent_dEyesYolk);
	bool b_dAngLYR = CheckInRange(AngLYR, mean_AngleLYR, RangePercent_AngleLYR);

	bool output = (b_dLR && b_dLEyeYolk && b_dREyeYolk && b_dAngLYR);
	if (output)
		updateRefFish(dLR, dRY, dLY, AngLYR);
	return output;
}
