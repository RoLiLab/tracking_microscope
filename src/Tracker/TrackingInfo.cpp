#include "Base/base.h"
#include "Tracker/TrackingInfo.h"
#include "ipp.h"
#include <numeric>


CntrData::CntrData(void)
{
	isTracking = 0;
	isPositionReadingSuccess = 0;
	isFishPosDetection = 0;
	isReadyMPCInput = 0;
	isReadyOnTime = 1;
	isFrameDropped = 0;
	isFishMoving = 0;
	DroppedFrameCount = 0;
	FrameSimulation = 0;
	GlobalMapIdx = 0;
	MovementIBICount = 0;
	isReadyTeensyIndex = 0;
	isReadyTeensyIndex_input = 0;
	n = CntrDataIndex_MAX;
	//data = (double *) malloc(CntrDataIndex_MAX*sizeof(double));
	for (int i = 0; i < CntrDataIndex_MAX;  i++)
		data[i] = 0;
}

CntrData::~CntrData(void) {
}

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
	b_recstart = false;
	b_recstop = false;
	b_recording = false;
	recFrms_Start = 0;
	recFrms_End = 0;
	recFrms = 0;
	recFrms_left = 0;
	FrmNo = 0;
	// -------------------------------------

	FLRRecEnabled = false;
	NIRRecEnabled = false;
	DataRecEnabled = false;
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
	for (int i = 0; i < harddrivecount_FLRsave; i++)
		pFLR_writer_binary[i] = NULL;
	for (int i = 0; i < harddrivecount_NIRsave; i++)
		pNIR_writer_binary[i] = NULL;

	RecDoneCounter = 0;
	RecDoneCounter_NIR = 0;
	RecDoneCounter_FLR = 0;
	preRecImageCount = 0;

}

TrackingMessageBuffer::~TrackingMessageBuffer(void)
{
	freeBuffer();
}


void TrackingMessageBuffer::allocateBuffer(unsigned int _MaxBufferSize) {
	MaxBufferSizeBeforeTrigger = _MaxBufferSize;
	MaxBufferSize = MaxBufferSizeBeforeTrigger + 2*RecActiveWindowSize;
	FrmNo = 0;
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
uint16 * TrackingMessageBuffer::getNIRImgPtrFrameNo(unsigned int _frameNo) {
	unsigned int idx = getIndexFromFrameNo(_frameNo);
	return (Data[idx].srcNIR->data);
}

void TrackingMessageBuffer::updateAllNIR(unsigned int _frameNo,
		XPSGatheringInfo pos,
		zebrafishInfo fishPos,
		CntrData _cntrData,
		uint16 * src, int rows, int cols)
{
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->frameNo = _frameNo;
		FrmNo = _frameNo;
		_data->stagePos = pos;
		_data->NIRfishPos = fishPos;
		_data->m_CntrData = _cntrData;
		POSCount++;
		if (POSCount > MaxBufferSizeBeforeTrigger  && !isRecording)
			POSCount = MaxBufferSizeBeforeTrigger;
		_data->srcNIR->data = src;
		_data->srcNIR->imgSize.height = rows;
		_data->srcNIR->imgSize.width = cols;
	}
}
void TrackingMessageBuffer::updateAllNIRcpy(unsigned int _frameNo,
	XPSGatheringInfo pos,
	zebrafishInfo fishPos,
	CntrData _cntrData,
	uint16 * src, int rows, int cols,
	FLRImageData _FLRdata)
{
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->frameNo = _frameNo;
		_data->stagePos = pos;
		_data->NIRfishPos = fishPos;
		_data->m_CntrData = _cntrData;
		_data->FLRdata = _FLRdata;
		POSCount++;
		if (POSCount > MaxBufferSizeBeforeTrigger  && !isRecording)
			POSCount = MaxBufferSizeBeforeTrigger;

		try {
			if (_data->srcNIR) {
				if (_data->srcNIR->data != 0) {
					free(_data->srcNIR->data);
					_data->srcNIR->data = NULL;
				}
				//delete(_data->srcNIR);
			}
			else {
				_data->srcNIR = new img_uint16();
			}

			if (src == NULL)
				_data->srcNIR->data = NULL;
			else
			{
				int _datasize = rows*cols*sizeof(uint16);
				//int _datasize = rows*cols*sizeof(uint16) * 3 / 4;
				uint16 * temp = (uint16 *)malloc(_datasize);
				memcpy(temp, src, _datasize);
				_data->srcNIR->data = temp;
				_data->srcNIR->imgSize.height = rows;
				_data->srcNIR->imgSize.width = cols;
				_data->srcNIR->_datasize = _datasize;
				NIRCount++;
				if (NIRCount > MaxBufferSizeBeforeTrigger && !isRecording)
					NIRCount = MaxBufferSizeBeforeTrigger;
			}
			DispIdx = getIndexFromFrameNo(_frameNo);
			RecBlockIdx = max(_frameNo - RecActiveWindowSize, (unsigned int)0);

			FrmNo = _frameNo;
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\updateAllNIRcpyError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "updateAllNIRcpy: %s\n", e.what());
			fclose(ofp);
		}
	}
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

void TrackingMessageBuffer::updateCntrData_lateinputreport(unsigned int _frameNo) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		_data->m_CntrData.data[XstgInputVel] = 0;
		_data->m_CntrData.data[YstgInputVel] = 0;
		_data->m_CntrData.isReadyOnTime = 0;
	}
}

void TrackingMessageBuffer::updateNIRImage(unsigned int _frameNo, img_uint16 *  src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if(_data->srcNIR) delete(_data->srcNIR);
		_data->srcNIR = src;
		NIRSize = src->imgSize;
		NIRCount++;
		if (NIRCount > MaxBufferSizeBeforeTrigger && !isRecording)
			NIRCount = MaxBufferSizeBeforeTrigger;
	}
}

void TrackingMessageBuffer::updateNIRImageProc(unsigned int _frameNo, img_uint16 *  src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		if(_data->srcNIRproc) delete(_data->srcNIRproc);
		_data->srcNIRproc = src;
	}
}


void TrackingMessageBuffer::updateFLRImage(unsigned int _frameNo, img_uint16 * src) {
	// frame number update
	if (Data) {
		TrackingMessage * _data = getDataPtrFrameNo(_frameNo);
		//if(_data->srcFLR) delete(_data->srcFLR);
		if(_data->srcFLR) {
			//free(_data->srcFLR->data);
			delete(_data->srcFLR);
		}
		_data->srcFLR = src;
		FLRSize = src->imgSize;
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

Point2d TrackingMessageBuffer::getFramePos_cv(unsigned int _frameNo) {
	double * pos = getFramePos(_frameNo);
	if (pos)
		return Point2d(pos[0], pos[1]);
	else
		return Point2d(-1,-1);
}

Point2d TrackingMessageBuffer::getFramePoschamber_cv(unsigned int _frameNo) {
	double * pos = getFramePosChamber(_frameNo);
	if (pos)
		return Point2d(pos[0], pos[1]);
	else
		return Point2d(-1,-1);
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
		minCount = POSCount;
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
	if (_PreTrgRecFrms < FrmCount) {
		RecStartIdx = _Frame - _PreTrgRecFrms;
	}
	else {
		RecStartIdx = _Frame - FrmCount + RecActiveWindowSize;
	}
	RecEndIdx = 0;
	preRecImageCount = RecStartIdx - _Frame;
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

unsigned int TrackingMessageBuffer::getRecCount_NIR(void) {
	return RecDoneCounter_NIR;
}unsigned int TrackingMessageBuffer::getRecCount_FLR(void) {
	return RecDoneCounter_FLR;
}
unsigned int TrackingMessageBuffer::getRecCount(void) {
	return RecDoneCounter;
}

void TrackingMessageBuffer::InitializeRecording(const char * drive, bool enableNIRhdf5, bool enableFLRhdf5, int binaryDriveCount) {
	char NAME[128];
	RecDoneCounter = 0;
	RecDoneCounter_NIR = 0;
	RecDoneCounter_FLR = 0;
	t_now = time(0); // get time now;
	struct tm * now = localtime(&t_now);
	sprintf(DATESTR, "%04d%02d%02d_%02d%02d%02d", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
	sprintf(NAME, "d:\\%s\\", DATESTR);
	std::string FilePath = std::string(NAME);
	if (CreateDirectoryA(FilePath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{

		sprintf(NAME, "d:\\%s\\HDF5_%s_DATA.h5", DATESTR, DATESTR);
		pData_writer = new HDF5DataWriter(NAME);

		if (false) {
			int chunkimg = 1;
			sprintf(NAME, "d:\\%s\\HDF5_%s_NIR.h5", DATESTR, DATESTR);
			pNIR_writer = new HDF5ImageWriter(NAME, NIRSize.width, NIRSize.height, IMAGE_UINT16, chunkimg); // 27
		}
		else{
			char drives[] = "efg";
			for (int i = 0; i < harddrivecount_NIRsave; i++) {
				sprintf(NAME, "%c:\\%s\\", drives[i], DATESTR);
				std::string FilePath = std::string(NAME);
				if (CreateDirectoryA(FilePath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {
					sprintf(NAME, "%c:\\%s\\HDF5_%s_NIR%d.dat", drives[i], DATESTR, DATESTR, i);
					wchar_t* wString = new wchar_t[128];
					MultiByteToWideChar(CP_ACP, 0, NAME, -1, wString, 128);
					pNIR_writer_binary[i] = CreateFile(wString,                // name of the write
						GENERIC_WRITE,          // open for writing
						0,                      // do not share
						NULL,                   // default security
						CREATE_NEW,             // create new file only
						FILE_ATTRIBUTE_NORMAL,  // normal file
						NULL);                  // no attr. template
				}
			}
		}
		{
			char drives[] = "hijkl"; //"efghijkl";
			for (int i = 0; i < binaryDriveCount; i++) {
				sprintf(NAME, "%c:\\%s\\", drives[i], DATESTR);
				std::string FilePath = std::string(NAME);
				if (CreateDirectoryA(FilePath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {
					sprintf(NAME, "%c:\\%s\\HDF5_%s_EPI%d.dat", drives[i], DATESTR, DATESTR, i);
					wchar_t* wString = new wchar_t[128];
					MultiByteToWideChar(CP_ACP, 0, NAME, -1, wString, 128);
					pFLR_writer_binary[i] = CreateFile(wString,                // name of the write
						GENERIC_WRITE,          // open for writing
						0,                      // do not share
						NULL,                   // default security
						CREATE_NEW,             // create new file only
						FILE_ATTRIBUTE_NORMAL,  // normal file
						NULL);                  // no attr. template
				}
			}
		}
	} //End of File/Folder setting for recording
}


void TrackingMessageBuffer::ReleaseRecording(void) {
	// close files
}
void TrackingMessageBuffer::ReleaseRecording_NIR(void) {
	try {
		// close files
		if (pNIR_writer) {
			pNIR_writer->flush();
			delete pNIR_writer;
			pNIR_writer = NULL;
		}
		if (pNIR_writer_binary[0]) {
			for (int i = 0; i < harddrivecount_NIRsave; i++) {
				CloseHandle(pNIR_writer_binary[i]);
				pNIR_writer_binary[i] = NULL;
			}
		}
		if (pData_writer) {
			pData_writer->flush();
			delete pData_writer; pData_writer = NULL;
		}
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\ErrorReport_releaseRecording.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "MPCrecordingThread (NIRRecordFrame): %s\n", e.getCDetailMsg());
		fclose(ofp);
	}
}
void TrackingMessageBuffer::ReleaseRecording_FLR(void) {
	// close files
	if (pFLR_writer_binary[0]) {
		for (int i = 0; i < harddrivecount_FLRsave; i++) {
			CloseHandle(pFLR_writer_binary[i]);
			pFLR_writer_binary[i] = NULL;
		}
	}
}
void TrackingMessageBuffer::RecordDataVector(int index, vector<Point2d> * _data) {
	 vector<Point2d>::iterator it; // declare an iterator to a vector of strings
	 int i = 0;
	 for(it=_data->begin() ; it < _data->end(); it++) {
		 pData_writer->PointData[index + i++]->write(&it);
    }
}


bool TrackingMessageBuffer::RecordFrame_FLR(uint16_t * src) {

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
	try {
		TrackingMessage * t = getFrameData(_Frame);
		// NIR image recording
		if (t->srcNIR && NIRRecEnabled) {
			if (pNIR_writer) {
				pNIR_writer->write(t->srcNIR->data );
				if (_Frame % 16 == 0 && _Frame > 0)
					pNIR_writer->flush();
			}
			else if (pNIR_writer_binary[0]) {
				uint64 driveNo = ((uint64)_Frame - recFrms_Start) % harddrivecount_NIRsave;
				DWORD dwBytesWritten = 0;
				bool bErrorFlag = WriteFile(
					pNIR_writer_binary[driveNo],           // open file handle
					(char*)(t->srcNIR->data),      // start of data to write
					t->srcNIR->_datasize,  // number of bytes to write
					&dwBytesWritten, // number of bytes that were written
					NULL);
			}
			RecDoneCounter_NIR++;
		}
		if (DataRecEnabled) {
			// Data Recording
			pData_writer->SingleData[R1_FrameNo]->write(&(t->frameNo));

			// Fish Data
			pData_writer->PointData[R2_FishPos_Leye_px]->write(&t->NIRfishPos.m_eyeLeft.center);
			pData_writer->PointData[R2_FishPos_Reye_px]->write(&t->NIRfishPos.m_eyeRight.center);
			pData_writer->PointData[R2_FishPos_Yolk_px]->write(&t->NIRfishPos.m_yolk.center);
			pData_writer->PointData[R2_FishPos_Brain_px]->write(&t->NIRfishPos.centerEyes);
			pData_writer->SingleData[R1_FishPos_Orientation_Deg]->write(&(t->NIRfishPos.orienation));
			pData_writer->SingleData[R1_FishPos_Heading_DegMean]->write(&t->m_CntrData.data[HeadingFishMean]);
			pData_writer->SingleData[R1_FishPos_MovingDirection_Deg]->write(&t->m_CntrData.data[MovingDirFish]);
			pData_writer->SingleData[R1_PredictionWeight]->write(&(t->m_CntrData.data[weight]));

			// Stage Pos Data
			pData_writer->PointData[R2_StagePos]->write(t->stagePos.CurrentPosition);//
			pData_writer->PointData[R2_StageSetVoltRead]->write(t->stagePos.InputVoltage);//
			pData_writer->PointData[R2_StageAcc_setpoint]->write(&t->stagePos.SetpointAcceleration);//

			// Thermal Input
			pData_writer->SingleData[R1_thermalcntrinput]->write(&t->stagePos.thermalcntrinput);
			pData_writer->SingleData[R1_thermalControlInput_Mode]->write(&t->stagePos.temperature_circle_rev);
			pData_writer->SingleData[R1_FishIBICount]->write(&t->m_CntrData.MovementIBICount);

			// temporal data saving
			//pData_writer->SingleData[R1_reserved1]->write(&t->NIRfishPos.m_eyeLeft.angle);
			//pData_writer->SingleData[R1_reserved2]->write(&t->NIRfishPos.m_eyeRight.angle);
			//pData_writer->SingleData[R1_reserved3]->write(&t->stagePos.CurrentVelocity[0]);
			//pData_writer->SingleData[R1_reserved4]->write(&t->stagePos.CurrentVelocity[1]);


			// Status
			pData_writer->SingleData[R1_GlobalMapIndex]->write(&(t->m_CntrData.GlobalMapIdx));
			pData_writer->SingleData[R1_isTracking]->write(&(t->m_CntrData.isTracking));
			pData_writer->SingleData[R1_isPositionReadingSuccess]->write(&(t->m_CntrData.isPositionReadingSuccess));
			pData_writer->SingleData[R1_isFishPosDetection]->write(&(t->m_CntrData.isFishPosDetection));
			pData_writer->SingleData[R1_isReadyMPCInput]->write(&(t->m_CntrData.isReadyMPCInput));
			pData_writer->SingleData[R1_isReadyOnTime]->write(&(t->m_CntrData.isReadyOnTime));
			pData_writer->SingleData[R1_isFrameDropped]->write(&(t->m_CntrData.isFrameDropped));
			pData_writer->SingleData[R1_isFishMoving]->write(&(t->m_CntrData.isFishMoving));
			pData_writer->SingleData[R1_DroppedFrameCount]->write(&(t->m_CntrData.DroppedFrameCount));
			pData_writer->SingleData[R1_FrameSimulation]->write(&(t->m_CntrData.FrameSimulation));


			pData_writer->PointData[R2_StageInputVolt]->write(&t->m_CntrData.data[XstgInputVolt]);//
			pData_writer->PointData[R2_StageSetInputVel]->write(&t->m_CntrData.data[XstgInputVel]);//
			pData_writer->PointData[R2_FishPos]->write(&t->m_CntrData.data[Xfish]);//
			pData_writer->PointData[R2_FishPosProj]->write(&t->m_CntrData.data[Xfishproj]);//
			for (int i = 0; i < 7; i++)
				pData_writer->PointData[R2_FishPosPred]->write(&t->m_CntrData.data[Xfishpred0 + 2 * i]);//
			pData_writer->PointData[R2_Err]->write(&t->m_CntrData.data[Xerr]);//
			pData_writer->PointData[R2_Err_px]->write(&t->m_CntrData.data[XerrPx]);//
			pData_writer->PointData[R2_FishVelPrlPpd]->write(&t->m_CntrData.data[VfishPrl]);//
			pData_writer->PointData[R2_FishVelWeighted]->write(&t->m_CntrData.data[VfishPrlWeighted]);//
			pData_writer->PointData[R2_StagePosMPC]->write(&t->m_CntrData.data[XstgMPC]);//
			pData_writer->PointData[R2_StageSetInputVelMPC]->write(&t->m_CntrData.data[XstgInputVelMPC]);//
			pData_writer->PointData[R2_PID_DesiredPos]->write(&t->m_CntrData.data[XstgPIDdesired]);//
			pData_writer->PointData[R2_PID_CurrentPos]->write(&t->m_CntrData.data[XstgPIDtarget]);//
			pData_writer->PointData[R2_StageSetInputVelPID]->write(&t->m_CntrData.data[XstgPIDInputVel]);//
			pData_writer->PointData[R2_Stage_L2Weight]->write(&t->m_CntrData.data[Xstg_L2weight]);//
			double temp[2]; temp[0] = (double)t->NIRfishPos.AreaSize; temp[1] = t->NIRfishPos.fitness_heading;
			pData_writer->PointData[R2_fish_fittness]->write(temp);//
			temp[0] = t->NIRfishPos.m_eyeLeft.angle; temp[1] = t->NIRfishPos.m_eyeRight.angle;
			pData_writer->PointData[R2_FishPos_EyeAngleDeg]->write(temp);//

			pData_writer->PointData[R2_TargetPos]->write(&t->m_CntrData.data[x_target_mm]);//
			pData_writer->PointData[R2_RefPos]->write(&t->m_CntrData.data[x_ref_px]);//
			pData_writer->PointData[R2_StageI2T]->write(&t->m_CntrData.data[XstgI2T]);//

			temp[0] = (double)(t->m_CntrData.isReadyTeensyIndex_input); temp[1] = (double)(t->m_CntrData.isReadyTeensyIndex);
			pData_writer->PointData[R2_teensyIdx]->write(temp);

			if (_Frame % 1024 == 0 && _Frame > 0)
				pData_writer->flush();
		}



		RecDoneCounter++;
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\NIRRecordFrameError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "MPCrecordingThread (NIRRecordFrame): %s\n", e.getCDetailMsg());
		fclose(ofp);
	}
}


bool TrackingMessageBuffer::VelocityCalibration(double * offset) {
	return true;
}


JitterFilter::JitterFilter(void)
{
	threshold = 0.001;
	CurrentValue = Point2d(0,0);
}

JitterFilter::~JitterFilter(void)
{
}

Point2d JitterFilter::update(Point2d _p, bool _isfishdetected)
{
	if (_isfishdetected) {
		Point2d cur = CurrentValue;
		if (fabs(_p.x - CurrentValue.x) > threshold)
			cur.x = _p.x;
		if (fabs(_p.y - CurrentValue.y) > threshold)
			cur.y = _p.y;
		//double ds2 = ((cur.x - CurrentValue.x)*(cur.x - CurrentValue.x) + (cur.y - CurrentValue.y)*(cur.y - CurrentValue.y));
		//double fishDisplacementLimit = 1; // 1mm;
		//if (ds2 < fishDisplacementLimit)
			CurrentValue = cur;
	}
	return CurrentValue;
}


double NormDiff_1D(double a, double ref) {
	return fabs((a-ref))/min(a, ref);
};
double NormDiff_cvPoint(Point2d * a, Point2d * ref) {
	double d1 = Diff_cvPoint(a[0], ref[0]);
	double d2 = Diff_cvPoint(a[1], ref[1]);
	double Ndiff = 1000;
	Ndiff = NormDiff_1D(d1, d2);
	return Ndiff;
};
double NormDiff_cvPoint(Point2i * a, Point2i * ref) {
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
double Diff_cvPoint(Point2d a, Point2d ref){
	return norm_Point2d(a - ref);
};
double Diff_cvPoint(Point2i a, Point2i ref){
	return Diff_cvPoint(Point2d((double)a.x, (double)a.y), Point2d((double)ref.x, (double)ref.y));
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


double Area_3Points(Point2i a, Point2i b, Point2i c){
	return abs(0.5*(a.cross(b) + b.cross(c) +c.cross(a)));
};
double Area_3Points(Point2d a, Point2d b, Point2d c){
	return fabs(0.5*(a.cross(b) + b.cross(c) +c.cross(a)));
};

double atan2_twoPoints(Point2i a, Point2i b){
	return atan2_twoPoints(Point2d((double)a.x, (double)a.y), Point2d((double)b.x, (double)b.y));
};
double atan2_twoPoints(Point2d a, Point2d b){
	double angle = atan2(a.y - b.y, a.x - b.x);
	return angle*180/PI;
};
double atan2_twoPoints_Right(Point2i c, Point2i LEye){
	return atan2_twoPoints_Right(Point2d((double)c.x, (double)c.y), Point2d((double)LEye.x, (double)LEye.y));
};
double atan2_twoPoints_Right(Point2d c, Point2d LEye){

	double angle = atan2(LEye.y - c.y, LEye.x - c.x) + PI/2;
	if (angle < -PI) angle = angle + 2*PI;
	if (angle > PI) angle = angle - 2*PI;
	return angle*180/PI;
};

//-------------------------------------------------
SpotInfo::SpotInfo(void) {
	center = Point2d();
	width = 1;
	height = 1;
	angle = 0; // deg
	updateValue();
};
SpotInfo::SpotInfo(Point2d _x)  {
	center = _x;
	width = 1;
	height = 1;
	angle = 0; // deg
	updateValue();
};
SpotInfo::~SpotInfo(void)  {
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
	m_eyeLeft = SpotInfo(Point2d(300, 220));
	m_eyeRight = SpotInfo(Point2d(340, 220));
	m_yolk = SpotInfo(Point2d(320, 260));

	centerEyes = (m_eyeLeft.center + m_eyeRight.center)*0.5;
	orienation = atan2_twoPoints(centerEyes, m_yolk.center);
	AreaSize = Area_3Points(m_eyeLeft.center, m_eyeRight.center, m_yolk.center);
	CosAngle_LYR = 0.866;
	fitness_heading = 0;
};
zebrafishInfo::zebrafishInfo(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk){
	ReplacePosition(_eyeLeft, _eyeRight, _yolk);
};

zebrafishInfo::zebrafishInfo(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk, double distance){
	m_eyeLeft = _eyeLeft;
	m_eyeRight = _eyeRight;
	m_yolk = _yolk;
	centerEyes = (m_eyeLeft.center + m_eyeRight.center)*0.5;
	Point2d headingVec = centerEyes - m_yolk.center;
	Point2d headingVec_perp = m_eyeLeft.center - m_eyeRight.center;
	orienation = atan2(headingVec.y, headingVec.x) * 180 / PI;


	AreaSize = Area_3Points(m_eyeLeft.center, m_eyeRight.center, m_yolk.center);
	fitness_heading = headingVec.dot(headingVec_perp) / (headingVec.norm() * headingVec_perp.norm());

	centerEyes.x = m_yolk.center.x + distance * (headingVec.x / headingVec.norm());
	centerEyes.y = m_yolk.center.y + distance * (headingVec.y / headingVec.norm());
};

zebrafishInfo::zebrafishInfo(vector<Point2d> fish_pt, double fish_size){ // yolk, left, right, brain, fish_center, size
	m_yolk = SpotInfo(fish_pt[0]);
	m_eyeLeft = SpotInfo(fish_pt[1]);
	m_eyeRight = SpotInfo(fish_pt[2]);
	centerEyes = fish_pt[3];
	centerFish = fish_pt[5];
	Point2d headingVec = fish_pt[4];
	AreaSize = fish_size;
	orienation = atan2(headingVec.y, headingVec.x) * 180 / PI;
	fitness_heading = 0;
};


void zebrafishInfo::ReplacePosition(SpotInfo _eyeLeft, SpotInfo _eyeRight, SpotInfo _yolk) {
	m_eyeLeft = _eyeLeft;
	m_eyeRight = _eyeRight;
	m_yolk = _yolk;
	centerEyes = (m_eyeLeft.center + m_eyeRight.center)*0.5;
	centerFish = (centerEyes - m_yolk.center)*0.5;
	Point2d headingVec = centerEyes - m_yolk.center;
	Point2d headingVec_perp = m_eyeLeft.center - m_eyeRight.center;
	orienation = atan2(headingVec.y, headingVec.x) * 180 / PI;
	//orienation = atan2_twoPoints_Right(centerEyes, m_eyeLeft.center);
	AreaSize = Area_3Points(m_eyeLeft.center, m_eyeRight.center, m_yolk.center);
	fitness_heading = headingVec.dot(headingVec_perp) / (headingVec.norm() * headingVec_perp.norm());
	//double distLY = Diff_cvPoint(_eyeLeft.center, _yolk.center);
	//double distRY = Diff_cvPoint(_eyeRight.center, _yolk.center);
	//double distLR = Diff_cvPoint(_eyeRight.center, _eyeLeft.center);
	//CosAngle_LYR = (distRY*distRY + distLY*distLY - distLR*distLR) / (2 * distRY * distLY);

	// adjust centerEyesPosition
	//double distFromYolk = 60;
	//centerEyes.x = m_yolk.center.x + 60 * (headingVec.x / headingVec.norm());
	//centerEyes.y = m_yolk.center.y + 60 * (headingVec.y / headingVec.norm());
}
zebrafishInfo::~zebrafishInfo(void){
};

void zebrafishInfo::getVaildation(int FishSize) {

}

vector<Point2d> estimateFuturePos(vector<double> orientationHist, vector<Point2d> fishPosHist) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<Point2d> estPos;
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPosHist[0]);

	// 1. get peaks from history
	double peaks[2];
	Point2d peaks_pos[2];
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
	Point2d dS = peaks_pos[1] - peaks_pos[0];
	double costh = cos(th0*PI/180);
	double sinth = sin(th0*PI/180);
	double dx = fabs((costh*dS.x + sinth*dS.y)/di);
	double dy = fabs(-sinth*dS.x + costh*dS.y);
	double alpha = a/fabs(a)*dy/2;
	double omega = w/dx;
	phi = t0*dx;
	vector<Point2d> p_org(EstSteps);
	for (int i = 0; i < EstSteps; i++) {
		double x_temp = i*dx;
		double y_temp = alpha*sin(omega*(x_temp-phi));
		p_org[i] = Point2d(x_temp, y_temp);
	}
	// 4. rotataion & translation
	vector<Point2d> p_rot(EstSteps);
	Point2d p_offset;
	for (int i = 0; i < EstSteps; i++) {
		double x_temp = costh*p_org[i].x - sinth*p_org[i].y;
		double y_temp = sinth*p_org[i].x + costh*p_org[i].y;
		if (i == 0)
			p_offset = Point2d(x_temp, y_temp) + fishPosHist[0];
		p_rot[i] = Point2d(x_temp, y_temp) - p_offset;
	}
	return estPos;
}


vector<Point2d> estimateFuturePos_linear(vector<double> orientationHist, vector<Point2d> fishPosHist) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<Point2d> estPos;
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPosHist[0]);
	//int pastSteps = orientationHist.size();
	int pastSteps = 5;

	// 1. get peaks from history
	double peaks[2];
	Point2d peaks_pos[2];
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
	Point2d p_org = (peaks_pos[1] + peaks_pos[0])*0.5;
	vector<Point2d> p_orgB(pastSteps);
	for (int i = 0; i < pastSteps; i++) {
		Point2d p_trans = fishPosHist[i] - p_org;
		double x_temp =  costh*p_trans.x + sinth*p_trans.y;
		double y_temp = -sinth*p_trans.x + costh*p_trans.y;
		p_orgB[i] = Point2d(x_temp, y_temp);
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

vector<Point2d> estimateFuturePos_Front(vector<double> orientationHist, vector<Point2d> fishPosHist, double gain) {

	// estimate the future pose of the fish
	int EstSteps = 8;
	vector<Point2d> estPos;

	// 1. get peaks from history
	//double gain = 0.5*0.014;
	double OriDiff_sum=0;
	double HistCount = 6;

	//for i = 1:1:orientationHist.size();
	for (int i = 1; i<HistCount; i++) {
		double A0 = norm_Point2d(fishPosHist[i] - fishPosHist[i-1]);
		//double A0 = (orientationHist[i] - orientationHist[i-1]);
		//if (A0 > 180) A0 = A0 - 360;
		//if (A0 <= -180) A0 = A0 + 360;
		OriDiff_sum += (abs(A0));
	}

	double A0 = OriDiff_sum/(HistCount-1);

	Point2d fishPos_adjust = fishPosHist[0];
	if (A0 > 0.01) {
		double costh = cos(orientationHist[0]*PI/180);
		double sinth = sin(orientationHist[0]*PI/180);
		Point2d offset = Point2d(gain*costh*A0, gain*sinth*A0);
		fishPos_adjust = fishPosHist[0]+offset;
	}
	for (int i = 0; i < EstSteps; i++)
		estPos.push_back(fishPos_adjust);

	return estPos;
}


//-------------------------------------------------

vector<Point2i> GetZebraFishROI_exact(Point2d p, double Orientation_deg, imSize imgSize) {
	vector<Point2d> v(12);
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

	vector<Point2i> v_new;
	for (size_t i = 0; i<v.size(); i++) {
		v_new.push_back(Point2i(cos_o*v[i].x -sin_o*v[i].y +p.x, sin_o*v[i].x + cos_o*v[i].y+p.y));
	}

	return v_new;
}

// ---------------------------------------------------------------------
// ------------- Moving Average class ----------------------------------
// ---------------------------------------------------------------------
MovingAverage_double::MovingAverage_double(void)  {
	stepMax = 100;
	stepMax_Hist = 100;
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

	auto it = max_element(begin(RawData), end(RawData));
	_MAX = *it;

	// std
	_stddev = 0;
	for (int i = 0; i<steps; i++) {
		_stddev += (RawData[i] - _MVA)*(RawData[i] - _MVA);
	}


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
Point2d MovingAverage_cvPoint2d::update(Point2d NewValue) {
	updateRawData(NewValue);
	MVA = updateMVA_Hist(stepMax);
	return MVA;
}

void MovingAverage_cvPoint2d::updateRawData(Point2d NewValue) {
	if (RawData.size() == 0)
		RawData.push_back(NewValue);
	else
		RawData.insert(RawData.begin(), NewValue);
	while (RawData.size() > stepMax)
		RawData.pop_back();
}

Point2d MovingAverage_cvPoint2d::updateMVA_Hist(int steps) {
	Point2d _MVA;
	steps = min(steps, (int)(RawData.size()));
	for (int i = 0; i<steps; i++) {
		_MVA = _MVA + RawData[i];
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
Point2d MovingAverage_cvPoint2d::MVA_Hist_FDD(void) {
	Point2d fdev(0,0);
	vector<Point2d> diff;
	for (int i = 1; i < MVA_Hist.size(); i++)
		diff.push_back(MVA_Hist[i-1] - MVA_Hist[i]);

	switch (diff.size()) {
		case 0:
			break;
		case 1:
			break;
		case 2:
			fdev = diff[1]*(-1.0) + diff[0];
			break;
		case 3:
			fdev = diff[2] * (-3.0 / 2.0) + diff[1] * 2.0 - diff[0] * 0.5;
			break;
		case 4:
			fdev = diff[3]*(-11.0/6.0) + diff[2]*3.0 - diff[1]*(3.0/2.0) - diff[0]* (3.0 / 2.0);
			break;
		default:
			fdev = diff[3] * (-11.0 / 6.0) + diff[2] * 3.0 - diff[1] * (3.0 / 2.0) - diff[0] * (3.0 / 2.0);
			break;
	}
	return fdev;
}

// ---------------------------------------------------------------------
// ------------- Fish Position Estimation class ------------------------
// ---------------------------------------------------------------------
zebrafishPositionEstimate::zebrafishPositionEstimate(void)
{
	// Real physical values
	VelocityUpdateCycle_ms = 4;
	TargetPositionUpdateCycle_ms = 20;
	AxisHeadingUpdateCycle_ms = 12;
	HeadingUpdateCycle_ms = 12;
	deActivationTime_ms = 8;
	AngularMomentumWindow_ms = 8;
	angularMomentum_DegPerSec_UPPER = 30;
	angularMomentum_DegPerSec_LOWER = 3;
	stopFishThreshold = 5;
	predX = NULL;
	predY = NULL;
	// Initialization
	init(0.004, 7);
};
zebrafishPositionEstimate::~zebrafishPositionEstimate(void)  {
	if (predX) free(predX);
	if (predY) free(predY);
};

void zebrafishPositionEstimate::init(double cycle_sec, int _steps) {
	isFishMoving = false;
	steps = _steps;
	if (predX) free(predX);
	if (predY) free(predY);
	predX = (double *) malloc (_steps* sizeof(double));
	predY = (double *) malloc (_steps* sizeof(double));

	Cycle_dt = cycle_sec;
	Cycle_dt_ms = Cycle_dt * 1000;
	angMomentum_maxThrehold = angularMomentum_DegPerSec_UPPER;//(angularMomentum_DegPerSec_UPPER*Cycle_dt)*(angularMomentum_DegPerSec_UPPER*Cycle_dt); //
	angMomentum_minThrehold = angularMomentum_DegPerSec_LOWER;//(angularMomentum_DegPerSec_LOWER*Cycle_dt)*(angularMomentum_DegPerSec_LOWER*Cycle_dt);
	angMomentum_Threhold = angMomentum_maxThrehold;
	weight_angMomentum.clear();

	for (int i = 0; i < (int)(deActivationTime_ms/Cycle_dt_ms); i++)
		weight_angMomentum.push_back(0);

	//for (int i = 0; i < (int)(deActivationTime_ms/Cycle_dt_ms); i++)
	//	weight_angMomentum.push_back(log10((double)i/(deActivationTime_ms/Cycle_dt_ms)*9+1));
	weight_angMomentum.push_back(0.5);
	weight_angMomentum.push_back(1);
	WindowSize_AngMTh = (int)(AngularMomentumWindow_ms/Cycle_dt_ms); //[steps]
	angMomentum_FrameCountAfterMoving = weight_angMomentum.size()-1;
	MVA_TargetSpeed_5Step = MovingAverage_double((int)(20/Cycle_dt_ms));
	MVA_TargetSpeed = MovingAverage_double((int)(VelocityUpdateCycle_ms/Cycle_dt_ms));
	MVA_TargetPosition = MovingAverage_cvPoint2d((int)(TargetPositionUpdateCycle_ms/Cycle_dt_ms));
	MVA_TargetPosition_Heading = MovingAverage_cvPoint2d((int)(AxisHeadingUpdateCycle_ms/Cycle_dt_ms));
	MVA_AxisHeading = MovingAverage_double((int)(AxisHeadingUpdateCycle_ms/Cycle_dt_ms));
	MVA_Heading = MovingAverage_double((int)(HeadingUpdateCycle_ms/Cycle_dt_ms));
	ProjPt = Point2d(0,0);
	TrgtPt = Point2d(0,0);
	BasePt = Point2d(0,0);
	AxisHeadingAngle = 0;
	ForwardVelocity = 0;
	cosH = 1;
	sinH = 0;
	FrameCountStopEnabled = true;
}
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

	MVA_TargetSpeed_5Step.clear();
	MVA_TargetSpeed.clear();
	MVA_TargetPosition.clear();
	MVA_TargetPosition_Heading.clear();
	MVA_AxisHeading.clear();
	MVA_Heading.clear();
}

vector<Point2d> zebrafishPositionEstimate::estimateFuturePosition(zebrafishInfo CurInfo) {
	vector<Point2d> estPts = estimateFuturePosition(CurInfo.centerEyes, CurInfo.orienation);
	return estPts;
};

vector<Point2d> zebrafishPositionEstimate::estimateFuturePosition(Point2d TargetPos, double _HeadingAngle) {
	// update target position and orientation
	double dt = 0.004;
	TrgtPt = TargetPos;
	BasePt = MVA_TargetPosition.update(TargetPos);
	MVA_Heading.update(_HeadingAngle);
	// update velocity
	//Point2d ForwardVelocity_v = MVA_TargetPosition.MVA_Hist_FDD();
	Point2d ForwardVelocity_v(0,0);
	if (MVA_TargetPosition.RawData.size() > 1)
		ForwardVelocity_v = MVA_TargetPosition.RawData[0] - MVA_TargetPosition.RawData[1];
	ForwardVelocity = MVA_TargetSpeed.update(norm_Point2d(ForwardVelocity_v)/dt);
	// forward veolocty cut
	if (ForwardVelocity < 10) // 2px/frame
		ForwardVelocity = 0;

	// option 0. (fitting) Orientation (angle) - not updated to the history  & Projection of Point & correct the heading vector and headingangle.
	//computeAxisHeading_fitting(); // update to the axis heading angle
	//computeProjPt(); // compute ProjPt

	// option 2. New developped
	computeAxisHeading_dth();
	ProjPt = MVA_TargetPosition.RawData[0];
	AxisHeadingVector = Point2d(-cosH, -sinH);

	// Angular momentum
	computeAngularMomentum();
	getWeight();

	vector<Point2d> estPts(steps);
	for (int i = 0; i < steps; i++) {
		estPts[i] = AxisHeadingVector*weight*ForwardVelocity*(i + 1)*dt + ProjPt;
	}
	return estPts;
};

vector<Point2d> zebrafishPositionEstimate::estimateFuturePosition_option3(Point2d TargetPos, double _HeadingAngle) {
	// update target position and orientation
	TrgtPt = TargetPos;

	// step 0. Angular momentum
	MVA_Heading.update(_HeadingAngle);
	computeAngularMomentum();
	getWeight();
	BasePt = MVA_TargetPosition.update(TargetPos);
	// step 3.  update velocity
	Point2d ForwardVelocity_v(0,0);
	if (MVA_TargetPosition.RawData.size() > 1)
		ForwardVelocity_v = MVA_TargetPosition.RawData[0] - MVA_TargetPosition.RawData[1];
	ForwardVelocity = MVA_TargetSpeed.update(norm_Point2d(ForwardVelocity_v)/Cycle_dt);
	// forward veolocty cut
	if (ForwardVelocity < 10) // 2px/frame
		ForwardVelocity = 0;

	if (weight == 0) {
		AxisHeadingVector = Point2d(0, 0);
		ForwardVelocity = 0;
		ProjPt = TargetPos;
	}
	else {
		// step 1. (fitting) Orientation (angle) - not updated to the history  & Projection of Point & correct the heading vector and headingangle.
		MVA_TargetPosition_Heading.update(TargetPos); // update when the weight is over 0
		double _HeadingAngle_rad = _HeadingAngle*PI/180;
		cosH = cos(_HeadingAngle_rad);
		sinH = sin(_HeadingAngle_rad);
		MVA_AxisHeading.update(_HeadingAngle_rad);
		Point2d AxisHeadingVector_Ref = Point2d(-cosH, -sinH);
		//if (MVA_TargetPosition.RawData.size() == MVA_TargetPosition.stepMax) {
		if (MVA_TargetPosition_Heading.RawData.size() == MVA_TargetPosition_Heading.stepMax) {
			//computeAxisHeading_fitting(); // update to the axis heading angle
			computeAxisHeading_fitting_SeparateHeadingPos();
			AxisHeadingVector = Point2d(-cosH, -sinH);
			Point2d temp = AxisHeadingVector_Ref + AxisHeadingVector;
			if (norm_Point2d(temp) < 1.5) {
				cosH = -cosH;
				sinH = -sinH;
				AxisHeadingVector = Point2d(-cosH, -sinH);
			}
		}
		else {
			AxisHeadingVector = AxisHeadingVector_Ref;
		}

		// step 2.
		computeProjPt(); // compute ProjPt
	}



	vector<Point2d> estPts(steps);
	for (int i = 0; i < steps; i++) {
		estPts[i] = AxisHeadingVector*weight*ForwardVelocity*Cycle_dt*(double)(i + 1) + ProjPt;
	}
	return estPts;
};

vector<Point2d> zebrafishPositionEstimate::estimateFuturePosition_MeanVelProjPt(Point2d TargetPos, double _HeadingAngle) {
	// update target position and orientation
	TrgtPt = TargetPos;
	BasePt = MVA_TargetPosition.update(TargetPos);
	_HeadingAngle = (_HeadingAngle+180);
	if (_HeadingAngle >360) _HeadingAngle -= 360;

	// step 0.  update velocity (Moving Average)
	Point2d ForwardVelocity_v(0,0);
	if (MVA_TargetPosition.RawData.size() > 1)
		ForwardVelocity_v = MVA_TargetPosition.RawData[0] - MVA_TargetPosition.RawData[1];
	ForwardVelocity = MVA_TargetSpeed.update(norm_Point2d(ForwardVelocity_v)/Cycle_dt);
	// forward veolocty cut
	//if (ForwardVelocity < 10) // 2px/frame
	//	ForwardVelocity = 0;

	// step 1. Heading angle update and Angular momentum
	MVA_Heading.update(_HeadingAngle);// *PI/180;
	computeAxisHeading_MeanTheta();
	Point2d AxisHeadingVector = Point2d(cosH, sinH);

	// step 2. update weight
	MVA_TargetSpeed_5Step.update(norm_Point2d(ForwardVelocity_v)/Cycle_dt);
	computeAngularMomentum();
	getWeight_FishMotion();

	// step 3. compute Heading
	if (weight == 0)
		ProjPt = TargetPos;
	else
		computeProjPt();

	// step 4. compute future points
	vector<Point2d> estPts(steps);
	for (int i = 0; i < steps; i++) {
		estPts[i] = AxisHeadingVector*weight*ForwardVelocity*Cycle_dt*(i + 1) + ProjPt;
	}
	return estPts;
};
void zebrafishPositionEstimate::estimateFuturePosition_Ver1(Point2d TargetPos, double _HeadingAngle) {
	// update target position and orientation
	MVA_TargetPosition.updateRawData(TargetPos);

	_HeadingAngle = (_HeadingAngle+180);
	if (_HeadingAngle >360) _HeadingAngle -= 360;
	MVA_Heading.update(_HeadingAngle);// *PI/180;
	computeAxisHeading_MINMAX();

}

void zebrafishPositionEstimate::estimateFuturePosition_MeanVelProjPt_double(Point2d TargetPos, double _HeadingAngle) {
	// update target position and orientation
	TrgtPt = TargetPos;
	BasePt = MVA_TargetPosition.update(TargetPos);
	_HeadingAngle = (_HeadingAngle+180);
	if (_HeadingAngle >360) _HeadingAngle -= 360;

	// step 0.  update velocity (Moving Average)
	Point2d ForwardVelocity_v(0,0);
	if (MVA_TargetPosition.RawData.size() > 1)
		ForwardVelocity_v = MVA_TargetPosition.RawData[0] - MVA_TargetPosition.RawData[1];
	ForwardVelocity = MVA_TargetSpeed.update(norm_Point2d(ForwardVelocity_v)/Cycle_dt);
	// forward veolocty cut
	//if (ForwardVelocity < 10) // 2px/frame
	//	ForwardVelocity = 0;

	// step 1. Heading angle update and Angular momentum
	MVA_Heading.update(_HeadingAngle);// *PI/180;
	computeAxisHeading_MINMAX();
	//computeAxisHeading_MeanTheta();
	Point2d AxisHeadingVector = Point2d(cosH, sinH);

	// step 2. update weight
	MVA_TargetSpeed_5Step.update(norm_Point2d(ForwardVelocity_v)/Cycle_dt);
	computeAngularMomentum();
	getWeight_FishMotion();

	// step 3. compute Heading
	if (weight == 0)
		ProjPt = TargetPos;
	else
		computeProjPt();

	// step 4. compute future points
	for (int i = 0; i < steps; i++) {
		predX[i] = (i+1)*AxisHeadingVector.x*weight*ForwardVelocity*Cycle_dt + ProjPt.x;
		predY[i] = (i+1)*AxisHeadingVector.y*weight*ForwardVelocity*Cycle_dt + ProjPt.y;
	}
};


void zebrafishPositionEstimate::computeAxisHeading_MeanTheta(void)  {
	AxisHeadingAngle = 0;

	int n = MVA_Heading.RawData.size();
	vector<double> Heading_deg;
	for (int i = 0; i < n; i++)
		Heading_deg.push_back(MVA_Heading.RawData[i]);

	for (int i = 0; i < n-1; i++) {
		double dth = Heading_deg[i+1] - Heading_deg[i];
        if (dth < -180) Heading_deg[i+1] = Heading_deg[i+1] + 360;
        if (dth > 180) Heading_deg[i+1] = Heading_deg[i+1] - 360;
	}
	double sum = std::accumulate(Heading_deg.begin(), Heading_deg.end(), 0.0);
	double AxisHeadingAngle = sum / Heading_deg.size();

	cosH = cos(AxisHeadingAngle*PI/180);
	sinH = sin(AxisHeadingAngle*PI/180);
	MVA_AxisHeading.update(AxisHeadingAngle);
}


void zebrafishPositionEstimate::computeAxisHeading_MINMAX(void)  {
	AxisHeadingAngle = 0;
	int n = MVA_Heading.RawData.size();
	vector<double> Heading_deg;
	for (int i = 0; i < n; i++)
		Heading_deg.push_back(MVA_Heading.RawData[i]);

	// change the angles in the same range
	double ref = MVA_Heading.RawData[0];
	for (int i = 1; i < n; i++) {
		if (Heading_deg[i] > ref) {
			while (fabs(Heading_deg[i] - ref) > 180)
				Heading_deg[i] -= 360;
		}
		else {
			while (fabs(Heading_deg[i] - ref) > 180)
				Heading_deg[i] += 360;
		}
	}
	double min_value = *std::min_element(Heading_deg.begin(), Heading_deg.end());
	double max_value = *std::max_element(Heading_deg.begin(), Heading_deg.end());

	double AxisHeadingAngle = (min_value + max_value) / 2;
	while (AxisHeadingAngle > 180) AxisHeadingAngle -= 360;
	while (AxisHeadingAngle < -180) AxisHeadingAngle += 360;

	cosH = cos(AxisHeadingAngle*PI/180);
	sinH = sin(AxisHeadingAngle*PI/180);

	MVA_AxisHeading.update(AxisHeadingAngle);
}


void zebrafishPositionEstimate::computeAxisHeading_dth(void)  {
	AxisHeadingAngle = 0;

	int n = MVA_Heading.RawData.size();
	double dtheta_sum = 0;
	for (int i = 1; i < n; i++) {
		// get angle difference b/t [-180, 180)
		double dth = MVA_Heading.RawData[i] - MVA_Heading.RawData[0];
		if (dth < -180) dth += 360;
		if (dth >= 180) dth -= 360;
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

void zebrafishPositionEstimate::computeAxisHeading_fitting_SeparateHeadingPos(void)  {
	AxisHeadingAngle = 0;
	if (MVA_TargetPosition_Heading.MVA_Hist.size() >= 3) {
		int n_item = MVA_TargetPosition_Heading.MVA_Hist.size();
		// allocate memory
		double * X = (double *) malloc (2 * n_item * sizeof(double));
		// --- fitting ---  y = a0*x + a1 =[a0 a1][x;1]
		for (int i = 0; i < n_item; i++) {
			X[2*i] = MVA_TargetPosition_Heading.MVA_Hist[i].x;
			X[2*i + 1] = 1;
		};
		double * Y = (double *) malloc (n_item * sizeof(double));
		for (int i = 0; i < n_item; i++) {
			Y[i] = MVA_TargetPosition_Heading.MVA_Hist[i].y;
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
	Point2d pt_o = TrgtPt - BasePt;
	// 2. rotate (-theta)
	Point2d pt_o_rot(0, 0);
	pt_o_rot.x = cosH * pt_o.x + sinH * pt_o.y;
	// 3. rotate (theta) and get back to original position
	Point2d pt_o_proj;
	ProjPt.x = cosH * pt_o_rot.x + BasePt.x;
	ProjPt.y = sinH * pt_o_rot.x + BasePt.y;

	//AxisHeadingVector = ProjPt - BasePt;
	//double dist =norm_Point2d(AxisHeadingVector);
	//if (dist != 0)
	//	AxisHeadingVector = AxisHeadingVector * (1/dist);
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
			isFishMoving = true;
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

void zebrafishPositionEstimate::getWeight_FishMotion(void) {
	angMomentum_FrameCountAfterMoving = min(angMomentum_FrameCountAfterMoving + 1, (int)weight_angMomentum.size()-1);
    if (FrameCountStopEnabled) {
        if (angularMomentum > angMomentum_Threhold) {
            angMomentum_Threhold = angMomentum_minThrehold;
            angMomentum_FrameCountAfterMoving = 0;
            FrameCountStopEnabled = false;
			isFishMoving = true;
		}
	}
	else {
        if (angularMomentum < angMomentum_Threhold) {
            angMomentum_Threhold = angMomentum_maxThrehold;
            FrameCountStopEnabled = true;
		}
	}
    weight = weight_angMomentum[angMomentum_FrameCountAfterMoving];
	// step 2-2. Determine the end of the movement
	if (weight == 1 && MVA_TargetSpeed_5Step.MVA < stopFishThreshold && isFishMoving == true) {
		isFishMoving = false;
		weight = 0;
	}
	if (isFishMoving == false)
		weight = 0;
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

	double dLR = norm_Point2d(ref->m_eyeLeft.center - ref->m_eyeRight.center);
	double dRY = norm_Point2d(ref->m_eyeLeft.center - ref->m_yolk.center);
	double dLY = norm_Point2d(ref->m_eyeRight.center - ref->m_yolk.center);
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
