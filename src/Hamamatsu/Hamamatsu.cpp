#include "Base/base.h"
#include "Hamamatsu/Hamamatsu.h"

Hamamatsu::Hamamatsu(void)
{
	b_stop = false;
	b_start = false;
	b_recstart = false;
	b_recstop = false;
	b_recording = false;
	// ----------------- cam -----------------//
	gain = 0;
	sample_per_us = 10;
	hdcam = NULL;
	buffer = NULL;
	buffer_indexNo = NULL;
	pImg = NULL;
	buffer_stack = NULL;
	buffer_nframe_sec = 30;
	buffer_nframe = 0;
	captureEnabled = false;
	recFrms = 0;
	FrmNo = 0;
	FrmNo_init = 0;
	FrmNo_postproc = 0;
	recFrms_left = 0;
	HalfExposureEnable = false;
	HalfExposureLocationToggle = false;
	isRecoding = false;
	pImg_double = NULL;
	pImg_double_buffer = NULL;
	// hdcam = dcamcon_init_open();
	// ----------------- NI -----------------//
	uint _ROI = Hamamatsu_ROIMax;
	double _freq = 90.0;
	uint64 _exp_us = 1000;
	reset_timeInterval = 1000;
	update(_ROI, _freq, _exp_us);
	getSignal();
	epiDisplay = false;
	img_width = 0;
	img_height = 0;
	img_px = 0;
	b_pImgSetting = true;
	pImgSetting = 0;
	binning = DCAMPROP_BINNING__2;
	sprintf(filesettingName, "d:\\TrackingMicroscopeData\\_setting\\HamamatsuSetting_200Hz_2x2.h5");
}

Hamamatsu::~Hamamatsu(void)
{
}

// ----------------- ================ -----------------//
// ----------------- HAMAMATSU CAMERA -----------------//
// ----------------- ================ -----------------//

HDCAM Hamamatsu::HCAM_init_open()
{
	if (hdcam)
		HCAM_close();

	hdcam = NULL; // failure

	captureEnabled = false;
	int32	nDevice;
	int32	iDevice;

	// initialize DCAM-API
	if( ! dcam_init( NULL, &nDevice, NULL ) ) {
		dcamcon_show_dcamerr( NULL, "dcam_init()" );
		return hdcam;// failure
	}
	if( nDevice <= 0 ) {	// nDevice must be larger than 0
		return hdcam;// failure
	}
	else {
		iDevice = 0; // only one device connected (in our application)
		if( dcam_open( &hdcam, iDevice, NULL ) )
		{
			// success
			return hdcam;
		}
		dcamcon_show_dcamerr( NULL, "dcam_open()", "index is %d\n", iDevice );
	}

	// failure to open the camera
	return hdcam;// failure
}

bool Hamamatsu::HCAM_close(void) {
	if (captureEnabled)
		HCAM_stopCapture();
	// close HDCAM handle
	if (dcam_close( hdcam )) {
		// uninitialize DCAM-API
		bool suc = dcam_uninit( NULL, NULL );
		if (suc) {
			captureEnabled = false;
			return suc;
		}
	}
	return false;
}

bool Hamamatsu::HCAM_settGain(int _gain) {
	gain = _gain;
	return dcam_setpropertyvalue(hdcam, DCAM_IDPROP_CONTRASTGAIN, (double)gain);
}
bool Hamamatsu::HCAM_setting_200Hz(void) {
	bool triggermode = dcam_settriggermode(hdcam, DCAM_TRIGMODE_SYNCREADOUT); // DCAM_TRIGMODE_SYNCREADOUT, DCAM_TRIGMODE_EDGE
	bool triggerpolarity = dcam_settriggerpolarity(hdcam, DCAM_TRIGPOL_NEGATIVE);
	bool  b_binning = dcam_setbinning(hdcam, binning); //
	//bool  b_binning = dcam_setbinning(hdcam, DCAMPROP_BINNING__2); //


	//offset = 128; ROI = 772;
	//bool roimode = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYMODE, 2.0);
	//bool setting_HROI = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYHSIZE, ROI);
	//bool setting_HPOS = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYHPOS, offset);
	double _roiset = 0;
	dcam_getpropertyvalue(hdcam, DCAM_IDPROP_IMAGE_WIDTH, &_roiset);
	img_width = (uint)_roiset;
	dcam_getpropertyvalue(hdcam, DCAM_IDPROP_IMAGE_HEIGHT, &_roiset);
	img_height = (uint)_roiset;
	img_px = img_width* img_height;
	ImageSizeByte = img_px * 2;
	bool exptime = dcam_setexposuretime(hdcam, signal_CamTrigger.n*0.001*0.001);
	bool bittype = dcam_setdatatype(hdcam, DCAM_DATATYPE_UINT16);
	bool sensorset = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SENSORMODE, DCAMPROP_SENSORMODE__AREA);

	bool suc = false;
	if (suc) {
		DCAM_PROPERTYATTR property_data = { 0 };
		property_data.cbSize = sizeof(DCAM_PROPERTYATTR);
		property_data.iProp = DCAM_IDPROP_HIGHDYNAMICRANGE_MODE;
		suc = dcam_getpropertyattr(hdcam, &property_data);
		double valuemin = property_data.valuemin;
		double valuemax = property_data.valuemax;
		double valuedefault = property_data.valuedefault;
	}


	suc = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_CONTRASTGAIN, (double)gain);
	//double a; suc = dcam_getpropertyvalue(hdcam, DCAM_IDPROP_CONTRASTGAIN, &a);

	_DWORD bufsize;
	dcam_getdataframebytes(hdcam, &bufsize);
	return (triggermode && triggerpolarity && exptime && bittype && sensorset);
	//return exptime;
	//return exptime;
}

bool Hamamatsu::HCAM_setting(void) {
	bool triggermode = dcam_settriggermode(hdcam, DCAM_TRIGMODE_SYNCREADOUT); // DCAM_TRIGMODE_SYNCREADOUT, DCAM_TRIGMODE_EDGE
	bool triggerpolarity = dcam_settriggerpolarity(hdcam, DCAM_TRIGPOL_NEGATIVE); //DCAM_TRIGPOL_POSITIVE
	bool exptime = dcam_setexposuretime(hdcam, signal_CamTrigger.n*0.001*0.001);
	bool bittype = dcam_setdatatype(hdcam, DCAM_DATATYPE_UINT16);
	bool sensorset = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SENSORMODE, DCAMPROP_SENSORMODE__AREA);
	bool roimode = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYMODE, 2.0);

	double gain = -1;
	bool suc = dcam_getpropertyvalue(hdcam, DCAM_IDPROP_SENSITIVITY, &gain);

	//bool b_binning = dcam_setbinning(hdcam, binning);
	//double _roiset = 0;
	//dcam_getpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYVSIZE, &_roiset);
	//dcam_getpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYHSIZE, &_roiset);
	//dcam_getpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYVPOS, &_roiset);
	//dcam_getpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYVPOS, &_roiset);

	bool setting_VROI = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYVSIZE, ROI);
	bool setting_HROI = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYHSIZE, ROI);
	int offset = (2048.0 - ROI) / 2;
	bool setting_VPOS = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYVPOS, offset);
	bool setting_HPOS = dcam_setpropertyvalue(hdcam, DCAM_IDPROP_SUBARRAYHPOS, offset);

	return (triggermode && triggerpolarity && exptime && bittype && sensorset && roimode && setting_HPOS && setting_HROI && setting_VPOS && setting_VROI);
	//return exptime;
}

bool Hamamatsu::HCAM_startCapture(void) {
	// Prepares for capturing
	pImg = NULL;
	if (!dcam_precapture(hdcam, DCAM_CAPTUREMODE_SEQUENCE))
		return false;

	return HCAM_resumeCapture();
}

bool Hamamatsu::HCAM_resumeCapture(void) {
	// prepare buffer and attach
	if (!buffer) {
		_DWORD bufsize;
		dcam_getdataframebytes(hdcam, & bufsize);
		buffer_nframe = buffer_nframe_sec*int32(freq);
		//buffer = (uInt16 *) malloc (buffer_nframe * ROI * Hamamatsu_ROIMax * sizeof(uInt16*));
		buffer = (uInt16 **) malloc (buffer_nframe * sizeof(uInt16*));
		buffer_indexNo = (int32 *)malloc(buffer_nframe * sizeof(int32));
		for (int i = 0; i < buffer_nframe; i++) {
			buffer[i] = (uInt16 *)malloc(bufsize);
			buffer_indexNo[i] = -1;
		}
		if (!dcam_attachbuffer(hdcam, (void**)(buffer), buffer_nframe * sizeof(*buffer) ))
			return false;
	}

	// Starts acquisition
	if (!dcam_capture(hdcam)) return false;
		captureEnabled = true;
	return true;
}

bool Hamamatsu::HCAM_stopCapture(void) {
	if (!HCAM_pauseCapture())
		return false;
	if (buffer) {
		dcam_releasebuffer(hdcam);
		for (int i  = 0; i < buffer_nframe; i++)
			free(buffer[i]);
		free(buffer);
		free(buffer_indexNo);
		buffer =NULL;
		buffer_indexNo = NULL;
	}
	return true;
}
bool Hamamatsu::HCAM_pauseCapture(void) {
	HCAM_retreiveBuffer(); // to get a last frame of recording
	// Aborts acquisition
	if (!dcam_idle(hdcam)) return false;
	captureEnabled = false;
	// Interrupts acquisition and sets system in standby status
	//BOOL dcam_wait( HDCAM h, _DWORD* pCode, _DWORD timeout, HDCAMSIGNAL abort );
	// Gets capturing status
	//BOOL dcam_getstatus( HDCAM h, _DWORD* pStatus);
	//BOOL dcam_gettransferinfo( HDCAM h, int32* pNewestFrameIndex, int32* pFrameCount);
	// Frees acquired frame
	//if (!dcam_freeframe(hdcam)) return false;
	return true;
}

bool Hamamatsu::HCAM_retreiveBuffer(void) { // return the pointer of the newest image
	bool bsuc = false;
	if (captureEnabled) {
		_DWORD	dw = 0;
		if(!dcam_wait( hdcam, &dw, DCAM_EVENT_FRAMEEND, NULL ) )
		{
			if (dcam_gettransferinfo(hdcam, &frameIndex, &frameCount )) {	// should be true. (zero based)
				if (frameIndex >= 0 && frameCount > 0) {
					if (frameCount > FrmNo + FrmNo_init) {
						bsuc = true;
						if (FrmNo_init==0)
							FrmNo_init = frameCount;
						else
							FrmNo_Drop += (frameCount - (FrmNo + FrmNo_init + 1));

						FrmNo = frameCount - FrmNo_init;
						buffer_indexNo[FrmNo%buffer_nframe] = frameIndex;
						if (b_pImgSetting || (FrmNo % 2 == pImgSetting))
							pImg = buffer[frameIndex]; // selecting A or B image
					}
				}
			}
		}
	}
	return bsuc;
}
void Hamamatsu::setPostTrigSet(double _offsettime_sec) {
	int32 _numofImage_before_trigger = INT32(_offsettime_sec*freq);
	if (frameCount > _numofImage_before_trigger)
		recFrms_Start = frameCount - _numofImage_before_trigger;
	else
		recFrms_Start = 0;
	recFrms = recFrms_Start;
	recFrms_End = 0;
	ImageSizeByte = sizeof(uInt16)*ROI*ROI;
	recFrms = 0;
	recFrms_left = _numofImage_before_trigger;
	isRecoding = true;
}
void Hamamatsu::setEndRecTrigSet(void) {
	recFrms_End = frameCount;
	isRecoding = false;
}
void Hamamatsu::HCAM_REC_binary_n(std::ofstream * pwriter, int32 CurRecIdx, int32 imgcount) { // record in binary
	if (CurRecIdx <= frameCount) {
		int32 curIdx = (CurRecIdx) % buffer_nframe;
		pwriter->write((char*)((char*)(buffer[buffer_indexNo[curIdx]])), ImageSizeByte*imgcount);
		if (recFrms_End > 0) {
			recFrms_left = recFrms_End - (CurRecIdx + imgcount);
		}
		else {
			recFrms_left = frameCount - (CurRecIdx + imgcount);
		}
	}
}
void Hamamatsu::HCAM_REC_binary_win32(HANDLE * pwriter, int32 CurRecIdx) { // record in binary
	BOOL bErrorFlag = FALSE;
	if (CurRecIdx <= frameCount) {
		int32 curIdx = (CurRecIdx) % buffer_nframe;
		int32 _imgsize_byte = ImageSizeByte;

		//pwriter->write((char*)(buffer[curIdx]), ImageSizeByte);
		DWORD dwBytesWritten = 0;
		bErrorFlag = WriteFile(
			*pwriter,           // open file handle
			(char*)(buffer[buffer_indexNo[curIdx]]),      // start of data to write
			ImageSizeByte,  // number of bytes to write
			&dwBytesWritten, // number of bytes that were written
			NULL);
	}
}
void Hamamatsu::HCAM_REC_binary(std::ofstream * pwriter, int32 CurRecIdx ) { // record in binary
	if (CurRecIdx <= frameCount) {
		int32 curIdx = (CurRecIdx)%buffer_nframe;
		pwriter->write((char*)(buffer[buffer_indexNo[curIdx]]), ImageSizeByte);
		if (recFrms_End > 0) {
			recFrms_left = recFrms_End - (CurRecIdx);
		}
		else {
			recFrms_left = frameCount - (CurRecIdx);
		}
	}
}

int32 Hamamatsu::next_recimgnumber(int32 idx, int32 idx_addition, int32 idx_max) {
	int32 delta = idx_addition;
	if (idx_max > 0)
		if ((idx_max - idx) < idx_addition)
			delta = (idx_max - idx);
	int32 delta2endframebuffer = (int32)(buffer_nframe - ((idx) % buffer_nframe));
	if (delta2endframebuffer < delta)
		delta = delta2endframebuffer;
	return delta;
}

void Hamamatsu::HCAM_HDF5REC(HDF5ImageWriter * pwriter, int32 CurRecIdx) { // record in HDF5 file
	try {
		if (CurRecIdx <= frameCount) {
			int32 curIdx = (CurRecIdx) % buffer_nframe;
			//pwriter->write((char*)(buffer[curIdx]), ImageSizeByte);
			pwriter->write((uint8_t*)(buffer[buffer_indexNo[curIdx]]));
			if (recFrms_End > 0) {
				recFrms_left = recFrms_End - (CurRecIdx);
			}
			else {
				recFrms_left = frameCount - (CurRecIdx);
			}
			if (CurRecIdx % 8 == 0 && CurRecIdx > 0)
				pwriter->flush();
		}
	}
	catch (exception& e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "Recording (Fluorescent) thread (HDF5 Error): %s\n", e.what());
		fclose(ofp);
	}
}

void Hamamatsu::HCAM_IntRangeUpdate(uint16 * _img) {
	if (_img == NULL)
		return;
	intensityMax = 0;
	if (pImg && captureEnabled) {
		for (int i = 0; i < ROI*ROI; i++) {
			if (pImg[i] > intensityMax)
				intensityMax = pImg[i];
		}
	}
}

// ----------------- ==  -----------------//
// ----------------- NI -----------------//
// ----------------- == -----------------//
void Hamamatsu::update(uint _ROI, double _freq, uInt64 _exp_us)
{
	updateROI(_ROI);
	updatefreq(_freq);
	updateExp(_exp_us);
	ImageSizeByte = sizeof(uInt16)*ROI*ROI;
}

void Hamamatsu::getSignal(uint _ROI, double _freq, uInt64 _exp_us)
{
	update(_ROI, _freq, _exp_us);
	getSignal();
}

void Hamamatsu::updateROI(uint _ROI)
{
	ROI = _ROI;
	// 1. set ROI -> get freq_max
	if (ROI%2 != 0) ROI = ROI + 1;
	if (ROI > Hamamatsu_ROIMax) ROI = Hamamatsu_ROIMax;
	delay_afterTrigger_us = (Hamamatsu_DELAYCYCLE+1.0)*Hamamatsu_H; // 1.0 is jittering (max)
	readout_us = Hamamatsu_H * (double)(ROI/2);
	globalExpWindowStart_us = (uInt64)ceil(delay_afterTrigger_us + readout_us);
	freq_max = 1000000.0 / (double)globalExpWindowStart_us;
}

void Hamamatsu::updatefreq(double _freq)
{
	freq = _freq;
	if (freq > freq_max)
		freq = freq_max;
	// 2. set freq -> get exp_max
	cycle_us = (uInt64)(1000000.0/freq);
	expmax_us = cycle_us - globalExpWindowStart_us - 2*Hamamatsu_EXPOFFSETMIN;
}

void Hamamatsu::updateExp(uInt64 _exp_us)
{
	exp_us = _exp_us;
	if (exp_us > expmax_us)
		exp_us = expmax_us;
	// 3. set exp_max -> ready for generating signal
	offset = (uInt64)((cycle_us - globalExpWindowStart_us - exp_us)/2);
}

void Hamamatsu::getSignal_trigger(void)
{
	uInt64 edgeNo[2] = {0};
	edgeNo[0] = globalExpWindowStart_us;
	edgeNo[1] = cycle_us; // 100 us margin at the end
	signal_CamTrigger.generateDigSignal(2, edgeNo, 1<<2);
}

void Hamamatsu::getSignal_Exp(void)
{
	uInt64 edgeNo[6] = { 0 };
	edgeNo[0] = 0;
	edgeNo[1] = globalExpWindowStart_us;
	if (HalfExposureEnable && !HalfExposureLocationToggle)
		edgeNo[2] = edgeNo[1] + exp_us/2;
	else
		edgeNo[2] = edgeNo[1] + exp_us;
	edgeNo[3] = globalExpWindowStart_us + cycle_us;
	if (HalfExposureEnable && HalfExposureLocationToggle)
		edgeNo[4] = edgeNo[1] + exp_us/2 + cycle_us;
	else
		edgeNo[4] = edgeNo[1] + exp_us + cycle_us;
	edgeNo[5] = 2*cycle_us;
	signal_LEDExp.generateDigSignal(6, edgeNo, 1<<3);
}

void Hamamatsu::getSignal_DMD(void)
{
	uInt64 edgeNo[5] = {0};

	edgeNo[0] = 0;
	edgeNo[1] = globalExpWindowStart_us - 106;
	edgeNo[2] = cycle_us;
	edgeNo[3] = cycle_us + globalExpWindowStart_us - 106;
	edgeNo[4] = 2*cycle_us;

	signal_DMD.generateDigSignal(5, edgeNo, 1<<4);
}

void Hamamatsu::getSignal(void)
{
	if (ROI > Hamamatsu_ROIMax) return;
	if ((double)freq > freq_max) return;
	if (exp_us > expmax_us) return;

	getSignal_trigger();
	getSignal_Exp();
	getSignal_DMD();
	signal.resetSignal(2*cycle_us, 0);
	signal.mergeSignal(&signal_CamTrigger, 0, 0, 0);
	signal.mergeSignal(&signal_LEDExp, 0, 0, 1<<3);
	signal.mergeSignal(&signal_DMD, 0, 0, 0);
}


// ----------------- ================ -----------------//
// ----------------- HAMAMATSU CAMERA -----------------//
// ----------------- ================ -----------------//
// show HDCAM camera information by text.
void dcamcon_show_camera_information( HDCAM hdcam )
{
	char	buf[ 256 ];

	dcam_getstring( hdcam, DCAM_IDSTR_VENDOR,			buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_VENDOR         = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_MODEL,			buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_MODEL          = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_CAMERAID,			buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_CAMERAID       = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_BUS,				buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_BUS            = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_CAMERAVERSION,	buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_CAMERAVERSION  = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_DRIVERVERSION,	buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_DRIVERVERSION  = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_MODULEVERSION,	buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_MODULEVERSION  = %s\n", buf );

	dcam_getstring( hdcam, DCAM_IDSTR_DCAMAPIVERSION,	buf, sizeof( buf ) );
	printf( "DCAM_IDSTR_DCAMAPIVERSION = %s\n", buf );
}

void dcamcon_show_dcamerr( HDCAM hdcam, const char* apiname, const char* fmt, ...  )
{
	char	buf[ 256 ];
	memset( buf, 0, sizeof( buf ) );

	// get error information
	int32	err = dcam_getlasterror( hdcam, buf, sizeof( buf ) );
	printf( "failure: %s returns 0x%08X\n", apiname, err );
	if( buf[ 0 ] )	printf( "%s\n", buf );

	if( fmt != NULL )
	{
		va_list	arg;
		va_start(arg,fmt);
		vprintf( fmt, arg );
		va_end(arg);
	}
}

// ---------------------------------


void Hamamatsu::getSignal_trigger_PG(uInt64 _cycle_us)
{
	uInt64 edgeNo[3] = { 0 };
	edgeNo[0] = 0; // fa edge trigger
	edgeNo[1] = (_cycle_us / 2)*sample_per_us; // rising edge trigger
	edgeNo[2] = _cycle_us*sample_per_us; // rising edge trigger
	signal_CamTrigger.generateDigSignal(3, edgeNo, 1 << 2);
}

void Hamamatsu::getSignal_Exp_PG(uInt64 _cycle_us, uInt64 _exp_us, double * LED_offset, double * _exp_us_diff)
{
	if ((_exp_us == _exp_us_diff[0]) && (_exp_us == _exp_us_diff[1])) {
		uInt64 edgeNo[2] = { 0 };
		edgeNo[0] = 0;
		edgeNo[1] = (2 * _cycle_us)*sample_per_us;
		signal_LEDExp.generateDigSignal(2, edgeNo, 1 << 3);
	}
	else if (_exp_us == _exp_us_diff[1]) {
		uInt64 edgeNo[4] = { 0 };
		edgeNo[0] = 0;
		edgeNo[1] = (_cycle_us - LED_offset[0] - _exp_us)*sample_per_us;
		edgeNo[2] = (_cycle_us - LED_offset[0] - _exp_us_diff[0])*sample_per_us;
		edgeNo[3] = (2 * _cycle_us)*sample_per_us;
		signal_LEDExp.generateDigSignal(4, edgeNo, 1 << 3);
	}
	else if (_exp_us == _exp_us_diff[0]) {
		uInt64 edgeNo[4] = { 0 };
		edgeNo[0] = 0;
		edgeNo[1] = (_cycle_us + LED_offset[1])*sample_per_us;
		edgeNo[2] = (_cycle_us + LED_offset[1] + _exp_us - _exp_us_diff[1])*sample_per_us;
		edgeNo[3] = (2 * _cycle_us)*sample_per_us;
		signal_LEDExp.generateDigSignal(4, edgeNo, 1 << 3);
	}
	else {
		uInt64 edgeNo[6] = { 0 };
		edgeNo[0] = 0;
		edgeNo[1] = (_cycle_us - LED_offset[0] - _exp_us)*sample_per_us;
		edgeNo[2] = (_cycle_us - LED_offset[0] - _exp_us_diff[0])*sample_per_us;
		edgeNo[3] = (_cycle_us + LED_offset[1])*sample_per_us;
		edgeNo[4] = (_cycle_us + LED_offset[1] + _exp_us - _exp_us_diff[1])*sample_per_us;
		edgeNo[5] = (2 * _cycle_us)*sample_per_us;
		signal_LEDExp.generateDigSignal(6, edgeNo, 1 << 3);
	}
}

void Hamamatsu::getSignal_blockingtime(uInt64 _cycle_us, double * _shutter_offtime)
{
	uInt64 edgeNo[4] = { 0 };
	edgeNo[0] = 0;
	edgeNo[1] = _shutter_offtime[0] * sample_per_us;
	edgeNo[2] = _shutter_offtime[1] * sample_per_us;
	edgeNo[3] = _cycle_us*sample_per_us;
	signal_shutterOff.generateDigSignal(4, edgeNo, 1 << 6);
}


void Hamamatsu::getSignal_DMD_PG(uInt64 _cycle_us, uInt64 _exp_us, double * LED_offset, double * DMD_offset)
{
	//uInt64 edgeNo[6] = { 0 };
	//edgeNo[0] = 0;
	//edgeNo[1] = _cycle_us - LED_offset[0] - _exp_us - DMD_offset[0];
	//edgeNo[2] = _cycle_us - LED_offset[0] - DMD_offset[0];
	//edgeNo[3] = _cycle_us + LED_offset[1] - DMD_offset[1];
	//edgeNo[4] = _cycle_us + LED_offset[1] + _exp_us - DMD_offset[1];
	//edgeNo[5] = 2 * _cycle_us;
	//signal_DMD.generateDigSignal(6, edgeNo, (1 << 4) | (1 << 6));

	uInt64 edgeNo[4] = { 0 };
	double LED_offset_middle = 0;// -LED_offset[0];// ;
	edgeNo[0] = 0;
	edgeNo[1] = (_cycle_us + LED_offset_middle - DMD_offset[0])*sample_per_us;
	edgeNo[2] = (_cycle_us + 30)*sample_per_us;
	edgeNo[3] = (2 * _cycle_us)*sample_per_us;

	signal_DMD.generateDigSignal(4, edgeNo, (1 << 4));
}

void Hamamatsu::getSignal_PG(double _freq, uInt64 _exp_us, double * LED_offset, double  * DMD_offset, double * _exp_us_diff, double * _shutter_offtime)
{
	cycle_us = (uInt64)(1000000.0 / _freq);
	exp_us = _exp_us;
	getSignal_trigger_PG(cycle_us);
	getSignal_Exp_PG(cycle_us, _exp_us, LED_offset, _exp_us_diff);
	getSignal_DMD_PG(cycle_us, _exp_us, LED_offset, DMD_offset);
	getSignal_blockingtime(cycle_us, _shutter_offtime);

	signal.resetSignal(2 * cycle_us * sample_per_us, 0);
	signal.mergeSignal(&signal_CamTrigger, 0, 0, 0);
	signal.mergeSignal(&signal_LEDExp, 0, 0, 0);
	signal.mergeSignal(&signal_DMD, 0, 0, 0);
	signal.mergeSignal(&signal_shutterOff, 0, 0, 0);

	globalExpWindowStart_us = cycle_us - LED_offset[0] - _exp_us;
}
