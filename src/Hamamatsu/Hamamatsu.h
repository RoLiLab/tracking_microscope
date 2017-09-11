#ifndef _Hamamatsu_H
#define _Hamamatsu_H
#include <Windows.h>
#include "../PointGreyCamera/PointGreyCamera.h"
#include "DAQmx/DigSignal.h"
#include "HDF5/HDF5ImageWriter.h"
#include "dcamapi.h"
#include "dcamprop.h"

#define Hamamatsu_H	9.74436
#define Hamamatsu_DELAYCYCLE 9
#define Hamamatsu_ROIMax 2048
#define Hamamatsu_EXPOFFSETMIN 10


class PointGreyCamera;
class Hamamatsu
{
public:
	// basic functions
	Hamamatsu(void);
	~Hamamatsu(void);
	char filesettingName[1024];
	volatile bool b_stop;
	volatile bool b_start;
	volatile bool b_recstart;
	volatile bool b_recstop;
	volatile bool b_recording;
	int32 FrmNo_init;
	int32 FrmNo_Drop;
	int32 FrmNo_postproc;
	int32 FrmNo;
	int32 recFrms_Start;
	int32 recFrms_End;
	int32 recFrms_left;
	int32 recFrms;
	int32 ImageSizeByte;
	uint64 reset_timeInterval;
	void HCAM_IntRangeUpdate(uint16 * _img);
	int32 getbufferindexNo(int32 * _frmNo);
	int32 sample_per_us;

	PointGreyCamera GS3;

	// ----------------- cam -----------------//
	HDCAM hdcam;
	HDCAM HCAM_init_open(void);
	bool HCAM_close(void);
	bool HCAM_setting(void);
	bool HCAM_setting_200Hz(void);
	bool HCAM_settGain(int _gain);
	bool HCAM_startCapture(void);
	bool HCAM_resumeCapture(void);
	bool HCAM_stopCapture(void);
	bool HCAM_pauseCapture(void);
	void HCAM_HDF5REC(HDF5ImageWriter * pwriter, int32 n);

	bool HCAM_retreiveBuffer(void);
	bool captureEnabled;
	bool epiDisplay;
	uInt16 ** buffer;
	uInt16 ** buffer_stack;
	uInt16 ** buffer_stack_raw;
	int32 * buffer_indexNo;
	//uInt16 * buffer;
	int32 buffer_nframe;
	int32 buffer_nframe_sec;
	int32 frameIndex;
	int32 frameCount;
	uInt16 * pImg; // current image buffer pointer
	uInt16 * pImg_double; // current image buffer pointer
	uInt16 ** pImg_double_buffer; // current image buffer pointer
	uInt16 intensityMax;
	volatile bool isRecoding;
	void setPostTrigSet(double _offsettime_sec);
	void setEndRecTrigSet(void);
	void HCAM_REC_binary(std::ofstream * pwriter, int32 CurRecIdx);
	void HCAM_REC_binary_win32(HANDLE * pwriter, int32 CurRecIdx);
	void HCAM_REC_binary_n(std::ofstream * pwriter, int32 CurRecIdx, int32 imgcount);
	int32 next_recimgnumber(int32 idx, int32 idx_addition, int32 idx_max);
	// ----------------- NI -----------------//
	uint ROI;
	uint img_width;
	uint img_height;
	uint img_px;
	double freq_max;
	double freq;
	uInt64 expmax_us;
	uInt64 exp_us;
	DigSignal signal_CamTrigger;
	DigSignal signal_LEDExp;
	DigSignal signal_DMD;
	DigSignal signal_PiezoTrigger;
	DigSignal signal;
	DigSignal signal_shutterOff;
	void getSignal(void);
	void getSignal(uint _ROI, double _freq, uInt64 _exp_us);
	void update(uint _ROI, double _freq, uInt64 _exp_us);
	bool HalfExposureEnable;
	bool HalfExposureLocationToggle;
	int binning;
	uInt64 globalExpWindowStart_us;

	int pImgSetting;
	bool b_pImgSetting;
	int gain;

	void getSignal_PG(double _freq, uInt64 _exp_us, double * LED_offset, double * DMD_offset, double * _exp_us_diff, double * _shutter_offtime);
	void getSignal_trigger_PG(uInt64 _cycle_us);
	void getSignal_Exp_PG(uInt64 _cycle_us, uInt64 _exp_us, double * LED_offset, double * _exp_us_diff);
	void getSignal_DMD_PG(uInt64 _cycle_us, uInt64 _exp_us, double * LED_offset, double * DMD_offset);
	void getSignal_blockingtime(uInt64 _cycle_us, double * _shutter_offtime);

private:
	// ----------------- NI -----------------//
	double delay_afterTrigger_us;
	double readout_us;
	uInt64 cycle_us;
	uInt64 offset;
	void updateROI(uint _ROI);
	void updatefreq(double _freq);
	void updateExp(uInt64 _exp_us);
	void getSignal_trigger(void);
	void getSignal_Exp(void);
	void getSignal_DMD(void);


};

void dcamcon_show_dcamerr( HDCAM hdcam, const char* apiname, const char* fmt = NULL, ...  );
void dcamcon_show_camera_information( HDCAM hdcam );
#endif
