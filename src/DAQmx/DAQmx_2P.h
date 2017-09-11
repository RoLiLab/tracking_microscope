#pragma once

#include "NIDAQmx.h"
#include "base/base.h"
#include "DAQmx/DigSignal.h"
#include "DAQmx/AnalogSignal.h"
#include "XPSQ8/XPS_NI.h"
#include "DAQmx/Piezo_NI.h"

#define DAQmxErrChk(functionCall) if( DAQmxFailed(error=(functionCall)) ) goto Error; else
#define PI	3.1415926535
#define ResFreq 7930

static int32 GetTerminalNameWithDevPrefix(TaskHandle taskHandle, const char terminalName[], char triggerName[]);
int32 CVICALLBACK DoneCallback(TaskHandle taskHandle, int32 status, void *callbackData);
int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);
int32 CVICALLBACK ChangeDetectionCallback(TaskHandle taskHandle, int32 signalID, void *callbackData);

class DAQmx_2P
{
public:
	// basic functions
	DAQmx_2P(void);
	~DAQmx_2P(void);

	int			error;
	char		errBuff[2048];
	//------------- NewDesign2015 -------------- //
	volatile bool b_start;
	volatile bool b_stop;
	XPS_NI * XPS_StageSignal;
	Piezo_NI Piezo_scanSingal;
	DigSignal hamamatsu_signal;
	XPS_NI * XPS_StageSignal_new;
	Piezo_NI * Piezo_scanSingal_new;
	DigSignal * hamamatsu_signal_new;
	AnalogSignal AO;
	DigSignal DO;
	uInt64 DO_hamamatsu_globalexpwindow_size;
	// sample count update
	uInt64 totalSamplecount;
	uInt64 totalSamplecount_AO;
	uInt64 writeposition_AO;
	uInt64 nexsamplecount_update;
	uInt64 NIsampleclock;
	uInt64 offsetcount;
	int samples_per_us;
	void initializeNIallSignal(uInt64 _hamamatsu_globalexpwindowsize);
	void updateNIallSignal(void);
	void updateNIallSignal_fixed(void);

	void update_PiezoAll(AnalogSignal * input);
	void update_PiezoHalf(AnalogSignal * input, int n, int offset);
	//------------- NewDesign2015 -------------- //


	// Handlers for New Setup (Zebrafish)
	int DO_ExtTrigger_NumCycleNIR;
	int DO_ExtTrigger_SignalWidth;
	unsigned int getCurrentFrameFromNISampleNumber(int * rem);

	// Signal generation
	int generateNIallSignal(DigSignal * input, AnalogSignal * inputAO);
	int stopNIalltSignal(void);
	int updateDOAOSignal(DigSignal * DO, AnalogSignal * AO, uInt64 writingPtr);

	TaskHandle taskDigPortSignal;
	TaskHandle taskAOPiezoSignal;
	TaskHandle taskAILEDSignal;
	bool taskAIEnabled;
	float64 * AI_LED_Data;
	float32 * AI_LED_Data_sort;
	int AI_LED_Data_sort_sz;
	uInt64 AI_size_per_ch;
	uInt64 AI_n_ch;
	uInt64 AI_offset;
	uInt64 AI_chunksize;
	uInt64 AI_n_sampleRepeat;
	int AI_freq;
	uInt64 getTotalSampleDigPortSignal(void);
	uInt64 getTotalSamplePiezoAO(void);
	uInt64 getWriteCurPosPiezoAO(void);

	// NIR camera sampling time >= 8.3 ms
	// LED camera sampling time >= 71.5 ms
	int DO_ExtTrigger_NumCycleEPI;
	int DO_ExtTrigger_NumCycleNIR_Min;
	int DO_ExtTrigger_NumCycleEPI_Min;
	int DO_ExtTrigger_NumCycleEPI_Readout;
	int DO_ExtTrigger_NumCycleEPI_Exposure;
	uInt8 DO_ExtTriggerForEPICamera_Exp[10000];
	uInt8 DO_ExtTriggerForStageToggle[10000];
	uInt8 DO_ExtTriggerForNIRCamera[10000];
	uInt8 DO_ExtTriggerForEPICamera[10000];
	uInt8 DO_ExtTrigger[10000];
	int DO_ExtTrigger_SampleCountPerChannel;
	int DO_ExtTrigger_NIRCyclesPerEPI;
	TaskHandle DO_ExtTrigger_TaskHandler;
	void getParametersForDOExtTrigger(int _NumCycleNIR, int _NumCycleEPI);



	// double buffer AO signal output (Stage and Piezo)
	TaskHandle AO_TaskHandler_DB;
	float64 AO_Stage_Data[10000];
	uInt64 totalSample; // start from 1;
	// for sawteeth
	TaskHandle  AO0taskHandle;
	bool AO0taskHandle_Enabled;


	// Apply Voltage on AO1

	TaskHandle stage_AO1;
	int InitiateAO1ForPiezo(void);
	int ClearAO1ForPiezo(void);
	void UpdateAO1(double z);

	// Apply Voltage on AO2 & AO3
	bool stage_AO23_Updated;
	bool stage_AO23_ZeroApplied;
	int stage_AO23_FrameNo;

	TaskHandle stage_AO23;
	int InitiateAOForStage(void);
	int ClearAOForStage(void);
	void UpdateAO23(double x, double y);
	int32 ReadAI0(void);
	TaskHandle stage_AI01;
	float64 data_ai[1000];
	float64 piezo_voltage;
	uInt64 totalSamplecount_ai;


	TaskHandle stage_AO23_Simulation;
	// Input Apply (for stage - Generation)
	int AOForStage_SampleNoPerChannel; // 10 samples
	float64 AOForStage_Data[200];
	int InitiateAOForStage_Buffer_WN(int SampleCount, const float64 * data);
	unsigned int getCurrentFrameFromNISampleNumber_WNBUFFER(void);
	int clearAOForStage_Buffer_WN(void);


	// Input reading AI0
	TaskHandle stage_AI23;
	int AIForStage_SampleNoPerChannel; // 300 samples/channel x 2 buffer x 2 channels
	float64 AIForStage_Data[2000];
	int InitiateAIForStage_Buffer(void);
	int UpdateAIForStage_Buffer_CurrRead(int rem); // 300 samples read
	int UpdateAIAOForStage_clear(void);


	// --- need to write ---
	void UpdateAO_DB(float64 vx, float64 vy, float64 vz);
	void getCurBufferCount(void);
	void UpdateAO_DBx(float64 vx);
	void UpdateAO_DBy(float64 vx);
	void UpdateAO_DBz(float64 vx);
	int CurrentBufferCount;
	int CurrentWrittingBufferIndex; // 0 or 1;

	// handler for AO signal for CRS & Piezo
	TaskHandle  Scanning_taskHandle;
	void Scanning_CreateWaveform();
	int Scanning_GenerateWaveform(char chan_output[256], char chan_ExtClk[256], char chan_DigTrig[256] );
	int Scanning_StopWaveform();
	float64 ScanningWaveform_Data[10000];
	int Scanning_NumPtPerCycle;

	// Scanning parameters
	int scanPx;
	bool SingleLayerScan; // true: single layer scan, false: multiple layer scan

	// Linear Scanner parameter
	int LNGV_scanPx;
	float64 LNGV_data_sawtooth[500];
	void LNGV_CreateSawtoothWaveform();
	int LNGV_NumPtPerCycle;
	int LNGV_SweepAngle;
	double LNGV_DegPerVolt;
	int LNGV_SingleLayer_offset_cycle;
	int LNGV_MultiLayer_offset_cycle;

	// Piezo scan
	float64 PZ_data[20];
	void PZ_CreateStairWaveForm();
	int PZ_indexer_Layer;
	int PZ_NumLayerPerCycle;
	int PZ_Pos_BottomLayer;
	int PZ_gap_btLayer;
	double PZ_umPerVolt;

	// Generate clocks (Trigger for NIR imaging)
	TaskHandle TrkClk_TaskHandler;
	uInt8 TrkClk_Data[3000];
	int TrkClk_cycle;
	void Trk_CreateClk();

	// Reading Analog signals
	TaskHandle AI_TaskHandler;

	int AI_NumPtPerCycle;

	double Piezo_VoltageRange;
	double Piezo_VoltageZero;

	// Bode plot (SineFunction Generator)
	int GenerateSineWaveAO0(double Amp, int dt) ;
	int GenerateSawteethAO0(float Amp, float center, int freq);

};
