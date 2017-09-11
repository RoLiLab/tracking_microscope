#include "Base/base.h"
// NI board 64bit
#include "DAQmx_2P.h"

std::mutex PosReading_mutex;

DAQmx_2P::DAQmx_2P(void)
{
	b_start = false;
	b_stop = false;
	totalSamplecount = 0;
	nexsamplecount_update = 0;
	samples_per_us = 10;
	NIsampleclock = 1000 * 1000 * samples_per_us;
	offsetcount = NIsampleclock;
	//Piezo_scanSingal = NULL;
	XPS_StageSignal = NULL;
	Piezo_scanSingal_new = NULL;
	XPS_StageSignal_new = NULL;
	hamamatsu_signal_new = NULL;
	error=0;
	taskDigPortSignal = NULL;
	taskAOPiezoSignal = NULL;
	taskAILEDSignal = NULL;
	// Handlers for New Setup (Zebrafish)
	// NIR camera sampling time >= 8.3 ms
	// LED camera sampling time >= 71.5 ms
	stage_AO23_Updated = false;
	DO_ExtTrigger_NumCycleNIR_Min = 21;
	DO_ExtTrigger_NumCycleEPI_Min = 100;
	DO_ExtTrigger_NIRCyclesPerEPI = 2;
	DO_ExtTrigger_NumCycleEPI_Readout = 84; // readout time for EPI camera
	DO_ExtTrigger_NumCycleEPI_Exposure = 0; // 10 = 1 ms;
	stage_AO23_FrameNo = 0;
	DO_ExtTrigger_TaskHandler = NULL;
	getParametersForDOExtTrigger(40, 100);
	//generateDOExtTriggerWaveform();
	stage_AO23 = NULL;
	stage_AO23_Simulation = NULL;
	stage_AO1 = NULL;

	AOForStage_SampleNoPerChannel = 2; // MAX 100
	//InitiateAOForStage();
	InitiateAO1ForPiezo();

	Piezo_VoltageRange = 10;
	Piezo_VoltageZero = 0;
	AO0taskHandle = 0;
	AO0taskHandle_Enabled = false;
	AIForStage_SampleNoPerChannel = 300;
	//InitiateAIForStage_Buffer();

	DO_hamamatsu_globalexpwindow_size = 0;
	// Ext
	AI_n_ch = 4;
	AI_freq = 100*1000;
	AI_size_per_ch = 50000;
	AI_n_sampleRepeat = 50;
	AI_offset = 0;
	AI_chunksize = 100;
	AI_LED_Data_sort_sz = AI_chunksize * AI_n_sampleRepeat * AI_n_ch* sizeof(float32);
	AI_LED_Data = (float64 *)malloc(AI_size_per_ch*AI_n_ch* sizeof(float64));
	AI_LED_Data_sort = (float32 *)malloc(AI_LED_Data_sort_sz);
	// 100000sample/sec * 500 us = 50 sample/window
	// 50K/100Ksample = 0.5sec * 100Hz = 50 times repeat
	taskAIEnabled = false;
}


DAQmx_2P::~DAQmx_2P(void)
{
	UpdateAO23(0.0, 0.0);
	free(AI_LED_Data);
}



int DAQmx_2P::generateNIallSignal(DigSignal * input, AnalogSignal * inputAO) {
	//uInt32      data=0xfffffff0;
	//char        errBuff[2048]={'\0'};
	stopNIalltSignal();

	int32		written;
	char    trigName[256];
	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&taskDigPortSignal));
	DAQmxErrChk (DAQmxCreateDOChan(taskDigPortSignal,"Dev1/port0","",DAQmx_Val_ChanForAllLines));
	DAQmxErrChk (DAQmxCfgSampClkTiming(taskDigPortSignal,"",NIsampleclock,DAQmx_Val_Rising,DAQmx_Val_ContSamps,input->n));
	DAQmxErrChk (GetTerminalNameWithDevPrefix(taskDigPortSignal,"do/SampleClock",trigName));

	/*********************************************/
	// DAQmx Configure, Write & Start Code (Analog signal)
	/*********************************************/
	// editing code
	if (taskAIEnabled) {
		DAQmxErrChk(DAQmxCreateTask("", &taskAILEDSignal));
		DAQmxErrChk(DAQmxCreateAIVoltageChan(taskAILEDSignal, "Dev1/ai0:3", "", DAQmx_Val_Diff, 0.0, 5.0, DAQmx_Val_Volts, NULL));
		DAQmxErrChk(DAQmxCfgSampClkTiming(taskAILEDSignal, "", float64(AI_freq), DAQmx_Val_Falling, DAQmx_Val_ContSamps, AI_size_per_ch));
		DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(taskAILEDSignal, trigName, DAQmx_Val_Rising));
	}
	/*********************************************/
	// DAQmx Configure, Write & Start Code (Analog signal)
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&taskAOPiezoSignal));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(taskAOPiezoSignal,"Dev1/ao2","",0.0,10.0,DAQmx_Val_Volts,NULL));
	DAQmxErrChk(DAQmxCfgSampClkTiming(taskAOPiezoSignal, "", Piezo_scanSingal.steps_update * 2, DAQmx_Val_Falling, DAQmx_Val_ContSamps, inputAO->n)); //"PFI1"
	DAQmxErrChk (DAQmxCfgDigEdgeStartTrig(taskAOPiezoSignal,trigName,DAQmx_Val_Rising));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk(DAQmxSetWriteRegenMode(taskDigPortSignal, DAQmx_Val_AllowRegen)); // DAQmx_Val_AllowRegen // DAQmx_Val_DoNotAllowRegen
	DAQmxErrChk(DAQmxSetWriteRegenMode(taskAOPiezoSignal, DAQmx_Val_AllowRegen)); // DAQmx_Val_AllowRegen // DAQmx_Val_DoNotAllowRegen
	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk(DAQmxWriteAnalogF64(taskAOPiezoSignal, (int32)inputAO->n, 0, 10.0, DAQmx_Val_GroupByChannel, inputAO->data, &written, NULL));
	DAQmxErrChk (DAQmxWriteDigitalU32(taskDigPortSignal,(int32)input->n,0,10.0,DAQmx_Val_GroupByChannel,input->data,&written,NULL));
	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	totalSamplecount = 1;
	DAQmxErrChk(DAQmxStartTask(taskAOPiezoSignal));
	DAQmxErrChk(DAQmxStartTask(taskDigPortSignal));

	return 1;
Error:
	if (DAQmxFailed(error))
		printf("DAQmx Error: %s\n", errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return stopNIalltSignal();
}



void DAQmx_2P::update_PiezoAll(AnalogSignal * AO) {
	//uint64 curPos;
	//int32 totalGenerated_sample = ((int32)Piezo_scanSingal.totalupdatedSample - (int32)Piezo_scanSingal.steps_per_sweep);
	//int32 offset_fromCurrentPosition = totalGenerated_sample + Piezo_scanSingal.steps_update*2;
	//DAQmxErrChk(DAQmxResetWriteOffset(taskAOPiezoSignal));
	//DAQmxErrChk(DAQmxGetWriteCurrWritePos(taskAOPiezoSignal, &curPos));
	//DAQmxErrChk(DAQmxSetWriteOffset(taskAOPiezoSignal, totalGenerated_sample));

	int32 written = 0;
	DAQmxErrChk(DAQmxWriteAnalogF64(taskAOPiezoSignal, (int32)AO->n, 0, 0.25, DAQmx_Val_GroupByChannel, AO->data, &written, NULL));

	//if (totalGenerated_sample > 50)
	//	curPos = 0;
	//DAQmxErrChk(DAQmxGetWriteCurrWritePos(taskAOPiezoSignal, &curPos));

	return;
Error:
	if (DAQmxFailed(error))
		printf("DAQmx Error: %s\n", errBuff);
	return;
}

void DAQmx_2P::update_PiezoHalf(AnalogSignal * AO, int n, int offset) {
	int32 written = 0;
	DAQmxErrChk(DAQmxWriteAnalogF64(taskAOPiezoSignal, (int32)n, 0, 10.0, DAQmx_Val_GroupByChannel, &AO->data[offset], &written, NULL));
Error:
	if (DAQmxFailed(error))
		printf("DAQmx Error: %s\n", errBuff);
	return;
}


int DAQmx_2P::stopNIalltSignal(void) {
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( taskDigPortSignal!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskDigPortSignal);
		DAQmxClearTask(taskDigPortSignal);
	}
	if( taskAOPiezoSignal!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskAOPiezoSignal);
		DAQmxClearTask(taskAOPiezoSignal);
	}

	if (taskAILEDSignal != 0) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskAILEDSignal);
		DAQmxClearTask(taskAILEDSignal);
	}
	DAQmxWaitUntilTaskDone(taskAILEDSignal, 10.0);
	DAQmxWaitUntilTaskDone(taskDigPortSignal, 10.0);
	DAQmxWaitUntilTaskDone(taskAOPiezoSignal, 10.0);
	taskAILEDSignal = NULL;
	taskDigPortSignal = NULL;
	taskAOPiezoSignal = NULL;
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return 0;
}
int DAQmx_2P::updateDOAOSignal(DigSignal * DO, AnalogSignal * AO,uInt64 writingPtr) {
	int32		written;
	if (DO != NULL && taskDigPortSignal!= NULL) {
		//DAQmxErrChk (DAQmxSetWriteRelativeTo(taskDigPortSignal, DAQmx_Val_FirstSample));
		//DAQmxErrChk (DAQmxSetWriteOffset(taskDigPortSignal, writingPtr));
		DAQmxErrChk (DAQmxWriteDigitalU32(taskDigPortSignal,(int32)DO->n,0,10.0,DAQmx_Val_GroupByChannel,DO->data,&written,NULL));
	}

	if (AO != NULL && taskAOPiezoSignal!= NULL) {
		//DAQmxErrChk (DAQmxSetWriteRelativeTo(taskAOPiezoSignal, DAQmx_Val_FirstSample));
		//DAQmxErrChk (DAQmxSetWriteOffset(taskAOPiezoSignal, writingPtr));
		//DAQmxErrChk (DAQmxWriteAnalogF64(taskAOPiezoSignal,(int32)AO->n,0,10.0,DAQmx_Val_GroupByChannel,AO->data,&written,NULL));
	}
	return 1;
Error:
	return stopNIalltSignal();
}

uInt64 DAQmx_2P::getTotalSamplePiezoAO(void) {
	//std::lock_guard<std::mutex> lock(PosReading_mutex);
	uInt64 totalSample_cur;

	if (taskAOPiezoSignal != 0) {
		// 1. get the total sample
		DAQmxGetWriteTotalSampPerChanGenerated(taskAOPiezoSignal, &totalSample_cur);
	}
	if (totalSample_cur > totalSamplecount_AO)
		totalSamplecount_AO = totalSample_cur;


	return totalSamplecount_AO;
}

uInt64 DAQmx_2P::getWriteCurPosPiezoAO(void) {
	//std::lock_guard<std::mutex> lock(PosReading_mutex);
	uInt64 totalSample_cur;

	if (taskAOPiezoSignal != 0) {
		// 1. get the total sample
		DAQmxGetWriteCurrWritePos(taskAOPiezoSignal, &writeposition_AO);
	}
	return writeposition_AO;
}

uInt64 DAQmx_2P::getTotalSampleDigPortSignal(void) {
	//std::lock_guard<std::mutex> lock(PosReading_mutex);
	uInt64 totalSample_cur;
	// check NI handler should be ready.
	if (taskDigPortSignal != 0) {
		// 1. get the total sample
		DAQmxGetWriteTotalSampPerChanGenerated(taskDigPortSignal, &totalSample_cur);
	}
	if (totalSample_cur > totalSamplecount)
		totalSamplecount = totalSample_cur;


	//if (taskAOPiezoSignal != 0) {
	//	// 1. get the total sample
	//	DAQmxGetWriteTotalSampPerChanGenerated(taskAOPiezoSignal, &totalSample_cur);
	//}
	//if (totalSample_cur > totalSamplecount_AO)
	//	totalSamplecount_AO = totalSample_cur;


	return totalSamplecount;
}
int32 DAQmx_2P::ReadAI0(void) {
	int32 read = 0;
	if (taskAILEDSignal && AI_LED_Data) {
		DAQmxReadAnalogF64(taskAILEDSignal, AI_size_per_ch, 10.0, DAQmx_Val_GroupByChannel, AI_LED_Data, AI_n_ch*AI_size_per_ch, &read, NULL);
		for (int i = 0; i < AI_n_sampleRepeat * AI_n_ch; i++) {
			int loc0 = i * 1000 + AI_offset; // 1000 -> every 10ms, 50 offset
			for (int j = 0; j < AI_chunksize; j++) {
				AI_LED_Data_sort[i*AI_chunksize + j] = (float32)AI_LED_Data[loc0 + j];
			}
		}
	}
	return read;
}

unsigned int DAQmx_2P::getCurrentFrameFromNISampleNumber(int * rem) {

	//std::lock_guard<std::mutex> lock(PosReading_mutex);
	// FrmNo is starting from 1 (1-based)
	// return 0 mean, it is not ready
	unsigned int FrmNo = 0;
	uInt64 totalSample;
	// check NI handler should be ready.
	if (taskDigPortSignal != 0) {
		// 1. get the total sample
		totalSample = getTotalSampleDigPortSignal()/100;
		if (totalSample <= DO_ExtTrigger_SignalWidth) {
			// at the beginning, to get where is the falling edge
			FrmNo = 0;
			rem[0] = 0;
		}
		else {
			// after fisrt edge, count the number after falling edge (Trigger)
			FrmNo  = (totalSample-DO_ExtTrigger_SignalWidth) / (DO_ExtTrigger_NumCycleNIR) + 1;
			rem[0] = (totalSample-DO_ExtTrigger_SignalWidth) % (DO_ExtTrigger_NumCycleNIR);
		}

	}
	return FrmNo;
}



void DAQmx_2P::getParametersForDOExtTrigger(int _NumCycleNIR, int _NumCycleEPI) {
	// 1. set the number of cycles for NIR camera
	if (_NumCycleNIR>=DO_ExtTrigger_NumCycleNIR_Min)
		DO_ExtTrigger_NumCycleNIR = _NumCycleNIR;
	else
		DO_ExtTrigger_NumCycleNIR = DO_ExtTrigger_NumCycleNIR_Min;

	// 1. set the number of cycles for FLR camera
	if (_NumCycleEPI>=DO_ExtTrigger_NumCycleEPI_Min)
		DO_ExtTrigger_NumCycleEPI = _NumCycleEPI;
	else
		DO_ExtTrigger_NumCycleEPI = DO_ExtTrigger_NumCycleEPI_Min;

	// 3. Set the length of the total buffer
	int Quotient;
	int remainder;
	if (DO_ExtTrigger_NumCycleEPI >= 2*DO_ExtTrigger_NumCycleNIR) {
		Quotient = DO_ExtTrigger_NumCycleEPI/(2*DO_ExtTrigger_NumCycleNIR);
		DO_ExtTrigger_SampleCountPerChannel = (2*DO_ExtTrigger_NumCycleNIR)*(Quotient + 1);
		DO_ExtTrigger_NumCycleEPI = DO_ExtTrigger_SampleCountPerChannel;
	}
	else {
		DO_ExtTrigger_SampleCountPerChannel = 2*DO_ExtTrigger_NumCycleNIR;
		DO_ExtTrigger_NumCycleEPI = 2*DO_ExtTrigger_NumCycleNIR;
	}
	/*
	DO_ExtTrigger_SampleCountPerChannel = DO_ExtTrigger_NumCycleEPI;
	while (DO_ExtTrigger_SampleCountPerChannel < 500) {
		remain = DO_ExtTrigger_SampleCountPerChannel % (2*DO_ExtTrigger_NumCycleNIR);
		if (remain == 0)
			break;
		else
			DO_ExtTrigger_SampleCountPerChannel += DO_ExtTrigger_NumCycleEPI;
	}
	if (remain != 0) { // default
		DO_ExtTrigger_NumCycleNIR = 25;
		DO_ExtTrigger_NumCycleEPI = 100;
		DO_ExtTrigger_SampleCountPerChannel = 400;
	}
	DO_ExtTrigger_SampleCountPerChannel = DO_ExtTrigger_NumCycleEPI;
	*/

	// 5. AO
	CurrentBufferCount = 0;
	CurrentWrittingBufferIndex = 0;

}


int DAQmx_2P::InitiateAOForStage_Buffer_WN(int SampleCount, const float64 * data) {

	ClearAOForStage();
	clearAOForStage_Buffer_WN();

	/*********************************************/
	// DAQmx Configure Code // connect port0:Line1 to PFI0
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&stage_AO23_Simulation));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(stage_AO23_Simulation,"Dev1/ao2:3","",-10.0,10.0,DAQmx_Val_Volts,NULL));
	DAQmxErrChk (DAQmxCfgSampClkTiming(stage_AO23_Simulation,"/Dev1/PFI0",100000.0/DO_ExtTrigger_NumCycleNIR,
		DAQmx_Val_Falling, DAQmx_Val_ContSamps,AOForStage_SampleNoPerChannel));

	DAQmxErrChk( DAQmxSetWriteRegenMode(stage_AO23_Simulation, DAQmx_Val_DoNotAllowRegen)); // DAQmx_Val_AllowRegen // DAQmx_Val_DoNotAllowRegen

	DAQmxErrChk (DAQmxWriteAnalogF64(stage_AO23_Simulation, SampleCount
		, 1, 10.0, DAQmx_Val_GroupByChannel,
		data,NULL,NULL));



	return 1;

Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( stage_AO23_Simulation!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO23_Simulation);
		DAQmxClearTask(stage_AO23_Simulation);
		stage_AO23_Simulation = 0;
	}
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return 0;
}

unsigned int DAQmx_2P::getCurrentFrameFromNISampleNumber_WNBUFFER(void) {
	uInt64 totalSample = 0;
	// check NI handler should be ready.
	if (stage_AO23_Simulation != 0) {
		// 1. get the total sample
		DAQmxGetWriteTotalSampPerChanGenerated(stage_AO23_Simulation, &totalSample);
	}
	return (unsigned int)totalSample;
}


int DAQmx_2P::clearAOForStage_Buffer_WN(void) {
	if( stage_AO23_Simulation!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO23_Simulation);
		DAQmxClearTask(stage_AO23_Simulation);
		stage_AO23_Simulation = 0;
		return 1;
	}
	return 0;
}


int DAQmx_2P::InitiateAO1ForPiezo(void) {
	error=0;
	float64     data[1] = {0};

	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&stage_AO1));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(stage_AO1,"Dev1/ao1","",0,10.0,DAQmx_Val_Volts,""));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(stage_AO1));

	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk (DAQmxWriteAnalogF64(stage_AO1,1,1,10.0,DAQmx_Val_GroupByChannel,data,NULL,NULL));

	return 1;
Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( stage_AO1!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO1);
		DAQmxClearTask(stage_AO1);
	}
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return 0;
}

int DAQmx_2P::ClearAO1ForPiezo(void) {
	if( stage_AO1!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO1);
		DAQmxClearTask(stage_AO1);
		stage_AO1 = 0;
		return 1;
	}
	return 0;
}

void DAQmx_2P::UpdateAO1(double z) {
	error=0;
	if( stage_AO1!=0 ) {
		DAQmxWriteAnalogF64(stage_AO1,1,1,10.0,DAQmx_Val_GroupByChannel,&z,NULL,NULL);
	}
}


int DAQmx_2P::InitiateAOForStage(void) {
	error=0;
	float64     data[2] = {5.0, 0.0};

	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&stage_AO23));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(stage_AO23,"Dev1/ao0:1","",0,10.0,DAQmx_Val_Volts,""));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(stage_AO23));

	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk (DAQmxWriteAnalogF64(stage_AO23,1,1,10.0,DAQmx_Val_GroupByChannel,data,NULL,NULL));

	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk(DAQmxCreateTask("", &stage_AI01));
	DAQmxErrChk(DAQmxCreateAIVoltageChan(stage_AI01, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, NULL));
	DAQmxErrChk(DAQmxCfgSampClkTiming(stage_AI01, "PFI1", 200, DAQmx_Val_Falling, DAQmx_Val_ContSamps, 1));
	//DAQmxErrChk(DAQmxSetReadAutoStart(stage_AI01, DAQmx_Val_OverwriteUnreadSamps));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk(DAQmxStartTask(stage_AI01));
	totalSamplecount_ai = 0;
	/*********************************************/
	// DAQmx Read Code
	/*********************************************/


	return 1;
Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( stage_AO23!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO23);
		DAQmxClearTask(stage_AO23);
		DAQmxStopTask(stage_AI01);
		DAQmxClearTask(stage_AI01);
	}
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return 0;
}

int DAQmx_2P::ClearAOForStage(void) {
	if( stage_AO23!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AO23);
		DAQmxClearTask(stage_AO23);
		stage_AO23 = 0;
		DAQmxStopTask(stage_AI01);
		DAQmxClearTask(stage_AI01);
		stage_AI01 = 0;
		return 1;
	}
	return 0;
}


void DAQmx_2P::UpdateAO23(double x, double y) {
	error=0;
	float64     data[2] = {0,0};
	if( stage_AO23!=0 ) {
		data[0] = (float64)x;
		data[1] = (float64)y;
		DAQmxWriteAnalogF64(stage_AO23, 1, 1, 10.0, DAQmx_Val_GroupByChannel, data, NULL, NULL);
		ReadAI0();
		//totalSamplecount_ai = temp_totalsamplecount;
	}
}


int DAQmx_2P::GenerateSawteethAO0(float Amp, float center, int freq) {
	error=0;
	float64     data[1000];
	float64     dataNULL[1000] = {0};
	int         i=0;

	if (AO0taskHandle != 0) {
		DAQmxStopTask(AO0taskHandle);
		DAQmxClearTask(AO0taskHandle);
		AO0taskHandle = 0;
	}
	if (!AO0taskHandle_Enabled) {
		for(;i<1000;i++) {
			data[i] = center + (Amp*(float)i/1000.0 - Amp/2);
			if (data[i] < 0) data[i] = 0;
			if (data[i] > 10) data[i] = 10;
		}
	}
	else {
		for(;i<1000;i++) {
			data[i] = center;
			if (data[i] < 0) data[i] = 0;
			if (data[i] > 10) data[i] = 10;
		}
	}
	AO0taskHandle_Enabled = !AO0taskHandle_Enabled;


	// create one sine waveform


	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&AO0taskHandle));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(AO0taskHandle,"Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,NULL));
	DAQmxErrChk (DAQmxCfgSampClkTiming(AO0taskHandle,"",
		freq*1000,
		DAQmx_Val_Rising,DAQmx_Val_ContSamps,1000));

	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk (DAQmxWriteAnalogF64(AO0taskHandle,1000,0,10.0,DAQmx_Val_GroupByChannel,data,NULL,NULL));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(AO0taskHandle));
	return 1;


Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( AO0taskHandle!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(AO0taskHandle);
		DAQmxClearTask(AO0taskHandle);
	}
	return 0;
}

int DAQmx_2P::GenerateSineWaveAO0(double Amp, int freq) {
	error=0;
	TaskHandle  taskHandle=0;
	float64     data[1000];
	float64     dataNULL[1000] = {0};
	int         i=0;

	// create one sine waveform
	for(;i<1000;i++) {
		data[i] = Amp*sin((double)i*(2.0*PI/1000.0));
	}
	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&taskHandle));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(taskHandle,"Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,NULL));
	DAQmxErrChk (DAQmxCfgSampClkTiming(taskHandle,"",freq*1000,DAQmx_Val_Rising,DAQmx_Val_ContSamps,1000));

	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk (DAQmxWriteAnalogF64(taskHandle,1000,0,10.0,DAQmx_Val_GroupByChannel,data,NULL,NULL));

	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(taskHandle));
	Sleep(3000);

	DAQmxStopTask(taskHandle);
	DAQmxClearTask(taskHandle);
	taskHandle= 0;


	TaskHandle  taskHandle2=0;
	float64     data2[1] = {0.0};
	/*********************************************/
	// DAQmx Configure Code
	/*********************************************/
	DAQmxErrChk (DAQmxCreateTask("",&taskHandle2));
	DAQmxErrChk (DAQmxCreateAOVoltageChan(taskHandle2,"Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,""));
	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(taskHandle2));
	/*********************************************/
	// DAQmx Write Code
	/*********************************************/
	DAQmxErrChk (DAQmxWriteAnalogF64(taskHandle2,1,1,10.0,DAQmx_Val_GroupByChannel,data2,NULL,NULL));


Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( taskHandle!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskHandle);
		DAQmxClearTask(taskHandle);
	}
	if( taskHandle2!=0 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskHandle2);
		DAQmxClearTask(taskHandle2);
	}
	return 0;
}


void DAQmx_2P::UpdateAO_DB(float64 vx, float64 vy, float64 vz) {
	error=0;
	uInt64  curPos;
	uInt64 curPos2;
	uInt64 totalSample;
	int32 n = DO_ExtTrigger_NumCycleNIR;
	int32 offset;
	int32 offset_allowed = 10;
	if( AO_TaskHandler_DB!=0 ) {
		DAQmxErrChk (DAQmxGetWriteTotalSampPerChanGenerated(AO_TaskHandler_DB, &totalSample));
		//DAQmxErrChk (DAQmxGetWriteCurrWritePos(AO_TaskHandler_DB, &curPos));
		CurrentBufferCount = totalSample % (2*DO_ExtTrigger_NumCycleNIR);
		if ((CurrentBufferCount < DO_ExtTrigger_NumCycleNIR + offset_allowed)
			&& (CurrentBufferCount > offset_allowed) ){
			CurrentWrittingBufferIndex = 1;
			offset = DO_ExtTrigger_NumCycleNIR;
			n = DO_ExtTrigger_NumCycleNIR;
			if (CurrentBufferCount < DO_ExtTrigger_NumCycleNIR)
				n += CurrentBufferCount;
		}
		else {
			CurrentWrittingBufferIndex = 0;
			offset = 0;
			n = max(DO_ExtTrigger_NumCycleNIR, CurrentBufferCount);
		}
		//for (int i = 0; i < n; i++) {
		//	AO_Stage_Data[0+i*3] = vx;
		//	AO_Stage_Data[1+i*3] = vy;
		//	AO_Stage_Data[2+i*3] = vz;
		//}

		DAQmxErrChk (DAQmxSetWriteOffset(AO_TaskHandler_DB, offset));
		DAQmxErrChk (DAQmxWriteAnalogF64(AO_TaskHandler_DB, 4, 0, 10.0, DAQmx_Val_GroupByScanNumber,AO_Stage_Data,NULL,NULL));
		//DAQmxErrChk (DAQmxSetWriteOffset(AO_TaskHandler_DB, 0));
		//DAQmxErrChk (DAQmxGetWriteCurrWritePos(AO_TaskHandler_DB, &curPos2));
		DAQmxErrChk (DAQmxGetWriteTotalSampPerChanGenerated(AO_TaskHandler_DB, &curPos2));
	}
	return;
Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
}

static int32 GetTerminalNameWithDevPrefix(TaskHandle taskHandle, const char terminalName[], char triggerName[])
{
	int32	error=0;
	char	device[256];
	int32	productCategory;
	uInt32	numDevices,i=1;

	DAQmxErrChk (DAQmxGetTaskNumDevices(taskHandle,&numDevices));
	while( i<=numDevices ) {
		DAQmxErrChk (DAQmxGetNthTaskDevice(taskHandle,i++,device,256));
		DAQmxErrChk (DAQmxGetDevProductCategory(device,&productCategory));
		if( productCategory!=DAQmx_Val_CSeriesModule && productCategory!=DAQmx_Val_SCXIModule ) {
			*triggerName++ = '/';
			strcat(strcat(strcpy(triggerName,device),"/"),terminalName);
			break;
		}
	}
Error:
	return error;
}
// -------------------------------------------------- //

void DAQmx_2P::PZ_CreateStairWaveForm(void) {
	for (int i = 0; i<20; i++)
		PZ_data[i] = 0;

	for(int i = 0;i<PZ_NumLayerPerCycle ;i++)
		PZ_data[i] = (PZ_Pos_BottomLayer + PZ_gap_btLayer*i) / PZ_umPerVolt;
}


int DAQmx_2P::InitiateAIForStage_Buffer(void) {
	error = 0;
	char    trigName[256];
	DAQmxErrChk (GetTerminalNameWithDevPrefix(DO_ExtTrigger_TaskHandler,"do/SampleClock",trigName));
	DAQmxErrChk (DAQmxCreateTask("",&stage_AI23));
	DAQmxErrChk (DAQmxCreateAIVoltageChan(stage_AI23,"Dev1/ai2:3","",DAQmx_Val_Cfg_Default
		,-10.0,10.0,DAQmx_Val_Volts,NULL));
	DAQmxErrChk (DAQmxCfgSampClkTiming(stage_AI23,"",10000.0,
		DAQmx_Val_Rising,DAQmx_Val_ContSamps,600));
	DAQmxErrChk (DAQmxCfgDigEdgeStartTrig(stage_AI23,trigName,DAQmx_Val_Falling));
	//DAQmxErrChk( DAQmxSetReadRegenMode(stage_AI23, DAQmx_Val_AllowRegen)); // DAQmx_Val_AllowRegen // DAQmx_Val_DoNotAllowRegen

    DAQmxErrChk (DAQmxSetReadOverWrite(stage_AI23, DAQmx_Val_OverwriteUnreadSamps));
	/*********************************************/
	// DAQmx Start Code
	/*********************************************/
	DAQmxErrChk (DAQmxStartTask(stage_AI23));
	return 1;

Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( stage_AI23 ) {
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(stage_AI23);
		DAQmxClearTask(stage_AI23);
		stage_AI23 = 0;
	}
	return 0;
}

int DAQmx_2P::UpdateAIForStage_Buffer_CurrRead(int rem) {
	int32 readoutCount;
	int32   DAQmxInternalAIbuffer_Offset = AIForStage_SampleNoPerChannel;
	error=0;
	/*********************************************/
	// DAQmx Read Code
	/*********************************************/
	if (rem == 0) {
		DAQmxErrChk (DAQmxSetReadOffset(stage_AO23, DAQmxInternalAIbuffer_Offset));
		DAQmxErrChk (DAQmxReadAnalogF64(stage_AI23,AIForStage_SampleNoPerChannel, 1.0,
			DAQmx_Val_GroupByChannel,AIForStage_Data,
			4*AIForStage_SampleNoPerChannel,&readoutCount,NULL));
	}
	else {
		DAQmxErrChk (DAQmxSetReadOffset(stage_AO23, 0));
		DAQmxErrChk (DAQmxReadAnalogF64(stage_AI23,AIForStage_SampleNoPerChannel, 1.0,
			DAQmx_Val_GroupByChannel,AIForStage_Data,
			4*AIForStage_SampleNoPerChannel,&readoutCount,NULL));
	}

	return 1;

	Error:
	if( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if( stage_AI23!=0 ) {
		DAQmxStopTask(stage_AI23);
		DAQmxClearTask(stage_AI23);
	}
	if( DAQmxFailed(error) )
		printf("DAQmx Error: %s\n",errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
	return 0;
}

int DAQmx_2P::UpdateAIAOForStage_clear(void) {
	/*********************************************/
	// DAQmx Stop Code
	/*********************************************/
	if( stage_AI23 ) {
		DAQmxStopTask(stage_AI23);
		DAQmxClearTask(stage_AI23);
		stage_AI23 = 0;
	}
	return 1;
}
void DAQmx_2P::updateNIallSignal(void) {
	DO.resetSignal(NIsampleclock, 0);
	DO.mergeSignal(&XPS_StageSignal->signal, offsetcount, nexsamplecount_update, 1 + 1<<1);
	DO.mergeSignal(&XPS_StageSignal->signal_CamTrigger, offsetcount - XPS_StageSignal->cam_offset, nexsamplecount_update, 1 << 5);
	DO.mergeSignal(&hamamatsu_signal, offsetcount - DO_hamamatsu_globalexpwindow_size, nexsamplecount_update, 0);
	//AO.resetSignal(NIsampleclock, 0.0);
	//AO.generateAnalogSignal(&Piezo_scanSingal.signal, offsetcount, nexsamplecount_update, 0);
	updateDOAOSignal(&DO, &AO, NULL);
	nexsamplecount_update += NIsampleclock;
}
void DAQmx_2P::initializeNIallSignal(uint64 _hamamatsu_globalexpwindowsize) {
	DO_hamamatsu_globalexpwindow_size = _hamamatsu_globalexpwindowsize;
	DO.resetSignal(2 * NIsampleclock, 0);
	DO.mergeSignal(&XPS_StageSignal->signal, offsetcount, 0, (1) + (1 << 1)); // stage & monitoring signal
	DO.mergeSignal(&XPS_StageSignal->signal_CamTrigger, offsetcount - XPS_StageSignal->cam_offset, 0, (1 << 5)); // NIR camera trigger
	DO.mergeSignal(&hamamatsu_signal, offsetcount - DO_hamamatsu_globalexpwindow_size, 0,  0); // FLR camera trigger
	//AO.resetSignal(2 * NIsampleclock, 0.0);
	//AO.generateAnalogSignal(&Piezo_scanSingal.signal, offsetcount, 0, 0);
	nexsamplecount_update = NIsampleclock;
}

void DAQmx_2P::updateNIallSignal_fixed(void) {
	DO.resetSignal(2*NIsampleclock, 0);
	DO.mergeSignal(&XPS_StageSignal->signal, offsetcount, NIsampleclock, 1 + 1 << 1);
	DO.mergeSignal(&XPS_StageSignal->signal_CamTrigger, offsetcount - XPS_StageSignal->cam_offset, NIsampleclock, 1 << 5);
	DO.mergeSignal(&hamamatsu_signal, offsetcount - DO_hamamatsu_globalexpwindow_size, NIsampleclock, 0);
}

//------------- NewDesign2015 -------------- //
