#pragma once
#include "Qt_Ext/Pixmap.h"
#include "XPSQ8/XPS_Q8.h"							// stage control
#include "VideoProcessor/UI/ui_MainWindow.h"		// Qt Design generated this file for our main window
#include <QtWidgets/qmainwindow.h>
// add opencv library - Is there another way to do?

#include "DSP/ImageProcess_NPP.h"
#include "DSP/GlobalMap.h"
#include "Tracker/AutoFocusVolumeScan.h"
#include "Tracker/TrackingInfo.h"
#include "Tracker/MPC.h"
#include "Tracker/FeedbackControlMode.h"
#include "Tracker/zebrafishpredictor.h"
// NI board 64bit
#include "NIDAQmx.h"
// ADC board
#include "PointGreyCamera/PointGreyCamera.h"
// teensy
#include "Teensy/PointGrey_SerialCOM.h"
#include "Teensy/XPS_SerialCOM.h"


#include <H5Cpp.h>
#include "HDF5/hdf5imagewriter.h"
#include "HDF5/hdf5imagereader.h"
#include "HDF5/hdf5replayer.h"

#include "Hamamatsu/Hamamatsu.h"
#include "XPSQ8/XPS_NI.h"
#include "DAQmx/Piezo_NI.h"
#include "Qt_Ext/Qtgraph.h"
#include "ExpControl/ThermalControl.h"
#include "hidapi/DMD.h"



//namespace ImageSource { class ImageSource; }

namespace UI {
	class QGLZoomableImageViewer;
	//! Our main window, inherits from QMainWindow, contains the class created by Qt Designer
	class MainWindow : public QMainWindow {
	public:
		MainWindow();		//!< Initialize
		virtual ~MainWindow();	//!< Non-inline destructor
	private:
		// UI objects
		Ui::MainWindow m_ui;			//!< Our UI generated in Qt Designer
		QGLZoomableImageViewer* m_pImage1, *m_pImage2, *m_pImage3, *m_pImage5, *m_pImageR1, *m_pImageR2, *m_pImageR3, *m_pImageR4;		//!< Multipurpose image viewers

		virtual void timerEvent( QTimerEvent* ) override { onTimer(); }	//!< UI update timer
		void onTimer();													//!< Updates our live video and status periodically

		int disp_totalcount;
		double * disp_data1; bool disp_data1_enable[2]; unsigned int disp_data1_idx;
		double * disp_data2; bool disp_data2_enable[2]; unsigned int disp_data2_idx;
		double * disp_data3; bool disp_data3_enable[2]; unsigned int disp_data3_idx;
		double ** disp_data4; bool disp_data4_enable[2]; unsigned int disp_data4_idx;
		double ** disp_data5; bool disp_data5_enable[2]; unsigned int disp_data5_idx;
		Qtgraph qtgraph0;
		Qtgraph qtgraph0_thermal;
		Qtgraph qtgraph0_gmap;
		Qtgraph qtgraph1;
		Qtgraph qtgraph2;
		Qtgraph qtgraph3;
		Qtgraph qtgraph4;
		Qtgraph qtgraph5_EPIintMean;
		Qtgraph qtgraph5_EPI_peizoZset;
		Qtgraph qtgraph5_EPI_peizoZreal;

		// global map related QT functions
		void GlobalMap_imgprocessenable_QCheckBoxtoggled(bool);
		void GlobalMap_globalmapbuildingenable_QCheckBoxtoggled(bool);
		void tracking_enable_QCheckBoxtoggled(bool);
		void tracking_MPCenable_QCheckBoxtoggled(bool);
		void followingerror_enable_QCheckBoxtoggled(bool);
		void bounderror_enable_QCheckBoxtoggled(bool);
		void responseEstimation_RandomoffsetEnable_QCheckBoxtoggled(bool);
		void jitterfilter_Enable_QCheckBoxtoggled(bool);
		void Piezo_trk_Enable_QCheckBoxtoggled(bool);
		void constanctdistantfromyolk_brainpos_Enable_QCheckBoxtoggled(bool);
		void display_Enable_QCheckBoxtoggled(bool);
		void fakefishtest_Enable_QCheckBoxtoggled(bool);
		void intensity_ratio_set_QCheckBoxtoggled(bool);
		void toggle_EPI_display_checkBox_epiDisplay(bool);

		// other functions
		void reset_failurecount(void);
		void tracking_fishpredictionenable_QCheckBoxtoggled(bool);
		//<!--- PointGreyCamera (NIR);
		PointGreyCamera NIRCameraPG;
		void PointGreyCameraNIRcapture();

		//! new camera (Point Grey Camera - Flea3)
		bool StartCamera_GreyPoint(void);
		bool m_Tracking_EnableLive;
		bool m_Recording_EnableLive;
		void NIR_UpdateParameters_QpushButtonClicked(void); //
		//! Snapshot images
		img_uint8 * Snapshot;
		Point2f Snapshot_Pos_mm;


		// Global Maps variables
		GlobalMap GlobalMap_NIR;
		// global map functions
		void GlobalMap_Initialization(void); // load pixel distance and angle, and reset global map
		void GlobalMap_ParameterLoading(void); // load pixel distance and angle, and reset global map
		void GlobalMap_SaveTIFF(void); // Measure pixel distance
		void GlobalMap_displayreset(void);
		void GlobalMap_updateFishMask(void);
		void GlobalMap_UpdateStillImageUpdate(void);
		void GlobalMap_UpdateEnable_QCheckBoxtoggled(bool);
		void GlobalMap_UpdateEnable_QCheckBoxtoggled2(bool);
		void GlobalMap_UpdateAllEnable_QCheckBoxtoggled(bool);
		void ImageProc_ParameterLoading(void);

		// replayer functions
		HDF5Replayer replayer;
		void replayer_openfolder(void);
		void replayer_openfile_nir(void);
		void replayer_openfile_flr(void);
		void replayer_openfile_data(void);
		void replayer_play(void);
		void replayer_prev(void);
		void replayer_next(void);
		void replayer_display(void);
		void replayer_play_thread(void);
		void replayer_proc(void);


		DAQmx_2P ControllHardWare_2P; // NI board controller
		XPS_Q8 ILS200LM; //! Stage (ILS200LM Newport)
		// !!!! ------- functions for calibration ------ !!!! //
		void calibrateNIAOOffset(void);

		// ----
		parameterSet ParaSet;
		void UpdateTrackingParameters(void); // < QT variable update
		void UpdateStageParameters(void); // < QT variable update
		void ReadQtVariables(void); // < QT variable update

		vector<XPSGatheringInfo> m_PositionData; // position data (stage, piezo - from XPS controller)

		// ---------------------------------------------------------------------------------------

		// ----------------------------- NI Signal generating thread ------------------------------
		unique_ptr<std::thread> m_pThreadNIDigSignal;
		void NIDigSignal_updateThread(void);
		void replaceNewNIDO(void);
		void replaceNewNIAO(void);
		void NI_DO_ParametricSearch(void);
		void NI_DO_update(void);

		// ------------------------ NI data savin thread ------------------------
		unique_ptr<std::thread> m_pThreadNIAIsave;
		void NIAISingal_saveThread();
		// ------------------------------- Teensy setting ----------------------------------------
		XPS_SerialCOM XPSAVController;
		unique_ptr<std::thread> m_pThreadStageTeensy;
		void teensyThread();
		unsigned int TeensyOnTimeErrCount;

		// ----------------------------- Stage Position Reading thread ----------------------------
		unique_ptr<std::thread> m_pThreadStagePositionReading;
		void StagePositionReadingThread();
		double manualInputVelStage[2];
		double manualInputVelStage_set[2];
		QImage * insetmap;
		double insetmap_x[2];
		double insetmap_y[2];
		double insetmap_pxdist;
		double insetmap_size_mm;

		void toggle_NIRp1Update(bool);
		void toggle_NIRp2Update(bool);
		void toggle_FLRp1Update(bool);
		void toggle_FLRp2Update(bool);
		Point2d NIR_p1, NIR_p2, NIR_px1, NIR_px2;
		Point2d FLR_p1, FLR_p2, FLR_px1, FLR_px2;
		// ----------------------------- hamamastsu fluorescent camera thread ---------------------
		Hamamatsu HamamatsuCamera_C11440;
		unique_ptr<std::thread> m_pThreadFLRcapture;
		void HamamatsuCamera_FLRCapture(void);
		void HamamatsuCamera_updateParameters(void);
		void HamamatsuCamera_updateSignals_PG(void);
		void HamamatsuCamera_updateSignals(void);
		void HamamatsuCamera_Snapshot(void);
		void HamamatsuCamera_setgain(void);
		void HamamatsuCamera_setting(int index);
		MovingAverage_double m_ThreadProcessTimeMills_FluorescentImgCycle;
		MovingAverage_double m_ThreadProcessTimeMills_FluorescentImgRecProc;
		void refStackCapture(void);
		void refStackDisplay(int frm);
		void refStackSave(void);
		void refStackLoad(void);
		void refStackClear(void);
		void refSetTargetLayer(void);
		void set_zmatcherlimit(void);
		void refStackTEST(void);
		void refStackSimulation(bool);
		void DIFFDisplay_enable(bool);
		void DIFFDisplayA_enable(bool);
		void refStack_reverse(void);


		// ------------------ hamamastsu fluorescent image recording thread -----------------------
		unique_ptr<std::thread> m_pThreadFLRRecord;
		void HamamatsuCamera_FLRRecord(void); // stop capture and record

		void HamamatsuCamera_FLRRecord_binary(int n);
		vector<vector<int32>> FLRRecord_frameNo;
		unique_ptr<std::thread> m_pThread_FLRRecord_binary[harddrivecount_FLRsave];
		// ------------------------ Recording setting ------------------------
		unique_ptr<std::thread> m_pThreadRecBuffer_CopyResult;
		void RecBufferCopyResultThread();
		unique_ptr<std::thread> m_pThreadRecBuffer;
		void MPCrecordingThread();
		struct RecBufferResults {
			unsigned int FrmNo;
			XPSGatheringInfo StgPos;
			zebrafishInfo DetectedZebraFish;
			CntrData ControlDataRec;
			uint16 * ImgSrc;
			int rows;
			int cols;
			FLRImageData FLRdata;
		};
		vector<RecBufferResults> m_queueResults;

		TrackingMessageBuffer RecBuffer;
		TrackingMessage * RecBufferPtr;

		DMD lc9000;
		bool lc9000_autoreset;
		MovingAverage_double m_autosetDMDtime_ms;
		void connectDMD();
		void startstopDMD();
		void autoresetDMD_QCheckBoxtoggled(bool);


		// ------------------------ Z-position control thread------------------------
		unique_ptr<std::thread> m_pThreadPiezo;
		void PiezoSignal_updateThread(void);
		MovingAverage_double m_ThreadProcessTimeMills_PiezoCycle;
		double PiezoOffset[1000];
		double PiezoDirset[1000];
		int PiezoOffset_concatenate;
		void setPiezoConcatenate(bool flag);
		void setPiezoMaxupdateEnable(bool flag);

		// ------------------------ Z-position control thread------------------------
		unique_ptr<std::thread> m_pThreadIRLaser;
		void IRLaserUpdate_updateThread(void);
		MovingAverage_double m_ThreadProcessTimeMills_IRlaser;


		// ----------------------------------------------------------------------------------------
		bool g_bNIRImageProcess;
		void ResetAllprocess(void); // update external trigger signal
		void StartStopRecording(void);
		// ------------------------ setting ------------------------
		void RecSettings(void);
		void setting_update(bool _updateNI, bool _updateCameraSetting);

		// ------------------------ fish position feeding  ------------------------
		void replayer_loadfishpos(void);
		void replayer_fishpos_enable_QCheckBoxtoggled(bool flag);
		void replayer_fishpos_start_QCheckBoxtoggled(bool flag);
		void replayer_fishpos_perfectprediction_QCheckBoxtoggled(bool flag);
		double * replayer_fishpos;
		double * replayer_fishpos_OrientationDeg;
		int replayer_fishpos_totalfrm;
		int replayer_fishpos_curfrm;
		bool replayer_fishpos_enable;
		bool replayer_fishpos_enable_perfectprediction;
		bool replayer_fishpos_enable_start;



		void RecBufferCopyResultThread_old();

		bool EnablePositionReading;
		unsigned int CurrentFrmNo; // recorded in position reading thread
		unsigned int LastUpdatedMPCFrmNo; // recorded in position reading thread
		XPSGatheringInfo StgPos_ReadingThread[2];
		int TotalThreadCount;





		XPS_NI XPS_StageSignal;
		Piezo_NI Piezo_scanSingal; // analog signal
		void generateNextSignal(void);
		DigSignal NIDigSignal;
		AnalogSignal NIAnalogSignal;
		uInt64 DigSignal_chunksize;
		uInt64 offsetIdx;
		uInt64 startIdx;
		uInt64 IdxAdvanced;
		MovingAverage_double m_ThreadProcessTimeMills_NIWritingCycle;

		// ----------------------------- Single Threads setting ------------------------
		std::thread m_pThreadMainProcess;
		unique_ptr<std::thread> m_pThreadNIRMPCtracking;
		void NIRProcessFrameSingleThread(void);
		int Readoutdelay_Frm;
		unsigned int NIRFrameDropFailureCount;
		unsigned int ReadyOnTimeMPCInputFailureCount;
		unsigned int PositionGatheringInFailureCount;
		unsigned int FishDetectionFailureCount;

		int isSync;

		bool isFishDetection(zebrafishInfo fish);

		MovingAverage_double m_ThreadProcessTimeMills_FramethreadCycle;
		MovingAverage_double m_ThreadProcessTimeMills_Framethread_Stg;
		MovingAverage_double m_ThreadProcessTimeMills_Framethread_ImgProcMPC;
		MovingAverage_double m_ThreadProcessTimeMills_Framethread_Total;
		MovingAverage_double m_ThreadProcessTimeMills_Framethread_nirImgRecProc;
		MovingAverage_double m_ThreadProcessTimeMills_TeensyCycle;
		MovingAverage_double m_ThreadProcessTimeMills_StagePositionReading;
		MovingAverage_double m_ThreadProcessTimeMills_StagePositionReadingCycle;
		MovingAverage_double m_ThreadProcessTimeMills_reserved1;
		MovingAverage_double m_ThreadProcessTimeMills_reserved2;


		struct TeensyMessage {
			TeensyMessage() : FrmNo(0), CntrInputReady(false),
				Teensy_StreamingEnabled(false), Teensy_FrmNo(0), Teensy_Status(0)
			{CntrInput[0] = 0; CntrInput[1] = 0;}
			// input (from MPC controller -> Teensy)
			unsigned int FrmNo;
			double CntrInput[2];
			bool CntrInputReady;
			// output (from Teensy -> MPC controller)
			bool Teensy_StreamingEnabled;
			unsigned int Teensy_FrmNo;
			unsigned int Teensy_Status; // 0 Success, Teensy_InitErr, Teensy_OnTimeErr
			enum{Teensy_Success, Teensy_FrmDroppedErr, Teensy_OntimeErr};
		};
		TeensyMessage TeensyStatus;
		bool EnableTeensyReset;
		unsigned int TeensyFrmMissedCount;

		//POINTGREY_SerialCOM FLRExpTimeController;
		// ------------------------ MPC Control setting ------------------------
		MPC_XPSAVT MPC_X;
		MPC_XPSAVT MPC_Y;
		FeedbackControlMode feedbackcontroller;
		//zebrafishPositionEstimate FishPosEstimator;
		zebrafishpredictor FishPosEstimator;
		JitterFilter globalFishPosAdjustor;
		void ParameterUpdate_FishPosEstimator(void);
		unsigned int lastUpdatedMPCFrame;
		double _MaxStageSpeed;

		int m_trk_targetSelection;
		int m_trk_bfishpospredictionenable;
		int m_trk_bfollowingerrorEnable;
		int m_trk_bboundaryerrorEnable;
		int m_trk_bjitterenable;
		int m_bconstanctdistantfromyolk_brainpos_enable;
		double m_constanctdistantfromyolk_brainpos;
		int m_Piezo_trk_enable;
		int m_trk_brespoffsetenable;
		double m_trk_brespoffsetrange;
		double m_trk_respoffsetfishpos[2];
		double m_trk_respoffsetfishpos_ref[2];
		int m_trk_brespoffsetrange_steps;

		int m_trk_bfakefishposition;
		int fakefish_idx;
		int fakefish_idxmax;
		double * fakefishposition_x;
		double * fakefishposition_y;
		double * fakefishposition_th;




		bool computeMPC_L(double * StgPosition, CntrData * ControlDataRec);

		void getFishGlobalPosition(zebrafishInfo * DetectedZebraFish, double * FishRefPosition, CntrData * ControlDataRec, bool _isfishdetected);

		void MPCParmeterUpdate(void);
		double * importImpRestInd(H5File* file, char * datasetName, int * ReadDataCount);
		// ------------------------ Display setting ------------------------
		MovingAverage_double m_ThreadProcessTimeMills_DisplayUpdate;
		MovingAverage_double m_ThreadProcessTimeMills_DisplayUpdateCycle;
		MovingAverage_double m_ThreadProcessTimeMills_Recording;
		MovingAverage_double m_ThreadProcessTimeMills_Recording_NIR;
		MovingAverage_double m_ThreadProcessTimeMills_Recording_FLR;
		MovingAverage_double m_ThreadProcessTimeMills_FluorescentImg;
		MovingAverage_double m_ThreadProcessTimeMills_trkPiezo;


		// ------------------------ piezo tracking setting ------------------------
		AutoFocusVolumeScan zaxis_controller;
		void piezoparameterupdate_buttonclicked(void);

		// ------------------------ Replayer setting ------------------------
		HDF5ImageReader * NIRImageDecoder;
		HDF5ImageReader * FLRImageDecoder;
		HDF5ImageReader * GlobalMapDecoder;
		HDF5DataReader * NIRDataDecoder;
		vector<Point2d> GPUImageProcess(uint16 * src, double * position, int * fishSize, bool FishDetectionSuccess_PrevFrms);
		vector<Point2d> GPUImageProcess_centroid(uint16 * src, double * position, int * fishSize, bool FishDetectionSuccess_PrevFrms);

		//void GPUBGupdate(int fishSize, int fishSizeRef); //
		//void RePlayer_OpenFileDialogClicked(void); //! Kvideo replayer open file dialog function
		//void RePlayerDATA_OpenFileDialogClicked(void); //! Kvideo replayer open file dialog function
		//void RePlayer_PlayDialogClicked(void);
		//void Replayer_VideoCurFrameValueChanged(int index);
		//void RePlayer_FULLSimulation_07315(void);

		// global map loading
		//cv::Mat RePlay_GlobalMap;
		//void RePlayer_GlobalMap1(void);
		//void RePlayer_GlobalMap2(void);
		//void Replayer_VideoStartFrameValueChanged(int index);
		//cv::Mat RePlayer_GlobalMapUpdating(cv::Mat * GMap_Src, cv::Mat * src, cv::Rect _ROI);
		QString RePlaydirname;

		// ------------------------ Calibration setting ------------------------
		double WN_vel_Range;
		double WN_resolutionPerMMPS;
		bool m_EnableWhiteNoiseMaker_X;
		bool m_EnableWhiteNoiseMaker_Y;

		bool WN_Enable;
		float64 * WN_DataBuffer;
		float64 * WN_DataBuffer_Vel;
		int WN_TotalSampleCountWhole;
		void Stage_WN_SampleGeneration_Clicked(void);
		void Stage_WN_RandomSampleGeneration_Clicked(void);
		void Stage_WN_SampleLoad_Clicked(void);
		void Stage_WN_Start_Clicked(void);
		void Stage_WN_ScanSampleGeneration(void);
		void Stage_Replay_VelSetLoad_Clicked(void);



		bool LoadedFishPos_Enable;
		double * LoadedFishPos;
		double * LoadedFishPosOrientation;
		double * LoadedFishPosError_mm;
		double LoadedFishPos_StageInitialPosition[2];
		double LoadedFishPos_StageInitialPosition_offset[2];
		void Stage_Replay_FishPosLoad_Clicked(void);
		void Stage_Replay_FishPosLoadInitStage_Clicked(void);
		void Stage_Replay_FishPosLoadStartSimulation_Clicked(void);
		double * importHDF5_Column(H5File* file, char * datasetName, int * ReadDataCount);

		void Stage_FishPos_SinePathGeneration(void);

		void Stage_I2T_setup(void);
		void Stage_I2T_clear(void);
		// ------------------------ Calibration setting ------------------------
		//<!--- GPU(NIR) Image Processing thread
		DivGradDetector_GPU NIRFishPosDetector_GPU;
		zebrafishInfoReference FishPosRef;
		void NIRImageproc_GPU(void);



		//<--- FLR image gathering and Recording thread
		unsigned int RecBuffer_FLR_PreTrigFrmCountMax;
		struct FLRImage{
			img_uint16 *  EPIFLImage; // PMT images (new & delete)
			unsigned int FrameNo_FLR;
			unsigned int FrameNo_NIRRef;
		};

		PointGreyCamera FLRCameraPG;
		//void PointGreyCameraFLRcapture();

		vector<FLRImage> * RecBuffer_FLR;
		FLRImage * RecBuffer_FLR_ptr;
		vector<FLRImage> * RecBuffer_FLR_RecordingOnly;
		void PostTriggerRec_FLR();
		bool EnableFLRRecording;


		std::thread m_pThreadGeneral;


		vector<XPSGatheringInfo> m_queuePOS; // delete later



		// functinos Using
		bool isDoingReset;




		// ----------------------------- Raster ------------------------
		void Raster3DSpace(void);
		void RasterLoadPos(void);





		// ----------------------------- END: lockfree Multithread setting ------------------------

		//!!! Qt component event functions
		void NIR_GPUResultDisplay_QcomboBoxCurrentIndexChanged(int index);
		void Data_Save_MainController_QpushButtonClicked(void);
		void Data_Clear_MainController_QpushButtonClicked(void);


		bool Stage_ConnectOnly_QpushButtonClicked(void);
		bool Stage_InitializationOnly_QpushButtonClicked(void);
		void Stage_ReadyHoming(void);
		void Stage_ADCEnable(void);

		void Stage_updatestatus(void); // connect & disconnect stage
		void Stage_MoveToPosition_QpushButtonClicked(void); // move the stage to the certain position
		void Stage_SaveCurrentPosition_QpushButtonClicked(void); // save the current position as a set position in the text file (Stage_Set_Position.txt)
		void Stage_MoveToSavedPosition_QpushButtonClicked(void); // load the set position saved in the text file & move the state to the set position (Stage_Set_Position.txt)
		void Stage_UpdataParameters_QpushButtonClicked(void); // Update all stage parameters (turn off the Analog velocity control -> update the parameters -> turn on the analog velocity control)
		void Stage_EditAdvancedParameters_QpushButtonClicked(void); // Enabling to edit the advanced parameters of stage
		void Stage_ReleaseStageOnly_QpushButtonClicked(void);
		void Stage_ReadyOnly_QpushButtonClicked(void);
		void Stage_HomeSearch_QpushButtonClicked(void);
		void Stage_EnableAnalogControlOnly_QpushButtonClicked(void);
		void Stage_UpdateStatus(int StateNumber);
		void pushButton_Stage_CalibrationImpulseStep_QpushButtonClicked(void);
		void CalibrateImpDataFromStepResp(void);
		void Stage_CalibrationSimple_Random_QpushButtonClicked(void);
		bool EnableCalibration;
		vector<XPSGatheringInfo> m_queuePOSCalibration;
		enum{STAGE_NotInitiated,
		STAGE_NotReferenced,
		STAGE_Referencing,
		STAGE_Homming,
		STAGE_Ready,
		STAGE_AnalogTracking};


		void NIR_PixelDistanceCalibration(void); // Measure pixel distance

		void NIR_SusanMaggotSizeCalibrate(void);
		void NIR_KCOSMaggotSizeCalibrate(void);
		void NIR_GFTTMaggotSizeCalibrate(void);
		void PiezoMode_QcomboBoxCurrentIndexChanged(int index);

		void Tracking_StartStopLive_QpushButtonClicked(void);
		void Tracking_StartStopRecording_QpushButtonClicked(void);
		void Tracking_TargetSelection_QcomboBoxCurrentIndexChanged(int index);


		Point2d SelectTrackingTarget(int option, zebrafishInfo * DetectedZebrafish);

		// ------------------------ replay variable and functinos ---------------
		void replayer_OpenDirDialog(void); //
		vector<vector<double>> replayer_ReadData(QString name); // data loading function
		// data vector [frame #][data type] -
		vector<vector<double>> m_replayer_NIRdata;
		enum{re_NIRFrame, re_NIR_Timesstamp,
			re_NIR_px, re_NIR_py,
			re_NIR_targetx, re_NIR_targety,
			re_NIR_BD_x, re_NIR_BD_y, re_NIR_width, re_NIR_height
			, rp_NIR_Time_ms};
		vector<vector<double>> m_replayer_PMTdata;
		vector<unsigned int> m_replayer_PMTNIRsync;
		enum{re_PMTFrameNo,
			re_PMT_FocalValue,
			re_PMT_PiezoInputVolt,
			re_PMT_time_ms,
			re_PMT_PiezoOutputVolt};
		vector<vector<double>> m_replayer_STGdata;
		enum{re_STGFrameNo,
			re_STG_px_mm, re_STG_py_mm,
			re_STG_InputVolt_x, re_STG_InputVolt_y,
			re_STG_Time_ms};

		//! Image loading using kvideo interface


		//! replayer common functions


		void pushButton_SaveParameters_clicked(void);
		void updataXPSVariables(void);

		void Piezo_ManualMoveDown(void);
		void Piezo_ManualMoveUp(void);
		void Piezo_ManualStop(void);

		//void DisplayImage(cv::Mat MatSource, QGLZoomableImageViewer* window);
		void DisplayImage(img_uint8 MatSource, QGLZoomableImageViewer* window);
		void DisplayImage(img_uint16 MatSource, QGLZoomableImageViewer* window);
		void DisplayImage_ROI(img_uint16 MatSource, QGLZoomableImageViewer* window, uint16 min, uint16 max);

		int m_Tracking_FishPosFutureTargetIdx;
		Point2d m_ImageTargetPosPx;
		bool m_bEnableStageRandomInputGenerator;
		int m_NIR_TargetMidPoint_Sel;
		bool m_Tracking_FishPosEstimationEnabled;
		double m_piezoInput;
		double m_piezoOutput;
		Point2d m_InputVolStage;
		int m_Videoframenumber;
		bool m_NIR_EnableImageProcessing;
		bool m_NIR_EnableImageBlur;
		int m_NIR_BlurMode;
		int m_NIR_DivResultDisplayIdx;
		bool m_NIR_ShowBlurImage;
		bool m_NIR_EnableImageBinarization;
		int m_NIR_BWThreshold;
		int m_NIR_BWMode;
		bool m_NIR_ShowBWImage;
		bool m_NIR_ShowSortedImage;
		bool m_NIR_EnableImageErodeDilate;
		int m_NIR_ErodeDilateIteration;
		bool m_NIR_ShowErodeDilateImage;
		double m_StageSimulation_StepError ;
		bool m_NIR_EnableCornerDetection;
		int m_NIR_CornerDetectionMode;
		int m_NIR_cornersize_KCOS;
		int m_NIR_MinDistanceBtPeaks_KCOS;
		int m_NIR_TargetDetection_MiddlePoints_NoPoints;
		int m_NIR_TargetDetection_MiddlePoints_SelNoPoints;
		double m_NIR_TargetDetection_DistanceFromHeading_Px;
		bool m_NIR_EnableTargetDetection;
		int m_NIR_TargetDetectionMode;
		int m_NIR_RecordEveryNFrame;
		//cv::Mat m_NIROrg;
		//cv::Mat m_NIROrg_Proc;
		bool m_bTracking_EnableRecording;
		bool m_bPMTs_isLive;
		unsigned int m_DroppedFrame_NIR;
		unsigned int m_DroppedFrame_NIRImageProc;
		unsigned int m_DroppedFrame_PMT;
		double m_AppPolyEpsilon;
		bool m_bADCRecord;

		// z-axi scan
		int m_Piezo_TrackingMode; // 0:Off, 1: Full Scan, 2: 3 Layer
		int m_Piezo_scanCenter;
		int m_Piezo_scanCenterCurrent;
		int m_Piezo_scanRange;
		int m_Piezo_scanStep;
		int m_Piezo_scanImgsPerLayer;
		bool m_bXYEnableTrackingInput;

		//
		int m_NIR_ContourDetectionMethod;
		bool bm_NIR_CannyEnable;
		bool bm_NIR_CannyResultEnable;
		double m_NIR_CannyThreshold_High;
		double m_NIR_CannyThreshold_Low;
		double m_NIR_Sigma;
		bool m_bNIR_CannyFillGapsEnable;
		bool m_bNIR_CannyFillGapsResultEnable;
		int m_NIR_CannyFillGapsIteration;
		bool m_bNIR_CannyObjectSortingEnable;
		bool m_bNIR_CannyObjectSortingResultEnable;
		int m_NIR_CannyObjectSortingMinSize;
		int m_NIR_CannyObjectSortingMaxSize;
		bool m_bNIR_CannyPolygonAppxEnable;
		bool m_bNIR_CannyPolygonAppxResultEnable;
		double m_NIR_CannyPolygonAppxEpsilon;
		bool m_bNIR_CannyRemovingWallEnable;
		bool m_bNIR_CannyRemovingWallResultEnable;
		int m_NIR_CannyRemovingWallIteration;

		// tracking parameters
		double m_trk_TargetPositionOffset_majorAxis;
		double m_trk_TargetPositionOffset_minorAxis;
		enum{Brain, Yolk, LeftEye, RightEye};
		Point2d Tracking_SimulationTarget;
		vector<Point2d> Tracking_SimulationVelSet;

		void pushButton_test1_Clicked(void);
		void pushButton_test2_Clicked(void);


		// Thermal Control functions
		ThermalControl thermalcontroller;
		void ThermalControl_update(void);
		void ThermalControl_loadmap(void);

		//
		uint64 interboutCount;
		MovingAverage_double * fishVelBuffer;
		MovingAverage_double * interboutInterval;
		Point2d interboutInitial_pos;
		Point2d interbout_pos;
		double interbout_std_threhold;
		uint64 interboutInitial_idx;
		bool interbout_motion_stopped;
		double interbout_trvdst;

	protected:
        virtual void keyPressEvent(QKeyEvent *event); // evnet function for stage manual control

	}; // end class MainWindow

} // end namespace UI


void QtDisplayMessageError(QString msg);
void QtDisplayMessageWarning(QString msg);
void QtDisplayMessageInfo(QString msg);
void QtComboBoxItemDisable(QComboBox * mComboBox, int index);
void QtComboBoxItemEnable(QComboBox * mComboBox, int index);
//shared_ptr<QImage> cvMat2QImage(const cv::Mat * src);
//shared_ptr<QImage> cvMat2QImage3(const cv::Mat * src);
