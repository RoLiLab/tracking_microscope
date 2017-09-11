#include "Base/base.h"
#include "VideoProcessor/UI/MainWindow.h"
#include "Qt_Ext/Pixmap.h"

#include "QT_Ext/QGLZoomableImageViewer.h"
#include "Qt_Ext/Qt_Ext.h"

#include "DSP/ImageProcess.h"

#include "XPSQ8/XPS_Q8.h"

#include "NIDAQmx.h"

#include <numeric>
#include <algorithm>
#include <math.h>


#include <H5Cpp.h>
#include "HDF5/hdf5imagewriter.h"
#include "HDF5/hdf5imagereader.h"
#include "HDF5/hdf5datareader.h"
#include "HDF5/hdf5datawriter.h"

#include "DSP/CUDA_kernels.h"

std::mutex Stage_mutex;
std::mutex Rec_mutex;
std::mutex NIRImgCopy_mutex;
//std::lock_guard<std::mutex> lock(Rec_mutex);
std::mutex Rec_teensy;
std::mutex GPUMPC_mutex;
std::mutex NIRCam_mutex;
std::mutex h5_rec_mutes;
bool g_bProcessRunning = false;	//!< Set to false when we're shutting down our application, this causes threads to shut down gracefully
//bool g_bNIRImageProcess = true;	//!< Set to false when we're shutting down our application, this causes threads to shut down gracefully


UI::MainWindow::MainWindow()
{

	// ---------------- QT Setup our UI ----------------
	Snapshot = NULL;
	m_ui.setupUi( this );

	//!!! Main Viewers (Place our image viewers inside their frames)
	m_pImage1= new QGLZoomableImageViewer( m_ui.frameImage_1 ); addMainChild( m_ui.frameImage_1, m_pImage1 );
	m_pImage2= new QGLZoomableImageViewer( m_ui.frameImage_2 );	addMainChild( m_ui.frameImage_2, m_pImage2 );
	m_pImage3= new QGLZoomableImageViewer( m_ui.frameImage_W4_1 );	addMainChild( m_ui.frameImage_W4_1, m_pImage3 );
	//m_pImage4= new QGLZoomableImageViewer( m_ui.frameImage_W4_2 );	addMainChild( m_ui.frameImage_W4_2, m_pImage4 );
	m_pImage5= new QGLZoomableImageViewer( m_ui.frameImage_W4_3 );	addMainChild( m_ui.frameImage_W4_3, m_pImage5 );
	//m_pImage6 = new QGLZoomableImageViewer(m_ui.frameImage_W4_4);	addMainChild(m_ui.frameImage_W4_4, m_pImage6);
	m_pImageR1 = new QGLZoomableImageViewer(m_ui.frameImage_replay_1);	addMainChild(m_ui.frameImage_replay_1, m_pImageR1);
	m_pImageR2 = new QGLZoomableImageViewer(m_ui.frameImage_replay_2);	addMainChild(m_ui.frameImage_replay_2, m_pImageR2);
	m_pImageR3 = new QGLZoomableImageViewer(m_ui.frameImage_replay_3);	addMainChild(m_ui.frameImage_replay_3, m_pImageR3);
	m_pImageR4 = new QGLZoomableImageViewer(m_ui.frameImage_replay_4);	addMainChild(m_ui.frameImage_replay_4, m_pImageR4);

	m_ui.toolBox->setCurrentIndex(0);
	// QT setting main contoller panel
	connect(m_ui.pushButton_refreshstart, &QPushButton::clicked, this, &MainWindow::ResetAllprocess);
	connect(m_ui.pushButton_recording, &QPushButton::clicked, this, &MainWindow::StartStopRecording);
	connect(m_ui.pushButton_saveSetting, &QPushButton::clicked, this, &MainWindow::RecSettings);
	connect(m_ui.pushButton_ResetError, &QPushButton::clicked, this, &MainWindow::reset_failurecount);
	connect(m_ui.checkBox_nir_imgproc_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {GlobalMap_imgprocessenable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_nir_globalmapbuilding_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {GlobalMap_globalmapbuildingenable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_nir_tracking_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {tracking_enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_nir_mpc_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {tracking_MPCenable_QCheckBoxtoggled(flag); });

	connect(m_ui.checkBox_stage_followingerror_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {followingerror_enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_stage_stageboundaryerror_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {bounderror_enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_resp_offsetEnable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {responseEstimation_RandomoffsetEnable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_tracking_fishposeJitter_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {jitterfilter_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_Piezo_trk_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {Piezo_trk_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_tracking_fishpose_distancefromyolk, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {constanctdistantfromyolk_brainpos_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_resp_fakefishEnable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {fakefishtest_Enable_QCheckBoxtoggled(flag); });

	connect(m_ui.checkBox_display_show_error, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_collect_error, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_show_fish, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_collect_fish, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_show_piezo, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_collect_piezo, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_show_stage, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_display_collect_stage, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {display_Enable_QCheckBoxtoggled(flag); });

	connect(m_ui.checkBox_intensity_ratio_set, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {intensity_ratio_set_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_epiDisplay, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {toggle_EPI_display_checkBox_epiDisplay(flag); });

	connect(m_ui.checkBox_piezoConcatenate_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {setPiezoConcatenate(flag); });
	connect(m_ui.checkBox_piezo_MaxCorEnable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {setPiezoMaxupdateEnable(flag); });

	// QT setting MPC panel
	connect(m_ui.checkBox_tracking_fishposestimation_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {tracking_fishpredictionenable_QCheckBoxtoggled(flag); });

	//! QT setting - Not sorted yet

	connect(m_ui.pushButton_DMDconnect, &QPushButton::clicked, this, &MainWindow::connectDMD);
	connect(m_ui.pushButton_DMDstartstop, &QPushButton::clicked, this, &MainWindow::startstopDMD);
	connect(m_ui.checkBox_DMDautoreset, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {autoresetDMD_QCheckBoxtoggled(flag); });
	m_ui.pushButton_DMDstartstop->setEnabled(false);
	connectDMD();
	startstopDMD();
	lc9000_autoreset = true;



	//! QT setting (PiezoControl)
	connect( m_ui.pushButton_Piezo_MoveUp, &QPushButton::clicked, this, &MainWindow::Piezo_ManualMoveUp);
	connect( m_ui.pushButton_Piezo_MoveDown, &QPushButton::clicked, this, &MainWindow::Piezo_ManualMoveDown);
	connect(m_ui.pushButton_Piezo_STOP, &QPushButton::clicked, this, &MainWindow::Piezo_ManualStop);

	connect(m_ui.pushButton_RefBufferCapture, &QPushButton::clicked, this, &MainWindow::refStackCapture);
	connect(m_ui.pushButton_RefBufferSave, &QPushButton::clicked, this, &MainWindow::refStackSave);
	connect(m_ui.pushButton_RefBufferLoad, &QPushButton::clicked, this, &MainWindow::refStackLoad);
	connect(m_ui.pushButton_RefBufferClear, &QPushButton::clicked, this, &MainWindow::refStackClear);
	connect(m_ui.pushButton_Ref_setTarget, &QPushButton::clicked, this, &MainWindow::refSetTargetLayer);
	connect(m_ui.pushButton_zmatcher_limit_set, &QPushButton::clicked, this, &MainWindow::refSetTargetLayer);
	connect(m_ui.pushButton_RefZtest, &QPushButton::clicked, this, &MainWindow::refStackTEST);
	connect(m_ui.spinBox_RefBufferDisplayNo, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &MainWindow::refStackDisplay);
	connect(m_ui.checkBox_piezoSimulation, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {refStackSimulation(flag); });
	connect(m_ui.pushButton_RefBufferZtracking_update_rev, &QPushButton::clicked, this, &MainWindow::refStack_reverse);

	//connect( m_ui.Button_Hamamatsu_updateMax, &QPushButton::clicked, this, &MainWindow::HamamatsuCamera_updateParameters);
	//connect(m_ui.Button_Hamamatsu_updateSignal, &QPushButton::clicked, this, &MainWindow::HamamatsuCamera_updateSignals);
	connect(m_ui.Button_FRLSnashot, &QPushButton::clicked, this, &MainWindow::HamamatsuCamera_Snapshot);
	connect(m_ui.Button_Hamamatsu_save, &QPushButton::clicked, this, &MainWindow::HamamatsuCamera_setgain);

	connect(m_ui.pushButton_globalmap_save, &QPushButton::clicked, this, &MainWindow::GlobalMap_SaveTIFF);
	connect(m_ui.pushButton_globalmap_resetdisplay, &QPushButton::clicked, this, &MainWindow::GlobalMap_displayreset);
	connect(m_ui.pushButton_globalmap_maskupdate, &QPushButton::clicked, this, &MainWindow::GlobalMap_updateFishMask);

	connect(m_ui.pushButton_Stage_I2T_update, &QPushButton::clicked, this, &MainWindow::Stage_I2T_setup);

	//! QT setting (replayer)
	connect(m_ui.toolButton_OpenFile_Folder, &QPushButton::clicked, this, &MainWindow::replayer_openfolder);
	connect(m_ui.toolButton_OpenFile_Data, &QPushButton::clicked, this, &MainWindow::replayer_openfile_nir);
	connect(m_ui.toolButton_OpenFile_NIR, &QPushButton::clicked, this, &MainWindow::replayer_openfile_flr);
	connect(m_ui.toolButton_OpenFile_FLR, &QPushButton::clicked, this, &MainWindow::replayer_openfile_data);
	connect(m_ui.toolButton_replay_seekprev, &QPushButton::clicked, this, &MainWindow::replayer_prev);
	connect(m_ui.toolButton_replay_play, &QPushButton::clicked, this, &MainWindow::replayer_play);
	connect(m_ui.toolButton_replay_seeknext, &QPushButton::clicked, this, &MainWindow::replayer_next);
	connect(m_ui.toolButton_replay_process, &QPushButton::clicked, this, &MainWindow::replayer_proc);
	//! QT setting (White noise Buffered Data)
	connect( m_ui.pushButton_Stage_WN_SampleGeneration, &QPushButton::clicked, this, &MainWindow::Stage_WN_SampleGeneration_Clicked);
	connect( m_ui.pushButton_Stage_WN_RandomInputGeneration, &QPushButton::clicked, this, &MainWindow::Stage_WN_RandomSampleGeneration_Clicked);
	connect( m_ui.pushButton_Stage_WN_SampleLoad, &QPushButton::clicked, this, &MainWindow::Stage_WN_SampleLoad_Clicked);
	connect( m_ui.pushButton_Stage_WN_VelSelLoad, &QPushButton::clicked, this, &MainWindow::Stage_Replay_VelSetLoad_Clicked);
	connect( m_ui.pushButton_Stage_Scan, &QPushButton::clicked, this, &MainWindow::Stage_WN_ScanSampleGeneration);
	connect( m_ui.pushButton_Stage_WN_Start, &QPushButton::clicked, this, &MainWindow::Stage_WN_Start_Clicked);

	//! QT setting (fish position feeding)
	connect(m_ui.toolButton_replay_fishpos, &QPushButton::clicked, this, &MainWindow::replayer_loadfishpos);
	connect(m_ui.checkBox_replay_fishpos_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {replayer_fishpos_enable_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_replay_fishpos_enablefeed, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {replayer_fishpos_start_QCheckBoxtoggled(flag); });
	connect(m_ui.checkBox_replay_fishpos_enableperfectpred, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {replayer_fishpos_perfectprediction_QCheckBoxtoggled(flag); });

	connect(m_ui.checkBox_nir_p1, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {toggle_NIRp1Update(flag); });
	connect(m_ui.checkBox_nir_p2, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {toggle_NIRp2Update(flag); });
	connect(m_ui.checkBox_flr_p1_2, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {toggle_FLRp1Update(flag); });
	connect(m_ui.checkBox_flr_p2_2, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {toggle_FLRp2Update(flag); });

	connect(m_ui.checkBox_DIFFDisplay_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {DIFFDisplay_enable(flag); });
	connect(m_ui.checkBox_DIFFDisplayA_enable, static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::toggled), [this](bool flag) {DIFFDisplayA_enable(flag); });


	WN_TotalSampleCountWhole = 0;
	WN_DataBuffer = NULL;
	WN_DataBuffer_Vel = NULL;
	WN_Enable = false;

	LoadedFishPos_Enable = false;
	LoadedFishPos = NULL;
	LoadedFishPosError_mm = NULL;
	LoadedFishPosOrientation = NULL;
	LoadedFishPos_StageInitialPosition[0] = 0; LoadedFishPos_StageInitialPosition[1] = 0;
	LoadedFishPos_StageInitialPosition_offset[0] = 0; LoadedFishPos_StageInitialPosition_offset[1] = 0;
	connect( m_ui.pushButton_Stage_FIshPosLoad, &QPushButton::clicked, this, &MainWindow::Stage_Replay_FishPosLoad_Clicked);
	connect( m_ui.pushButton_Stage_FIshPosLoad_InitStagePos, &QPushButton::clicked, this, &MainWindow::Stage_Replay_FishPosLoadInitStage_Clicked);
	connect(m_ui.pushButton_Stage_FIshPosLoad_StartSimulation, &QPushButton::clicked, this, &MainWindow::Stage_Replay_FishPosLoadStartSimulation_Clicked);

	connect( m_ui.pushButton_Stage_FIshPosLoad_SineWaveGeneration, &QPushButton::clicked, this, &MainWindow::Stage_FishPos_SinePathGeneration);
	connect( m_ui.pushButton_Stage_FIshPosLoad_SineWaveStart, &QPushButton::clicked, this, &MainWindow::Stage_Replay_FishPosLoadStartSimulation_Clicked);

	connect( m_ui.pushButton_Stage_I2T_update, &QPushButton::clicked, this, &MainWindow::Stage_I2T_setup);
	connect( m_ui.pushButton_Stage_I2T_clear, &QPushButton::clicked, this, &MainWindow::Stage_I2T_clear);
	// Raster mode function
	connect( m_ui.pushButton_RasterGo, &QPushButton::clicked, this, &MainWindow::Raster3DSpace);
	connect( m_ui.pushButton_Raster_LoadCurPos, &QPushButton::clicked, this, &MainWindow::RasterLoadPos);

	//! Stage connection
	connect(m_ui.Button_Stage_updatestutus, &QPushButton::clicked, this, &MainWindow::Stage_updatestatus);
	//! pushbutton for Moving the stage to the set pointion
	//! pushbutton for saving the current stage location
	connect( m_ui.Button_Stage_SaveCurrentPosition, &QPushButton::clicked, this, &MainWindow::Stage_SaveCurrentPosition_QpushButtonClicked);
	m_ui.Button_Stage_SaveCurrentPosition->setDisabled(true);
	//! pushbutton for moving the stage to the loaded location location
	connect( m_ui.Button_Stage_MoveToSavedPosition, &QPushButton::clicked, this, &MainWindow::Stage_MoveToSavedPosition_QpushButtonClicked);
	m_ui.Button_Stage_MoveToSavedPosition->setDisabled(true);
	//! pushbutton for moving the stage to the loaded location location
	//! pushbutton for Update all stage parameters (turn off the Analog velocity control -> update the parameters -> turn on the analog velocity control)
	connect( m_ui.Button_Stage_AVT_UpdateParameters, &QPushButton::clicked, this, &MainWindow::Stage_UpdataParameters_QpushButtonClicked);
	//! Enabling to edit the advanced parameters of stage
	connect( m_ui.Button_Stage_AVT_EnableEditParameters, &QPushButton::clicked, this, &MainWindow::Stage_EditAdvancedParameters_QpushButtonClicked);
	//! stage calibration
	connect( m_ui.pushButton_Stage_ImpulseCalibration_Run2, &QPushButton::clicked, this, &MainWindow::pushButton_Stage_CalibrationImpulseStep_QpushButtonClicked);
	//connect( m_ui.pushButton_Stage_CalibrationSimple_Random, &QPushButton::clicked, this, &MainWindow::Stage_CalibrationSimple_Random_QpushButtonClicked);
	connect(m_ui.pushButton_Stage_CalibrationSimple_Random, &QPushButton::clicked, this, &MainWindow::calibrateNIAOOffset);
	connect(m_ui.pushButton_NI_parametric, &QPushButton::clicked, this, &MainWindow::NI_DO_ParametricSearch);
	connect(m_ui.Button_doubleshot_exp_update, &QPushButton::clicked, this, &MainWindow::NI_DO_update);

	//! Stage update gathering data (manually)


	//! Stage connection only
	connect( m_ui.Button_Stage_ConnectOnly, &QPushButton::clicked, this, &MainWindow::Stage_ConnectOnly_QpushButtonClicked);
	//! Stage Release only (Kill All)
	connect( m_ui.Button_Stage_ReleaseStageOnly, &QPushButton::clicked, this, &MainWindow::Stage_ReleaseStageOnly_QpushButtonClicked);
	//! Stage Initialization
	connect( m_ui.Button_Stage_InitializationOnly, &QPushButton::clicked, this, &MainWindow::Stage_InitializationOnly_QpushButtonClicked);
	//! Stage Ready
	//connect( m_ui.Button_Stage_ReadyOnly, &QPushButton::clicked, this, &MainWindow::Stage_ReadyOnly_QpushButtonClicked);
	connect(m_ui.Button_Stage_ReadyOnly, &QPushButton::clicked, this, &MainWindow::Stage_ReadyHoming);
	//! Stage Ready
	//connect( m_ui.Button_Stage_SearchHomming, &QPushButton::clicked, this, &MainWindow::Stage_HomeSearch_QpushButtonClicked);
	//! Stage Enable Analog Velocity Control
	//connect( m_ui.Button_Stage_EnableAnalogControlOnly, &QPushButton::clicked, this, &MainWindow::Stage_EnableAnalogControlOnly_QpushButtonClicked);
	connect( m_ui.Button_Stage_EnableAnalogControlOnly, &QPushButton::clicked, this, &MainWindow::	Stage_UpdataParameters_QpushButtonClicked);
	//!!! ToolBax - Digital Output External Trigger UI components -----


	QList<QString> TextList;
	QComboBox * mComboBox;

	mComboBox = m_ui.comboBox_imgproc_intimgdispaly;
	TextList.append("Original");
	TextList.append("BG");
	TextList.append("Subtracted");
	TextList.append("G_Subtracted");
	TextList.append("BG-updated");
	//TextList.append("Linear filter y");
	//TextList.append("GRAD Linear filter x");
	//TextList.append("GRAD Linear filter y");
	//TextList.append("Divergency");
	//TextList.append("GlobalMap");
	//TextList.append("bGlobalMapExploreded");
	//connect(mComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [this](int index) {NIR_GPUResultDisplay_QcomboBoxCurrentIndexChanged(index);});
	mComboBox->addItems(TextList); TextList.clear(); mComboBox->setEnabled(true); mComboBox->setCurrentIndex(0);

	m_ui.comboBox_Stage_I2T_MODE->setCurrentIndex(2);

	//! comboBox of Target
	mComboBox = m_ui.comboBox_tracking_fishpostarget;
	connect(mComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [this](int index) {Tracking_TargetSelection_QcomboBoxCurrentIndexChanged(index);});
	TextList.append("Brain"); TextList.append("Yolk"); TextList.append("Left eye"); TextList.append("Right eye"); 	mComboBox->addItems(TextList); TextList.clear(); mComboBox->setEnabled(true); mComboBox->setCurrentIndex(0);
	//! comboBox of seleting tracking method


	//! comboBox of thermal control
	mComboBox = m_ui.comboBox_ThermalControl_mode;
	TextList.append("Time"); TextList.append("Cross"); TextList.append("Map"); TextList.append("half"); TextList.append("half inv"); TextList.append("half repeat");
	TextList.append("half grad"); TextList.append("half grad inv"); TextList.append("half grad repeat");
	TextList.append("circle"); TextList.append("circle_rev");
	mComboBox->addItems(TextList); TextList.clear(); mComboBox->setEnabled(true); mComboBox->setCurrentIndex(0);

	//! comboBox of thermal control
	mComboBox = m_ui.comboBox_Hamamatsu_binning;
	TextList.append("200Hz_2x2"); TextList.append("40Hz_1x1"); mComboBox->addItems(TextList); TextList.clear();
	mComboBox->setEnabled(true); mComboBox->setCurrentIndex(0);
	connect(mComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [this](int index) {HamamatsuCamera_setting(index); });


	//! pushbutton to start to capture epi-fluorescent image based on the updated parameters
	connect( m_ui.pushButton_test1, &QPushButton::clicked, this, &MainWindow::pushButton_test1_Clicked);
	connect( m_ui.pushButton_test2, &QPushButton::clicked, this, &MainWindow::pushButton_test2_Clicked);


	connect(m_ui.pushButton_Piezo_parameterupdate, &QPushButton::clicked, this, &MainWindow::piezoparameterupdate_buttonclicked);

	//m_ui.spinBox_nircam_roix->setValue(248);
	//m_ui.spinBox_nircam_roiy->setValue(308);

	// ----------------------------------------------------------------------------------

	// ----------------- Read initial values from QT -----------------
	// 1. Load all defaults variables
	// 2. Write on the QT
	// 3. Read the current value from QT (Dynamic)
	// 3. Read the current value from QT (Static)
	ReadQtVariables();

	insetmap_pxdist = 0.4;
	insetmap_size_mm = 45;
	int pixels_all = int(insetmap_size_mm / insetmap_pxdist);
	insetmap = new QImage(pixels_all, pixels_all, QImage::Format_Mono);
	insetmap->fill(0);


	// ----------------- GPU SETUP -----------------
	GlobalMap_ParameterLoading(); // load pxdistance and angle and update parameters (640x480 image only)
	GlobalMap_NIR.gpuMalloc();

	// ----------------- Parameters for Calibration -----------------
	EnableCalibration = false;
	Tracking_SimulationTarget = Point2d(0,0);
	WN_vel_Range = 0;
	WN_resolutionPerMMPS = 0;
	m_EnableWhiteNoiseMaker_X = false;
	m_EnableWhiteNoiseMaker_Y = false;
	m_trk_respoffsetfishpos[0] = 0;
	m_trk_respoffsetfishpos[1] = 0;
	m_trk_respoffsetfishpos_ref[0] = 0;
	m_trk_respoffsetfishpos_ref[1] = 0;
	m_trk_brespoffsetenable = 0;
	m_trk_bjitterenable = 1;
	m_bconstanctdistantfromyolk_brainpos_enable = 0 ;
	m_constanctdistantfromyolk_brainpos = 50;
	m_Piezo_trk_enable = 0;
	m_trk_bfollowingerrorEnable = 1;
	m_trk_bboundaryerrorEnable = 1;
	m_trk_bfishpospredictionenable = 1;
	m_trk_brespoffsetrange_steps = 250;

	m_trk_bfakefishposition = 0;
	fakefish_idx = 0;
	fakefish_idxmax = 0;
	fakefishposition_x = NULL;
	fakefishposition_y = NULL;
	fakefishposition_th = NULL;



	// ----------------------------- Double Threads setting ------------------------

	// ----------------- Parameters for Recording -----------------
	//RecBuffer.allocateBuffer(m_ui.spinBox_MaxPreTriggerRecFrms->value());
	//RecBuffer_FLR_PreTrigFrmCountMax = 600;
	//EnableFLRRecording = false;
	//RecBuffer_FLR_ptr = NULL;

	// ----------------- Parameters for Tracking -----------------
	m_trk_targetSelection = Brain;
	_MaxStageSpeed = 100;
	m_piezoInput = 0;

	// ----------------- Video and Data Display for Tracking -----------------
	NIRImageDecoder = NULL;
	FLRImageDecoder = NULL;
	NIRDataDecoder = NULL;
	GlobalMapDecoder = NULL;

	// ----------------- Others for Tracking -----------------

	m_DroppedFrame_NIR = 0;
	m_DroppedFrame_NIRImageProc = 0;
	m_DroppedFrame_PMT = 0;
	m_bPMTs_isLive = false;

	m_Piezo_scanCenter = 0;
	m_Piezo_scanRange = 10;
	m_Piezo_scanStep = 2;
	m_Piezo_TrackingMode = 0;
	m_Piezo_scanImgsPerLayer =1;

	//if (!FLRExpTimeController.handler.open("\\\\.\\COM4"))
	////	QtDisplayMessageError("FLR control teensy is not able to be connected (COM4)");
	//if (!XPSAVController.handler.open("\\\\.\\COM3"))
	//	QtDisplayMessageError("XPS control teensy is not able to be connected (COM3)");
	//XPSAVController.stop();
	// 	---------------- Multithread start ----------------

	//// Position reading thread
	//m_pThreadStagePositionReading.reset(new std::thread([this] { StagePositionReadingThread(); }));
	//// NI signal generation thread
	//m_pThreadNIDigSignal.reset(new std::thread([this] { NIDigSignal_updateThread(); }));
	//// FLR Image thread
	//m_pThreadFLRcapture.reset(new std::thread([this] { HamamatsuCamera_FLRCapture(); }));
	//// FLR Image Rec thread
	//m_pThreadFLRRecord.reset(new std::thread([this] { HamamatsuCamera_FLRRecord(); }));
	//// recording thread
	//m_pThreadRecBuffer_CopyResult.reset(new std::thread([this] { RecBufferCopyResultThread(); }));
	//// MPC recording thread
	//m_pThreadRecBuffer.reset(new std::thread([this] { MPCrecordingThread(); }));
	//// MPC thread
	//m_pThreadNIRMPCtracking.reset(new std::thread([this] { NIRProcessFrameSingleThread(); }));
	//// MPC DAC input
	//m_pThreadStageTeensy.reset(new std::thread([this] { teensyThread(); }));

	disp_data1 = NULL;
	disp_data2 = NULL;
	disp_data3 = NULL;
	disp_data4 = NULL;
	disp_data5 = NULL;

	 EnablePositionReading = false;
	 TotalThreadCount = 2;
	 EnableTeensyReset = false;

	 //ResetAllTracking();;
	 g_bNIRImageProcess = false;
	 m_queueResults.clear();
	 // parameter need to be set before start
	 manualInputVelStage[0] = 0; manualInputVelStage[1] = 0;
	 manualInputVelStage_set[0] = 0; manualInputVelStage_set[1] = 0;

	 connect(m_ui.pushButton_ThermalControl_Update, &QPushButton::clicked, this, &MainWindow::ThermalControl_update);
	 connect(m_ui.pushButton_ThermalControl_loadmap, &QPushButton::clicked, this, &MainWindow::ThermalControl_loadmap);

	 // binary parallel save
	 for (int i = 0; i < harddrivecount_FLRsave; i++)
		 FLRRecord_frameNo.push_back(vector<int32>(0));
	 replayer_fishpos = NULL;
	 replayer_fishpos_OrientationDeg = NULL;
	 replayer_fishpos_enable = false;
	 replayer_fishpos_enable_start = false;
	 replayer_fishpos_enable_perfectprediction = false;
	 replayer_fishpos_totalfrm = 0;
	 replayer_fishpos_curfrm = 0;


	 interboutInterval = new MovingAverage_double(10);
	 fishVelBuffer = new MovingAverage_double(10);
	 interboutCount = 0;
	 interboutInitial_idx = 0;
	 interboutInitial_pos = Point2d(0, 0);
	 interbout_motion_stopped = true;
	 m_ui.tabWidget_Window->setCurrentIndex(2);


	// 	---------------- Start Display update timer ----------------
	const int TIMER_UPDATE_INTERVAL_MILLIS= 50;
	(void)startTimer( TIMER_UPDATE_INTERVAL_MILLIS );
	srand(time(NULL));
	isDoingReset = false;

	//g_bProcessRunning = true;
	//m_pThreadIRLaser.reset(new std::thread([this] { IRLaserUpdate_updateThread(); }));

} // end UI::MainWindow::MainWindow()

UI::MainWindow::~MainWindow() {
	// Clear our running flag and wait for all of our threads to complete
	GlobalMap_NIR.subEnabled = false; // disable the global map update
	GlobalMap_NIR.UpdateEnabled = false; // disable the global map update
	GlobalMap_NIR.UpdateAllEnabled = false; // disable the global map update


	if (g_bProcessRunning) {
	g_bProcessRunning = false;
		m_pThreadStagePositionReading->join();
		// FLR Image thread
		m_pThreadFLRcapture->join();
		// MPC DAC input
		m_pThreadStageTeensy->join();
		// FLR Image Rec thread
		m_pThreadFLRRecord->join();
		// FLR Image Rec thread
		m_pThreadRecBuffer->join();
		// recording thread
		m_pThreadRecBuffer_CopyResult->join();
		// recording thread
		m_pThreadNIDigSignal->join();
		// MPC thread
		m_pThreadNIRMPCtracking->join();
		// Piezo thread
		m_pThreadPiezo->join();
		// IR laser thread
		m_pThreadIRLaser->join();
		for (int i = 0; i < harddrivecount_FLRsave; i++)
			m_pThread_FLRRecord_binary[i]->join();
	}
	delete insetmap;
}

// Manual Control Key Events (QT)
void UI::MainWindow::keyPressEvent(QKeyEvent *event) {
	if (ILS200LM.AnalogTrackingMode == XPS_Q8::AnnalogPositionTracking)
		return;

	if (!m_Tracking_EnableLive) { // if the stage is not on tracking mode, DO manual control
		// value update
		manualInputVelStage[0] = m_InputVolStage.x;
		manualInputVelStage[1] = m_InputVolStage.y;
		double piezoInput = m_piezoInput;
		if (m_ui.checkBox_Stage_ManualControlVelocity_sync->isChecked()) {
			ILS200LM.ManualControlVeloicty_x = m_ui.doubleSpinBox_Stage_ManualControlVelocity_x->value();
			ILS200LM.ManualControlVeloicty_y = ILS200LM.ManualControlVeloicty_x;
			m_ui.doubleSpinBox_Stage_ManualControlVelocity_y->setValue(ILS200LM.ManualControlVeloicty_y);
		}
		else{
			ILS200LM.ManualControlVeloicty_x = m_ui.doubleSpinBox_Stage_ManualControlVelocity_x->value();
			ILS200LM.ManualControlVeloicty_y = m_ui.doubleSpinBox_Stage_ManualControlVelocity_y->value();
		}
		double ManualMotionVelocity_x = ILS200LM.ManualControlVeloicty_x;
		double ManualMotionVelocity_y = ILS200LM.ManualControlVeloicty_y;
		double ManualMotionVelocity_ZeroX = ILS200LM.Velocity2Volt_x(0);
		double ManualMotionVelocity_ZeroY = ILS200LM.Velocity2Volt_y(0);
		manualInputVelStage_set[0] = ManualMotionVelocity_ZeroX;
		manualInputVelStage_set[1] = ManualMotionVelocity_ZeroY;
		switch(event->key()) { // switch: get the key (1-9 : number pad)
		case 0x32: // DOWN
			manualInputVelStage[0] = ManualMotionVelocity_ZeroX; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = -ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x38: // UP
			manualInputVelStage[0] = ManualMotionVelocity_ZeroX; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x34: // LEFT
			manualInputVelStage[0] = ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_ZeroY; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x36: // RIGHT
			manualInputVelStage[0] = -ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_ZeroY; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x39: // UP-RIGHT
			manualInputVelStage[0] = -ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x37: // UP-LEFT
			manualInputVelStage[0] = ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x33: // DOWN-RIGHT
			manualInputVelStage[0] = -ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = -ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		case 0x31: // DOWN-LEFT
			manualInputVelStage[0] = ManualMotionVelocity_x; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = -ManualMotionVelocity_y; // Velocity of the X positioner demanded during second jog op
			break;
		default: // ANYKEY to STOP
			manualInputVelStage[0] = ManualMotionVelocity_ZeroX; // Velocity of the X positioner demanded during second jog op
			manualInputVelStage[1] = ManualMotionVelocity_ZeroY; // Velocity of the X positioner demanded during second jog op
			break;
		} // end switch

		// Disable tracking (delete the input voltage and replace w/ the previous voltage
		{
			//ControllHardWare_2P.UpdateAO23(
			//	ILS200LM.Velocity2Volt_x(voltageInputStage[0]),
			//	ILS200LM.Velocity2Volt_y(voltageInputStage[1]));
			//double volX =  ILS200LM.Velocity2Volt_x(voltageInputStage[0]);
			//double volY =  ILS200LM.Velocity2Volt_x(voltageInputStage[1]);
			//std::lock_guard<std::mutex> lock(Rec_teensy);
			//XPSAVController.sendVolInput(0, volX, volY);
			//ControllHardWare_2P.UpdateAO1(piezoInput);
		}
		m_ui.centralwidget->setFocus(); // if it is in manual control
	}// endif the stage is not on tracking mode, DO manual control
} // END keyPressEvent function

//------------------------------------------------------------------------------
// ONTIMER function (Display update)
//------------------------------------------------------------------------------
void UI::MainWindow::onTimer() {
	try {
		double fstarttime, fEndSeconds, fstarttime2 = 0;
		fstarttime = GetCycleCountSeconds();
		if (fstarttime2 != 0) m_ThreadProcessTimeMills_DisplayUpdateCycle.update((fstarttime - fstarttime2)*1000);
		UpdateTrackingParameters();
		int toolboxidx = m_ui.toolBox->currentIndex();
		if (!g_bProcessRunning || RecBuffer.FrmNo > 0) {
			TrackingMessage * DispMsg = RecBuffer.getDispFrameRec();
			if (DispMsg) {
				if (DispMsg->srcNIR->data) {
					int _min = m_ui.spinBox_NIRDisplay_min->value();
					int _max = m_ui.spinBox_NIRDisplay_max->value();
					auto pImage = toQImage_shift((DispMsg->srcNIR), (uInt16)_min, (UINT16)_max, 4, m_ui.checkBox_nir_globalmapdisplay_enable->isChecked());
					//auto pImage = toQImage_u12((DispMsg->srcNIR), (uInt16)_min, (UINT16)_max);
					if (DispMsg->NIRfishPos.m_yolk.center.x > 0 && pImage != NULL) {
						paint_cvpoint2(DispMsg->NIRfishPos.m_yolk.center, Qt::red, pImage);
						paint_cvpoint2(DispMsg->NIRfishPos.m_eyeLeft.center, Qt::green, pImage);
						paint_cvpoint2(DispMsg->NIRfishPos.m_eyeRight.center, Qt::yellow, pImage);
						if (qtgraph2.enabledraw) {
							paint_cvpoint2(DispMsg->NIRfishPos.centerEyes, Qt::cyan, pImage);
							paint_cvline(DispMsg->NIRfishPos.centerEyes, DispMsg->NIRfishPos.orienation*PI / 180, Qt::cyan, pImage);
							paint_Ellipse(DispMsg->NIRfishPos.centerEyes, 3, Qt::cyan, pImage);
							paint_Ellipse(m_ImageTargetPosPx, 9, Qt::darkCyan, pImage);
						}
						m_ui.lcdNumber_Fish_Pos_x->display(getfixeddecimal(DispMsg->m_CntrData.data[Xfish], 3));
						m_ui.lcdNumber_Fish_Pos_y->display(getfixeddecimal(DispMsg->m_CntrData.data[Yfish], 3));
						m_ui.lcdNumber_Fish_Leye_Angle->display(getfixeddecimal(DispMsg->NIRfishPos.m_eyeLeft.angle, 1));
						m_ui.lcdNumber_Fish_Reye_Angle->display(getfixeddecimal(DispMsg->NIRfishPos.m_eyeRight.angle, 1));
						m_ui.lcdNumber_Fish_eye_AngleDiff->display(getfixeddecimal(DispMsg->NIRfishPos.m_yolk.angle, 1));
						m_ui.progressBar_Fish_eye_AngleDiff->setValue(int(DispMsg->NIRfishPos.m_yolk.angle));

						m_ui.lcdNumber_Fish_Pos_theta->display(getfixeddecimal(DispMsg->m_CntrData.data[HeadingFish], 1));
						m_ui.lcdNumber_Fish_Pos_x_filtered->display(getfixeddecimal(FishPosEstimator.xproj.data[0], 3));
						m_ui.lcdNumber_Fish_Pos_y_filtered->display(getfixeddecimal(FishPosEstimator.yproj.data[0], 3));
						double filteredAngle_Deg180 = FishPosEstimator.thfiltered.data[0];
						while (filteredAngle_Deg180 > 180)filteredAngle_Deg180 = filteredAngle_Deg180 - 360;
						while (filteredAngle_Deg180 < -180) filteredAngle_Deg180 = filteredAngle_Deg180 + 360;
						m_ui.lcdNumber_Fish_Pos_theta_filtered->display(getfixeddecimal(filteredAngle_Deg180, 1));

						/*if (disp_data1 && disp_data1_enable[0]) {
							int x0 = 70;
							int y0 = 120;
							int dx = 1;
							double ygain = 100;
							paint_line(Point2d(x0, y0), Point2d(x0 + 500, y0), Qt::white, pImage);
							paint_line(Point2d(x0, y0 - 100), Point2d(x0 + 500, y0 - 100), Qt::white, pImage);
							paint_line(Point2d(x0, y0 - 10), Point2d(x0 + 500, y0 - 10), Qt::white, pImage);
							paint_line(Point2d(x0, y0 - 20), Point2d(x0 + 500, y0 - 20), Qt::white, pImage);
							paint_line(Point2d(x0, y0 - 30), Point2d(x0 + 500, y0 - 30), Qt::white, pImage);
							for (int i = 0; i < 21; i++)
							paint_line(Point2d(x0 + dx * 25 * i, y0), Point2d(x0 + dx * 25 * i, y0 - 100), Qt::white, pImage);
							//paint_graph(disp_data1, disp_totalcount, disp_data1_idx, x0, y0, dx, ygain, 0, Qt::cyan, pImage);
							paint_graph(disp_data1, disp_totalcount, -1, x0, y0, dx, ygain, 0, Qt::cyan, pImage);
							}*/

						//if (disp_data4 && disp_data4_enable[0]) {
						//	paint_graph(disp_data4[0], disp_data4[1], disp_totalcount, disp_data4_idx,
						//		DispMsg->NIRfishPos.centerEyes.x, DispMsg->NIRfishPos.centerEyes.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx,
						//		DispMsg->m_CntrData.data[Xfish], DispMsg->m_CntrData.data[Yfish], Qt::magenta, pImage);
						//}
						//if (disp_data5 && disp_data5_enable[0]) {
						//	paint_graph(disp_data5[0], disp_data5[1], disp_totalcount, disp_data5_idx,
						//		m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx,
						//		DispMsg->stagePos.CurrentPosition[0], DispMsg->stagePos.CurrentPosition[1], Qt::yellow, pImage);
						//}
						//double gmap_offset[] = { 0 , 250};
						//double gmap_offset2[] = { -m_ui.doubleSpinBox_Map_cx_mm->value() / insetmap_pxdist,  m_ui.doubleSpinBox_Map_cy_mm->value() / insetmap_pxdist };
						//paint_Ellipse(Point2i(int((-DispMsg->m_CntrData.data[Xfish] + insetmap_size_mm / 2) / insetmap_pxdist + gmap_offset[0]),
						//	int((-DispMsg->m_CntrData.data[Yfish] + insetmap_size_mm / 2) / insetmap_pxdist + gmap_offset[1])), 3, Qt::red, pImage);
						//double gridxy_gmap[] = {40, 90, 140, 190};
						//for (int i = 0; i < 4; i++) {
						//	paint_line(Point2d(gridxy_gmap[i] + gmap_offset[0], gridxy_gmap[0] + gmap_offset[1]), Point2d(gridxy_gmap[i] + gmap_offset[0], gridxy_gmap[3] + gmap_offset[1]), QColor(100, 0, 0), pImage);
						//	paint_line(Point2d(gridxy_gmap[0] + gmap_offset[0], gridxy_gmap[i] + gmap_offset[1]), Point2d(gridxy_gmap[3] + gmap_offset[0], gridxy_gmap[i] + gmap_offset[1]), QColor(100, 0, 0), pImage);
						//}
						qtgraph0_gmap.drawnow(100, 380, 0,0,pImage);
						//if (thermalcontroller.mode == 1)
						//	qtgraph0_thermal.drawnow(m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, DispMsg->stagePos.CurrentPosition[0], DispMsg->stagePos.CurrentPosition[1], pImage);
						if (thermalcontroller.mode >= 9 || thermalcontroller.mode == 8) {
							Point2d globalCenter;
							globalCenter.x = (DispMsg->stagePos.CurrentPosition[0] - thermalcontroller.cx_mm) / GlobalMap_NIR.PxDist_mmppx + NIRCameraPG.ROIWidth / 2;
							globalCenter.y = (DispMsg->stagePos.CurrentPosition[1] - thermalcontroller.cy_mm) / GlobalMap_NIR.PxDist_mmppx + NIRCameraPG.ROIHeight / 2;
							paint_Ellipse(globalCenter, thermalcontroller.dx_mm / GlobalMap_NIR.PxDist_mmppx, Qt::red, pImage);
							paint_Ellipse(globalCenter, thermalcontroller.dy_mm / GlobalMap_NIR.PxDist_mmppx, Qt::green, pImage);

							paint_Ellipse(Point2d(100 - thermalcontroller.cx_mm / insetmap_pxdist, 380 - thermalcontroller.cy_mm / insetmap_pxdist), thermalcontroller.dx_mm / insetmap_pxdist, QColor(100, 100, 100), pImage);
							paint_Ellipse(Point2d(100 - thermalcontroller.cx_mm / insetmap_pxdist, 380 - thermalcontroller.cy_mm / insetmap_pxdist), thermalcontroller.dy_mm / insetmap_pxdist, QColor(100, 100, 100), pImage);
							paint_Ellipse(Point2d(100 - thermalcontroller.cx_mm / insetmap_pxdist, 380 - thermalcontroller.cy_mm / insetmap_pxdist), thermalcontroller.dx_mm / insetmap_pxdist / 2, QColor(100, 100, 100), pImage);
							paint_Ellipse(Point2d(100 - thermalcontroller.cx_mm / insetmap_pxdist, 380 - thermalcontroller.cy_mm / insetmap_pxdist), thermalcontroller.dy_mm / insetmap_pxdist / 2, QColor(100, 100, 100), pImage);
						}
						qtgraph0.drawnow(m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, DispMsg->stagePos.CurrentPosition[0], DispMsg->stagePos.CurrentPosition[1], pImage);
						qtgraph1.drawnow(m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, DispMsg->stagePos.CurrentPosition[0], DispMsg->stagePos.CurrentPosition[1], pImage);
						qtgraph2.drawnow(DispMsg->NIRfishPos.centerEyes.x, DispMsg->NIRfishPos.centerEyes.y, DispMsg->m_CntrData.data[Xfish], DispMsg->m_CntrData.data[Yfish], pImage);
						qtgraph3.drawnow1D(70.0, 120.0, pImage, 0);
						//qtgraph4.drawnow1D(70.0, 120.0, pImage, -2);
					}
					// always display in main panel
					double temp = sqrt(DispMsg->m_CntrData.data[Xerr] * DispMsg->m_CntrData.data[Xerr] + DispMsg->m_CntrData.data[Yerr] * DispMsg->m_CntrData.data[Yerr]);
					m_ui.lcdNumber_Stage_err_x->display(getfixeddecimal(DispMsg->m_CntrData.data[Xerr] * 1000,1));
					m_ui.lcdNumber_Stage_err_y->display(DispMsg->m_CntrData.data[Yerr] * 1000);
					m_ui.lcdNumber_Stage_err->display(temp * 1000);
					m_ui.lcdNumber_Stage_Pos_x->display(ILS200LM.CurrentStatus.CurrentPosition[0]);
					m_ui.lcdNumber_Stage_Pos_y->display(ILS200LM.CurrentStatus.CurrentPosition[1]);
					m_ui.lcdNumber_nir_freqHz->display(1000.0 / (double)m_ThreadProcessTimeMills_FramethreadCycle.MVA);
					m_ui.lcdNumber_failure_lateinputc->display((int)ReadyOnTimeMPCInputFailureCount);
					m_ui.lcdNumber_failure_positionreading->display((int)(PositionGatheringInFailureCount));
					m_ui.lcdNumber_failure_fishdetection->display((int)(FishDetectionFailureCount));

					m_ui.lcdNumber_interboutinterval_curtime->display((int)(interboutCount*0.004));
					m_ui.lcdNumber_interboutinterval_freq->display(1.0/(interboutInterval->MVA*0.004));
					m_ui.lcdNumber_interboutinterval_trvdist->display((double)(interbout_trvdst));


					int totalproctime_sec = (int)((double)(RecBuffer.FrmNo)*((double)ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR) / 10000.0);
					int totalrectime_sec = (int)((double)(RecBuffer.recFrms - RecBuffer.recFrms_Start)*((double)(ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR) / 10000.0));
					if (HamamatsuCamera_C11440.b_recording == true || RecBuffer.b_recording == true) m_ui.progressBar_recording->setValue(1); else m_ui.progressBar_recording->setValue(0);
					if (RecBuffer.b_recording) m_ui.progressBar_recording_nir->setValue((int)(100.0*(double)RecBuffer.recFrms_left / (double)RecBuffer.MaxBufferSize));
					if (ILS200LM.bRangeError) m_ui.progressBar_stage_rangeerror->setValue(1); else m_ui.progressBar_stage_rangeerror->setValue(0);
					if (MPC_X.controlmanager.b_followingerrorfailure) m_ui.progressBar_stage_followingerror->setValue(1); else m_ui.progressBar_stage_followingerror->setValue(0);
					switch (toolboxidx) {
					case 0:
						m_ui.lcdNumber_time_live_min->display(totalproctime_sec / 60);
						m_ui.lcdNumber_time_live_sec->display(totalproctime_sec % 60);
						m_ui.lcdNumber_time_rec_min->display(totalrectime_sec / 60);
						m_ui.lcdNumber_time_rec_sec->display(totalrectime_sec % 60);
						m_ui.lcdNumber_tracking_cycle_mean->display(m_ThreadProcessTimeMills_FramethreadCycle.MVA);
						m_ui.lcdNumber_tracking_cycle_max->display(m_ThreadProcessTimeMills_FramethreadCycle._MAX);
						m_ui.lcdNumber_tracking_stgpos_mean->display(m_ThreadProcessTimeMills_Framethread_Stg.MVA);
						m_ui.lcdNumber_tracking_stgpos_max->display(m_ThreadProcessTimeMills_Framethread_Stg._MAX);
						m_ui.lcdNumber_tracking_imgproc_mean->display(m_ThreadProcessTimeMills_Framethread_ImgProcMPC.MVA);
						m_ui.lcdNumber_tracking_imgproc_max->display(m_ThreadProcessTimeMills_Framethread_ImgProcMPC._MAX);
						m_ui.lcdNumber_tracking_totalproc_mean->display(m_ThreadProcessTimeMills_Framethread_Total.MVA);
						m_ui.lcdNumber_tracking_totalproc_max->display(m_ThreadProcessTimeMills_Framethread_Total._MAX);
						m_ui.lcdNumber_tracking_rec_mean->display(m_ThreadProcessTimeMills_Framethread_nirImgRecProc.MVA);
						m_ui.lcdNumber_tracking_rec_max->display(m_ThreadProcessTimeMills_Framethread_nirImgRecProc._MAX);
						m_ui.lcdNumber_tracking_rec_frames->display((int)(RecBuffer.recFrms - RecBuffer.recFrms_Start));
						m_ui.lcdNumber_tracking_rec_remainedframes->display((int)RecBuffer.recFrms_left);
						m_ui.lcdNumber_NI_cycle_mean->display(m_ThreadProcessTimeMills_NIWritingCycle.MVA);
						m_ui.lcdNumber_NI_cycle_max->display(m_ThreadProcessTimeMills_NIWritingCycle._MAX);
						m_ui.lcdNumber_stage_cycle_mean->display(m_ThreadProcessTimeMills_StagePositionReadingCycle.MVA);
						m_ui.lcdNumber_stage_cycle_max->display(m_ThreadProcessTimeMills_StagePositionReadingCycle._MAX);
						m_ui.lcdNumber_teensy_cycle_mean->display(m_ThreadProcessTimeMills_TeensyCycle.MVA);
						m_ui.lcdNumber_teensy_cycle_max->display(m_ThreadProcessTimeMills_TeensyCycle._MAX);
						m_ui.lcdNumber_displayupdate_cycle_mean->display(m_ThreadProcessTimeMills_DisplayUpdate.MVA);
						m_ui.lcdNumber_displayupdate_cycle_max->display(m_ThreadProcessTimeMills_DisplayUpdate._MAX);
						m_ui.lcdNumber_piezo_cycle_mean->display(m_ThreadProcessTimeMills_PiezoCycle.MVA);
						m_ui.lcdNumber_piezo_cycle_max->display(m_ThreadProcessTimeMills_PiezoCycle._MAX);
						m_ui.lcdNumber_IRlaser_cycle_mean->display(m_ThreadProcessTimeMills_IRlaser.MVA);
						m_ui.lcdNumber_IRlaser_cycle_max->display(m_ThreadProcessTimeMills_IRlaser._MAX);

						m_ui.lcdNumber_reserved1_mean->display(ILS200LM.I2T_X.I2T());
						m_ui.lcdNumber_reserved1_max->display(ILS200LM.I2T_Y.I2T());
						m_ui.lcdNumber_reserved2_mean->display(ILS200LM.I2T_X.AccelerationLimit);
						m_ui.lcdNumber_reserved2_max->display(ILS200LM.I2T_Y.AccelerationLimit);
						m_ui.lcdNumber_reserved3_mean->display(m_autosetDMDtime_ms.MVA);
						m_ui.lcdNumber_reserved3_max->display(m_autosetDMDtime_ms._MAX);



						break;
					case 1:
						m_ui.lcdNumber_zebrafish_size_px->display(DispMsg->NIRfishPos.AreaSize);
						m_ui.lcdNumber_zebrafish_headingfitnessCOSphi->display(DispMsg->NIRfishPos.fitness_heading);

						break;
					case 2:
						break;
					case 3:
						break;
					case 4:
						break;
					case 5:
						break;
					case 6:
						m_ui.lcdNumber_replay_curfrm->display(replayer_fishpos_curfrm);
						break;
					default:
						break;
					}

					m_ui.doubleSpinBox_ThermalControl_currentValue->setValue(DispMsg->stagePos.thermalcntrinput);
					m_ui.progressBar_ThermalControl_currentVolts->setValue(int(DispMsg->stagePos.thermalcntrinput * 100));
					m_ui.progressBar_ThermalControl_circle_rev_on->setValue(int(((double)thermalcontroller.circle_inv_counter) * 100));
					if (thermalcontroller.circle_inv_cooldown_enable)
						m_ui.progressBar_ThermalControl_circle_rev_off->setValue(int(((double)thermalcontroller.circle_inv_counter - thermalcontroller.circle_inv_counter_cooldown) * 100));
					else
						m_ui.progressBar_ThermalControl_circle_rev_off->setValue(0);
					if (m_ui.tabWidget_Window->currentIndex() == 0)
						m_pImage1->SetImage(pImage);
					else if (m_ui.tabWidget_Window->currentIndex() == 2)
						m_pImage3->SetImage(pImage);

					if (m_ui.tabWidget_Window->currentIndex() == 0) {
						if (m_ui.checkBox_NIR_center->isChecked())
							paint_Ellipse(Point2i(m_pImage1->curpos.x(), m_pImage1->curpos.y()), 1, Qt::yellow, pImage);
						m_ui.lcdNumber_NIR_px_x->display(m_pImage1->curpos.x());
						m_ui.lcdNumber_NIR_px_y->display(m_pImage1->curpos.y());
						if (m_ui.checkBox_nir_p1->isChecked()) paint_Ellipse(Point2i((int)NIR_p1.x, (int)NIR_p1.y), 1, Qt::green, pImage);
						if (m_ui.checkBox_nir_p2->isChecked()) paint_Ellipse(Point2i((int)NIR_p2.x, (int)NIR_p2.y), 1, Qt::green, pImage);
						m_pImage1->SetImage(pImage);
					}
					else if (m_ui.tabWidget_Window->currentIndex() == 2) {
						if (m_ui.checkBox_NIR_center->isChecked())
							paint_Ellipse(Point2i(m_pImage3->curpos.x(), m_pImage3->curpos.y()), 1, Qt::yellow, pImage);
						m_ui.lcdNumber_NIR_px_x->display(m_pImage3->curpos.x());
						m_ui.lcdNumber_NIR_px_y->display(m_pImage3->curpos.y());
						if (m_ui.checkBox_nir_p1->isChecked()) paint_Ellipse(Point2i((int)NIR_p1.x, (int)NIR_p1.y), 1, Qt::green, pImage);
						if (m_ui.checkBox_nir_p2->isChecked()) paint_Ellipse(Point2i((int)NIR_p2.x, (int)NIR_p2.y), 1, Qt::green, pImage);
						m_pImage3->SetImage(pImage);
					}
				}
			}
		}
		// Hamamastu camera related parameters
		if (g_bProcessRunning && HamamatsuCamera_C11440.pImg && HamamatsuCamera_C11440.FrmNo > 0) {
			uint16 * curImg = HamamatsuCamera_C11440.pImg;
			if ((HamamatsuCamera_C11440.pImg_double) && (HamamatsuCamera_C11440.epiDisplay))
				curImg = HamamatsuCamera_C11440.pImg_double;
			// variable update
			int _min = m_ui.spinBox_Hamamatsu_IntMinSet->value();
			int _max = m_ui.spinBox_Hamamatsu_IntMaxSet->value();
			if (_min >= _max) {
				_max = _min + 1;
				m_ui.spinBox_Hamamatsu_IntMaxSet->setValue(_max);
			}

			// image display
			img_uint16 FLRimageDisp = img_uint16(curImg, imSize(HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height));
			auto pImage_FLR = toQImage_tf(&FLRimageDisp, (uint16)_min, (uint16)_max);

			//HamamatsuCamera_C11440.HCAM_IntRangeUpdate(curImg); // update image intensity maximum value
			m_ui.spinBox_Hamamatsu_IntMax->setValue((int)zaxis_controller.imgdata_cur->intensity_max);
			m_ui.spinBox_Hamamatsu_IntMean->setValue((int)zaxis_controller.imgdata_cur->intensity_mean);
			//if (zaxis_controller.imgdata_cur)
			//	m_ui.spinBox_Hamamatsu_IntMax->setValue((int)zaxis_controller.imgdata_cur->intensity_max); // update image intensity maximum value




			qtgraph5_EPIintMean.drawnow1D(100, 600, pImage_FLR, 0);
			qtgraph5_EPI_peizoZset.drawnow1D(100, 600, pImage_FLR, 0);
			qtgraph5_EPI_peizoZreal.drawnow1D(100, 600, pImage_FLR, 0);

			double filteredAngle = FishPosEstimator.thfiltered.data[0]*3.141592/180;
			double cx1 = cos(filteredAngle); double sx1 = sin(filteredAngle);
			double cx2 = cos(filteredAngle + 3.141592); double sx2 = sin(filteredAngle + 3.141592);
			paint_line(Point2d(sx1 * 1500 + pImage_FLR->width() / 2, cx1 * 1500 + pImage_FLR->height() / 2),
				Point2d(sx2 * 1500 + pImage_FLR->width() / 2, cx2 * 1500 + pImage_FLR->height() / 2), Qt::yellow, pImage_FLR);

			double cx3 = cos(filteredAngle + 3.141592 / 2); double sx3 = sin(filteredAngle + 3.141592 / 2);
			double cx4 = cos(filteredAngle + 3.141592 * 3 / 2); double sx4 = sin(filteredAngle + 3.141592 * 3 / 2);
			paint_line(Point2d(sx3 * 1500 + pImage_FLR->width() / 2, cx3 * 1500 + pImage_FLR->height() / 2),
				Point2d(sx4 * 1500 + pImage_FLR->width() / 2, cx4 * 1500 + pImage_FLR->height() / 2), Qt::magenta, pImage_FLR);
			/*int x0 = 150;
			int y0 = 1500;
			int dx = 3;
			double ygain = 100;
			paint_graph(zaxis_controller.z, x0, y0, dx, 0.25, 0, Qt::red, pImage_FLR);
			paint_graph(zaxis_controller.z_real, x0, y0, dx, 0.25, 0, Qt::blue, pImage_FLR);
			paint_line(Point2d(x0, y0), Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0), Qt::yellow, pImage_FLR);
			paint_line(Point2d(x0, y0 - 0.25 * 400), Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0 - 0.25 * 400), Qt::yellow, pImage_FLR);
			paint_line(Point2d(x0, y0 - zaxis_controller.z_vmean_max_last / 4), Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0 - zaxis_controller.z_vmean_max_last / 4), Qt::green, pImage_FLR);
			paint_line(Point2d(0, HamamatsuCamera_C11440.ROI / 2), Point2d(HamamatsuCamera_C11440.ROI, HamamatsuCamera_C11440.ROI / 2), Qt::yellow, pImage_FLR);
			paint_line(Point2d(HamamatsuCamera_C11440.ROI / 2, 0), Point2d(HamamatsuCamera_C11440.ROI / 2, HamamatsuCamera_C11440.ROI), Qt::yellow, pImage_FLR);
			*/
			//paint_line(Point2d(x0, y0 - (zaxis_controller.mu_high - zaxis_controller.mu_low)*ygain),
			//	Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0 - (zaxis_controller.mu_high - zaxis_controller.mu_low)*ygain),
			//	Qt::blue, pImage_FLR);
			//paint_line(Point2d(x0, y0), Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0), Qt::yellow, pImage_FLR);
			//paint_line(Point2d(x0, y0 - (zaxis_controller.mu_threshold - zaxis_controller.mu_low)*ygain),
			//	Point2d(x0 + (zaxis_controller.z.size() - 1)*dx, y0 - (zaxis_controller.mu_threshold - zaxis_controller.mu_low)*ygain),
			//	Qt::green, pImage_FLR);
			//paint_graph(zaxis_controller.mu, x0, y0, dx, ygain / (zaxis_controller.vmean_max_last), zaxis_controller.vmean_min_last, Qt::cyan, pImage_FLR);
			//paint_Contour(contourPoints, Qt::red, pImage_FLR, false);

			if (m_ui.tabWidget_Window->currentIndex() == 1) {
				if (disp_data1 && disp_data1_enable[0]) {
					int x0 = 70;
					double ygain = 300;
					int y0 = ygain + 100;
					int dx = 3;
					paint_line(Point2d(x0, y0), Point2d(x0 + disp_totalcount*dx, y0), Qt::white, pImage_FLR);
					paint_line(Point2d(x0, y0 - ygain), Point2d(x0 + disp_totalcount*dx, y0 - ygain), Qt::white, pImage_FLR);
					paint_line(Point2d(x0, y0 - ygain*0.1), Point2d(x0 + disp_totalcount*dx, y0 - ygain*0.1), Qt::white, pImage_FLR);
					paint_line(Point2d(x0, y0 - ygain*0.2), Point2d(x0 + disp_totalcount*dx, y0 - ygain*0.2), Qt::white, pImage_FLR);
					paint_line(Point2d(x0, y0 - ygain*0.3), Point2d(x0 + disp_totalcount*dx, y0 - ygain*0.3), Qt::white, pImage_FLR);
					for (int i = 0; i < 21; i++)
						paint_line(Point2d(x0 + dx * disp_totalcount / 20 * i, y0), Point2d(x0 + dx * disp_totalcount / 20 * i, y0 - ygain), Qt::white, pImage_FLR);
					paint_graph(disp_data1, disp_totalcount, disp_data1_idx, x0, y0, dx, ygain, 0, Qt::cyan, pImage_FLR);
				}
				if (m_ui.checkBox_FLR_center->isChecked())
					paint_Ellipse(Point2i(m_pImage2->curpos.x(), m_pImage2->curpos.y()), 1, Qt::yellow, pImage_FLR);

				m_pImage2->SetImage(pImage_FLR);
				m_ui.lcdNumber_FLR_px_x->display(m_pImage2->curpos.x());
				m_ui.lcdNumber_FLR_px_y->display(m_pImage2->curpos.y());
			}
			else if (m_ui.tabWidget_Window->currentIndex() == 2) {
				if (m_ui.checkBox_FLR_center->isChecked())
					paint_Ellipse(Point2i(m_pImage5->curpos.x(), m_pImage5->curpos.y()), 1, Qt::yellow, pImage_FLR);
				m_pImage5->SetImage(pImage_FLR);
				m_ui.lcdNumber_FLR_px_x->display(m_pImage5->curpos.x());
				m_ui.lcdNumber_FLR_px_y->display(m_pImage5->curpos.y());
			}
			//pImage_FLR.reset();

			m_ui.lcdNumber_fluorescent_freqHz->display(1000.0 / (double)m_ThreadProcessTimeMills_FluorescentImgCycle.MVA);

			m_ui.lcdNumber_intensity_pattern1->display(zaxis_controller.imgMeanIntensity_DIFF[0]);
			m_ui.lcdNumber_intensity_pattern2->display(zaxis_controller.imgMeanIntensity_DIFF[1]);
			m_ui.lcdNumber_intensity_ratio->display(zaxis_controller.imgMeanIntensity_DIFF_ratio);
			m_ui.progressBar_intensity_pattern1->setValue(zaxis_controller.imgMeanIntensity_DIFF[0]);
			m_ui.progressBar_intensity_pattern2->setValue(zaxis_controller.imgMeanIntensity_DIFF[1]);
			if (zaxis_controller.imgMeanIntensity_DIFF_ratio >= 0)
				m_ui.progressBar_intensity_ratio->setInvertedAppearance(false);
			else
				m_ui.progressBar_intensity_ratio->setInvertedAppearance(true);
			double intensity_ratio_temp = fabs(zaxis_controller.imgMeanIntensity_DIFF_ratio);
			if (intensity_ratio_temp > 5.0) intensity_ratio_temp = 5.0;
			m_ui.progressBar_intensity_ratio->setValue(intensity_ratio_temp*100);


			if (HamamatsuCamera_C11440.b_recording) m_ui.progressBar_recording_fluorescent->setValue((int)(100.0*(double)HamamatsuCamera_C11440.recFrms_left / (double)HamamatsuCamera_C11440.buffer_nframe));
			switch (toolboxidx) {
			case 0:
				m_ui.lcdNumber_fluorescent_cycle_mean->display(m_ThreadProcessTimeMills_FluorescentImgCycle.MVA);
				m_ui.lcdNumber_fluorescent_cycle_max->display(m_ThreadProcessTimeMills_FluorescentImgCycle._MAX);
				m_ui.lcdNumber_fluorescent_rec_mean->display(m_ThreadProcessTimeMills_FluorescentImgRecProc.MVA);
				m_ui.lcdNumber_fluorescent_rec_max->display(m_ThreadProcessTimeMills_FluorescentImgRecProc._MAX);
				m_ui.lcdNumber_fluorescent_rec_frames->display((int)(HamamatsuCamera_C11440.recFrms - HamamatsuCamera_C11440.recFrms_Start));
				m_ui.lcdNumber_fluorescent_rec_remainedframes->display((int)HamamatsuCamera_C11440.recFrms_left);
				m_ui.lcdNumber_fluorescent_DroppedFrms->display((int)HamamatsuCamera_C11440.FrmNo_Drop);
				//m_ui.lcdNumber_reserved2_mean->display(m_ThreadProcessTimeMills_trkPiezo.MVA);
				//m_ui.lcdNumber_reserved2_max->display(m_ThreadProcessTimeMills_trkPiezo._MAX);
				break;
			case 1:

				break;
			case 2:

				break;
			case 3:
				m_ui.lcdNumber_Stage_Pos_z_center->display((double)ControllHardWare_2P.Piezo_scanSingal.center_um);
				m_ui.lcdNumber_Stage_Pos_z->display((double)ControllHardWare_2P.Piezo_scanSingal.z_result);
				m_ui.lcdNumber_Stage_Pos_z_adjust->display((double)ControllHardWare_2P.Piezo_scanSingal.delta_z);
				break;
			case 4:

				break;
			default:
				break;
			}
		}



		//----------------------------------------temp-------------------------------------------
		bool flag_u12 = m_ui.checkBox_display_u12->isChecked();
		if (flag_u12 && (g_bProcessRunning || HamamatsuCamera_C11440.GS3.b_start) && HamamatsuCamera_C11440.pImg && HamamatsuCamera_C11440.FrmNo > 0) {
			uint16 * curImg = HamamatsuCamera_C11440.pImg;
			if ((HamamatsuCamera_C11440.pImg_double) && (HamamatsuCamera_C11440.epiDisplay))
				curImg = HamamatsuCamera_C11440.pImg_double;
			// variable update
			int _min = m_ui.spinBox_Hamamatsu_IntMinSet->value();
			int _max = m_ui.spinBox_Hamamatsu_IntMaxSet->value();
			if (_min >= _max) {
				_max = _min + 1;
				m_ui.spinBox_Hamamatsu_IntMaxSet->setValue(_max);
			}

			// image display
			img_uint16 FLRimageDisp = img_uint16(curImg, imSize(HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height));
			auto pImage_FLR = toQImage_u12(&FLRimageDisp, (uint16)_min, (uint16)_max);

			if (m_ui.tabWidget_Window->currentIndex() == 1) {

				m_pImage2->SetImage(pImage_FLR);
				m_ui.lcdNumber_FLR_px_x->display(m_pImage2->curpos.x());
				m_ui.lcdNumber_FLR_px_y->display(m_pImage2->curpos.y());
			}
			else if (m_ui.tabWidget_Window->currentIndex() == 2) {
				m_pImage5->SetImage(pImage_FLR);
				m_ui.lcdNumber_FLR_px_x->display(m_pImage5->curpos.x());
				m_ui.lcdNumber_FLR_px_y->display(m_pImage5->curpos.y());
			}
			//pImage_FLR.reset();/**/

			m_ui.lcdNumber_fluorescent_freqHz->display(1000.0 / (double)m_ThreadProcessTimeMills_FluorescentImgCycle.MVA);

			m_ui.lcdNumber_intensity_pattern1->display(zaxis_controller.imgMeanIntensity_DIFF[0]);
			m_ui.lcdNumber_intensity_pattern2->display(zaxis_controller.imgMeanIntensity_DIFF[1]);
			m_ui.lcdNumber_intensity_ratio->display(zaxis_controller.imgMeanIntensity_DIFF_ratio);
			m_ui.progressBar_intensity_pattern1->setValue(zaxis_controller.imgMeanIntensity_DIFF[0]);
			m_ui.progressBar_intensity_pattern2->setValue(zaxis_controller.imgMeanIntensity_DIFF[1]);
			if (zaxis_controller.imgMeanIntensity_DIFF_ratio >= 0)
				m_ui.progressBar_intensity_ratio->setInvertedAppearance(false);
			else
				m_ui.progressBar_intensity_ratio->setInvertedAppearance(true);
			double intensity_ratio_temp = fabs(zaxis_controller.imgMeanIntensity_DIFF_ratio);
			if (intensity_ratio_temp > 5.0) intensity_ratio_temp = 5.0;
			m_ui.progressBar_intensity_ratio->setValue(intensity_ratio_temp * 100);



			if (HamamatsuCamera_C11440.b_recording) m_ui.progressBar_recording_fluorescent->setValue((int)(100.0*(double)HamamatsuCamera_C11440.recFrms_left / (double)HamamatsuCamera_C11440.buffer_nframe));
			switch (toolboxidx) {
			case 0:
				m_ui.lcdNumber_fluorescent_cycle_mean->display(m_ThreadProcessTimeMills_FluorescentImgCycle.MVA);
				m_ui.lcdNumber_fluorescent_cycle_max->display(m_ThreadProcessTimeMills_FluorescentImgCycle._MAX);
				m_ui.lcdNumber_fluorescent_rec_mean->display(m_ThreadProcessTimeMills_FluorescentImgRecProc.MVA);
				m_ui.lcdNumber_fluorescent_rec_max->display(m_ThreadProcessTimeMills_FluorescentImgRecProc._MAX);
				m_ui.lcdNumber_fluorescent_rec_frames->display((int)(HamamatsuCamera_C11440.recFrms - HamamatsuCamera_C11440.recFrms_Start));
				m_ui.lcdNumber_fluorescent_rec_remainedframes->display((int)HamamatsuCamera_C11440.recFrms_left);
				//m_ui.lcdNumber_reserved2_mean->display(m_ThreadProcessTimeMills_trkPiezo.MVA);
				//m_ui.lcdNumber_reserved2_max->display(m_ThreadProcessTimeMills_trkPiezo._MAX);
				break;
			case 1:

				break;
			case 2:

				break;
			case 3:
				m_ui.lcdNumber_Stage_Pos_z->display((double)zaxis_controller.z[zaxis_controller.histlen - 1]);

				break;
			case 4:

				break;
			default:
				break;
			}
		}
		// replay display
		replayer_play_thread();


		fEndSeconds = GetCycleCountSeconds();
		m_ThreadProcessTimeMills_DisplayUpdate.update((fEndSeconds - fstarttime) * 1000.0);
	}
	catch (exception& e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "Main thread (display): %s\n", e.what());
		fclose(ofp);
	}
}

void UI::MainWindow::toggle_NIRp1Update(bool flag) {
	if (flag){
		NIR_px1.x = m_ui.lcdNumber_NIR_px_x->value();
		NIR_px1.y = m_ui.lcdNumber_NIR_px_y->value();
		NIR_p1.x = ILS200LM.CurrentStatus.CurrentPosition[0];
		NIR_p1.y = ILS200LM.CurrentStatus.CurrentPosition[1];
		if (m_ui.checkBox_nir_p2->isChecked()) {
			double d1 = (NIR_p1 - NIR_p2).norm();
			double d2 = (NIR_px1 - NIR_px2).norm();
			double _computed_pxdist = d1 / d2 * 1000;
			m_ui.doubleSpinBox_nir_pixeldistance_umppx_2->setValue(_computed_pxdist);
		}
	}
}

void UI::MainWindow::toggle_NIRp2Update(bool flag) {
	if (flag){
		NIR_px2.x = m_ui.lcdNumber_NIR_px_x->value();
		NIR_px2.y = m_ui.lcdNumber_NIR_px_y->value();
		NIR_p2.x = ILS200LM.CurrentStatus.CurrentPosition[0];
		NIR_p2.y = ILS200LM.CurrentStatus.CurrentPosition[1];
		if (m_ui.checkBox_nir_p1->isChecked()) {
			double d1 = (NIR_p1 - NIR_p2).norm();
			double d2 = (NIR_px1 - NIR_px2).norm();
			double _computed_pxdist = d1 / d2 * 1000;
			m_ui.doubleSpinBox_nir_pixeldistance_umppx_2->setValue(_computed_pxdist);
		}
	}
}

void UI::MainWindow::toggle_FLRp1Update(bool flag) {
	if (flag){
		FLR_px1.x = m_ui.lcdNumber_FLR_px_x->value();
		FLR_px1.y = m_ui.lcdNumber_FLR_px_y->value();
		FLR_p1.x = ILS200LM.CurrentStatus.CurrentPosition[0];
		FLR_p1.y = ILS200LM.CurrentStatus.CurrentPosition[1];
		if (m_ui.checkBox_flr_p2_2->isChecked()) {
			double d1 = (FLR_p1 - FLR_p2).norm();
			double d2 = (FLR_px1 - FLR_px2).norm();
			double _computed_pxdist = d1 / d2 * 1000;
			m_ui.doubleSpinBox_flr_pixeldistance_umppx_2->setValue(_computed_pxdist);
		}
	}
}

void UI::MainWindow::toggle_FLRp2Update(bool flag) {
	if (flag){
		FLR_px2.x = m_ui.lcdNumber_FLR_px_x->value();
		FLR_px2.y = m_ui.lcdNumber_FLR_px_y->value();
		FLR_p2.x = ILS200LM.CurrentStatus.CurrentPosition[0];
		FLR_p2.y = ILS200LM.CurrentStatus.CurrentPosition[1];
		if (m_ui.checkBox_flr_p1_2->isChecked()) {
			double d1 = (FLR_p1 - FLR_p2).norm();
			double d2 = (FLR_px1 - FLR_px2).norm();
			double _computed_pxdist = d1 / d2 * 1000;
			m_ui.doubleSpinBox_flr_pixeldistance_umppx_2->setValue(_computed_pxdist);
		}
	}
}

// ----------------!!! add functions before this line ------------------------------------
// ----------------!!! Qt component event functions only ---------------------------------

void UI::MainWindow::NIR_GPUResultDisplay_QcomboBoxCurrentIndexChanged(int index) {
	QComboBox * mComboBox = m_ui.comboBox_imgproc_intimgdispaly;
	if (!GlobalMap_NIR.subEnabled) {
		int imgType = 0;
		uInt16 * Img = (uInt16 *)malloc(GlobalMap_NIR.height*GlobalMap_NIR.width * sizeof(uInt16));
		float * Img32f = (float *)malloc(GlobalMap_NIR.height*GlobalMap_NIR.width * sizeof(float));

		cudaError_t error = cudaSuccess;
		switch(index) { // index 0: off, 1: on
		case 0: // Updated Image (original)
			error = cudaMemcpy(Img,
				GlobalMap_NIR.d_src_float,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(uInt16), cudaMemcpyDeviceToHost);
			break;
		case 1: // 'SubstractedImage'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				GlobalMap_NIR.d_sub,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float), cudaMemcpyDeviceToHost);
			break;
		case 2: // 'Gaussian'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPimg_Gaussian,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		case 3: // 'Linear filter x'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPLN_x,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		case 4: // 'Linear filter y'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPLN_y,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		case 5: // 'GRAD Linear filter x'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPGRD_x,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		case 6: // 'GRAD Linear filter y'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPGRD_y,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		case 7: // 'Divergency'
			imgType = 1;
			error = cudaMemcpy(Img32f,
				NIRFishPosDetector_GPU.p_NPPimDivF,
				(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float) , cudaMemcpyDeviceToHost);
			break;
		//case 8: // 'GlobalMap' // copy to uploading image
			//double temp_pos[2] = { Snapshot_Pos_mm.x, Snapshot_Pos_mm.y };
			//GlobalMap_NIR.cudacopyBG(GlobalMap_NIR.d, temp_pos);
			//error = cudaMemcpy(Img, Img32f,(size_t)(GlobalMap_NIR.height * GlobalMap_NIR.width)*sizeof(float), cudaMemcpyDeviceToHost);
			//break;
		default:
			break;
		} // end switch
		if (error == cudaSuccess) {
			if (imgType == 1) {
				IppiSize tempSize;
				tempSize.width = GlobalMap_NIR.width;
				tempSize.height = GlobalMap_NIR.height;

				time_t t = time(0); // get time now;
				struct tm * now = localtime(&t);
				char NAME[1024];
				sprintf(NAME, "d:\\TrackingMicroscopeData\\_Snapshot\\testimage%d_%04d%02d%02d_%02d%02d%02d_GMap.dat", index, now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
				std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
				pwriter_binary->write((char*)Img32f, tempSize.width * tempSize.height *  sizeof(float));
				pwriter_binary->close();
				delete(pwriter_binary);
			}
			else {
				IppiSize tempSize;
				tempSize.width = GlobalMap_NIR.width;
				tempSize.height = GlobalMap_NIR.height;
				//ippiConvert_32f8u_C1RSfs(Img32f, 4 * GlobalMap_NIR.windowSize.width, Img, GlobalMap_NIR.windowSize.width, tempSize, ippRndNear, 0);

				time_t t = time(0); // get time now;
				struct tm * now = localtime(&t);
				char NAME[1024];
				sprintf(NAME, "d:\\TrackingMicroscopeData\\_Snapshot\\testimage%d_%04d%02d%02d_%02d%02d%02d_GMap.dat", index, now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
				std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
				pwriter_binary->write((char*)Img, tempSize.width * tempSize.height *  sizeof(uint16));
				pwriter_binary->close();
				delete(pwriter_binary);

			}

			img_uint16 imageDisp8 = img_uint16(Img, imSize(GlobalMap_NIR.width, GlobalMap_NIR.height));
			int _min = m_ui.spinBox_NIRDisplay_min->value();
			int _max = m_ui.spinBox_NIRDisplay_max->value();
			auto pImage = toQImage_shift(&imageDisp8, (uInt16)_min, (UINT16)_max, 4, m_ui.checkBox_nir_globalmapdisplay_enable->isChecked());
			m_pImageR1->SetImage(pImage);
		}

		free(Img);
		free(Img32f);

	}
}

// Stage controller | Connect button
void UI::MainWindow::Stage_updatestatus(void) { // connect & disconnect stage

	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
}

// Stage controller | Connect button
void UI::MainWindow::Stage_MoveToPosition_QpushButtonClicked(void) { // move the stage to the certain position
}
// Stage controller | Connect button
void UI::MainWindow::Stage_SaveCurrentPosition_QpushButtonClicked(void) { // save the current position as a set position in the text file (Stage_Set_Position.txt)
}
// Stage controller | Connect button
void UI::MainWindow::Stage_MoveToSavedPosition_QpushButtonClicked(void) { // load the set position saved in the text file & move the state to the set position (Stage_Set_Position.txt)
}
// Stage controller | Updating-Parameter button
void UI::MainWindow::Stage_UpdataParameters_QpushButtonClicked(void) { // Update all stage parameters (turn off the Analog velocity control -> update the parameters -> turn on the analog velocity control)
	ILS200LM.MyXYGroupControllerStatusRead();
	updataXPSVariables(); // update variables

	if (ILS200LM.ControllerStatus == 48) {
		if(ILS200LM.EnableAnalogTracking()) {
			QtDisplayMessageInfo("Analog Tracking Disabled");
			return;
		}
	}
	if (ILS200LM.ControllerStatus >= 10 && ILS200LM.ControllerStatus <= 19) {
		if(ILS200LM.EnableAnalogTracking()) {
			QtDisplayMessageInfo("Analog Tracking Enabled");
			return;
		}
	}
	ILS200LM.MyXYGroupControllerStatusRead();
	XPSAVController.SetDefault(ILS200LM.Velocity2Volt_x(0), ILS200LM.Velocity2Volt_y(0));
	//QtDisplayMessageError(qstr("Stage Status: %1%") % ILS200LM.ControllerStatus));
}

void UI::MainWindow::Stage_ADCEnable(void) { // Update all stage parameters (turn off the Analog velocity control -> update the parameters -> turn on the analog velocity control)
	ILS200LM.MyXYGroupControllerStatusRead();
	updataXPSVariables(); // update variables

	if (ILS200LM.ControllerStatus == 48) {
		if (ILS200LM.EnableAnalogTracking()) {
			//QtDisplayMessageInfo("Analog Tracking Disabled");
			return;
		}
	}
	if (ILS200LM.ControllerStatus >= 10 && ILS200LM.ControllerStatus <= 19) {
		if (ILS200LM.EnableAnalogTracking()) {
			//QtDisplayMessageInfo("Analog Tracking Enabled");
			return;
		}
	}
	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
	//QtDisplayMessageError(qstr("Stage Status: %1%") % ILS200LM.ControllerStatus));
}

// Stage controller | Enable/Disable to edit the advanced parameters
void UI::MainWindow::Stage_EditAdvancedParameters_QpushButtonClicked(void) { // Enabling to edit the advanced parameters
	if (m_ui.doubleSpinBox_Stage_AVT_Scale_x->isEnabled()) {
		m_ui.Button_Stage_AVT_EnableEditParameters->setText("Edit"); // change the text on the button
		m_ui.doubleSpinBox_Stage_AVT_Scale_x->setDisabled(true); // make a variable editable
		m_ui.doubleSpinBox_Stage_AVT_Scale_y->setDisabled(true); // make a variable editable
		m_ui.doubleSpinBox_Stage_AVT_Offset_x->setDisabled(true); // make a variable editable
		m_ui.doubleSpinBox_Stage_AVT_Offset_y->setDisabled(true);
		m_ui.spinBox_Stage_AVT_Order_x->setDisabled(true);
		m_ui.spinBox_Stage_AVT_Order_y->setDisabled(true);
		m_ui.spinBox_Stage_AVT_VelMax->setDisabled(true);
		m_ui.spinBox_Stage_AVT_AccMax->setDisabled(true);
		m_ui.spinBox_Stage_AVT_InputVoltMax->setDisabled(true);
	}
	else {
		m_ui.Button_Stage_AVT_EnableEditParameters->setText("Lock");
		m_ui.doubleSpinBox_Stage_AVT_Scale_x->setEnabled(true);
		m_ui.doubleSpinBox_Stage_AVT_Scale_y->setEnabled(true);
		m_ui.doubleSpinBox_Stage_AVT_Offset_x->setEnabled(true);
		m_ui.doubleSpinBox_Stage_AVT_Offset_y->setEnabled(true);
		m_ui.spinBox_Stage_AVT_Order_x->setEnabled(true);
		m_ui.spinBox_Stage_AVT_Order_y->setEnabled(true);
		m_ui.spinBox_Stage_AVT_VelMax->setEnabled(true);
		m_ui.spinBox_Stage_AVT_AccMax->setEnabled(true);
		m_ui.spinBox_Stage_AVT_InputVoltMax->setEnabled(true);
	}
}

bool UI::MainWindow::Stage_ConnectOnly_QpushButtonClicked(void){
	if (ILS200LM.SocketID == -1) {	// Initialize stage
		ILS200LM.TCPConnection();
		ILS200LM.MyXYGroupDACSet();
		if (-1 > ILS200LM.error) {
			ILS200LM.SocketID = -1;
			return false;
		};
		Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
		return true; // Connect
	}
	return false;
}
void UI::MainWindow::Stage_ReleaseStageOnly_QpushButtonClicked(void){
	if (ILS200LM.SocketID != -1) {	// Initialize stage
		ILS200LM.DisableAnalogTracking();
		if (0 != ILS200LM.error) return; // Kill Group
	}
	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
}
bool UI::MainWindow::Stage_InitializationOnly_QpushButtonClicked(void){
	if (ILS200LM.SocketID != -1) {	// Initialize stage
		ILS200LM.Initialization();
		if (0 != ILS200LM.error)
			return false; // Kill Group
		Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
		return true;
	}
	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
	return false;
}
void UI::MainWindow::Stage_ReadyOnly_QpushButtonClicked(void){
	if (ILS200LM.SocketID != -1) {	// Initialize stage
		ILS200LM.ReferencingReady();
		if (0 != ILS200LM.error) return; // Kill Group
	}
	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
}

void UI::MainWindow::Stage_ReadyHoming(void)  {
	m_pThreadGeneral = std::thread([this]{Stage_ReadyOnly_QpushButtonClicked();}); // call function in other thread
	m_pThreadGeneral.detach(); // detach the thread for next recording
	return;
}

void UI::MainWindow::Stage_EnableAnalogControlOnly_QpushButtonClicked(void){
	if (ILS200LM.SocketID != -1) {	// Initialize stage
		//ILS200LM.MyXYGroupDACSet();
		ILS200LM.EnableAnalogTracking();
		if (0 != ILS200LM.error) return; // Kill Group
	}
	Stage_UpdateStatus(ILS200LM.MyXYGroupControllerStatusRead());
}
void UI::MainWindow::Stage_UpdateStatus(int StateNumber){
	int Status = 0;
	if (0 <= StateNumber && StateNumber < 10 )
		Status = STAGE_NotInitiated;
	else if (StateNumber == 42)
		Status = STAGE_NotReferenced;
	else if (StateNumber == 46)
		Status = STAGE_Referencing;
	else if (10 <= StateNumber && StateNumber < 20 )
		Status = STAGE_Ready;
	else if (StateNumber == 48)
		Status = STAGE_AnalogTracking;
	else if (StateNumber == 43)
		Status = STAGE_Homming;

	switch(Status) {
	case STAGE_NotInitiated: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Not Initiated");
		break;
	case STAGE_NotReferenced: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Not Referenced");
		break;
	case STAGE_Referencing: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Referencing");
		break;
	case STAGE_Ready: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Ready");
		break;
	case STAGE_AnalogTracking: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Analog Tracking Enabled");
		break;
	case STAGE_Homming: //
		m_ui.label_Stage_StatusDisplay->setText("Status: Homming");
		break;
	default:
		m_ui.label_Stage_StatusDisplay->setText("Status: Not informed");
		break;
	}; // end switch
}



void UI::MainWindow::pushButton_Stage_CalibrationImpulseStep_QpushButtonClicked(void)  {
	m_pThreadGeneral = std::thread([this]{CalibrateImpDataFromStepResp();}); // call function in other thread
	m_pThreadGeneral.detach(); // detach the thread for next recording
	return;
}

void UI::MainWindow::CalibrateImpDataFromStepResp(void)  {
}
void UI::MainWindow::Stage_CalibrationSimple_Random_QpushButtonClicked(void)  {
	m_pThreadGeneral = std::thread([this]{calibrateNIAOOffset();}); // call function in other thread
	m_pThreadGeneral.detach(); // detach the thread for next recording
	return;
}

void UI::MainWindow::calibrateNIAOOffset(void)  {

	// 0. Check the stage is in Analog velocity tracking mode
	ILS200LM.MyXYGroupControllerStatusRead(); // check status of stage
	if (!(ILS200LM.ControllerStatus == 48)) // if it is not ready status
		return;

	double offset_calibration[2] = {0};
	RecBuffer.VelocityCalibration(offset_calibration);


	QtDisplayMessageInfo("Offset Calibrated");
	return;
}

void UI::MainWindow::ParameterUpdate_FishPosEstimator(void) {
	int H_pred = m_ui.spinBox_MPC_predhrz->value();
	int halfcycle_fishswim = m_ui.spinBox_fishpred_swimhalfcycle_ms->value();
	int dt_ms = (int)((double)m_ui.spinBox_DO_ExtTrigger_NIR->value() / 10.0);
	double weightupperlimit = PI / (double)m_ui.spinBox_fishpred_weightupperthreshold_den->value();
	double weightlowerlimit = PI / (double)m_ui.spinBox_fishpred_weightlowerthreshold_den->value();
	double teminatevelocity = (double)m_ui.doubleSpinBox_fishpred_terminatevelocity_mmps->value();
	double slopegain = (double)m_ui.doubleSpinBox_fishpred_slopegain->value();
	FishPosEstimator.reset(halfcycle_fishswim, dt_ms, H_pred);
	FishPosEstimator.resetParameter(weightlowerlimit, weightupperlimit, teminatevelocity, slopegain);

}


// NIR controller | Button_NIR_CalibratePixelDistance_QpushButtonClicked
void UI::MainWindow::NIR_PixelDistanceCalibration(void) {

	if (!g_bProcessRunning || RecBuffer.FrmNo > 0){
		TrackingMessage * DispMsg = RecBuffer.getDispFrameRec();
		if (DispMsg) {
			if (DispMsg->srcNIR) {
				char NAME[256];
				time_t t;
				struct tm * now;
				t = time(0); // get time now
				now = localtime(&t);
				sprintf(NAME, "D:\\TrackingMicroscopeData\\_Snapshot\\Snapshot_x%3.5f_y%3.5f_%04d%02d%02d_%02d%02d%02d.dat",
					DispMsg->stagePos.CurrentPosition[0], DispMsg->stagePos.CurrentPosition[1],
					now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
				std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
				int32 ImageSizeByte = DispMsg->srcNIR->imgSize.height * DispMsg->srcNIR->imgSize.width;
				pwriter_binary->write((char *)(DispMsg->srcNIR->data), ImageSizeByte);
				pwriter_binary->close();
				delete(pwriter_binary);
			}
		}
	}
}

void UI::MainWindow::GlobalMap_SaveTIFF(void) {
	m_pThreadGeneral = std::thread([this]{GlobalMap_NIR.cudasaveBG();}); // call function in other thread
	m_pThreadGeneral.detach(); // detach the thread for next recording
}
void UI::MainWindow::GlobalMap_updateFishMask(void) {
	double rxpos = (double)m_ui.doubleSpinBox_globalmap_fishmask_rxpos->value();
	double rxneg= (double)m_ui.doubleSpinBox_globalmap_fishmask_rxneg->value();
	double ry= (double)m_ui.doubleSpinBox_globalmap_fishmask_ry->value();
	GlobalMap_NIR.updateFishMask(rxpos, rxneg, ry);
}

void UI::MainWindow::GlobalMap_displayreset(void) {

	uInt16 * img_temp = (uint16 *)malloc(GlobalMap_NIR.byte_per_image);
	//cudaError_t error = cudaMemcpy(img_temp, GlobalMap_NIR.d_src, GlobalMap_NIR.byte_per_image, cudaMemcpyDeviceToHost); //GlobalMap_NIR.byte_per_image
	//
	float * img_temp_float = (float *)malloc(GlobalMap_NIR.byte_per_image * 2);
	int index = m_ui.comboBox_imgproc_intimgdispaly->currentIndex();

	switch (index) { // index 0: off, 1: on
	case 0:
		cudaMemcpy(img_temp_float, GlobalMap_NIR.d_src_float, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	case 1:
		cudaMemcpy(img_temp_float, GlobalMap_NIR.d_save1, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	case 2:
		cudaMemcpy(img_temp_float, GlobalMap_NIR.d_sub, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	case 3:
		cudaMemcpy(img_temp_float, NIRFishPosDetector_GPU.p_NPPimg_Gaussian, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	case 4:
		cudaMemcpy(img_temp_float, GlobalMap_NIR.d_save2, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	case 5:
		cudaMemcpy(img_temp_float, GlobalMap_NIR.d_save3, GlobalMap_NIR.byte_per_image * 2, cudaMemcpyDeviceToHost);
		break;
	default:
		break;
	}


	for (int i = 0; i < GlobalMap_NIR.width*GlobalMap_NIR.height; i++)
		img_temp[i] = (uint16)img_temp_float[i];

	img_uint16 Img_u16 = img_uint16(img_temp, imSize(GlobalMap_NIR.height, GlobalMap_NIR.width));

	int _min = m_ui.spinBox_NIRDisplay_min->value();
	int _max = m_ui.spinBox_NIRDisplay_max->value();
	//auto pImage = toQImage_shift(&Img_u16, (uInt16)_min, (UINT16)_max, 4, m_ui.checkBox_nir_globalmapdisplay_enable->isChecked());
	auto pImage = toQImage(&Img_u16, (uInt16)_min, (UINT16)_max);
	m_pImageR1->SetImage(pImage);
}

void UI::MainWindow::GlobalMap_ParameterLoading(void) {
	// Import Data (h5 File)
	try {
		int nx=0; int ny = 0;
		double * _Angle = NULL;
		double * _PXDIST = NULL;
		double * _NIR_ROI = NULL;
		double * _NIR_center = NULL;
		double * _LED_offset = NULL;
		double * _POS_center = NULL;
		double * _NIR_exp = NULL;
		H5File* _file = new H5File ("D:\\TrackingMicroscopeData\\_setting\\Default_PxDist.h5", H5F_ACC_RDONLY);
		_Angle = importImpRestInd(_file, "NIRCameraAlignAngle", &nx);
		_PXDIST = importImpRestInd(_file, "NIRCameraPxDist", &ny);
		_NIR_ROI = importImpRestInd(_file, "NIR_roi", &ny);
		_NIR_center = importImpRestInd(_file, "NIR_center", &ny);
		_LED_offset = importImpRestInd(_file, "LED_offset", &ny);
		_POS_center = importImpRestInd(_file, "target_majoraxis_offset", &ny);
		_NIR_exp = importImpRestInd(_file, "NIR_exp", &ny);

		m_ui.doubleSpinBox_nir_pixeldistance_umppx->setValue(_PXDIST[0]);
		m_ui.doubleSpinBox_nir_adjustmentangle_rad->setValue(_Angle[0]);

		if (m_ui.checkBox_autoLoading->isChecked()) {
			m_ui.spinBox_nircam_roix->setValue(_NIR_ROI[0]);
			m_ui.spinBox_nircam_roiy->setValue(_NIR_ROI[1]);
			m_ui.spinBox_tracking_targetcenterx_px->setValue(_NIR_center[0]);
			m_ui.spinBox_tracking_targetcentery_px->setValue(_NIR_center[1]);
			m_ui.doubleSpinBox_intensity_ratio_set->setValue(_LED_offset[0]);
			m_ui.spinBox_tracking_posoffset_major->setValue(_POS_center[0]);
			m_ui.spinBox_nircam_exptime_us->setValue(_NIR_exp[0]);
		}

		GlobalMap_NIR.updateParameters(_PXDIST[0] / 1000, _Angle[0], NIRCameraPG.ROIWidth, NIRCameraPG.ROIHeight, 16);
		//GlobalMap_NIR.updateParameters(_PXDIST[0] / 1000, _Angle[0], NIRCameraPG.ROIWidth, NIRCameraPG.ROIHeight, 12);

		delete _file;
		if (_Angle != NULL) free(_Angle);
		if (_PXDIST != NULL) free(_PXDIST);
		if (_NIR_ROI != NULL) free(_NIR_ROI);
		if (_NIR_center != NULL) free(_NIR_center);
		if (_LED_offset != NULL) free(_LED_offset);
		if (_POS_center != NULL) free(_POS_center);
		if (_NIR_exp != NULL) free(_NIR_exp);
		m_ui.checkBox_autoLoading->setChecked(false);
	}
	catch(DataSetIException error) {
		error.printError();
	}
}
void UI::MainWindow::GlobalMap_Initialization(void) {
	GlobalMap_ParameterLoading();
	GlobalMap_NIR.gpuFree();
	GlobalMap_NIR.gpuMalloc();
	GlobalMap_NIR.gpuInit();
}
void UI::MainWindow::ImageProc_ParameterLoading(void) {
	NIRFishPosDetector_GPU.updateParameter((int)m_ui.spinBox_imgproc_kernelradius->value(),
		(int)m_ui.spinBox_imgproc_sigma->value(),
		(int)m_ui.spinBox_imgproc_searcharea1->value(),
		NIRCameraPG.ROIHeight, NIRCameraPG.ROIWidth);
}


void UI::MainWindow::GlobalMap_UpdateStillImageUpdate(void) {
	if (!Snapshot->data) {
		QtDisplayMessageError("Updating still Image Failed");
		return;
	}
	// ---- USE this function as a temperal snapshot
	QtDisplayMessageInfo("Still Image updated on GPU");
}


void UI::MainWindow::Tracking_TargetSelection_QcomboBoxCurrentIndexChanged(int index) {
	QComboBox * mComboBox = m_ui.comboBox_tracking_fishpostarget;
	switch(index) {
		case 0: // off
			m_trk_targetSelection = Brain;
			break;
		case 1: // Full scan tracking
			m_trk_targetSelection = Yolk;
			break;
		case 2: // 3 layer scan
			m_trk_targetSelection = LeftEye;
			break;
		case 3: // 3 layer scan
			m_trk_targetSelection = RightEye;
			break;
		default:
			break;
		}; // end switch
}

void  UI::MainWindow::updataXPSVariables(void) {
	// import values
	ILS200LM.AVT_deadbandPosition = m_ui.doubleSpinBox_Stage_AVT_deadbandMM->value(); // mm (for setting deadbandthreshold)
	ILS200LM.AVT_Deadbandthreshold = m_ui.doubleSpinBox_Stage_AVT_deadbandMM->value(); // mm (for setting deadbandthreshold)
	ILS200LM.AVT_scale_x  = m_ui.doubleSpinBox_Stage_AVT_Scale_x->value();// scaling
	ILS200LM.AVT_offset_x  = m_ui.doubleSpinBox_Stage_AVT_Offset_x->value();; // offset voltage
	ILS200LM.AVT_scale_y  = m_ui.doubleSpinBox_Stage_AVT_Scale_y->value(); // scaling
	ILS200LM.AVT_offset_y  = m_ui.doubleSpinBox_Stage_AVT_Offset_y->value();; // offset voltage
	ILS200LM.AVT_order = m_ui.spinBox_Stage_AVT_Order_x->value();; // Order 1st or 2nd
	ILS200LM.AVT_Velocity_max = m_ui.spinBox_Stage_AVT_VelMax->value();// mm/s
	ILS200LM.AVT_acceleration = m_ui.spinBox_Stage_AVT_AccMax->value(); // mm/s^2
	ILS200LM.AVT_InputVoltage_max = m_ui.spinBox_Stage_AVT_InputVoltMax->value();// Volts
	ILS200LM.AnalogTrackingMode = m_ui.toolBox_AnalogTrackingControlMode->currentIndex();
	ILS200LM.APT_scale_x  = m_ui.doubleSpinBox_Stage_AVT_Scale_x->value();// scaling
	ILS200LM.APT_offset_x  = m_ui.doubleSpinBox_Stage_AVT_Offset_x->value();; // offset voltage
	ILS200LM.APT_scale_y  = m_ui.doubleSpinBox_Stage_AVT_Scale_y->value(); // scaling
	ILS200LM.APT_offset_y  = m_ui.doubleSpinBox_Stage_AVT_Offset_y->value();; // offset voltage
	ILS200LM.APT_Velocity_max = m_ui.spinBox_Stage_AVT_VelMax->value();// mm/s
	ILS200LM.APT_acceleration = m_ui.spinBox_Stage_AVT_AccMax->value(); // mm/s^2



	//ILS200LM.MyXyAnalogTrakcingInternalValueUpdate();
	// display results
	m_ui.doubleSpinBox_Stage_AVT_deadbandThresholdVolts->setValue(ILS200LM.DeadBandThresholdVolts());
	m_ui.doubleSpinBox_Stage_AVT_VoltPerVelMMPS->setValue(ILS200LM.VoltsPerMMPS());
}


void UI::MainWindow::pushButton_SaveParameters_clicked(void) {
}

void UI::MainWindow::UpdateTrackingParameters(void) {
	GlobalMap_NIR.max_intensity_set = (float)m_ui.doubleSpinBox_imgproc_centroid_maxIntensitySet->value();

	NIRFishPosDetector_GPU.fishRefSize = (double)m_ui.spinBox_zebrafish_refsize_px->value();
	NIRFishPosDetector_GPU.fishRefSize_variation = m_ui.doubleSpinBox_zebrafish_refsize_variation->value();
	NIRFishPosDetector_GPU.fitnessRef_heading = m_ui.doubleSpinBox_zebrafish_headingfitnessCOSphi_Ref->value();

	NIRFishPosDetector_GPU.centroid_BW_threshold = (float)m_ui.doubleSpinBox_imgproc_centroid_threshold->value();
	NIRFishPosDetector_GPU.centroid_srch_radius = (float)m_ui.doubleSpinBox_imgproc_centroid_srch->value();
	NIRFishPosDetector_GPU.centroid_distC2Y = (double)m_ui.doubleSpinBox_imgproc_centroid_distC2Y->value();
	NIRFishPosDetector_GPU.centroid_distC2B = (double)m_ui.doubleSpinBox_imgproc_centroid_distC2B->value();
	NIRFishPosDetector_GPU.centroid_distB2E = (double)m_ui.doubleSpinBox_imgproc_centroid_distB2E->value();


	m_ImageTargetPosPx = Point2d(m_ui.spinBox_tracking_targetcenterx_px->value(), m_ui.spinBox_tracking_targetcentery_px->value()); // get a trakcing center position from UI
	m_trk_TargetPositionOffset_majorAxis = double(m_ui.spinBox_tracking_posoffset_major->value()) / 1000; // mm
	m_trk_TargetPositionOffset_minorAxis = double(m_ui.spinBox_tracking_posoffset_minor->value()) / 1000; // mm
	interbout_std_threhold = double(m_ui.doubleSpinBox_fish_interboutInterval_velstd_threshold->value());



}



void UI::MainWindow::UpdateStageParameters(void) {
	ILS200LM.AVT_deadbandPosition = m_ui.doubleSpinBox_Stage_AVT_deadbandMM->value(); // mm (for setting deadbandthreshold)
	ILS200LM.AVT_scale_x  = m_ui.doubleSpinBox_Stage_AVT_Scale_x->value();// scaling
	ILS200LM.AVT_scale_y  = m_ui.doubleSpinBox_Stage_AVT_Scale_y->value(); // scaling
	ILS200LM.AVT_offset_x  = m_ui.doubleSpinBox_Stage_AVT_Offset_x->value();; // offset voltage
	ILS200LM.AVT_offset_y  = m_ui.doubleSpinBox_Stage_AVT_Offset_y->value();; // offset voltage
	ILS200LM.AVT_order = m_ui.spinBox_Stage_AVT_Order_x->value(); // Order 1st or 2nd
	ILS200LM.AVT_Velocity_max = m_ui.spinBox_Stage_AVT_VelMax->value();// mm/s
	ILS200LM.AVT_acceleration = m_ui.spinBox_Stage_AVT_AccMax->value(); // mm/s^2
	ILS200LM.AVT_InputVoltage_max = m_ui.spinBox_Stage_AVT_InputVoltMax->value();// Volts
}

void UI::MainWindow::ReadQtVariables(void) {
}

// END: ----------------!!! Qt component event functions only ---------------------------------

void QtComboBoxItemDisable(QComboBox * mComboBox, int index_i) {
	QModelIndex index = mComboBox->model()->index(index_i,0); // Get the index of the value to disable
	QVariant v(0); // This is the effective 'disable' flag
	mComboBox->model()->setData( index, v, Qt::UserRole -1);
}

void QtComboBoxItemEnable(QComboBox * mComboBox, int index_i) {
	QModelIndex index = mComboBox->model()->index(index_i,0); // Get the index of the value to disable
	QVariant v(Qt::ItemIsSelectable | Qt::ItemIsEnabled); // This is the effective 'enable' flag
	mComboBox->model()->setData( index, v, Qt::UserRole -1);
}

//void UI::MainWindow::DisplayImage(cv::Mat MatSource, QGLZoomableImageViewer* window) {
//	auto pPixmap = toPixmap(MatSource);
//	auto pImage = toQImage( *pPixmap );
//	window->SetImage( pImage );
//}

void UI::MainWindow::DisplayImage(img_uint8 MatSource, QGLZoomableImageViewer* window) {
	auto pImage = toQImage(&MatSource);
	window->SetImage(pImage);
}

void UI::MainWindow::DisplayImage(img_uint16 MatSource, QGLZoomableImageViewer* window) {
	auto pImage = toQImage(&MatSource);
	window->SetImage(pImage);
}

void UI::MainWindow::DisplayImage_ROI(img_uint16 MatSource, QGLZoomableImageViewer* window, uint16 _min, uint16 _max) {
	auto pImage = toQImage(&MatSource, _min, _max);
	window->SetImage(pImage);
}

void UI::MainWindow::Piezo_ManualMoveDown(void) {
	float64 _step_um = (float64)m_ui.doubleSpinBox_Piezo_trk_dz->value();
	float64 _center_um = (float64)m_ui.spinBox_Piezo_trk_initz->value();
	ControllHardWare_2P.Piezo_scanSingal.center_um = _center_um - _step_um;
	m_ui.spinBox_Piezo_trk_initz->setValue((int)(_center_um - _step_um));
}
void UI::MainWindow::Piezo_ManualMoveUp(void) {
	float64 _step_um = (float64)m_ui.doubleSpinBox_Piezo_trk_dz->value();
	float64 _center_um = (float64)m_ui.spinBox_Piezo_trk_initz->value();
	ControllHardWare_2P.Piezo_scanSingal.center_um = _center_um + _step_um;
	m_ui.spinBox_Piezo_trk_initz->setValue((int)(_center_um + _step_um));
}
void UI::MainWindow::Piezo_ManualStop(void) {
	float64 _step_um = (float64)m_ui.doubleSpinBox_Piezo_trk_dz->value();
	float64 _center_um = (float64)m_ui.spinBox_Piezo_trk_initz->value();
	ControllHardWare_2P.Piezo_scanSingal.center_um = _center_um;
	ControllHardWare_2P.Piezo_scanSingal.step_um = _step_um;
	ControllHardWare_2P.Piezo_scanSingal.getSignal_stop();
	if (m_ui.checkBox_piezoConcatenate_enable->isChecked())
		setPiezoConcatenate(true);
}


vector<Point2d> UI::MainWindow::GPUImageProcess(uint16 * src, double * position, int * fishSize, bool FishDetectionSuccess_PrevFrms) {

	GlobalMap_NIR.cudaBGsub(src, position);
	if (GlobalMap_NIR.subEnabled)
		NIRFishPosDetector_GPU.detect_CUDA(GlobalMap_NIR.d_sub);
	else
		NIRFishPosDetector_GPU.detect_CUDA(GlobalMap_NIR.d_src_float);
	vector<Point2d> fishPos;
	fishPos = NIRFishPosDetector_GPU.findFishPositionFromDivImg_CUDA(fishSize, FishDetectionSuccess_PrevFrms);

	return fishPos;
}

vector<Point2d> UI::MainWindow::GPUImageProcess_centroid(uint16 * src, double * position, int * fishSize, bool FishDetectionSuccess_PrevFrms) {

	GlobalMap_NIR.cudaBGsub(src, position);
	if (GlobalMap_NIR.subEnabled)
		NIRFishPosDetector_GPU.detect_CUDA_gaussianOnly(GlobalMap_NIR.d_sub);//d_sub
	else
		NIRFishPosDetector_GPU.detect_CUDA_gaussianOnly(GlobalMap_NIR.d_src_float);

	float p_imgMoment[8] = { 0 };
	gpuErrchk(cudaMemcpy(p_imgMoment, NIRFishPosDetector_GPU.p_pt_xy_f, sizeof(float) * 8, cudaMemcpyDeviceToHost));
	vector<Point2d> fishPos;
	Point2d fish_centroid, yolk, eyeR, eyeL, brain;
	fish_centroid = Point2d(p_imgMoment[0] / p_imgMoment[2], p_imgMoment[1] / p_imgMoment[2]);
	double mu_x, mu_y, mu_xy;
	mu_x = p_imgMoment[3] / p_imgMoment[2] - fish_centroid.x*fish_centroid.x;
	mu_y = p_imgMoment[4] / p_imgMoment[2] - fish_centroid.y*fish_centroid.y;
	mu_xy = p_imgMoment[5] / p_imgMoment[2] - fish_centroid.x*fish_centroid.y;
	fishSize[0] = (int)(p_imgMoment[2]/4);
	double th = (atan2(2 * mu_xy, mu_x - mu_y) / 2);
	Point2d e1 = Point2d(cos(th), sin(th));
	Point2d e2 = Point2d(-sin(th), cos(th));
	Point2d fish_max;
	fish_max.x = (double)((int)(p_imgMoment[6]) % NIRFishPosDetector_GPU.imgSize.height);
	fish_max.y = (p_imgMoment[6] - fish_max.x) / NIRFishPosDetector_GPU.imgSize.height;
	yolk = e1*(-NIRFishPosDetector_GPU.centroid_distC2Y) + fish_centroid;
	if (yolk.norm(fish_max) > NIRFishPosDetector_GPU.centroid_distC2Y) {
		e1 = e1*(-1);
		e2 = e2*(-1);
		th = (th - PI);
		if (th < PI) th += 2 * PI;
		yolk = e1*(-NIRFishPosDetector_GPU.centroid_distC2Y) + fish_centroid;
	}
	brain = e1*(NIRFishPosDetector_GPU.centroid_distC2B) + fish_centroid;
	eyeR = e2*(NIRFishPosDetector_GPU.centroid_distB2E) + brain;
	eyeL = e2*(-NIRFishPosDetector_GPU.centroid_distB2E) + brain;
	double _offset = 0.0;// NIRFishPosDetector_GPU.Kernelradius;
	fishPos.push_back(yolk + _offset);
	fishPos.push_back(eyeL + _offset);
	fishPos.push_back(eyeR + _offset);
	fishPos.push_back(brain + _offset);
	fishPos.push_back(e1);
	fishPos.push_back(fish_centroid + _offset);

	return fishPos;
}



void UI::MainWindow::pushButton_test1_Clicked(void) {
	XPSGatheringInfo SyncPos =ILS200LM.CurrentStatus; // the latest position data (n + 1)
	Tracking_SimulationTarget = Point2d(SyncPos.CurrentPosition[0], SyncPos.CurrentPosition[1]) + Point2d(m_StageSimulation_StepError,m_StageSimulation_StepError);

	//Tracking_SimulationVelSet.clear();
	//
	//double VSet_X[] = {-50.0000000000,-50.0000000000,-50.0000000000,13.5065493034,9.9206838733,0.3628735620,-0.1318020851,-4.1044595548,4.5834759125,-4.3777107142,3.6456813413,-3.4073654072,1.5738173544,-0.3933987087,-0.0635851975,0.7288830748,-1.2193149620,1.3176598489,-1.2243127285,1.0148343329,-0.7201827014,0.3986550543,-0.1184335593,-0.1062594477,0.2691555177,-0.3549784241,0.3702656652,-0.3347837136,0.2655602163,-0.1790321525,0.0910443367,-0.0134692661,-0.0465521596,0.0855883983,-0.1036682498,0.1037857296,-0.0905041129,0.0688538815,-0.0437106634,0.0192155039,0.0016256627,-0.0170369876,0.0263827532,-0.0299737132,0.0287903407,-0.0241817561,0.0176005201,-0.0103919976,0.0036459816,0.0018811940,-0.0057803133,0.0079550693,-0.0085563246,0.0079005992,-0.0063875775,0.0044276787,-0.0023863127,0.0005478560,0.0009008175,-0.0018700497,0.0023546364,-0.0024133458,0.0021452893,-0.0016669981,0.0010929917,-0.0005214737,0.0000257200,0.0003491490,-0.0005849485,0.0006860323,-0.0006730826,0.0005763449,-0.0004293207,0.0002636202,-0.0001053485,-0.0000268904,0.0001225200,-0.0001783473,0.0001971261,-0.0001857106,0.0001531420,-0.0001089315,0.0000617125,-0.0000183445,-0.0000165297,0.0000405288,-0.0000532745,0.0000559381,-0.0000507013,0.0000402207,-0.0000271652,0.0000138682,-0.0000021096,-0.0000069758,0.0000128828,-0.0000156440,0.0000156906,-0.0000136966,0.0000104312,-0.0000066351};
	//double VSet_Y[] = {-50.0000000000,-50.0000000000,-50.0000000000,17.9994321223,7.6017360111,2.8617145117,-0.9851600101,-1.0990153716,0.2765918929,-0.9325392365,1.0879798998,-1.1245253099,1.4240166765,-1.2232643758,0.9291969943,-0.7667791805,0.5876559911,-0.4244390381,0.2992751970,-0.1913399486,0.1040335372,-0.0368898491,-0.0119109029,0.0423179519,-0.0589209716,0.0659963610,-0.0660013224,0.0613892480,-0.0540984425,0.0454366107,-0.0364359053,0.0278406253,-0.0201313440,0.0135744601,-0.0082510606,0.0041226403,-0.0010812317,-0.0010226050,0.0023537907,-0.0030754176,0.0033392467,-0.0032790850,0.0030072419,-0.0026136369,0.0021667475,-0.0017159166,0.0012942847,-0.0009218663,0.0006085345,-0.0003567115,0.0001636731,-0.0000234517,-0.0000716630,0.0001299970,-0.0001596975,0.0001682165,-0.0001620161,0.0001464425,-0.0001257211,0.0001030319,-0.0000806332,0.0000600099,-0.0000420264,0.0000270723,-0.0000151933,0.0000062029,0.0000002275,-0.0000044974,0.0000070258,-0.0000082149,0.0000084273,-0.0000079738,0.0000071084,-0.0000060297,0.0000048849,-0.0000037769,0.0000027717,-0.0000019060,0.0000011944,-0.0000006357,0.0000002185,0.0000000751,-0.0000002654,0.0000003735,-0.0000004192,0.0000004201,-0.0000003910,0.0000003440,-0.0000002883,0.0000002309,-0.0000001763,0.0000001275,-0.0000000859,0.0000000522,-0.0000000260,0.0000000067,0.0000000066,-0.0000000151,0.0000000196,-0.0000000212};
	//for (int i = 0; i<100; i++)
	//	Tracking_SimulationVelSet.push_back(Point2d(VSet_X[i], VSet_Y[i]));
};

void UI::MainWindow::pushButton_test2_Clicked(void) {
	XPSGatheringInfo SyncPos =ILS200LM.CurrentStatus; // the latest position data (n + 1)
	Tracking_SimulationTarget = Point2d(SyncPos.CurrentPosition[0], SyncPos.CurrentPosition[1]) + Point2d(-m_StageSimulation_StepError,-m_StageSimulation_StepError);
};


Point2d UI::MainWindow::SelectTrackingTarget(int option, zebrafishInfo * DetectedZebrafish) {
	// target pose select
	Point2d targetPose(0,0);
	switch(option) {
	case Brain: //
		targetPose = DetectedZebrafish->centerEyes;
		break;
	case Yolk: //
		targetPose = DetectedZebrafish->m_yolk.center;
		break;
	case LeftEye: //
		targetPose = DetectedZebrafish->m_eyeLeft.center;
		break;
	case RightEye: //
		targetPose = DetectedZebrafish->m_eyeRight.center;
		break;
	default:
		targetPose = DetectedZebrafish->centerEyes;
		break;
	};
	return targetPose;
};


bool UI::MainWindow::StartCamera_GreyPoint(void) {
	//NIRCameraPG.setSerialNumber(NIRCameraPG.getSerialNumber());
	NIRCameraPG.setExpTime((float)m_ui.spinBox_nircam_exptime_us->value()*0.001);
	NIRCameraPG.setROI((unsigned int)m_ui.spinBox_nircam_roix->value(), (unsigned int)m_ui.spinBox_nircam_roiy->value(), NIRCameraPG.ROIWidth, NIRCameraPG.ROIHeight);
	if (!NIRCameraPG.connectCamera())
		return false;
	if (!NIRCameraPG.setConfigurations())
		return false;
	if (!NIRCameraPG.startCapture())
		return false;
	return true;
}


void UI::MainWindow::generateNextSignal() {
	startIdx += DigSignal_chunksize;// replace startIdx + 1000000;
	NIDigSignal.resetSignal(DigSignal_chunksize, 0);
	NIDigSignal.mergeSignal(&XPS_StageSignal.signal, offsetIdx, startIdx, 0);
	NIDigSignal.mergeSignal(&HamamatsuCamera_C11440.signal, offsetIdx, startIdx, 1<<3);

	//NIAnalogSignal.resetSignal(DigSignal_chunksize, 0.0);
	//NIAnalogSignal.generateAnalogSignal(&Piezo_scanSingal.signal, offsetIdx, startIdx, 0);
}


void UI::MainWindow::NIAISingal_saveThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Read NI AI save");

	std::ofstream *pwriter_binary = NULL;
	while (g_bProcessRunning) {
		int32 _readbuf = ControllHardWare_2P.ReadAI0();
		if (_readbuf > 0) {
			if (!pwriter_binary) {
				char NAME[256];
				time_t t;
				struct tm * now;
				t = time(0); // get time now
				now = localtime(&t);
				sprintf(NAME, "D:\\TrackingMicroscopeData\\_LED\\led_%04d%02d%02d_%02d%02d%02d.dat", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
				pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
			}
			pwriter_binary->write((char *)(ControllHardWare_2P.AI_LED_Data_sort), ControllHardWare_2P.AI_LED_Data_sort_sz);
		}
		else {
			if (pwriter_binary) {
				pwriter_binary->close();
				delete(pwriter_binary);
				pwriter_binary = NULL;
			}
		}
	}
}

void UI::MainWindow::RecBufferCopyResultThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Copy Results to Record Buffer ");

	double fstarttime, fEndSeconds, fstarttime2 = 0;
	display_Enable_QCheckBoxtoggled(true);
	disp_totalcount = 500;
	disp_data1 = (double *)malloc(disp_totalcount * sizeof(double));
	disp_data1_idx = 0;
	disp_data2 = (double *)malloc(disp_totalcount * sizeof(double));
	disp_data2_idx = 0;
	disp_data3 = (double *)malloc(disp_totalcount * sizeof(double));
	disp_data3_idx = 0;
	disp_data4 = (double **)malloc(2 * sizeof(double*));
	disp_data4[0] = (double *)malloc(disp_totalcount * sizeof(double*));
	disp_data4[1] = (double *)malloc(disp_totalcount * sizeof(double*));
	disp_data4_idx = 0;
	disp_data5 = (double **)malloc(2 * sizeof(double*));
	disp_data5[0] = (double *)malloc(disp_totalcount * sizeof(double*));
	disp_data5[1] = (double *)malloc(disp_totalcount * sizeof(double*));
	disp_data5_idx = 0;
	for (int i = 0; i < disp_totalcount; i++) {
		disp_data1[i] = 0;
		disp_data2[i] = 0;
		disp_data3[i] = 0;
		disp_data4[0][i] = 0;
		disp_data4[1][i] = 0;
		disp_data5[0][i] = 0;
		disp_data5[1][i] = 0;
	}

	while (g_bProcessRunning) {
		bool b_isavailableData = false;
		RecBufferResults currentNIRFrame;
		{
			std::lock_guard<std::mutex> lock(NIRImgCopy_mutex);
			if (m_queueResults.size() > 0) {
				currentNIRFrame = m_queueResults.front();
				m_queueResults.erase(m_queueResults.begin());
				b_isavailableData = true;
			}
			if ((NIRCameraPG.b_stop == true) && (m_queueResults.size() > 0))
				m_queueResults.clear();
		}
		// add some safe transition when the recording is reset
		if (b_isavailableData && currentNIRFrame.FrmNo > 0) {
			RecBuffer.updateAllNIRcpy(
				currentNIRFrame.FrmNo,
				currentNIRFrame.StgPos,
				currentNIRFrame.DetectedZebraFish,
				currentNIRFrame.ControlDataRec,
				currentNIRFrame.ImgSrc,
				currentNIRFrame.rows,
				currentNIRFrame.cols,
				currentNIRFrame.FLRdata);
			if (disp_data1_enable[1]) {
				disp_data1[currentNIRFrame.FrmNo%disp_totalcount] = sqrt(currentNIRFrame.ControlDataRec.data[Xerr] * currentNIRFrame.ControlDataRec.data[Xerr] + currentNIRFrame.ControlDataRec.data[Yerr] * currentNIRFrame.ControlDataRec.data[Yerr]);
				disp_data1_idx = currentNIRFrame.FrmNo;
			}
			//if (disp_data4_enable[1]) {
			//	disp_data4[0][currentNIRFrame.FrmNo%disp_totalcount] = currentNIRFrame.ControlDataRec.data[Xfish];
			//	disp_data4[1][currentNIRFrame.FrmNo%disp_totalcount] = currentNIRFrame.ControlDataRec.data[Yfish];
			//	disp_data4_idx = currentNIRFrame.FrmNo;
			//}
			//if (disp_data5_enable[1]) {
			//	disp_data5[0][currentNIRFrame.FrmNo%disp_totalcount] = currentNIRFrame.StgPos.CurrentPosition[0];
			//	disp_data5[1][currentNIRFrame.FrmNo%disp_totalcount] = currentNIRFrame.StgPos.CurrentPosition[1];
			//	disp_data5_idx = currentNIRFrame.FrmNo;
			//}
			qtgraph0_gmap.update(currentNIRFrame.ControlDataRec.data[Xfish], currentNIRFrame.ControlDataRec.data[Yfish], currentNIRFrame.FrmNo);
			//qtgraph0_gmap.update(currentNIRFrame.StgPos.CurrentPosition[0], currentNIRFrame.StgPos.CurrentPosition[1], currentNIRFrame.FrmNo);
			qtgraph1.update(currentNIRFrame.StgPos.CurrentPosition[0], currentNIRFrame.StgPos.CurrentPosition[1], currentNIRFrame.FrmNo);
			qtgraph2.update(currentNIRFrame.ControlDataRec.data[Xfish], currentNIRFrame.ControlDataRec.data[Yfish], currentNIRFrame.FrmNo);
			qtgraph3.update1D(
				sqrt(currentNIRFrame.ControlDataRec.data[Xerr] * currentNIRFrame.ControlDataRec.data[Xerr] + currentNIRFrame.ControlDataRec.data[Yerr] * currentNIRFrame.ControlDataRec.data[Yerr]),
				currentNIRFrame.FrmNo);
			//qtgraph4.update1D(currentNIRFrame.FLRdata.intensity_mean, currentNIRFrame.FrmNo);

			// global map updating
			int xpx = (-currentNIRFrame.ControlDataRec.data[Xfish] + insetmap_size_mm / 2) / insetmap_pxdist;
			int ypx = (-currentNIRFrame.ControlDataRec.data[Yfish] + insetmap_size_mm / 2) / insetmap_pxdist;
			if (xpx >= 0 && xpx < insetmap->width() && ypx >= 0 && ypx < insetmap->height()) {
				//insetmap_x[currentNIRFrame.FrmNo % 2] = xpx;
				//insetmap_y[currentNIRFrame.FrmNo % 2] = ypx;
				//if (insetmap_x[0] >= 0 && insetmap_x[1] >= 0 && insetmap_y[0] >= 0 && insetmap_y[1] >= 0)
					//paint_line(insetmap_x, insetmap_y, QColor(255,255,255), insetmap);
					insetmap->setPixel(xpx, ypx, 1);
			}
		}
	}
}

void UI::MainWindow::RecSettings(void) {
	struct tm * now = localtime(&RecBuffer.t_now);
	char NAME[128];
	{
		std::lock_guard<std::mutex> lock(Rec_mutex);
		sprintf(NAME, "d:\\%s\\HDF5_%s_setting.h5", RecBuffer.DATESTR, RecBuffer.DATESTR);
		HDFWriter * settingwriter = new HDFWriter(NAME);
		if (settingwriter->err) {
			delete settingwriter;
			sprintf(NAME, "d:\\TrackingMicroscopeData\\_setting\\current_setting.h5");
			settingwriter = new HDFWriter(NAME);
		}

		double v_double[64] = { 0 };
		int v_int[64] = { 0 };
		uint v_uint[64] = { 0 };
		uint v_uint64[64] = { 0 };
		settingwriter->write("NIR_ExpTime_us", H5_double, v_double);
		v_int[0] = NIRCameraPG.ROIWidth;
		v_int[1] = NIRCameraPG.ROIHeight;
		settingwriter->write("spinBox_nircam_roix", H5_INT, v_int, 2);
		settingwriter->write("fishReferenceSize", H5_double, &(NIRFishPosDetector_GPU.fishRefSize));
		settingwriter->write("fishReferenceSize_variation", H5_double, &(NIRFishPosDetector_GPU.fishRefSize_variation));
		settingwriter->write("fishReferenceHeading_Cosphi", H5_double, &(NIRFishPosDetector_GPU.fitnessRef_heading));
		v_double[0] = m_ImageTargetPosPx.x;
		v_double[1] = m_ImageTargetPosPx.y;
		settingwriter->write("imagetarget_px", H5_INT, v_double, 2);
		settingwriter->write("fshRefSize_var", H5_double, &(NIRFishPosDetector_GPU.fishRefSize_variation));

		settingwriter->write("targetposoffset_majoraxis", H5_INT, &m_trk_TargetPositionOffset_majorAxis);
		settingwriter->write("GPU_Kernelsize", H5_INT, &NIRFishPosDetector_GPU.Kernelradius);
		settingwriter->write("GPU_Sigma", H5_INT, &NIRFishPosDetector_GPU.Sigma);
		settingwriter->write("GPU_SearchRadius", H5_INT, &NIRFishPosDetector_GPU.SrchRadius);

		settingwriter->write("NIR_pxdist", H5_double, &GlobalMap_NIR.PxDist_mmppx);
		settingwriter->write("NIR_theta", H5_double, &GlobalMap_NIR.theta);
		v_double[0] = GlobalMap_NIR.width;
		v_double[1] = GlobalMap_NIR.height;
		settingwriter->write("windowSize", H5_INT, v_double, 2);
		// record
		settingwriter->write("Rec_NIR_FrmStart", H5_UINT64, &RecBuffer.recFrms_Start);
		settingwriter->write("Rec_NIR_FrmEnd", H5_UINT64, &RecBuffer.recFrms_End);
		settingwriter->write("Rec_EPI_FrmStart", H5_UINT64, &HamamatsuCamera_C11440.recFrms_Start);
		settingwriter->write("Rec_EPI_FrmEnd", H5_UINT64, &HamamatsuCamera_C11440.recFrms_End);


		v_double[0] = m_ui.doubleSpinBox_Map_imsize_x->value();
		v_double[1] = m_ui.doubleSpinBox_Map_imsize_y->value();
		settingwriter->write("Chamber_size_mm", H5_double, v_double, 2);
		v_double[0] = thermalcontroller.cx_mm;
		v_double[1] = thermalcontroller.cy_mm;
		settingwriter->write("Chamber_offset_mm", H5_double, v_double, 2);
		v_double[0] = thermalcontroller.dx_mm;
		v_double[1] = thermalcontroller.dy_mm;
		settingwriter->write("Chamber_thermalstimulus_band_mm", H5_double, v_double, 2);

		/**/
		delete settingwriter;
	}
}

bool UI::MainWindow::isFishDetection(zebrafishInfo fish) {
	bool flagsize = false;
	bool flagangle = false;

	// size checking (
	double FishDetectionFailureRatio = NIRFishPosDetector_GPU.fishRefSize_variation;
	if (NIRFishPosDetector_GPU.fishRefSize*(1.0 - FishDetectionFailureRatio) < fish.AreaSize
		&& fish.AreaSize < NIRFishPosDetector_GPU.fishRefSize*(1.0 + FishDetectionFailureRatio))
		flagsize = true;
	else
		return false;

	if (fabs(fish.fitness_heading) < NIRFishPosDetector_GPU.fitnessRef_heading)
		flagangle = true;
	else
		return false;

	return true;
}


void UI::MainWindow::getFishGlobalPosition(zebrafishInfo * DetectedZebraFish, double * FishRefPosition, CntrData * ControlDataRec, bool _isfishdetected) {
	// 1. convert a local fish pose (in image) to the global pose (in mm - stage position added)
	Point2d tempPt = SelectTrackingTarget(m_trk_targetSelection, DetectedZebraFish);
	DetectedZebraFish->target_px = Point2i(tempPt.x, tempPt.y);
	ControlDataRec->data[XerrPx] = 	m_ImageTargetPosPx.x - DetectedZebraFish->target_px.x;
	ControlDataRec->data[YerrPx] = 	m_ImageTargetPosPx.y - DetectedZebraFish->target_px.y;
	// transpose error
	double Pos_error_px[2];
	//Pos_error_px[0] = GlobalMap_NIR.cth*ControlDataRec->data[XerrPx] - GlobalMap_NIR.sth*ControlDataRec->data[YerrPx];
	//Pos_error_px[1] = GlobalMap_NIR.sth*ControlDataRec->data[XerrPx] + GlobalMap_NIR.cth*ControlDataRec->data[YerrPx];
	Pos_error_px[0] = ControlDataRec->data[XerrPx];
	Pos_error_px[1] = ControlDataRec->data[YerrPx];
	ControlDataRec->data[Xerr] = (Pos_error_px[0])*GlobalMap_NIR.PxDist_mmppx;
	ControlDataRec->data[Yerr] = (Pos_error_px[1])*GlobalMap_NIR.PxDist_mmppx;

	// jitter filter for zebrafish global position
	if (m_trk_bjitterenable) {
		Point2d globalfishpos = globalFishPosAdjustor.update(Point2d(FishRefPosition[0] + ControlDataRec->data[Xerr], FishRefPosition[1] + ControlDataRec->data[Yerr]), _isfishdetected);
		ControlDataRec->data[Xfish] = globalfishpos.x;
		ControlDataRec->data[Yfish] = globalfishpos.y;
		//ControlDataRec->data[Xerr] = -FishRefPosition[0] + ControlDataRec->data[Xfish];
		//ControlDataRec->data[Yerr] = -FishRefPosition[1] + ControlDataRec->data[Yfish];
	}
	else {
		// raw value of the global position (it may include some noise from the stage - less than 1/2 pixel distance (~ 7um)
		ControlDataRec->data[Xfish] = FishRefPosition[0] + ControlDataRec->data[Xerr];
		ControlDataRec->data[Yfish] = FishRefPosition[1] + ControlDataRec->data[Yerr];
	}
	if (m_trk_brespoffsetenable) {
		ControlDataRec->data[Xfish] = m_trk_respoffsetfishpos_ref[0] + m_trk_respoffsetfishpos[0];
		ControlDataRec->data[Yfish] = m_trk_respoffsetfishpos_ref[1] + m_trk_respoffsetfishpos[1]; // offset when for calculate responses.
	}
	ControlDataRec->data[HeadingFish] = DetectedZebraFish->orienation;

	if (m_trk_bfakefishposition) { //
		fakefish_idx = fakefish_idx%fakefish_idxmax;
		ControlDataRec->data[Xfish] = fakefishposition_x[fakefish_idx];
		ControlDataRec->data[Yfish] = fakefishposition_y[fakefish_idx];
		ControlDataRec->data[HeadingFish] = fakefishposition_th[fakefish_idx];
		fakefish_idx++;

		ControlDataRec->data[Xerr] = FishRefPosition[0] - ControlDataRec->data[Xfish];
		ControlDataRec->data[Yerr] = FishRefPosition[1] - ControlDataRec->data[Yfish];
		ControlDataRec->data[XerrPx] = ((ControlDataRec->data[Xerr])/GlobalMap_NIR.PxDist_mmppx);
		ControlDataRec->data[YerrPx] = ((ControlDataRec->data[Yerr])/GlobalMap_NIR.PxDist_mmppx);
		DetectedZebraFish->target_px.x = m_ImageTargetPosPx.x - int(ControlDataRec->data[XerrPx]);
		DetectedZebraFish->target_px.y = m_ImageTargetPosPx.y - int(ControlDataRec->data[YerrPx]);
		DetectedZebraFish->centerEyes.x = DetectedZebraFish->target_px.x;
		DetectedZebraFish->centerEyes.y = DetectedZebraFish->target_px.y;
		DetectedZebraFish->m_eyeLeft.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		DetectedZebraFish->m_eyeRight.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		DetectedZebraFish->m_yolk.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		_isfishdetected = true;
	}

	if (replayer_fishpos_enable && replayer_fishpos) {
		if (replayer_fishpos_curfrm >= replayer_fishpos_totalfrm)
			replayer_fishpos_curfrm = replayer_fishpos_totalfrm - 1;
		if (!replayer_fishpos_enable_start)
			replayer_fishpos_curfrm = 0;
		ControlDataRec->data[Xfish] = replayer_fishpos[2 * replayer_fishpos_curfrm];
		ControlDataRec->data[Yfish] = replayer_fishpos[2 * replayer_fishpos_curfrm + 1];
		ControlDataRec->data[HeadingFish] = replayer_fishpos_OrientationDeg[replayer_fishpos_curfrm];
		replayer_fishpos_curfrm++;

		ControlDataRec->data[Xerr] = ControlDataRec->data[Xfish] - FishRefPosition[0];
		ControlDataRec->data[Yerr] = ControlDataRec->data[Yfish] - FishRefPosition[1];
		ControlDataRec->data[XerrPx] = ((ControlDataRec->data[Xerr]) / GlobalMap_NIR.PxDist_mmppx);
		ControlDataRec->data[YerrPx] = ((ControlDataRec->data[Yerr]) / GlobalMap_NIR.PxDist_mmppx);
		DetectedZebraFish->target_px.x = m_ImageTargetPosPx.x - int(ControlDataRec->data[XerrPx]);
		DetectedZebraFish->target_px.y = m_ImageTargetPosPx.y - int(ControlDataRec->data[YerrPx]);
		DetectedZebraFish->centerEyes.x = DetectedZebraFish->target_px.x;
		DetectedZebraFish->centerEyes.y = DetectedZebraFish->target_px.y;
		DetectedZebraFish->m_eyeLeft.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		DetectedZebraFish->m_eyeRight.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		DetectedZebraFish->m_yolk.center = Point2d(m_ImageTargetPosPx.x, m_ImageTargetPosPx.x);
		DetectedZebraFish->orienation = ControlDataRec->data[HeadingFish];
		_isfishdetected = true;
	}
}



void UI::MainWindow::MPCParmeterUpdate(void) {
	// Import Data (h5 File)
	try {
		int nx=0; int ny = 0;
		double * ImpRspX = NULL;
		double * ImpRspY = NULL;
		H5File* _file = new H5File ("D:\\TrackingMicroscopeData\\_setting\\Default_StepResp.h5", H5F_ACC_RDONLY);
		ImpRspX = importImpRestInd(_file, "ImpRspX", &nx);
		ImpRspY = importImpRestInd(_file, "ImpRspY", &ny);

		double * ImpRspX_Acc = NULL;
		int nx_acc = 0;
		ImpRspX_Acc = importImpRestInd(_file, "ImpRspX_Acc", &nx_acc);

		delete _file;





		double MPC_MaxVel_x = m_ui.spinBox_MPC_maxvel_mmps_x->value();
		double MPC_MaxVel_y = m_ui.spinBox_MPC_maxvel_mmps_y->value();
		double MPC_MaxdVel_x = m_ui.spinBox_MPC_maxdvel_mmps_x->value();
		double MPC_MaxdVel_y = m_ui.spinBox_MPC_maxdvel_mmps_y->value();
		double MPC_MaxAcc_x = ScalingAcc_X / 1.05;
		double MPC_MaxAcc_y = ScalingAcc_Y / 1.05;
		unsigned int C_pred = m_ui.spinBox_MPC_cntrhrz->value();
		unsigned int H_pred = m_ui.spinBox_MPC_predhrz->value();
		double w = 1;

		double wL1x = 0;
		double wL1y = 0;
		double wL2x = 0;
		double wL2y = 0;


		if (MPC_X.impRsp_ALL != NULL) free(MPC_X.impRsp_ALL);
		if (m_ui.checkBox_tracking_NL->isChecked()) {
			ifstream myFile("D:\\TrackingMicroscopeData\\_setting\\Default_StepResp_allX.dat", ios::in | ios::binary);
			size_t nx2 = 40 * 17 * 8;
			double * ImpRspX_ALL = (double *)malloc(nx2);
			myFile.read(reinterpret_cast<char*>(ImpRspX_ALL), nx2);
			MPC_X.impRsp_ALL = ImpRspX_ALL;
			MPC_X.impRsp_ALL_n = 17;
			MPC_X.impRsp_ALL_stride = 40;
		}

		MPC_X.Init((int)C_pred, wL1x, wL1y, MPC_MaxVel_x, MPC_MaxdVel_x, nx, ImpRspX);
		MPC_X.updateImpRsp_Acc(nx_acc, ImpRspX_Acc, MPC_MaxAcc_x);
		MPC_Y.Init((int)C_pred, wL2x, wL2y, MPC_MaxVel_y, MPC_MaxdVel_y, ny, ImpRspY);
		MPC_Y.updateImpRsp_Acc(nx_acc, ImpRspX_Acc, MPC_MaxAcc_y);
		MPC_X.terminate();
		MPC_Y.terminate();
		if (ImpRspX != NULL) free(ImpRspX);
		if (ImpRspY != NULL) free(ImpRspY);
		if (ImpRspX_Acc != NULL) free(ImpRspX_Acc);



	}
	catch(DataSetIException error) {
		error.printError();
		return;
	}
	return;
}


double * UI::MainWindow::importImpRestInd(H5File* file, char * datasetName, int * ReadDataCount) {
	try {
		DataSet* _dataset = new DataSet(file->openDataSet(datasetName));
		DataSpace* _file_dataspace = new DataSpace(_dataset->getSpace());
		auto rank = _file_dataspace->getSimpleExtentNdims();
		if (rank == 2) {
			hsize_t _total_size[2];
			_file_dataspace->getSimpleExtentDims(_total_size);
			DataSpace* _memory_dataspace = new DataSpace(2, _total_size);

			double * ReadData = (double *) malloc(_total_size[0]*_total_size[1]*sizeof(double));
			_dataset->read(ReadData, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
			delete _memory_dataspace;
			delete _dataset;
			delete _file_dataspace;
			*ReadDataCount = _total_size[0]*_total_size[1];
			return ReadData;
		}
		else if (rank == 1) {
			hsize_t _total_size[1];
			_file_dataspace->getSimpleExtentDims(_total_size);
			DataSpace* _memory_dataspace = new DataSpace(1, _total_size);
			double * ReadData = (double *)malloc(_total_size[0]*sizeof(double));
			_dataset->read(ReadData, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
			delete _memory_dataspace;
			delete _dataset;
			delete _file_dataspace;
			*ReadDataCount = _total_size[0];
			return ReadData;
		}

	}
	catch(DataSetIException error) {
		error.printError();
		return NULL;
	}
	return NULL;
}




//shared_ptr<QImage> cvMat2QImage(const cv::Mat * src) {
//	return Mat2QImage(src->data, (unsigned int)src->cols, (unsigned int)src->rows, (unsigned int)src->step);
//}
//shared_ptr<QImage> cvMat2QImage3(const cv::Mat * src) {
//	return Mat2QImage3(src->data, (unsigned int)src->cols, (unsigned int)src->rows, (unsigned int)src->step);
//}
// Display message
void QtDisplayMessageError(QString msg) {
	QMessageBox messageBox;
	messageBox.critical(0,"Error",msg);
}

void QtDisplayMessageWarning(QString msg) {
	QMessageBox messageBox;
	messageBox.warning(0,"Warning",msg);
}

void QtDisplayMessageInfo(QString msg) {
	QMessageBox messageBox;
	messageBox.information(0,"Information",msg);
}
void UI::MainWindow::Stage_WN_SampleGeneration_Clicked(void) {

	int StartFrm = m_ui.spinBox_Stage_WN_StartFrame->value();
	int OnDurationFrm = m_ui.spinBox_Stage_WN_OnDurationFrm->value();
	int RepeatFrm = m_ui.spinBox_Stage_WN_RepeatFrm->value();
	int inputFlipped = 1;
	if (RepeatFrm < 0) {
		inputFlipped = -1;
		RepeatFrm = -RepeatFrm;
	}
	float MaxVel = (float)m_ui.spinBox_Stage_WN_MaxVolts->value();
	int VelResolution = (float)m_ui.spinBox_Stage_WN_Resolution->value();
	int iter= m_ui.spinBox_Stage_WN_Iteration->value();
	int StepsPerCycle = m_ui.spinBox_Stage_WN_DutyRatio->value();
	bool xEnable = m_ui.checkBoxStage_Stage_WN_EnableX->isChecked();
	bool yEnable = m_ui.checkBoxStage_Stage_WN_EnableY->isChecked();

	WN_TotalSampleCountWhole = (MaxVel/VelResolution)*StepsPerCycle*2*iter + StartFrm + 50; // 100 steps per moiton | + and - | iteration



	if (WN_DataBuffer != NULL) {
		free(WN_DataBuffer); free(WN_DataBuffer_Vel);
		WN_DataBuffer = NULL; WN_DataBuffer_Vel = NULL;
	}

	WN_DataBuffer = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64)); // x and y
	WN_DataBuffer_Vel = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64));
	for (int i = 0; i < 2*WN_TotalSampleCountWhole; i++) {
		WN_DataBuffer[i] = 0;
		WN_DataBuffer_Vel[i] = 0;
	}
	float64 vel = 0;
	for (int i = StartFrm; i < WN_TotalSampleCountWhole; i += 2*StepsPerCycle) {
		vel += (float64)VelResolution;
		if (vel > MaxVel) {
			vel = (float64)VelResolution;
		}
		int flippedSign = 1;
		for (int j=0; j < RepeatFrm ; j++) {
			flippedSign = flippedSign*inputFlipped;
			float velSet = flippedSign*vel;
			for (int k=0; k < OnDurationFrm ; k++) {
				WN_DataBuffer_Vel[i+j*OnDurationFrm+k] = velSet ;
				WN_DataBuffer_Vel[i+j*OnDurationFrm+k+WN_TotalSampleCountWhole] = velSet ;
				WN_DataBuffer_Vel[i+j*OnDurationFrm+k+StepsPerCycle] = -velSet ;
				WN_DataBuffer_Vel[i+j*OnDurationFrm+k+StepsPerCycle+WN_TotalSampleCountWhole] = -velSet ;
			}
		}
	}

	m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);


};

void UI::MainWindow::Stage_WN_RandomSampleGeneration_Clicked(void) {

	int StartFrm = m_ui.spinBox_Stage_WN_StartFrame->value();
	int OnDurationFrm = m_ui.spinBox_Stage_WN_OnDurationFrm->value();
	int StepsPerCycle = 250*11;
	float MaxVel = (float)m_ui.spinBox_Stage_WN_MaxVolts->value();
	int iter= m_ui.spinBox_Stage_WN_Iteration->value();

	float VelSet[3];
	VelSet[0] = 0;
	VelSet[1] = MaxVel;
	VelSet[2] = -VelSet[1];

	WN_TotalSampleCountWhole = StepsPerCycle*iter + StartFrm; // 100 steps per moiton | + and - | iteration

		if (WN_DataBuffer != NULL) {
		free(WN_DataBuffer); free(WN_DataBuffer_Vel);
		WN_DataBuffer = NULL; WN_DataBuffer_Vel = NULL;
	}

	WN_DataBuffer = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64)); // x and y
	WN_DataBuffer_Vel = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64));
	for (int i = 0; i < 2*WN_TotalSampleCountWhole; i++) {
		WN_DataBuffer[i] = 0;
		WN_DataBuffer_Vel[i] = 0;
	}

	for (int i = StartFrm; i < WN_TotalSampleCountWhole; i += StepsPerCycle) {
		for (int k=0; k < OnDurationFrm ; k++) {
			WN_DataBuffer_Vel[i+k] = VelSet[rand()%3] ;
			WN_DataBuffer_Vel[i+k+WN_TotalSampleCountWhole] = VelSet[rand()%3] ;
		}
	}

	m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);


};

void UI::MainWindow::Stage_WN_ScanSampleGeneration(void) {

	double Horizontal_mm = 2*m_ui.spinBox_Stage_GlobalMapScanRadius_mm->value();
	double VerticlaScan_mm = 10.0 - m_ui.spinBox_Stage_GlobalMapOverlap_mm->value();
	float64 Speed_mmps = m_ui.spinBox_Stage_GlobalMapScanspeed_mmps->value();


	int sweepingCount = (int)(Horizontal_mm/VerticlaScan_mm + 1);
	int H_step = (int)(Horizontal_mm/Speed_mmps*10000/ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR);
	int V_step = (int)(VerticlaScan_mm/Speed_mmps*10000/ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR);

	WN_TotalSampleCountWhole = 2 * (sweepingCount * (V_step + H_step) + (int)(V_step/2)*3 + 100);



	if (WN_DataBuffer != NULL) {
		free(WN_DataBuffer); free(WN_DataBuffer_Vel);
		WN_DataBuffer = NULL; WN_DataBuffer_Vel = NULL;
	}
	float64 * H_Buff_F = (float64 *) malloc(H_step * sizeof(float64));
	float64 * H_Buff_B = (float64 *) malloc(H_step * sizeof(float64));
	float64 * V_Buff = (float64 *) malloc(V_step * sizeof(float64));
	float64 * InitV_Buff = (float64 *) malloc((int)(H_step/2) * sizeof(float64));

	for (int i = 0; i <H_step; i++) {
		H_Buff_F[i] = Speed_mmps;
		H_Buff_B[i] = -Speed_mmps;
	}
	for (int i = 0; i <V_step; i++) {
		V_Buff[i] = Speed_mmps;
	}
	for (int i = 0; i <(int)(H_step/2); i++) {
		InitV_Buff[i] = -Speed_mmps;
	}

	WN_TotalSampleCountWhole = 2*WN_TotalSampleCountWhole; // + & - direction
	WN_DataBuffer = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64)); // x and y
	WN_DataBuffer_Vel = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64));
	for (int i = 0; i < 2*WN_TotalSampleCountWhole; i++) {
		WN_DataBuffer[i] = 0;
		WN_DataBuffer_Vel[i] = 0;
	}


	int Count = (int)(H_step/2);
	memcpy(&WN_DataBuffer_Vel[0], InitV_Buff, Count*sizeof(float64));
	memcpy(&WN_DataBuffer_Vel[WN_TotalSampleCountWhole], InitV_Buff, Count*sizeof(float64));

	for (int i = 0; i < sweepingCount; i++) {
		if (i%2 == 0)
			memcpy(&WN_DataBuffer_Vel[Count], H_Buff_F, H_step*sizeof(float64));
		else
			memcpy(&WN_DataBuffer_Vel[Count], H_Buff_B, H_step*sizeof(float64));
		Count = Count + H_step;
		if (i < sweepingCount-1) {
			memcpy(&WN_DataBuffer_Vel[Count + WN_TotalSampleCountWhole], V_Buff, V_step*sizeof(float64));
			Count = Count + V_step;
		}
	}
	memcpy(&WN_DataBuffer_Vel[Count], InitV_Buff, (int)(H_step/2)*sizeof(float64));
	memcpy(&WN_DataBuffer_Vel[Count + WN_TotalSampleCountWhole], InitV_Buff, (int)(H_step/2)*sizeof(float64));

	free(H_Buff_F);
	free(H_Buff_B);
	free(V_Buff);
	free(InitV_Buff);

	m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);
	// load
	//ControllHardWare_2P.InitiateAOForStage_Buffer_WN(WN_TotalSampleCountWhole, WN_DataBuffer);*/
};

void UI::MainWindow::replayer_openfolder(void) {
	QString defaulname = m_ui.lineEdit_foldername->text();
	if (defaulname.isEmpty())
		defaulname = QString("d:\\");
	QString RePlaydirname = QFileDialog::getExistingDirectory(this, tr("Select a Directory"), defaulname);
	if (RePlaydirname.isEmpty()) return;
	m_ui.lineEdit_foldername->setText(RePlaydirname);

	if (m_ui.checkBox_replay_nir->isChecked()) {
		QString Replay_fn_NIRVideo = QFileDialog::getOpenFileName(this, "Select NIR video file", RePlaydirname + "/*_NIR.h5", tr("Videos (*.h5)"));
		if (Replay_fn_NIRVideo.isEmpty()) return;
		replayer.openNIR(Replay_fn_NIRVideo.toUtf8().constData());
		m_ui.lineEdit_nirfilename->setText(Replay_fn_NIRVideo);
		m_ui.lcdNumber_replay_totalframe_nir->display((int)replayer.totalframe_NIR);
	}

	if (m_ui.checkBox_replay_nir->isChecked()) {
		QString Replay_fn_FLRdata = QFileDialog::getOpenFileName(this, "Select FLR data file", RePlaydirname + "/*_EPI.h5", tr("Videos(*.h5)"));
		if (Replay_fn_FLRdata.isEmpty()) return;
		replayer.openFLR(Replay_fn_FLRdata.toUtf8().constData());
		m_ui.lineEdit_flrfilename->setText(Replay_fn_FLRdata);
		m_ui.lcdNumber_replay_totalframe_flr->display((int)replayer.totalframe_FLR);
	}

	if (m_ui.checkBox_replay_data->isChecked()) {
		QString Replay_fn_NIRdata = QFileDialog::getOpenFileName(this, "Select NIR data file", RePlaydirname + "/*_DATA.h5", tr("data file(*.h5)"));
		if (Replay_fn_NIRdata.isEmpty()) return;
	}
	m_ui.label_replaytime->setText(tr("00:00.000"));
	m_ui.lcdNumber_replay_curframe_nir->display(0);
	m_ui.lcdNumber_replay_curframe_flr->display(0);
}
void UI::MainWindow::replayer_openfile_nir(void) {
	QString Replay_fn_NIRVideo = QFileDialog::getOpenFileName(this, "Select NIR video file", RePlaydirname + "/*_NIR.h5", tr("Videos (*.h5)"));
	if (Replay_fn_NIRVideo.isEmpty()) return;
	replayer.openNIR(Replay_fn_NIRVideo.toUtf8().constData());
	m_ui.lineEdit_nirfilename->setText(Replay_fn_NIRVideo);
	m_ui.lcdNumber_replay_totalframe_nir->display((int)replayer.totalframe_NIR);
}
void UI::MainWindow::replayer_openfile_flr(void) {
	QString Replay_fn_FLRdata = QFileDialog::getOpenFileName(this, "Select FLR data file", RePlaydirname + "/*_EPI.h5", tr("Videos(*.h5)"));
	if (Replay_fn_FLRdata.isEmpty()) return;
	replayer.openFLR(Replay_fn_FLRdata.toUtf8().constData());
	m_ui.lineEdit_flrfilename->setText(Replay_fn_FLRdata);
	m_ui.lcdNumber_replay_totalframe_flr->display((int)replayer.totalframe_FLR);
}
void UI::MainWindow::replayer_openfile_data(void) {

}
void UI::MainWindow::replayer_next(void) {
	replayer.stepframe = m_ui.spinBox_replay_skipframes->value();
	//replayer.offsetframe_FLR = m_ui.spinBox_replay_offsetframeFLR->value();
	if (m_ui.checkBox_replay_masterNIR->isChecked()) {
		replayer.samplingtime_NIRus = m_ui.spinBox_nir_samplingtime_us->value();
		replayer.updatingtime_us = replayer.stepframe*replayer.samplingtime_NIRus;
	}
	else {
		replayer.samplingtime_FLRus = m_ui.spinBox_flr_samplingtime_us->value();
		replayer.updatingtime_us = replayer.stepframe*replayer.samplingtime_FLRus;
	}
	replayer.update();
	replayer_display();
}
void UI::MainWindow::replayer_prev(void) {
	replayer.stepframe = m_ui.spinBox_replay_skipframes->value();
	//replayer.offsetframe_FLR = m_ui.spinBox_replay_offsetframeFLR->value();
	if (m_ui.checkBox_replay_masterNIR->isChecked()) {
		replayer.samplingtime_NIRus = m_ui.spinBox_nir_samplingtime_us->value();
		replayer.updatingtime_us = -replayer.stepframe*replayer.samplingtime_NIRus;
	}
	else {
		replayer.samplingtime_FLRus = m_ui.spinBox_flr_samplingtime_us->value();
		replayer.updatingtime_us = -replayer.stepframe*replayer.samplingtime_FLRus;
	}
	replayer.update();
	replayer_display();
}
void UI::MainWindow::replayer_play(void) {
	if (!replayer.b_playing) {
		//m_pThreadGeneral = std::thread([this]{replayer_play_thread(); }); // call function in other thread
		//m_pThreadGeneral.detach(); // detach the thread for next recording
		replayer.b_playing = true;
		m_ui.toolButton_replay_play->setText("=");
	}
	else {
		replayer.b_playing = false;
		m_ui.toolButton_replay_play->setText(">");
	}
}
void UI::MainWindow::replayer_play_thread(void) {
	if (replayer.b_playing) {
		replayer_next();
		if (replayer.currentplaytime_us >= replayer.currentplaytime_usMax)
			replayer.b_playing = false;
	}
}
void UI::MainWindow::replayer_display(void) {
	QGLZoomableImageViewer* m_pImage_NIR, *m_pImage_FLR;
	if (m_ui.checkBox_replay_displaymasterFLR->isChecked()) {
		m_pImage_NIR = m_pImageR2;
		m_pImage_FLR = m_pImageR1;
	}
	else{
		m_pImage_NIR = m_pImageR1;
		m_pImage_FLR = m_pImageR2;
	}
	{
		auto pImage = toQImage(&replayer.curNIRimg);
		m_pImage_NIR->SetImage(pImage);
	}
	{
		int _min = m_ui.spinBox_Hamamatsu_IntMinSet->value();
		int _max = m_ui.spinBox_Hamamatsu_IntMaxSet->value();
		if (_min >= _max) {
			_max = _min + 1;
			m_ui.spinBox_Hamamatsu_IntMaxSet->setValue(_max);
		}
		DisplayImage_ROI(replayer.curFLRimg, m_pImage_FLR, (uint16)_min, (uint16)_max);
	}
	m_ui.lcdNumber_replay_curframe_nir->display((int)replayer.curframe_NIR);
	m_ui.lcdNumber_replay_curframe_flr->display((int)replayer.curframe_FLR);
	m_ui.lcdNumber_replayer_freqHz->display(1000.0 / (double)m_ThreadProcessTimeMills_DisplayUpdateCycle.MVA * (double)replayer.stepframe);
	QString str;
	str.sprintf("%03d:%02d.%03d", replayer.currentplaytime_us / 1000000 / 60, (replayer.currentplaytime_us / 1000000) % 60, (replayer.currentplaytime_us / 1000) % 1000);
	m_ui.label_replaytime->setText(str);

	// add test on z-axis
	try {
		piezoparameterupdate_buttonclicked();

		double fstarttime = GetCycleCountSeconds();
		zaxis_controller.compute_mu(replayer.curFLRimg.data, 0 );
		zaxis_controller.update_steps();
		double fendtime = GetCycleCountSeconds();
		m_ThreadProcessTimeMills_trkPiezo.update((fendtime - fstarttime) * 1000);

		auto pImage = Mat2QImage((uchar*)zaxis_controller.mask, zaxis_controller.imgsize.height, zaxis_controller.imgsize.width, zaxis_controller.imgsize.width);
		m_pImageR3->SetImage(pImage);

	}
	catch (...)
	{
		int a = 1;
	}
}


void UI::MainWindow::replayer_proc(void) {
	replayer.stepframe = m_ui.spinBox_replay_skipframes->value();
	//replayer.offsetframe_FLR = m_ui.spinBox_replay_offsetframeFLR->value();
	if (m_ui.checkBox_replay_masterNIR->isChecked()) {
		//replayer.samplingtime_NIRus = m_ui.spinBox_nir_samplingtime_us->value();
		//replayer.updatingtime_us = replayer.stepframe*replayer.samplingtime_NIRus;
		int FishSize = 0;
		double stgPos[2] = { 0 };
		vector<Point2d> fishPos;// = GPUImageProcess_centroid(replayer.curNIRimg.data, stgPos, &FishSize, 1);

		uint8_t * ImgSrc = replayer.curNIRimg.data;
		if (true) // compute eye angles
		{
			// add the computing eye angle
			int x_eyeL = (int)(fishPos[0].x);
			int y_eyeL = (int)(fishPos[0].y);
			int x_eyeR = (int)(fishPos[1].x);
			int y_eyeR = (int)(fishPos[1].y);

			int r = 10;
			double M00[2] = { 0 };
			double M11[2] = { 0 };
			double M02[2] = { 0 };
			double M20[2] = { 0 };
			if ((x_eyeL > 2 * r) && (y_eyeL > 2 * r) && (x_eyeR > 2 * r) && (y_eyeR > 2 * r) &&
				(x_eyeL < NIRFishPosDetector_GPU.imgSize.width - 2 * r) && (y_eyeL < NIRFishPosDetector_GPU.imgSize.height - 2 * r) &&
				(x_eyeR < NIRFishPosDetector_GPU.imgSize.width - 2 * r) && (y_eyeR < NIRFishPosDetector_GPU.imgSize.height - 2 * r)) {

				for (int j = -r; j < r; j++) {
					int idL = (y_eyeL + j)*NIRFishPosDetector_GPU.imgSize.width + x_eyeL;
					int idR = (y_eyeR + j)*NIRFishPosDetector_GPU.imgSize.width + x_eyeR;
					for (int i = -r; i < r; i++) {
						double IL = double(ImgSrc[idL + i]);
						double IR = double(ImgSrc[idR + i]);
						M00[0] += IL; M00[1] += IR;
						M11[0] += IL*i*j; M11[1] += IR*i*j;
						M20[0] += IL*i*i; M20[1] += IR*i*i;
						M02[0] += IL*j*j; M02[1] += IR*j*j;
					}
				}
				double mu11[2] = { 0 };
				double mu20[2] = { 0 };
				double mu02[2] = { 0 };
				double th_Eye[2] = { 0 };
				for (int i = 0; i < 2; i++) {
					mu11[i] = M11[i] / M00[i];
					mu20[i] = M20[i] / M00[i];
					mu02[i] = M02[i] / M00[i];
					th_Eye[i] = atan2(2 * mu11[i], mu20[i] - mu02[i]) * 90.0/3.141592;
				}
				//DetectedZebraFish_new.m_eyeLeft.angle = th_Eye[0];
				//DetectedZebraFish_new.m_eyeRight.angle = th_Eye[1];
				//DetectedZebraFish_new.m_yolk.angle = fabs(th_Eye[0] - th_Eye[1]);
				//if (DetectedZebraFish_new.m_yolk.angle > 180)
				//	DetectedZebraFish_new.m_yolk.angle -= 180;
			}
		}

	}



	replayer.update();
	replayer_display();
}

void UI::MainWindow::Stage_Replay_VelSetLoad_Clicked(void) {
	// Import Data (h5 File)
	try {
		QString Replay_VelSet = QFileDialog::getOpenFileName(this,"Select h5 data file","d:\\TrackingMicroscopeData\\*.h5",tr("data file(*.h5)"));
		if( Replay_VelSet.isEmpty() ) return;

		int nx=0; int ny = 0;
		double * ImpRspX = NULL;
		double * ImpRspY = NULL;
		H5File* _file = new H5File (Replay_VelSet.toUtf8().constData(), H5F_ACC_RDONLY);
		ImpRspX = importImpRestInd(_file, "VelSet_X", &nx);
		ImpRspY = importImpRestInd(_file, "VelSet_Y", &ny);
		delete _file;

		int StartFrm = m_ui.spinBox_Stage_WN_StartFrame->value();

		WN_TotalSampleCountWhole = max(nx, ny) + StartFrm + 100; // 100 steps per moiton | + and - | iteration

		if (WN_DataBuffer != NULL) {
		free(WN_DataBuffer); free(WN_DataBuffer_Vel);
		WN_DataBuffer = NULL; WN_DataBuffer_Vel = NULL;
		}

		WN_DataBuffer = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64)); // x and y
		WN_DataBuffer_Vel = (float64 *) malloc(2 * WN_TotalSampleCountWhole * sizeof(float64));
		for (int i = 0; i < 2*WN_TotalSampleCountWhole; i++) {
			WN_DataBuffer[i] = 0;
			WN_DataBuffer_Vel[i] = 0;
		}

		memcpy(&WN_DataBuffer_Vel[StartFrm], ImpRspX, nx*sizeof(float64));
		memcpy(&WN_DataBuffer_Vel[StartFrm + WN_TotalSampleCountWhole], ImpRspY, ny*sizeof(float64));

		m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);

		if (ImpRspX != NULL) free(ImpRspX);
		if (ImpRspY != NULL) free(ImpRspY);
	}
	catch(DataSetIException error) {
		error.printError();
		return;
	}
	return;
}


void UI::MainWindow::Stage_WN_SampleLoad_Clicked(void) {
	//RePlayer_OpenFileDialogClicked();

	if(NIRDataDecoder == NULL) return;

	int StartFrm = m_ui.spinBox_Stage_WN_StartFrame->value();
	WN_TotalSampleCountWhole = StartFrm + NIRDataDecoder->PointData[0]->Data_Length();
	if (WN_DataBuffer != NULL) {
		free(WN_DataBuffer); free(WN_DataBuffer_Vel);
		WN_DataBuffer = NULL; WN_DataBuffer_Vel = NULL;
	}
	WN_DataBuffer = (float64 *) malloc(2*WN_TotalSampleCountWhole * sizeof(float64));
	WN_DataBuffer_Vel = (float64 *) malloc(2*WN_TotalSampleCountWhole * sizeof(float64));
	for (int i = 0; i < 2*WN_TotalSampleCountWhole; i++) {
		WN_DataBuffer[i] = 0;
		WN_DataBuffer_Vel[i] = 0;
	}
	int k = 0;
	for (int i = StartFrm; i < WN_TotalSampleCountWhole; i++) {
		Point2d vel_set;
		NIRDataDecoder->PointData[R2_StageSetInputVel]->read(k++, &vel_set);
		WN_DataBuffer_Vel[i] = vel_set.x;
		WN_DataBuffer_Vel[i + WN_TotalSampleCountWhole] = vel_set.y;
		WN_DataBuffer[i] = ILS200LM.Velocity2Volt_x(vel_set.x);
		WN_DataBuffer[i + WN_TotalSampleCountWhole] = ILS200LM.Velocity2Volt_y(vel_set.y);

	}

	m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole/2);
	// load

};

void UI::MainWindow::Stage_WN_Start_Clicked(void) {
	if (WN_DataBuffer_Vel != NULL) {
		//XPSAVController.stop();
		//XPSAVController.start();
		WN_Enable = true;
	}
	else
		WN_Enable = false;
}


void UI::MainWindow::RasterLoadPos(void) {
	m_ui.doubleSpinBox_Raster_X0->setValue(ILS200LM.CurrentStatus.CurrentPosition[0]);
	m_ui.doubleSpinBox_Raster_Y0->setValue(ILS200LM.CurrentStatus.CurrentPosition[1]);
	m_ui.doubleSpinBox_Raster_Z0->setValue(m_piezoInput*40.0);
}

void UI::MainWindow::Raster3DSpace(void) {
	// check the status (stage should be ready)
	if (ILS200LM.ControllerStatus < 10 || ILS200LM.ControllerStatus > 19) {
		QtDisplayMessageInfo("Stage Status is not ready");
		return;
	}
	if (RecBuffer_FLR == NULL) {
		QtDisplayMessageInfo("A camera is not ready");
		return;
	}
	else {
		if (RecBuffer_FLR->size() == 0) {
			QtDisplayMessageInfo("A camera is not capture the image yet");
			return;
		}
	}


	// 1. generate the positions from the input
	double SleepTime =  m_ui.spinBox_Raster_SettlingTime_ms->value(); // [ms]

	double p0[3] = {0};
	double dp[3] = {0};
	int steps[3] = {0};
	int TotalCount = 0;

	p0[0] = m_ui.doubleSpinBox_Raster_X0->value()*1000.0; //[um]
	p0[1] = m_ui.doubleSpinBox_Raster_Y0->value()*1000.0; //[um]
	p0[2] = m_ui.doubleSpinBox_Raster_Z0->value(); //[um]

	dp[0] = m_ui.doubleSpinBox_Raster_dx->value(); //[um]
	dp[1] = m_ui.doubleSpinBox_Raster_dy->value(); //[um]
	dp[2] = m_ui.doubleSpinBox_Raster_dz->value(); //[um]

	steps[0] = m_ui.spinBox_Raster_StepX->value(); //[um]
	steps[1] = m_ui.spinBox_Raster_StepY->value(); //[um]
	steps[2] = m_ui.spinBox_Raster_StepZ->value(); //[um]

	TotalCount = steps[0] * steps[1] * steps[2];


	double * RasterPositions = (double *) malloc(3 * TotalCount * sizeof(double)); // x and y
	for (int i = 0; i < steps[2]; i++) { //z
		for (int j = 0; j < steps[0]; j++) { //x
			for (int k = 0; k < steps[1]; k++) { //y
				RasterPositions[i*steps[0]*steps[1] + j*steps[0] + k] = p0[0] + dp[0]*(double)j;
				RasterPositions[i*steps[0]*steps[1] + j*steps[0] + k + TotalCount] = p0[1] + dp[1]*(double)k;
				RasterPositions[i*steps[0]*steps[1] + j*steps[0] + k + 2*TotalCount] = p0[2] + dp[2]*(double)i;
			}
		}
	}

	// 3. Move to the position and save data
	char NAME[256]; char NAME2[256];
	time_t t = time(0); // get time now;
	struct tm * now = localtime(&t);
	sprintf(NAME,"d:\\TrackingMicroscopeData\\_Snapshot\\Raster3D_%04d%02d%02d_%02d%02d%02d_EPI.h5",
		now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
	sprintf(NAME2,"d:\\TrackingMicroscopeData\\_Snapshot\\Raster3D_%04d%02d%02d_%02d%02d%02d_Data.h5",
		now->tm_year + 1900, (now->tm_mon + 1),now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);

	HDF5ImageWriter * temp_writer;
	temp_writer = new HDF5ImageWriter(NAME, FLRCameraPG.getcols(), FLRCameraPG.getrows(), IMAGE_UINT16, 1);
	H5File * _file = new H5File(std::string(NAME2).c_str(), H5F_ACC_TRUNC);
    vector<HDF5SingleDataSet *> SingleData;
	vector<HDF5PointDataSet *> PointData;
	SingleData.push_back(new HDF5SingleDataSet(_file, "FrameNo", H5_UINT));
	SingleData.push_back(new HDF5SingleDataSet(_file, "Raster_ZPos_um", H5_double));
	SingleData.push_back(new HDF5SingleDataSet(_file, "Raster_SetZPos_um", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "Raster_Pos_mm", H5_double));
	PointData.push_back(new HDF5PointDataSet(_file, "Raster_SetPos_mm", H5_double));
	//SingleData.push_back(new HDF5SingleDataSet(_file, "StageZPos_Volt", H5_double));
	//SingleData.push_back(new HDF5SingleDataSet(_file, "StageZSetPos_um", H5_double));
	//SingleData.push_back(new HDF5SingleDataSet(_file, "StageZSetPos_Volt", H5_double));


	for (int i = 0; i < TotalCount; i++) {
		double TargetPos[2];
		TargetPos[0]= RasterPositions[i]/1000.0; // conversion um -> mm
		TargetPos[1]= RasterPositions[i+ TotalCount]/1000.0; // conversion um -> mm
		double z = RasterPositions[i+ 2*TotalCount]/40.0; // conversion um -> volt

		ILS200LM.MyXYGroupMoveAbsolute(TargetPos);
		ControllHardWare_2P.UpdateAO1(z);
		Sleep(SleepTime);
		// Record the data
		if (RecBuffer_FLR->size() > 0) {
			temp_writer->write(RecBuffer_FLR_ptr->EPIFLImage->data);	// image
			SingleData[0]->write(&i); // frame number
			SingleData[1]->write(&ILS200LM.CurrentStatus.Pos_Z); // z-axis position
			SingleData[2]->write(&RasterPositions[i+ 2*TotalCount]); // z-axis position
			PointData[0]->write(ILS200LM.CurrentStatus.CurrentPosition); // xy position;
			PointData[1]->write(TargetPos); // xy set position;
		}
	}


	// delete buffer
	for (int i = 0; i < SingleData.size(); i++)
		delete SingleData[i];
	for (int i = 0; i < PointData.size(); i++)
		delete PointData[i];
	if (_file)
		delete _file;
	delete temp_writer;
	free(RasterPositions );
}
void UI::MainWindow::Stage_Replay_FishPosLoad_Clicked(void) {
	// Import Data (h5 File)
	try {
		QString Replay_VelSet = QFileDialog::getOpenFileName(this,"Select h5 data file","d:\\TrackingMicroscopeData\\*.h5",tr("data file(*.h5)"));
		if( Replay_VelSet.isEmpty() ) return;

		int nPos, nOri, nErr, nStg;
		if (LoadedFishPos != NULL) {
		free(LoadedFishPos); free(LoadedFishPosOrientation);
		//free(LoadedFishPosError_mm);
		}
		LoadedFishPos = NULL;
		LoadedFishPosOrientation = NULL;
		LoadedFishPosError_mm = NULL;
		double * LoadedStagePos = NULL;
		H5File* _file = new H5File (Replay_VelSet.toUtf8().constData(), H5F_ACC_RDONLY);
		LoadedFishPosOrientation = importHDF5_Column(_file, "FishPos_Orientation_Deg", &nOri);
		LoadedFishPos = importHDF5_Column(_file, "FishPos_mmN2", &nPos);
		//LoadedFishPosError_mm = importHDF5_Column(_file, "ErrPos_mmN2", &nErr);
		LoadedStagePos = importHDF5_Column(_file, "StagePos", &nStg);
		delete _file;

		WN_TotalSampleCountWhole = nOri; // 100 steps per moiton | + and - | iteration
		m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);
		LoadedFishPos_StageInitialPosition[0] = LoadedStagePos[0];
		LoadedFishPos_StageInitialPosition[1] = LoadedStagePos[1];
		free(LoadedStagePos);

	}
	catch(DataSetIException error) {
		error.printError();
		return;
	}
	return;
}
void UI::MainWindow::Stage_Replay_FishPosLoadInitStage_Clicked(void) {
}
void UI::MainWindow::Stage_Replay_FishPosLoadStartSimulation_Clicked(void) {
	if (LoadedFishPos != NULL) {
		LoadedFishPos_Enable = true;
	}
	else
		LoadedFishPos_Enable = false;
}

double * UI::MainWindow::importHDF5_Column(H5File* file, char * datasetName, int * ReadDataCount) {
	try {
		DataSet* _dataset = new DataSet(file->openDataSet(datasetName));
		DataSpace* _file_dataspace = new DataSpace(_dataset->getSpace());
		auto rank = _file_dataspace->getSimpleExtentNdims();
		if (rank == 2) {
			hsize_t _total_size[2];
			_file_dataspace->getSimpleExtentDims(_total_size);
			DataSpace* _memory_dataspace = new DataSpace(2, _total_size);

			double * ReadData = (double *) malloc(_total_size[0]*_total_size[1]*sizeof(double));
			_dataset->read(ReadData, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
			delete _memory_dataspace;
			delete _dataset;
			delete _file_dataspace;
			*ReadDataCount = _total_size[0]*_total_size[1];
			return ReadData;
		}
		else if (rank == 1) {
			hsize_t _total_size[1];
			_file_dataspace->getSimpleExtentDims(_total_size);
			DataSpace* _memory_dataspace = new DataSpace(1, _total_size);
			double * ReadData = (double *) malloc(_total_size[0]*sizeof(double));
			_dataset->read(ReadData, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
			delete _memory_dataspace;
			delete _dataset;
			delete _file_dataspace;
			*ReadDataCount =  _total_size[0];
			return ReadData;
		}

	}
	catch(DataSetIException error) {
		error.printError();
		return NULL;
	}
	return NULL;
}

void UI::MainWindow::Stage_FishPos_SinePathGeneration(void) {
	// Import Data (h5 File)
	double dt = (double)ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR*0.1*0.001; //[ms]
	double period = m_ui.doubleSpinBox_FIshPosLoad_SineWavePeriod->value(); //[ms]
	double Amplitude = m_ui.doubleSpinBox_FIshPosLoad_SineWaveAmp->value(); //[mm]
	double FwdVel = m_ui.doubleSpinBox_FIshPosLoad_SineWaveFwdVel->value(); //[mm]
	int iter = m_ui.spinBox_FIshPosLoad_SineWaveIteration->value(); //
	int RotAngDeg = m_ui.spinBox_FIshPosLoad_SineWaveRotAngle_Deg->value(); //[deg]
	double RotAng = (double)RotAngDeg*PI/180; //[rad]

	if (LoadedFishPos != NULL) {
		free(LoadedFishPos); free(LoadedFishPosOrientation); free(LoadedFishPosError_mm);
	}
	LoadedFishPos = NULL;
	LoadedFishPosOrientation = NULL;
	LoadedFishPosError_mm = NULL;

	WN_TotalSampleCountWhole = (int)(period * (double)iter / dt) + 1; //
	LoadedFishPos = (double *) malloc(2 * WN_TotalSampleCountWhole * sizeof(double)); // x and y
	LoadedFishPosOrientation = (double *) malloc(WN_TotalSampleCountWhole * sizeof(double)); // x and y
	LoadedFishPosError_mm = (double *) malloc(2 * WN_TotalSampleCountWhole * sizeof(double)); // x and y
	LoadedFishPos_StageInitialPosition[0] = 0;
	LoadedFishPos_StageInitialPosition[1] = 0;

	double omega = 2*PI/period;
	double cosRotAng = cos(RotAng);
	double sinRotAng = sin(RotAng);
	// x = A * sin(2*pi*w*t);
	for (int i = 0; i < WN_TotalSampleCountWhole; i++) {
		double t = dt*(double)i;
		double x = FwdVel*t;
		double y = Amplitude*sin(omega*t);
		double heading = 0;
		LoadedFishPos[2*i] = x*cosRotAng + y*sinRotAng;
		LoadedFishPos[2*i + 1] = x*sinRotAng - y*cosRotAng;
		LoadedFishPosOrientation[i] = RotAngDeg;
	}

	m_ui.lcdNumber_Stage_WN_TotalFrame->display(WN_TotalSampleCountWhole);
	return;
}

void UI::MainWindow::Stage_I2T_setup(void) {
	//double dt = (double)ControllHardWare_2P.DO_ExtTrigger_NumCycleNIR*0.1*0.001; //[ms]
	//double a0 = m_ui.doubleSpinBox_Stage_I2T_gainY->value(); //[ms]
	//int mode = m_ui.comboBox_Stage_I2T_MODE->currentIndex();
	if (m_ui.checkBox_tracking_update->isChecked()) {
		ILS200LM.I2T_X.set_L2w(m_ui.doubleSpinBox_Stage_I2T_minX->value(), m_ui.doubleSpinBox_Stage_I2T_maxX->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_X->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_X->value());
		ILS200LM.I2T_Y.set_L2w(m_ui.doubleSpinBox_Stage_I2T_minY->value(), m_ui.doubleSpinBox_Stage_I2T_maxY->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_Y->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_Y->value());
		ILS200LM.I2T_X.set_limitCurrent(m_ui.spinBox_MPC_maxAcc_x->value());
		ILS200LM.I2T_Y.set_limitCurrent(m_ui.spinBox_MPC_maxAcc_y->value());

		MPC_X.velMax = m_ui.spinBox_MPC_maxvel_mmps_x->value();
		MPC_Y.velMax = m_ui.spinBox_MPC_maxvel_mmps_y->value();
		MPC_X.dvelMax = m_ui.spinBox_MPC_maxdvel_mmps_x->value();
		MPC_Y.dvelMax = m_ui.spinBox_MPC_maxdvel_mmps_y->value();

		MPC_X.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_x->value(), m_ui.doubleSpinBox_Trk_b_x->value(), m_ui.doubleSpinBox_Trk_c_x->value());
		MPC_Y.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_y->value(), m_ui.doubleSpinBox_Trk_b_y->value(), m_ui.doubleSpinBox_Trk_c_y->value());

		ParameterUpdate_FishPosEstimator();
	}
	else
	{
		double *mu_min, *mu_max, *minErr, *maxErr;
		double *set_vel, *set_dvel, *set_maxAcc;
		double *PIDx, *PIDy;

		ILS200LM.I2T_X.set_L2w(m_ui.doubleSpinBox_Stage_I2T_minX->value(), m_ui.doubleSpinBox_Stage_I2T_maxX->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_X->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_X->value());
		ILS200LM.I2T_Y.set_L2w(m_ui.doubleSpinBox_Stage_I2T_minY->value(), m_ui.doubleSpinBox_Stage_I2T_maxY->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_Y->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_Y->value());
		ILS200LM.I2T_X.set_limitCurrent(m_ui.spinBox_MPC_maxAcc_x->value());
		ILS200LM.I2T_Y.set_limitCurrent(m_ui.spinBox_MPC_maxAcc_y->value());

		MPC_X.velMax = m_ui.spinBox_MPC_maxvel_mmps_x->value();
		MPC_Y.velMax = m_ui.spinBox_MPC_maxvel_mmps_y->value();
		MPC_X.dvelMax = m_ui.spinBox_MPC_maxdvel_mmps_x->value();
		MPC_Y.dvelMax = m_ui.spinBox_MPC_maxdvel_mmps_y->value();

		MPC_X.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_x->value(), m_ui.doubleSpinBox_Trk_b_x->value(), m_ui.doubleSpinBox_Trk_c_x->value());
		MPC_Y.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_y->value(), m_ui.doubleSpinBox_Trk_b_y->value(), m_ui.doubleSpinBox_Trk_c_y->value());

		ParameterUpdate_FishPosEstimator();
	}

}


void UI::MainWindow::Stage_I2T_clear(void) {
	ILS200LM.I2T_X.clear();
	ILS200LM.I2T_Y.clear();
}


bool UI::MainWindow::computeMPC_L(double * StgPosition, CntrData * ControlDataRec) {
	bool isMPCEnabled = MPC_X.controlmanager.isMPC();
	// TODO: MODIFY "make one file and save in a different disc"
	double Pos_Stage_OpenloopMPC[2];
	double Pos_Stage_OpenloopMPCNext[2];
	Pos_Stage_OpenloopMPC[0] = StgPosition[0];
	Pos_Stage_OpenloopMPC[1] = StgPosition[1];

	// 0. Initialization of MPC
	if (isMPCEnabled) {
		if (!MPC_X.isInitiated()) {
			MPC_X.InitPos(StgPosition[0]);
			MPC_Y.InitPos(StgPosition[1]);
		}
	}
	else {
		MPC_X.terminate();
		MPC_Y.terminate();
	}

	// 2. fish pose estimation
	if (m_trk_bfishpospredictionenable)
		if (replayer_fishpos_enable && replayer_fishpos) {
			if (replayer_fishpos_enable_perfectprediction) {
				for (int i = 0; i < FishPosEstimator.steps; i++) {
					int k = replayer_fishpos_curfrm + i;
					if (k >= replayer_fishpos_totalfrm)
						k = replayer_fishpos_totalfrm - 1;
					FishPosEstimator.predX[i] = replayer_fishpos[2 * k];
					FishPosEstimator.predY[i] = replayer_fishpos[2 * k + 1];
				}
			}
			else
			{
				FishPosEstimator.update(replayer_fishpos[2 * replayer_fishpos_curfrm], replayer_fishpos[2 * replayer_fishpos_curfrm + 1], replayer_fishpos_OrientationDeg[replayer_fishpos_curfrm]);
			}
		}
		else
			FishPosEstimator.update(ControlDataRec->data[Xfish], ControlDataRec->data[Yfish], ControlDataRec->data[HeadingFish]);
	else{
		for (int i = 0; i < FishPosEstimator.steps; i++) {
			FishPosEstimator.predX[i] = ControlDataRec->data[Xfish];
			FishPosEstimator.predY[i] = ControlDataRec->data[Yfish];
		}
	}

	// 2-1. offset for predict position
	double th_rad = ControlDataRec->data[HeadingFish]*PI/180;
	double th_rad_RightAngle = (ControlDataRec->data[HeadingFish] + 90)*PI/180;
	double dx = m_trk_TargetPositionOffset_majorAxis*cos(th_rad) +  m_trk_TargetPositionOffset_minorAxis*cos(th_rad_RightAngle);
	double dy = m_trk_TargetPositionOffset_majorAxis*sin(th_rad) +  m_trk_TargetPositionOffset_minorAxis*sin(th_rad_RightAngle);
	for (int i = 0; i < FishPosEstimator.steps; i++) {
		FishPosEstimator.predX[i] += dx;
		FishPosEstimator.predY[i] += dy;
	}

	ControlDataRec->data[x_target_mm] = ControlDataRec->data[Xfish] + dx;
	ControlDataRec->data[y_target_mm] = ControlDataRec->data[Yfish] + dy;
	ControlDataRec->data[x_ref_px] = m_ImageTargetPosPx.x;
	ControlDataRec->data[y_ref_px] = m_ImageTargetPosPx.y;

	//ControlDataRec->data[XerrPx] = m_ImageTargetPosPx.x - (ControlDataRec->data[fish] + dx);
	//ControlDataRec->data[YerrPx] = m_ImageTargetPosPx.y - (ControlDataRec->data[XerrPx] + dx);

	// 3. computing new inputs (MPC controller) & PID Adding convert to voltage from velocity setting
	double NextVelInput_Stage_MPC[2] = {0, 0};
	double NextVelInput_Stage[2] = {0, 0};
	double w1_x = 0.0;
	double w1_y = 0.0;
	double w2_x = 0.0;
	double w2_y = 0.0;

	if (MPC_X.isInitiated()) {
		//w2_x = ILS200LM.I2T_X.L2weight();
		//w2_y = ILS200LM.I2T_Y.L2weight();
		w2_x = ILS200LM.I2T_X.L2weight_Err(fabs(ControlDataRec->data[Xerr]));
		w2_y = ILS200LM.I2T_X.L2weight_Err(fabs(ControlDataRec->data[Yerr]));
		MPC_X.Acc_limit = ILS200LM.I2T_X.AccelerationLimit;
		MPC_Y.Acc_limit = ILS200LM.I2T_Y.AccelerationLimit;
		Pos_Stage_OpenloopMPC[0] = MPC_X.getCurPos();
		Pos_Stage_OpenloopMPC[1] = MPC_Y.getCurPos();
		MPC_X.step_weight(FishPosEstimator.steps, FishPosEstimator.predX, w1_x, w2_x, &Pos_Stage_OpenloopMPCNext[0], &NextVelInput_Stage_MPC[0]);
		MPC_Y.step_weight(FishPosEstimator.steps, FishPosEstimator.predY, w1_y, w2_y, &Pos_Stage_OpenloopMPCNext[1], &NextVelInput_Stage_MPC[1]);
	}


	//4. PID controller
	double PIDcorrVelSet[2] = {0, 0};
	if (!isMPCEnabled) {
		MPC_X.set_pidvel_limit_max();
		MPC_Y.set_pidvel_limit_max();
		//PIDcorrVelSet[0] = MPC_X.step_PID(-ControlDataRec->data[Xerr], 0.0);
		//PIDcorrVelSet[1] = MPC_Y.step_PID(-ControlDataRec->data[Yerr], 0.0);
		PIDcorrVelSet[0] = MPC_X.step_PID(StgPosition[0], ControlDataRec->data[Xfish]);
		PIDcorrVelSet[1] = MPC_Y.step_PID(StgPosition[1], ControlDataRec->data[Yfish]);
		//MPC_X.PIDCntr.setCurrentPos(-ControlDataRec->data[Xerr]);
		//MPC_Y.PIDCntr.setCurrentPos(-ControlDataRec->data[Yerr]);
		//MPC_X.PIDCntr.setDesiredPos(0);
		//MPC_Y.PIDCntr.setDesiredPos(0);
	}
	else {
		PIDcorrVelSet[0] = MPC_X.step_PID(StgPosition[0], Pos_Stage_OpenloopMPC[0]);
		PIDcorrVelSet[1] = MPC_Y.step_PID(StgPosition[1], Pos_Stage_OpenloopMPC[1]);
		//MPC_X.PIDCntr.setCurrentPos(StgPosition[0]);
		//MPC_Y.PIDCntr.setCurrentPos(StgPosition[1]);
		//MPC_X.PIDCntr.setDesiredPos(Pos_Stage_OpenloopMPC[0]);
		//MPC_Y.PIDCntr.setDesiredPos(Pos_Stage_OpenloopMPC[1]);
	}
	//PIDcorrVelSet[0] = MPC_X.PIDCntr.computeNewInputPID();
	//PIDcorrVelSet[1] = MPC_Y.PIDCntr.computeNewInputPID();

	NextVelInput_Stage[0] = NextVelInput_Stage_MPC[0] + PIDcorrVelSet[0]; // MPC control only have a initial value in NextVelInput_Stage
	NextVelInput_Stage[1] = NextVelInput_Stage_MPC[1] + PIDcorrVelSet[1]; // MPC control only have a initial value in NextVelInput_Stage




	if (m_trk_bfollowingerrorEnable){
		//MPC_X.controlmanager.checkingfollowingerror(StgPosition, Pos_Stage_OpenloopMPC);
		MPC_X.controlmanager.b_followingerrorfailure = ILS200LM.I2T_X.currentlimitedmodeenabled;
		MPC_Y.controlmanager.b_followingerrorfailure = ILS200LM.I2T_Y.currentlimitedmodeenabled;
		//MPC_X.controlmanager.checkingfollowingerror_I2T(ILS200LM.I2T_X.I2T());
		//if (MPC_X.controlmanager.b_followingerrorfailure)
		//	MPC_X.Acc_limit = 1700;
		//else
		//	MPC_X.Acc_limit = MPC_X.Acc_limit_set;
	}

	//5. Voltage Setting
	double voltageInputStage[2] = {0, 0};
	voltageInputStage[0] = ILS200LM.Velocity2Volt_x(NextVelInput_Stage[0]);
	voltageInputStage[1] = ILS200LM.Velocity2Volt_y(NextVelInput_Stage[1]);

	// 6.!!! record message

	// MPC stage position
	ControlDataRec->data[XstgMPC] = Pos_Stage_OpenloopMPC[0];
	ControlDataRec->data[YstgMPC] = Pos_Stage_OpenloopMPC[1];
	int n_fishPred = FishPosEstimator.steps;
	for (int i = 0; i < FishPosEstimator.steps; i++) {
		ControlDataRec->data[Xfishpred0+2*i] = FishPosEstimator.predX[i];
		ControlDataRec->data[Yfishpred0+2*i] = FishPosEstimator.predY[i];
	}

	ControlDataRec->data[Xerr] = StgPosition[0] - (ControlDataRec->data[Xfish] + dx);
	ControlDataRec->data[Yerr] = StgPosition[1] - (ControlDataRec->data[Yfish] + dy);

	ControlDataRec->data[CntrDataIndex::HeadingFishMean] = FishPosEstimator.thfiltered.data[0];
	ControlDataRec->data[CntrDataIndex::MovingDirFish] = FishPosEstimator.thetamoving;
	ControlDataRec->data[CntrDataIndex::weight] = FishPosEstimator.weight.data[0];
	ControlDataRec->data[CntrDataIndex::Xfishproj] = FishPosEstimator.xproj.data[0];
	ControlDataRec->data[CntrDataIndex::Yfishproj] = FishPosEstimator.yproj.data[0];
	//ControlDataRec->data[CntrDataIndex::VfishPrl] = FishPosEstimator.vparallel.data[0];
	//ControlDataRec->data[CntrDataIndex::VfishPpd] = FishPosEstimator.vperpendicular;
	ControlDataRec->data[CntrDataIndex::VfishPrl] = FishPosEstimator.vparallel.mean();
	ControlDataRec->data[CntrDataIndex::VfishPpd] = FishPosEstimator.vppd.mean();
	ControlDataRec->data[CntrDataIndex::VfishPrlWeighted] = FishPosEstimator.weightedVel.data[0];
	ControlDataRec->data[CntrDataIndex::dVfishPrlWeighted] = FishPosEstimator.vslopepred;

	ControlDataRec->data[XstgInputVelMPC] = NextVelInput_Stage_MPC[0];
	ControlDataRec->data[YstgInputVelMPC] = NextVelInput_Stage_MPC[1];
	ControlDataRec->data[XstgPIDInputVel] = PIDcorrVelSet[0];
	ControlDataRec->data[YstgPIDInputVel] = PIDcorrVelSet[1];
	ControlDataRec->data[XstgInputVel] = NextVelInput_Stage[0];
	ControlDataRec->data[YstgInputVel] = NextVelInput_Stage[1];
	ControlDataRec->data[XstgInputVolt] = voltageInputStage[0];
	ControlDataRec->data[YstgInputVolt] = voltageInputStage[1];

	ControlDataRec->data[XstgPIDdesired] = MPC_X.PIDCntr.getDesiredPos();;
	ControlDataRec->data[YstgPIDdesired] = MPC_Y.PIDCntr.getDesiredPos();;
	ControlDataRec->data[XstgPIDtarget] = MPC_X.PIDCntr.getCurrentPos();;
	ControlDataRec->data[YstgPIDtarget] = MPC_Y.PIDCntr.getCurrentPos();;

	ControlDataRec->data[XstgI2T] = ILS200LM.I2T_X.I2T();
	ControlDataRec->data[YstgI2T] = ILS200LM.I2T_Y.I2T();
	ControlDataRec->data[Xstg_L2weight] = w2_x;
	ControlDataRec->data[Ystg_L2weight] = w2_y;




	if(FishPosEstimator.isFishMoving == true)
		ControlDataRec->isFishMoving = 1;
	else
		ControlDataRec->isFishMoving = 0;
	// -------------------End of Control ------------------------------
	return true;
}



//------------- NewDesign2015 -------------- //
void UI::MainWindow::teensyThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Teensy Thread");
	double fstarttime, fEndSeconds, fstarttime2 = 0;

	if (!XPSAVController.handler.open("\\\\.\\COM3")) {
		QtDisplayMessageError("XPS control teensy is not able to be connected (COM3)");
		abort();
	}

	while (g_bProcessRunning) {
		try {
			if (XPSAVController.b_stop) { // requested stop
				// stop teensy
				XPSAVController.sendVolInput(0, 0, 0);
				XPSAVController.stop();
				XPSAVController.b_start = false;
				XPSAVController.b_stop = false;
				// waiting the other threads are ready to start
				while (XPSAVController.b_start == false) { ; }
				// start teensy process
				XPSAVController.reset();
				XPSAVController.start();
				m_ThreadProcessTimeMills_TeensyCycle.clear();
				XPSAVController.msg_hsndshake.b_write = true;
				XPSAVController.b_start = false;
				TeensyOnTimeErrCount = 0;
				// read first dummy message
				XPSAVController.readMsg();
				XPSAVController.readMsg();
				XPSAVController.msg_hsndshake.b_write = false;

			}
			else { // started
				// waiting until MPC-thread read the messageMPCFrmIdxR
				while (XPSAVController.msg_hsndshake.b_write == false) { ; }
				// write
				if (!ILS200LM.b_emergencystop) { // checking the range error
					XPSAVController.sendVolInput(
						XPSAVController.msg_hsndshake.FrmNo, // there are two frame delay for all process
						XPSAVController.msg_hsndshake.CntrInput[0], XPSAVController.msg_hsndshake.CntrInput[1]);
				}
				else
					XPSAVController.sendVolInput(XPSAVController.msg_hsndshake.FrmNo, 0, 0);

				// read file
				XPSAVController.readMsg();

				// complete
				XPSAVController.msg_hsndshake.b_write = false;
				fstarttime = GetCycleCountSeconds(); if (fstarttime2 != 0) m_ThreadProcessTimeMills_TeensyCycle.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;

			}; // end of if (XPSAVController.b_stop)
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Teensy Thread: %s\n", e.what());
			fclose(ofp);
		}
	} // end of while
}


//------------- NewDesign2015 -------------- //
void UI::MainWindow::PiezoSignal_updateThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Piezo Thread");
	double fstarttime, fEndSeconds, fstarttime2 = 0;
	bool _ready_for_update = true;
	int gap = 1000000;
	for (int i = 0; i < 10; i++) {
		PiezoOffset[i] = 0;
		PiezoDirset[i] = 1;
	}
	PiezoOffset_concatenate = 0;
	uint64 PiezoUpdateCounter = 0;
	while (g_bProcessRunning) {
		try {
			if (ControllHardWare_2P.taskAOPiezoSignal) {
				uint64 cur = ControllHardWare_2P.getTotalSamplePiezoAO();
				uint64 target = ControllHardWare_2P.getWriteCurPosPiezoAO();
				bool flag = ControllHardWare_2P.Piezo_scanSingal.isUpdateReady(cur, target);
				//if (gap >= (int)(target - cur))
				//	gap = (int)(target - cur);
				//else
				//	gap = 0;
				//int stepcount = (ControllHardWare_2P.totalSamplecount / (ControllHardWare_2P.Piezo_scanSingal.steps_update)) % ControllHardWare_2P.Piezo_scanSingal.steps_per_sweep;
				//bool flag = ControllHardWare_2P.Piezo_scanSingal.isUpdateReady(stepcount);
				if (flag && (_ready_for_update)) { //  && (ControllHardWare_2P.Piezo_scanSingal.isdetaReset==false)
					fstarttime2 = GetCycleCountSeconds();
					//float64 _center[2] = { 200, 150 };
					//ControllHardWare_2P.Piezo_scanSingal.getSignal_sawteeth_update(_center[(int)((HamamatsuCamera_C11440.FrmNo % 200) / 100)]);
					//ControllHardWare_2P.Piezo_scanSingal.getSignal_sawteeth_update((int)(rand() % 300) + 50);
					//ControllHardWare_2P.Piezo_scanSingal.getSignal_sawteeth_update(((HamamatsuCamera_C11440.FrmNo/50)%10)*25 + 100);
					double _tempoffset = 0;
					double _tempDir = 1;
					PiezoUpdateCounter++;
					if (PiezoOffset_concatenate > 0) {
						_tempDir = PiezoDirset[PiezoUpdateCounter%PiezoOffset_concatenate] * ControllHardWare_2P.Piezo_scanSingal.step_um_gain2;
						_tempoffset = PiezoOffset[PiezoUpdateCounter%PiezoOffset_concatenate] * ControllHardWare_2P.Piezo_scanSingal.step_um_gain2;
					}
					ControllHardWare_2P.Piezo_scanSingal.step_um_gain = _tempDir;
					ControllHardWare_2P.Piezo_scanSingal.getSignal_sawteeth_update(ControllHardWare_2P.Piezo_scanSingal.center_um + _tempoffset);
					ControllHardWare_2P.Piezo_scanSingal.center_um -= _tempoffset;
					ControllHardWare_2P.update_PiezoAll(&ControllHardWare_2P.Piezo_scanSingal.signalbase);

					ControllHardWare_2P.Piezo_scanSingal.isdetaReset = true;

					fstarttime = GetCycleCountSeconds(); if (fstarttime2 != 0) m_ThreadProcessTimeMills_PiezoCycle.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;
					_ready_for_update = false;
				}
				if ((flag == false) && (_ready_for_update==false))
					_ready_for_update = true;
			}
			else {
				//m_ThreadProcessTimeMills_PiezoCycle.clear();
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Piezo Thread: %s\n", e.what());
			fclose(ofp);
		}
	} // end of while
}

//------------- NewDesign2015 -------------- //
void UI::MainWindow::IRLaserUpdate_updateThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("IR laser Thread");
	double fstarttime, fEndSeconds, fstarttime2 = 0;

	while (g_bProcessRunning) {
		try {
			ControllHardWare_2P.UpdateAO1(thermalcontroller.voltage); // UpdateAO23
			fstarttime = GetCycleCountSeconds(); if (fstarttime2 != 0) m_ThreadProcessTimeMills_IRlaser.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "IRLaser Thread: %s\n", e.what());
			fclose(ofp);
		}
	} // end of while
}

void UI::MainWindow::StagePositionReadingThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Position Reading thread");

	double fstarttime, fEndSeconds, fstarttime2 = 0;

	// connect XPS, and set events. If fail, exit program
	if (!Stage_ConnectOnly_QpushButtonClicked()) {
		QtDisplayMessageError("XPS controller (XPSQ8) is not able to be connected (192.168.254.254)");
		abort();
	}
	srand(time(NULL));
	// process start
	while (g_bProcessRunning) {
		try {
			fstarttime = GetCycleCountSeconds();
			if (ILS200LM.b_stop) {// requested stop
				// stop XPS
				ILS200LM.b_start = false;
				ILS200LM.b_stop = false;
				// waiting the other threads are ready to start
				while (ILS200LM.b_start == false) { ; }
				// Initialization
				ILS200LM.b_emergencystop = false;
				ILS200LM.msg_positionreading.clear();
				// initialization go home position, set AVC,
				if (!Stage_InitializationOnly_QpushButtonClicked()) QtDisplayMessageError("XPS controller initialization failure");
				Stage_ReadyOnly_QpushButtonClicked();
				Stage_ADCEnable();
				GatheringReset(ILS200LM.SocketID);
				ILS200LM.isHighLow = m_ui.checkBox_stage_HighLow->isChecked();
				ILS200LM.I2T_X.set(ScalingAcc_X, 0.004, m_ui.doubleSpinBox_Stage_I2T_minX->value(), m_ui.doubleSpinBox_Stage_I2T_maxX->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_X->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_X->value());
				ILS200LM.I2T_X.set_AccelerationLimit_normal(m_ui.spinBox_MPC_maxAcc_x->value());
				ILS200LM.I2T_X.AccelerationLimit = m_ui.spinBox_MPC_maxAcc_x->value();
				ILS200LM.I2T_Y.set(ScalingAcc_Y, 0.004, m_ui.doubleSpinBox_Stage_I2T_minY->value(), m_ui.doubleSpinBox_Stage_I2T_maxY->value(), m_ui.doubleSpinBox_Stage_I2T_minErr_Y->value(), m_ui.doubleSpinBox_Stage_I2T_maxErr_Y->value());
				ILS200LM.I2T_Y.set_AccelerationLimit_normal(m_ui.spinBox_MPC_maxAcc_y->value());
				ILS200LM.I2T_Y.AccelerationLimit = m_ui.spinBox_MPC_maxAcc_y->value();
				MPC_X.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_x->value(), m_ui.doubleSpinBox_Trk_b_x->value(), m_ui.doubleSpinBox_Trk_c_x->value());
				MPC_Y.PIDCntr.updatePIDparameter(m_ui.doubleSpinBox_Trk_a_y->value(), m_ui.doubleSpinBox_Trk_b_y->value(), m_ui.doubleSpinBox_Trk_c_y->value());

				ILS200LM.frmNo = 0;
				ILS200LM.CurrentStatus.DOToggle = -1;
				PositionGatheringInFailureCount = 0;
				if (ILS200LM.ControllerStatus != 48) QtDisplayMessageError("XPS ADC control mode enable failure");
				// send signal to start
				m_ThreadProcessTimeMills_StagePositionReadingCycle.clear();
				ILS200LM.b_start = false;
			}
			else {
				// wait until the current buffer is filled
				int readNo = 0;
				int maximumNo = 0;
				while (readNo == 0)
					GatheringCurrentNumberGet(ILS200LM.SocketID, &readNo, &maximumNo);
				ILS200LM.MyXyGroupGathering();
				GatheringReset(ILS200LM.SocketID); // reset buffer (readNo = 0)
				for (int i = 0; i < readNo; i++) {
					ILS200LM.CurrentStatusBuffer[1] = ILS200LM.CurrentStatusBuffer[0];//
					ILS200LM.CurrentStatusBuffer[0] = ILS200LM.CurrentStatus;//
				}



				fstarttime = GetCycleCountSeconds(); if (fstarttime2 != 0) m_ThreadProcessTimeMills_StagePositionReadingCycle.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;


				// checking the emergency stop
				if (m_trk_bboundaryerrorEnable)
					if (ILS200LM.isXpsDataInRange())
						ILS200LM.b_emergencystop = true;

				// other calculation (emergency stop and I2T value)
				ILS200LM.I2T_X.update(ILS200LM.CurrentStatus.CurrentAcceleration[0]);
				ILS200LM.I2T_Y.update(ILS200LM.CurrentStatus.CurrentAcceleration[1]);

				// input applied on AO0 and AO1

				// thermal experiment control
				ILS200LM.CurrentStatusBuffer[0].thermalcntrinput = thermalcontroller.getinput(ILS200LM.CurrentStatus.CurrentPosition, ILS200LM.frmNo);
				ILS200LM.CurrentStatusBuffer[0].temperature_circle_rev = thermalcontroller.circle_inv_counter - thermalcontroller.circle_inv_counter_cooldown;

				//ControllHardWare_2P.UpdateAO23(zaxis_controller.piezo_inputVolts, ILS200LM.CurrentStatus.thermalcntrinput);

				// offset (Every 250)
				if (m_trk_brespoffsetenable) {
					if (ILS200LM.frmNo % m_trk_brespoffsetrange_steps == 0) {// every 250
						if (m_trk_respoffsetfishpos[0] == 0.0) {
							int steps = int(m_trk_brespoffsetrange * 2000 + 1);
							m_trk_respoffsetfishpos[0] = ((double)(rand() % steps) / 1000) - m_trk_brespoffsetrange;
							m_trk_respoffsetfishpos[1] = ((double)(rand() % steps) / 1000) - m_trk_brespoffsetrange;
							manualInputVelStage[0] = manualInputVelStage_set[0];
							manualInputVelStage[1] = manualInputVelStage_set[1];

						}
						else {
							m_trk_respoffsetfishpos[0] = 0.0;
							m_trk_respoffsetfishpos[1] = 0.0;
							manualInputVelStage[0] = ILS200LM.Velocity2Volt_x(0);
							manualInputVelStage[1] = ILS200LM.Velocity2Volt_y(0);
							manualInputVelStage_set[0] = -manualInputVelStage_set[0];
							manualInputVelStage_set[1] = -manualInputVelStage_set[1];
						}
					}
				}
				ILS200LM.frmNo += readNo;
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Position Reading thread: %s\n", e.what());
			fclose(ofp);
		}
	}
}

void UI::MainWindow::NIDigSignal_updateThread() {
	SetThreadName("NI Signal Update Thread");
	double fstarttime, fEndSeconds, fstarttime2 = 0;
	// check whether there is dev1 exist??

	while (g_bProcessRunning) {
		try {
			if (ControllHardWare_2P.b_stop) { // stop requested
				// stop & clear the task
				ControllHardWare_2P.taskAIEnabled = m_ui.checkBox_LEDMonitor->isChecked();
				ControllHardWare_2P.stopNIalltSignal();
				ControllHardWare_2P.b_start = false;
				ControllHardWare_2P.b_stop = false;
				// wait the writing enable signal
				while (ControllHardWare_2P.b_start == false) { ; }
				// update parameter, generate unitAO, unitDO, nextsampleclock = nirate, and initialize signal
				ControllHardWare_2P.NIsampleclock = 1000 * 1000 * HamamatsuCamera_C11440.sample_per_us;
				ControllHardWare_2P.nexsamplecount_update = 0;
				ControllHardWare_2P.totalSamplecount = 0;
				ControllHardWare_2P.totalSamplecount_AO = 0;
				// unitDO generation
				if (ControllHardWare_2P.XPS_StageSignal) delete(ControllHardWare_2P.XPS_StageSignal);
				ControllHardWare_2P.XPS_StageSignal = new XPS_NI((uInt64)((int)m_ui.spinBox_DO_ExtTrigger_NIR->value() * 100 * HamamatsuCamera_C11440.sample_per_us));
				HamamatsuCamera_updateSignals_PG();// HamamatsuCamera_updateSignals();
				ControllHardWare_2P.hamamatsu_signal.deepcopy(&HamamatsuCamera_C11440.signal);
				m_ThreadProcessTimeMills_NIWritingCycle.clear();
				m_autosetDMDtime_ms.clear();
				// unit AO generation
				int _steps_per_sweep = (int)m_ui.spinBox_Piezo_trk_lockdown->value();
				float64 _step_um = (float64)m_ui.doubleSpinBox_Piezo_trk_dz->value();
				float64 _center_um = (float64)m_ui.spinBox_Piezo_trk_initz->value();
				ControllHardWare_2P.Piezo_scanSingal.getSignal_sawteeth_init(_steps_per_sweep, _step_um, _center_um );


				// generate AO, DO signal
				ControllHardWare_2P.XPS_StageSignal->cam_offset = (uint64)m_ui.spinBox_nircam_triggerDelay->value() * HamamatsuCamera_C11440.sample_per_us;
				HamamatsuCamera_C11440.globalExpWindowStart_us = (5000 + 330);// ControllHardWare_2P.hamamatsu_signal.n / 2;
				ControllHardWare_2P.initializeNIallSignal(HamamatsuCamera_C11440.globalExpWindowStart_us * HamamatsuCamera_C11440.sample_per_us);
				// update
				ControllHardWare_2P.generateNIallSignal(&ControllHardWare_2P.DO, &ControllHardWare_2P.Piezo_scanSingal.signalbase);
				// update start signal
				ControllHardWare_2P.b_start = true;
			}
			else {
				ControllHardWare_2P.getTotalSampleDigPortSignal();
				// check whether the NI signal need to be updated (update every second)
				//if (ControllHardWare_2P.Piezo_scanSingal.isUpdateReady(HamamatsuCamera_C11440.FrmNo)) {
				if ((ControllHardWare_2P.totalSamplecount > ControllHardWare_2P.nexsamplecount_update) &&
					(ControllHardWare_2P.totalSamplecount < 3 * ControllHardWare_2P.NIsampleclock)){
					ControllHardWare_2P.updateNIallSignal();
					fstarttime = GetCycleCountSeconds(); if (fstarttime2 != 0) m_ThreadProcessTimeMills_NIWritingCycle.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;
				}
				else {
					//ControllHardWare_2P.totalSamplecount
					if (ControllHardWare_2P.totalSamplecount > ControllHardWare_2P.offsetcount) {
						uInt64 sampleCount = ControllHardWare_2P.totalSamplecount;
						uInt64 rmdcount = (sampleCount - ControllHardWare_2P.offsetcount) % (HamamatsuCamera_C11440.signal_CamTrigger.n * 4);
						if (rmdcount < HamamatsuCamera_C11440.reset_timeInterval) { // less than 1 ms
							if (lc9000_autoreset) {
								lc9000.stop();
								lc9000.start();
								ControllHardWare_2P.getTotalSampleDigPortSignal();
								m_autosetDMDtime_ms.update((ControllHardWare_2P.totalSamplecount - sampleCount));
							}
						}
					}
				}
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "NI Signal Update Thread: %s\n", e.what());
			fclose(ofp);
		}
	} // end of while
}
void UI::MainWindow::NI_DO_ParametricSearch(void) {
	// Generate DO
	int nx = 0;
	double * _exp_us = NULL;
	double * _LED_offset_us = NULL;
	double * _freq = NULL;

	H5File* _file = new H5File("D:\\TrackingMicroscopeData\\_setting\\PG3Setting_150Hz.h5", H5F_ACC_RDONLY);
	_exp_us = importImpRestInd(_file, "exposure_us", &nx);
	if (m_ui.checkBox_ExptimeManual->isChecked()) _exp_us[0] = (double)m_ui.spinBox_Hamamatsu_ExpTime_us->value();
	_LED_offset_us = importImpRestInd(_file, "LED_offset_us", &nx);
	_freq = importImpRestInd(_file, "freq", &nx);


	uInt64 pos_offset_1st = (_exp_us[0] + _LED_offset_us[0])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 pos_offset_2nd = (_LED_offset_us[1])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 period = (uInt64)(1000000.0 / _freq[0])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 period_double = period * 2;
	uint64 exp0 = _exp_us[0] * HamamatsuCamera_C11440.sample_per_us;
	uint64 exp_serise[150];
	uint64 interval_100ns = m_ui.spinBox__NI_parametric_gap->value() / 100;


	if (_exp_us[0] * HamamatsuCamera_C11440.sample_per_us - interval_100ns * 75 < 0) {
		m_ui.spinBox__NI_parametric_gap->setValue((int)floor(_exp_us[0] * HamamatsuCamera_C11440.sample_per_us) / 75);
		return;
	}

	if (_exp_us[0] * HamamatsuCamera_C11440.sample_per_us + interval_100ns * 75 > 3000){
		m_ui.spinBox__NI_parametric_gap->setValue((int)floor(3000 - _exp_us[0] * HamamatsuCamera_C11440.sample_per_us) / 75);
		return;
	}

	for (uint64 i = 0; i < 150; i++) {
		exp_serise[i] = exp0 + interval_100ns*(i - 75);
	}
	ControllHardWare_2P.updateNIallSignal_fixed();

	uInt32 temp = (ControllHardWare_2P.DO.data[0] & 0x08) >> 3;
	uInt32 pos = 0;
	for (int i = 0; i < ControllHardWare_2P.DO.n; i++) {
		if (temp != (ControllHardWare_2P.DO.data[i] & 0x08) >> 3) {
			pos = i + pos_offset_1st; break;
			//temp = (ControllHardWare_2P.DO.data[i] & 0x08) >> 3;
		}
	}


	for (uint64 i = 0; i < 150; i++) {
		for (uint64 j = 0; j < 300 * HamamatsuCamera_C11440.sample_per_us; j++) {
			if (j < exp_serise[i])
				ControllHardWare_2P.DO.data[pos + pos_offset_2nd + period_double*i + j] &= 0xf7;
			else
				ControllHardWare_2P.DO.data[pos + pos_offset_2nd + period_double*i + j] |= 0x08;
			//if (j < exp0)
			//	ControllHardWare_2P.DO.data[pos - pos_offset_1st + period_double*i - j] &= 0xf7;
			//else
			//	ControllHardWare_2P.DO.data[pos - pos_offset_1st + period_double*i - j] |= 0x08;
		}
	}
	ControllHardWare_2P.updateDOAOSignal(&ControllHardWare_2P.DO, NULL, NULL);
}


void UI::MainWindow::NI_DO_update(void) {
	// Generate DO
	int nx = 0;
	double * _exp_us = NULL;
	double * _LED_offset_us = NULL;
	double * _freq = NULL;

	H5File* _file = new H5File("D:\\TrackingMicroscopeData\\_setting\\PG3Setting_150Hz.h5", H5F_ACC_RDONLY);
	_exp_us = importImpRestInd(_file, "exposure_us", &nx);
	if (m_ui.checkBox_ExptimeManual->isChecked()) _exp_us[0] = (double)m_ui.spinBox_Hamamatsu_ExpTime_us->value();
	_LED_offset_us = importImpRestInd(_file, "LED_offset_us", &nx);
	_freq = importImpRestInd(_file, "freq", &nx);


	uInt64 pos_offset_1st = (_exp_us[0] + _LED_offset_us[0])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 pos_offset_2nd = (_LED_offset_us[1])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 period = (uInt64)(1000000.0 / _freq[0])*HamamatsuCamera_C11440.sample_per_us;
	uInt64 period_double = period * 2;
	uint64 exp0 = _exp_us[0] * HamamatsuCamera_C11440.sample_per_us;
	uint64 exp_serise[150];
	int64 interval_100ns = (int)(m_ui.doubleSpinBox_doubleshot_exp_value_us->value() *HamamatsuCamera_C11440.sample_per_us);


	for (uint64 i = 0; i < 150; i++) {
		exp_serise[i] = exp0 + interval_100ns;
	}
	ControllHardWare_2P.updateNIallSignal_fixed();

	uInt32 temp = (ControllHardWare_2P.DO.data[0] & 0x08) >> 3;
	uInt32 pos = 0;
	for (int i = 0; i < ControllHardWare_2P.DO.n; i++) {
		if (temp != (ControllHardWare_2P.DO.data[i] & 0x08) >> 3) {
			pos = i + pos_offset_1st; break;
			//temp = (ControllHardWare_2P.DO.data[i] & 0x08) >> 3;
		}
	}


	for (uint64 i = 0; i < 150; i++) {
		for (uint64 j = 0; j < 300 * HamamatsuCamera_C11440.sample_per_us; j++) {
			if (j < exp_serise[i])
				ControllHardWare_2P.DO.data[pos + pos_offset_2nd + period_double*i + j] &= 0xf7;
			else
				ControllHardWare_2P.DO.data[pos + pos_offset_2nd + period_double*i + j] |= 0x08;
			//if (j < exp0)
			//	ControllHardWare_2P.DO.data[pos - pos_offset_1st + period_double*i - j] &= 0xf7;
			//else
			//	ControllHardWare_2P.DO.data[pos - pos_offset_1st + period_double*i - j] |= 0x08;
		}
	}
	ControllHardWare_2P.updateDOAOSignal(&ControllHardWare_2P.DO, NULL, NULL);
}


void UI::MainWindow::replaceNewNIDO(void) {
	if (ControllHardWare_2P.XPS_StageSignal_new) {
		QtDisplayMessageError("Fail to update the piezo scan signal");
		return;
	}
	HamamatsuCamera_updateSignals_PG();// HamamatsuCamera_updateSignals();

	ControllHardWare_2P.hamamatsu_signal_new = &ControllHardWare_2P.hamamatsu_signal;

}

void UI::MainWindow::replaceNewNIAO(void) {
	if (ControllHardWare_2P.Piezo_scanSingal_new) {
		QtDisplayMessageError("Fail to update the piezo scan signal");
		return;
	}
};

void UI::MainWindow::HamamatsuCamera_updateSignals(void) {
	uint _ROI = m_ui.spinBox_Hamamatsu_ROI->value();
	uint _freq = m_ui.doubleSpinBox_Hamamatsu_freq->value();
	uint _exp_us = m_ui.spinBox_Hamamatsu_ExpTime_us->value();

}

void UI::MainWindow::HamamatsuCamera_updateSignals_PG(void) {
	//uint _ROI = m_ui.spinBox_Hamamatsu_ROI->value();
	//uint _freq = m_ui.doubleSpinBox_Hamamatsu_freq->value();
	//uint _exp_us = m_ui.spinBox_Hamamatsu_ExpTime_us->value();
	//HamamatsuCamera_C11440.update(_ROI, _freq, _exp_us);
	//m_ui.doubleSpinBox_Hamamatsu_freqMax->setValue(HamamatsuCamera_C11440.freq_max);
	//m_ui.spinBox_Hamamatsu_ExpTimeMax_us->setValue(HamamatsuCamera_C11440.expmax_us);
	//HamamatsuCamera_C11440.HalfExposureEnable = m_ui.checkBox_Hamamatsu_ExpHiLowEnable->isChecked();
	//HamamatsuCamera_C11440.HalfExposureLocationToggle = m_ui.checkBox_Hamamatsu_ExpHiLowToggle->isChecked();
	// checking the maximum values
	//if (_ROI > Hamamatsu_ROIMax) _ROI = Hamamatsu_ROIMax;
	//if (_freq > HamamatsuCamera_C11440.freq_max) _freq = floor(HamamatsuCamera_C11440.freq_max);
	//if (_exp_us	> HamamatsuCamera_C11440.expmax_us) _exp_us = HamamatsuCamera_C11440.expmax_us;

	setting_update(true, true);
	HamamatsuCamera_C11440.buffer_nframe_sec = m_ui.spinBox_Hamamatsu_buffersize_sec->value();
	//m_ui.doubleSpinBox_DO_ExtTrigger_EPIFL_Hz->setValue(_freq);
	//m_ui.spinBox_DO_ExtTrigger_EPIFL->setValue((int)(10000.0 / _freq));
}

void UI::MainWindow::HamamatsuCamera_FLRCapture() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Hamamatsu Camera FLR capture thread");
	double fstarttime, fEndSeconds, fstarttime2 = 0;

	int devNum;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&devNum);
	cudaStatus = cudaSetDevice(GPU1);

	int version = 0;
	cudaDriverGetVersion(&version);
	cudaRuntimeGetVersion(&version);

	gpuErrchk(cudaFree(0));

	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.initialization(772);
	/**/


	char NAME[256];
	time_t t;
	struct tm * now;
	t = time(0); // get time now
	now = localtime(&t);
	sprintf(NAME, "D:\\TrackingMicroscopeData\\_Snapshot\\FLRSnapshot_x%3.5f_y%3.5f_%04d%02d%02d_%02d%02d%02d_%d.dat", Snapshot_Pos_mm.x, Snapshot_Pos_mm.y, now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, HamamatsuCamera_C11440.FrmNo);
	std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
	int save_endframe = 4;

	while (g_bProcessRunning) {
		try {
			if (HamamatsuCamera_C11440.b_stop) {
				//if (!HamamatsuCamera_C11440.hdcam)
				HamamatsuCamera_C11440.HCAM_stopCapture();
				ControllHardWare_2P.ClearAOForStage();
				HamamatsuCamera_C11440.b_start = false;
				HamamatsuCamera_C11440.b_stop = false;

				HamamatsuCamera_setgain();
				// waiting the other threads are ready to start
				while (HamamatsuCamera_C11440.b_start == false) { ; }
				// Initialization
				if (!HamamatsuCamera_C11440.hdcam)
					if (!HamamatsuCamera_C11440.HCAM_init_open())
						;// QtDisplayMessageError("Fail to open camera");
				//HamamatsuCamera_C11440.binning = m)
				if (HamamatsuCamera_C11440.hdcam) {
					if (HamamatsuCamera_C11440.HCAM_setting_200Hz()) {
						if (!HamamatsuCamera_C11440.HCAM_startCapture()) { // HamamatsuCamera_C11440.FrmNo
							;// QtDisplayMessageError("Fail to start camera");
						}
					}
				}


				if (HamamatsuCamera_C11440.pImg_double_buffer) {
					for (int temp_i = 0; temp_i < 5; temp_i++)
						free(HamamatsuCamera_C11440.pImg_double_buffer[temp_i]);
					free(HamamatsuCamera_C11440.pImg_double_buffer);
				};
				HamamatsuCamera_C11440.pImg_double_buffer = (uInt16 **)malloc(50 * sizeof(uInt16*));
				for (int temp_i = 0; temp_i < 50; temp_i++)
					HamamatsuCamera_C11440.pImg_double_buffer[temp_i] = (uInt16 *)malloc(HamamatsuCamera_C11440.img_width* HamamatsuCamera_C11440.img_height * sizeof(uInt16));
				HamamatsuCamera_C11440.pImg_double = NULL;



				zaxis_controller.updateparameters_imgSetup(HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height);
				HamamatsuCamera_C11440.FrmNo = 0;
				HamamatsuCamera_C11440.FrmNo_init = 0;
				HamamatsuCamera_C11440.FrmNo_Drop = 0;
				HamamatsuCamera_C11440.FrmNo_postproc = 0;
				m_ThreadProcessTimeMills_FluorescentImgCycle.clear();
				ControllHardWare_2P.InitiateAOForStage(); // temp AO

				// send signal to start
				HamamatsuCamera_C11440.b_start = false;
			}
			else {
				if (HamamatsuCamera_C11440.hdcam) {
					bool _suc = HamamatsuCamera_C11440.HCAM_retreiveBuffer();
					while (ControllHardWare_2P.totalSamplecount < ControllHardWare_2P.offsetcount - HamamatsuCamera_C11440.globalExpWindowStart_us) {
						HamamatsuCamera_C11440.FrmNo = 0;
						HamamatsuCamera_C11440.FrmNo_init = 0;
						HamamatsuCamera_C11440.FrmNo_postproc = 0;
						HamamatsuCamera_C11440.FrmNo_Drop = 0;
						_suc = HamamatsuCamera_C11440.HCAM_retreiveBuffer();
					}
					if (_suc) {
						// capture succeed
						fstarttime = GetCycleCountSeconds();
						if (fstarttime2 != 0)
							m_ThreadProcessTimeMills_FluorescentImgCycle.update((fstarttime - fstarttime2) * 1000.0);
						fstarttime2 = fstarttime;

						if (HamamatsuCamera_C11440.FrmNo > 10){
							int sum_mode = HamamatsuCamera_C11440.FrmNo % 2;
							if (sum_mode == 0) { //new
								int sum_img_id = ((HamamatsuCamera_C11440.FrmNo-1) / 2) % 50;
								int _n1 = HamamatsuCamera_C11440.buffer_indexNo[(HamamatsuCamera_C11440.FrmNo) % HamamatsuCamera_C11440.buffer_nframe];
								int _n2 = HamamatsuCamera_C11440.buffer_indexNo[(HamamatsuCamera_C11440.FrmNo-1) % HamamatsuCamera_C11440.buffer_nframe];
								ippsAdd_16u_Sfs(HamamatsuCamera_C11440.buffer[_n1],
									HamamatsuCamera_C11440.buffer[_n2],
									HamamatsuCamera_C11440.pImg_double_buffer[sum_img_id],
									HamamatsuCamera_C11440.img_px, 1);

								HamamatsuCamera_C11440.pImg_double = HamamatsuCamera_C11440.pImg_double_buffer[sum_img_id];
								if (m_Piezo_trk_enable && ControllHardWare_2P.Piezo_scanSingal.isready_tracking) {
									ControllHardWare_2P.Piezo_scanSingal.get_adjust_z_um((int)HamamatsuCamera_C11440.FrmNo, HamamatsuCamera_C11440.pImg);
								}
								else {
									if (ControllHardWare_2P.Piezo_scanSingal.isdetaReset == true) {
										ControllHardWare_2P.Piezo_scanSingal.cor_max = 0;
										ControllHardWare_2P.Piezo_scanSingal.isdetaReset = false;
									}
								}


							}
						}
						zaxis_controller.compute_mean(HamamatsuCamera_C11440.pImg, (int)HamamatsuCamera_C11440.FrmNo);
						qtgraph5_EPIintMean.update1D(200-ControllHardWare_2P.Piezo_scanSingal.delta_z_res, (int)HamamatsuCamera_C11440.FrmNo); // red
						qtgraph5_EPI_peizoZset.update1D(400-ControllHardWare_2P.Piezo_scanSingal.getValue((uint64)HamamatsuCamera_C11440.FrmNo), (int)HamamatsuCamera_C11440.FrmNo); // margenta
						qtgraph5_EPI_peizoZreal.update1D(-ControllHardWare_2P.Piezo_scanSingal.delta_z, (int)HamamatsuCamera_C11440.FrmNo); // result position
						//qtgraph5_EPI_peizoZreal.update1D(-ControllHardWare_2P.Piezo_scanSingal.id_result*8, (int)HamamatsuCamera_C11440.FrmNo); // result position


						HamamatsuCamera_C11440.FrmNo_postproc = HamamatsuCamera_C11440.FrmNo;
						if ((HamamatsuCamera_C11440.FrmNo <= save_endframe) && (pwriter_binary) && false) {
							if (HamamatsuCamera_C11440.FrmNo == 0)
								m_ui.lcdNumber_intensity_ratio_2->display((int)(ControllHardWare_2P.totalSamplecount - 10000000));
							int _n1 = HamamatsuCamera_C11440.buffer_indexNo[(HamamatsuCamera_C11440.FrmNo) % HamamatsuCamera_C11440.buffer_nframe];

							pwriter_binary->write((char *)(HamamatsuCamera_C11440.buffer[_n1]), HamamatsuCamera_C11440.ImageSizeByte);
							if (HamamatsuCamera_C11440.FrmNo == save_endframe) {
								pwriter_binary->close();
								delete(pwriter_binary);
								pwriter_binary = NULL;
							}
						}
					}
				}
				SleepThread(1);
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Hamamatsu Camera FLR capture thread: %s\n", e.what());
			fclose(ofp);
		}
	}
}

void UI::MainWindow::HamamatsuCamera_updateParameters(void) {

	uint _ROI = m_ui.spinBox_Hamamatsu_ROI->value();
	uint _freq = m_ui.doubleSpinBox_Hamamatsu_freq->value();
	uint _exp_us = m_ui.spinBox_Hamamatsu_ExpTime_us->value();
	HamamatsuCamera_C11440.update(_ROI, _freq, _exp_us);
	//m_ui.doubleSpinBox_Hamamatsu_freqMax->setValue(HamamatsuCamera_C11440.freq_max);
	//m_ui.spinBox_Hamamatsu_ExpTimeMax_us->setValue(HamamatsuCamera_C11440.expmax_us);
}

void UI::MainWindow::HamamatsuCamera_setgain(void){
	int _gain = m_ui.spinBox_Hamamatsu_Gain_dB->value();
	HamamatsuCamera_C11440.HCAM_settGain(_gain);
}

void UI::MainWindow::HamamatsuCamera_Snapshot(void) {
	if (HamamatsuCamera_C11440.pImg) {
		char NAME[256];
		time_t t;
		struct tm * now;
		t = time(0); // get time now
		now = localtime(&t);
		sprintf(NAME, "D:\\TrackingMicroscopeData\\_Snapshot\\FLRSnapshot_x%3.5f_y%3.5f_%04d%02d%02d_%02d%02d%02d.dat", Snapshot_Pos_mm.x, Snapshot_Pos_mm.y, now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
		if (!HamamatsuCamera_C11440.epiDisplay) {
			pwriter_binary->write((char *)(HamamatsuCamera_C11440.pImg), HamamatsuCamera_C11440.ImageSizeByte);
			pwriter_binary->close();
			delete(pwriter_binary);
		}
		else{
				pwriter_binary->write((char *)(HamamatsuCamera_C11440.pImg_double), HamamatsuCamera_C11440.ImageSizeByte);
				pwriter_binary->close();
				delete(pwriter_binary);
			}
		//cv::Mat snapFLR = cv::Mat(HamamatsuCamera_C11440.ROI, HamamatsuCamera_C11440.ROI, CV_16U, HamamatsuCamera_C11440.pImg);
		//cv::imwrite(NAME,snapFLR);
	}
	else {
		QtDisplayMessageWarning("capturing fluorescent image failed");
	}
	// ---- USE this function as a temperal snapshot
}



void UI::MainWindow::NIRProcessFrameSingleThread(void) {

	SetThreadName("Single thread MPC thread");


	bool flag;
	HANDLE hRealHandle = 0;
	DuplicateHandle(GetCurrentProcess(), // Source Process Handle.
		GetCurrentThread(),  // Source Handle to dup.
		GetCurrentProcess(), // Target Process Handle.
		&hRealHandle,        // Target Handle pointer.
		0,                   // Options flag.
		TRUE,                // Inheritable flag
		DUPLICATE_SAME_ACCESS);// Options
	HANDLE currentProcess = GetCurrentProcess();
	flag = SetPriorityClass(currentProcess, REALTIME_PRIORITY_CLASS);
	flag = SetThreadPriorityBoost(hRealHandle, true);//  THREAD_ALL_ACCESS
	flag = SetThreadPriority(hRealHandle, THREAD_PRIORITY_HIGHEST);

	double fstarttime, fEndSeconds, ftimeGPUMPC, fstarttime2 = 0;
	unsigned int FrmNoRemain = 0; // keep using this to get a position data
	// error flag

	// read frame number (3.2 ms readout delay compensation)
	int FrmOffset = 2; //Readoutdelay_Frm;
	unsigned int FrmNo_MPC = 0;
	zebrafishInfo DetectedZebraFish; // detected zebrafish
	RecBufferResults ResultPrevFrm; ResultPrevFrm.FrmNo = 0;

	cudaSetDevice(GPU0);
	while (g_bProcessRunning) { // process enable flags. g_bProcessRunning(whole program)
		try {
			if (NIRCameraPG.b_stop) { // initiation
				cudaSetDevice(GPU0);
				// turn off the camera
				NIRCameraPG.releaseCamera();
				NIRCameraPG.b_start = false;
				NIRCameraPG.b_stop = false;
				// waiting the other threads are ready to start
				while (NIRCameraPG.b_start == false) { ; }
				// buffer setting
				RecBuffer.allocateBuffer(m_ui.spinBox_MaxPreTriggerRecFrms->value());
				// initialization
				if (!StartCamera_GreyPoint()) {// include parameter update and start function
					//QtDisplayMessageWarning("Connecting NIR camera failed");
					abort();
				}
				RecBuffer.NIRSize.width = NIRCameraPG.ROIWidth;
				RecBuffer.NIRSize.height = NIRCameraPG.ROIHeight;
				// reset MPC
				MPCParmeterUpdate(); // reset MPC
				// gpu image proce init
				GlobalMap_Initialization(); // global map and image processing parameters update
				ImageProc_ParameterLoading();
				// fish estimator init
				ParameterUpdate_FishPosEstimator();
				globalFishPosAdjustor.threshold = m_ui.doubleSpinBox_fishpred_JitterThrehold->value();
				globalFishPosAdjustor.CurrentValue = Point2d(0.0, 0.0);

				// failure count reset
				ReadyOnTimeMPCInputFailureCount = 0;
				FishDetectionFailureCount = 0;
				// frame number reset
				FrmNo_MPC = 0;
				// reset time information
				m_ThreadProcessTimeMills_Framethread_Stg.clear();
				m_ThreadProcessTimeMills_Framethread_ImgProcMPC.clear();
				m_ThreadProcessTimeMills_Framethread_Total.clear();
				m_ThreadProcessTimeMills_FramethreadCycle.clear();
				// send start signal
				//m_ui.progressBar_NIREnabled->setValue(1);
				//m_queueResults.clear(); Sleep(0.1);
				NIRCameraPG.b_start = false;
			}
			else {
				if (ControllHardWare_2P.totalSamplecount > 0) {
					// initialize data set
					RecBufferResults ResultCurFrm;
					ResultCurFrm.FrmNo = FrmNo_MPC;
					ResultCurFrm.ControlDataRec.data[XstgInputVel] = manualInputVelStage[0];
					ResultCurFrm.ControlDataRec.data[YstgInputVel] = manualInputVelStage[1];
					// Background map should not be updated when it fails.
					bool isFishPosAvailable = false;
					bool isFishPosDetected = false;
					// Detection error of the fish (Area detected is different 50% than the set value)
					// MAJOR ERROR: APPLY 0 mm/s set velocity as a input when MPC is enabled.
					// Background map should not be update when it fails.

					// retreive buffer
					bool _suc = NIRCameraPG.grabImage();
					//while (ControllHardWare_2P.totalSamplecount < ControllHardWare_2P.offsetcount - ControllHardWare_2P.XPS_StageSignal->cam_offset) {
					//	NIRCameraPG.resetcount();
					//	_suc = NIRCameraPG.grabImage();
					//}
					uint64 A0 = ControllHardWare_2P.offsetcount - XPS_StageSignal.cam_offset * 10 + XPS_StageSignal.cycle_us * 5 + FrmNo_MPC * 40000;
					uInt64 A = ControllHardWare_2P.getTotalSampleDigPortSignal();
					m_ThreadProcessTimeMills_Framethread_Stg.update((double(A - A0) / 10 / 1000));
					fstarttime = GetCycleCountSeconds();
					if (fstarttime2 != 0) m_ThreadProcessTimeMills_FramethreadCycle.update((fstarttime - fstarttime2) * 1000); fstarttime2 = fstarttime;
					uInt16 * ImgSrc = NIRCameraPG.getDataPointer();

					// wait the wrote signal from XPS stage
					XPSGatheringInfo StgPos = ILS200LM.CurrentStatusBuffer[1];//
					bool b_stagePosVaild = StgPos.GatheringCompleted;

					zebrafishInfo DetectedZebraFish_new;
					// image processing (substraction and Image processing GPU)
					if (GlobalMap_NIR.subEnabled && !ILS200LM.bRangeError) {
						// 3-3-1. Image processing & and detect the fish
						int FishSize = 0;

						if (m_bconstanctdistantfromyolk_brainpos_enable) {
							vector<Point2d> fishPos = GPUImageProcess(ImgSrc, StgPos.CurrentPosition, &FishSize, ResultPrevFrm.ControlDataRec.isFishPosDetection);
							DetectedZebraFish_new = zebrafishInfo(SpotInfo(fishPos[1]), SpotInfo(fishPos[2]), SpotInfo(fishPos[0]));
						}
						else {
							vector<Point2d> fishPos = GPUImageProcess_centroid(ImgSrc, StgPos.CurrentPosition, &FishSize, ResultPrevFrm.ControlDataRec.isFishPosDetection);
							DetectedZebraFish_new = zebrafishInfo(fishPos, (double)FishSize);
							fEndSeconds = GetCycleCountSeconds();
							m_ThreadProcessTimeMills_Framethread_ImgProcMPC.update((fEndSeconds - fstarttime) * 1000.0);
						}


						if (true) // compute eye angles
						{
							// add the computing eye angle
							int x_eyeL = (int)(NIRFishPosDetector_GPU.imgSize.width - 1 - DetectedZebraFish_new.m_eyeLeft.center.y);
							int y_eyeL = (int)(DetectedZebraFish_new.m_eyeLeft.center.x);
							int x_eyeR = (int)(NIRFishPosDetector_GPU.imgSize.width - 1 - DetectedZebraFish_new.m_eyeRight.center.y);
							int y_eyeR = (int)(DetectedZebraFish_new.m_eyeRight.center.x);

							int r = 10;
							double M00[2] = { 0 };
							double M11[2] = { 0 };
							double M02[2] = { 0 };
							double M20[2] = { 0 };
							if ((x_eyeL > 2 * r) && (y_eyeL > 2 * r) && (x_eyeR > 2 * r) && (y_eyeR > 2 * r) &&
								(x_eyeL < NIRFishPosDetector_GPU.imgSize.height - 2 * r) && (y_eyeL < NIRFishPosDetector_GPU.imgSize.width - 2 * r) &&
								(x_eyeR < NIRFishPosDetector_GPU.imgSize.height - 2 * r) && (y_eyeR < NIRFishPosDetector_GPU.imgSize.width - 2 * r)) {

								for (int j = -r; j < r; j++) {
									int idL = (y_eyeL + j)*NIRFishPosDetector_GPU.imgSize.height + x_eyeL;
									int idR = (y_eyeR + j)*NIRFishPosDetector_GPU.imgSize.height + x_eyeR;
									for (int i = -r; i < r; i++) {
										double IL = double(ImgSrc[idL + i]);
										double IR = double(ImgSrc[idR + i]);
										M00[0] += IL; M00[1] += IR;
										M11[0] += IL*i*j; M11[1] += IR*i*j;
										M20[0] += IL*i*i; M20[1] += IR*i*i;
										M02[0] += IL*j*j; M02[1] += IR*j*j;
									}
								}
								double mu11[2] = { 0 };
								double mu20[2] = { 0 };
								double mu02[2] = { 0 };
								double th_Eye[2] = { 0 };
								for (int i = 0; i < 2; i++) {
									mu11[i] = M11[i] / M00[i];
									mu20[i] = M20[i] / M00[i];
									mu02[i] = M02[i] / M00[i];
									th_Eye[i] = atan2(2 * mu11[i], mu20[i] - mu02[i]) * 90.0 / 3.141592;
								}
								DetectedZebraFish_new.m_eyeLeft.angle = th_Eye[0] + 90;
								DetectedZebraFish_new.m_eyeRight.angle = th_Eye[1] + 90;
								DetectedZebraFish_new.m_yolk.angle = fabs(th_Eye[0] - th_Eye[1]);
								if (DetectedZebraFish_new.m_yolk.angle > 180)
									DetectedZebraFish_new.m_yolk.angle = fabs(th_Eye[0] - th_Eye[1] + 180);
								if (DetectedZebraFish_new.m_yolk.angle > 180)
									DetectedZebraFish_new.m_yolk.angle = fabs(th_Eye[0] - th_Eye[1] - 180);
							}
						}

						isFishPosDetected = isFishDetection(DetectedZebraFish_new); // determine whether the detected fish is okay or not
						isFishPosAvailable = true;
						// 3-3-2. Background update
						if (b_stagePosVaild) // when the stage position is vaild, update the global map only
							GlobalMap_NIR.cudaupdateBG(DetectedZebraFish_new.centerFish.x, DetectedZebraFish_new.centerFish.y, DetectedZebraFish_new.orienation, isFishPosDetected);

						// update zebrafish position based on the detection failure.
						if (isFishPosDetected) {
							DetectedZebraFish = DetectedZebraFish_new;
						}
						else {
							FishDetectionFailureCount++;
						}
					}

					// fish position prediction
					if (isFishPosAvailable) {
						getFishGlobalPosition(&DetectedZebraFish, StgPos.CurrentPosition, &ResultCurFrm.ControlDataRec, isFishPosDetected);
						if (!isFishPosDetected)
						{
							DetectedZebraFish.AreaSize = DetectedZebraFish_new.AreaSize;
							DetectedZebraFish.fitness_heading = DetectedZebraFish_new.fitness_heading;
							DetectedZebraFish.m_eyeLeft = DetectedZebraFish_new.m_eyeLeft;
							DetectedZebraFish.m_eyeRight = DetectedZebraFish_new.m_eyeRight;
							DetectedZebraFish.m_yolk = DetectedZebraFish_new.m_yolk;
						}
					}

					// waiting until MPC-thread read the message
					while (XPSAVController.msg_hsndshake.b_write == true) { ; }
					// update teensy report
					if (FrmNo_MPC > 0) {
						if (XPSAVController.msg_hsndshake.b_lateinput) {
							ReadyOnTimeMPCInputFailureCount++;
							MPC_X.replaceNextInput(0); // replace last input in MPC
							MPC_Y.replaceNextInput(0);
							ResultCurFrm.ControlDataRec.data[XstgMPC] = MPC_X.getCurPos(); // replace the current position of the stage in MPC
							ResultCurFrm.ControlDataRec.data[YstgMPC] = MPC_Y.getCurPos();
							ResultPrevFrm.ControlDataRec.data[XstgInputVel] = 0; // update result
							ResultPrevFrm.ControlDataRec.data[YstgInputVel] = 0;
						}
						ResultPrevFrm.ControlDataRec.isReadyOnTime = (unsigned int)XPSAVController.msg_hsndshake.b_lateinput;
						ResultPrevFrm.ControlDataRec.isReadyTeensyIndex= XPSAVController.msg_hsndshake.MPCFrmIdxR; // false report
						ResultPrevFrm.ControlDataRec.isReadyTeensyIndex_input = XPSAVController.msg_hsndshake.MPCFrmIdxW; // false report
						// update inputs (SAVE input) - send to save thread
						{
							std::lock_guard<std::mutex> lock(NIRImgCopy_mutex);
							m_queueResults.push_back(ResultPrevFrm);
						}
					}

					ResultCurFrm.ControlDataRec.data[XstgMPC] = StgPos.CurrentPosition[0];
					ResultCurFrm.ControlDataRec.data[YstgMPC] = StgPos.CurrentPosition[1];

					// Feedback control process (MPC process)
					if (isFishPosAvailable && MPC_X.controlmanager.istracking()) { // MPC & PID all included
						computeMPC_L(StgPos.CurrentPosition, &ResultCurFrm.ControlDataRec);
					}
					ftimeGPUMPC = GetCycleCountSeconds();
					//m_ThreadProcessTimeMills_Framethread_ImgProcMPC.update((ftimeGPUMPC - fEndSeconds) * 1000.0);
					// send voltage input to the teensy
					XPSAVController.msg_hsndshake.CntrInput[0] = ILS200LM.Velocity2Volt_x(ResultCurFrm.ControlDataRec.data[XstgInputVel]);
					XPSAVController.msg_hsndshake.CntrInput[1] = ILS200LM.Velocity2Volt_y(ResultCurFrm.ControlDataRec.data[YstgInputVel]);
					XPSAVController.msg_hsndshake.FrmNo = FrmNo_MPC + FrmOffset;
					XPSAVController.msg_hsndshake.b_write = true;

					// copy the results
					ResultCurFrm.StgPos = StgPos;
					ResultCurFrm.DetectedZebraFish = DetectedZebraFish;
					ResultCurFrm.ImgSrc = ImgSrc;
					ResultCurFrm.rows = NIRCameraPG.getrows();
					ResultCurFrm.cols = NIRCameraPG.getcols();
					ResultCurFrm.ControlDataRec.isTracking = (unsigned int)m_Tracking_EnableLive;
					ResultCurFrm.ControlDataRec.isFishPosDetection = (unsigned int)isFishPosDetected;
					ResultCurFrm.ControlDataRec.GlobalMapIdx = GlobalMap_NIR.BG_x0 + GlobalMap_NIR.BG_y0*GlobalMap_NIR.bg_width;
					ResultCurFrm.ControlDataRec.isPositionReadingSuccess = (unsigned int)StgPos.DOToggle;

					// interbout interval calculation
					double trav_dist = (interbout_pos - globalFishPosAdjustor.CurrentValue).norm();
					double MVA_fshVel = fishVelBuffer->update(trav_dist);
					if (fishVelBuffer->_stddev > interbout_std_threhold && trav_dist > 0.1) {
						// moving
						if (interbout_motion_stopped && interboutCount > 24) {
							interboutInterval->update((double)interboutCount);
							interboutCount = 0;
							interbout_motion_stopped = false;
							interbout_trvdst = trav_dist;
							interbout_pos = globalFishPosAdjustor.CurrentValue;
						}
						else {
							interboutCount++;
						}
					}
					else {
						interboutCount++;
						interbout_motion_stopped = true;
						//interboutInitial_pos = globalFishPosAdjustor.CurrentValue;
						interbout_trvdst = (interbout_pos - globalFishPosAdjustor.CurrentValue).norm();
					}
					ResultCurFrm.ControlDataRec.MovementIBICount = (double)interboutCount;


					ResultPrevFrm = ResultCurFrm; // keep data for saving after result report from teensy
					FrmNo_MPC++;

					fEndSeconds = GetCycleCountSeconds();
					m_ThreadProcessTimeMills_Framethread_Total.update((fEndSeconds - fstarttime) * 1000.0);
				}
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Single thread MPC thread: %s\n", e.what());
			fclose(ofp);
		}
	}
}


void UI::MainWindow::ResetAllprocess(void) { // update external trigger signal
	bool isinitialtrial = g_bNIRImageProcess;
	if (RecBuffer.b_recording || HamamatsuCamera_C11440.isRecoding) {
		QtDisplayMessageWarning("RECORDING NOW: STOP or FINISH recording first");
		return;
	}
	ILS200LM.b_start = false;
	HamamatsuCamera_C11440.b_start = false;
	NIRCameraPG.b_start = false;
	XPSAVController.b_start = false;
	ILS200LM.b_stop = true;
	HamamatsuCamera_C11440.b_stop = true;
	NIRCameraPG.b_stop = true;
	XPSAVController.b_stop = true;

	// turn on all thread at beginning
	if (!g_bProcessRunning) {
		g_bProcessRunning = true;
		g_bNIRImageProcess = true;
		// Position reading thread
		m_pThreadStagePositionReading.reset(new std::thread([this] { StagePositionReadingThread(); }));
		// FLR Image thread
		m_pThreadFLRcapture.reset(new std::thread([this] { HamamatsuCamera_FLRCapture(); }));
		// MPC thread
		m_pThreadNIRMPCtracking.reset(new std::thread([this] { NIRProcessFrameSingleThread(); }));
		// MPC DAC input
		m_pThreadStageTeensy.reset(new std::thread([this] { teensyThread(); }));
		// piezo buffer control
		m_pThreadPiezo.reset(new std::thread([this] { PiezoSignal_updateThread(); }));
		// IR laser control
		m_pThreadIRLaser.reset(new std::thread([this] { IRLaserUpdate_updateThread(); }));

		// FLR Image Rec thread
		m_pThreadFLRRecord.reset(new std::thread([this] { HamamatsuCamera_FLRRecord(); }));
		// MPC recording thread
		m_pThreadRecBuffer.reset(new std::thread([this] { MPCrecordingThread(); }));
		// recording thread
		m_pThreadRecBuffer_CopyResult.reset(new std::thread([this] { RecBufferCopyResultThread(); }));
		// NI AI save thread
		m_pThreadNIAIsave.reset(new std::thread([this] { NIAISingal_saveThread(); }));

		for (int i = 0; i < harddrivecount_FLRsave; ++i) {
			m_pThread_FLRRecord_binary[i].reset(new std::thread([this](int index){HamamatsuCamera_FLRRecord_binary(index); }, i));
		}


	}

	// wait until all device stopped.
	while (ILS200LM.b_stop || NIRCameraPG.b_stop || XPSAVController.b_stop || HamamatsuCamera_C11440.b_stop) { Sleep(100); }

	// stop NI signal
	ControllHardWare_2P.b_start = false;
	ControllHardWare_2P.b_stop = true;
	if (!isinitialtrial) {
		// NI signal generation thread
		m_pThreadNIDigSignal.reset(new std::thread([this] { NIDigSignal_updateThread(); }));
	}

	while (ControllHardWare_2P.b_stop) { Sleep(100); }

	insetmap_x[0] = -1;	insetmap_x[1] = -1;
	insetmap_y[0] = -1;	insetmap_y[1] = -1;
	QRgb value = qRgb(0, 0, 0);
	insetmap->fill(value);

	// start signal for all device
	ILS200LM.b_start = true;
	HamamatsuCamera_C11440.b_start = true;
	NIRCameraPG.b_start = true;
	XPSAVController.b_start = true;

	// wait until all device start.
	while (ILS200LM.b_start || NIRCameraPG.b_start || XPSAVController.b_start || HamamatsuCamera_C11440.b_start) { Sleep(100); }

	// start NI signal
	ControllHardWare_2P.b_start = true;
}

void UI::MainWindow::StartStopRecording(void) {
	if (!g_bProcessRunning) {
		QtDisplayMessageWarning("not all thread started yet");
		return;
	}
	if (HamamatsuCamera_C11440.hdcam) {
		if (!HamamatsuCamera_C11440.b_recording && !RecBuffer.b_recording) { // start recording
			//file setting
			RecBuffer.FLRSize.height = HamamatsuCamera_C11440.img_height;
			RecBuffer.FLRSize.width = HamamatsuCamera_C11440.img_width;
			bool NIRhdf5Enable = m_ui.checkBox_rec_nirimage_hdf5->isChecked();
			bool FLRhdf5Enable = m_ui.checkBox_rec_fluorescent_hdf5->isChecked();
			RecBuffer.InitializeRecording("e", NIRhdf5Enable, FLRhdf5Enable, harddrivecount_FLRsave); // Initialize file setting
			HamamatsuCamera_C11440.b_recstart = true;
			RecBuffer.b_recstart = true;
			m_ui.pushButton_recording->setText(QApplication::translate("MainWindow", "Recording Stop", 0));
		}
		else if (HamamatsuCamera_C11440.b_recording && RecBuffer.b_recording && !RecBuffer.b_recstop && !HamamatsuCamera_C11440.b_recstop) {
			// recording
			// request end
			RecBuffer.b_recstop = true;
			HamamatsuCamera_C11440.b_recstop = true;
			m_ui.pushButton_recording->setText(QApplication::translate("MainWindow", "Recording", 0));
		}
	}
	else {
		if (!RecBuffer.b_recording) { // start recording
			//file setting
			RecBuffer.FLRSize.height = HamamatsuCamera_C11440.img_height;
			RecBuffer.FLRSize.width = HamamatsuCamera_C11440.img_width;
			bool NIRhdf5Enable = m_ui.checkBox_rec_nirimage_hdf5->isChecked();
			bool FLRhdf5Enable = m_ui.checkBox_rec_nirimage_hdf5->isChecked();
			RecBuffer.InitializeRecording("e", NIRhdf5Enable, FLRhdf5Enable, harddrivecount_FLRsave); // Initialize file setting
			RecBuffer.b_recstart = true;
			m_ui.pushButton_recording->setText(QApplication::translate("MainWindow", "Recording Stop", 0));
		}
		else if (RecBuffer.b_recording && !RecBuffer.b_recstop) {
			// recording
			// request end
			RecBuffer.b_recstop = true;
			m_ui.pushButton_recording->setText(QApplication::translate("MainWindow", "Recording", 0));
		}
	}
}

void UI::MainWindow::HamamatsuCamera_FLRRecord(void) {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Recording FLR (Fluorescent) thread");
	double fstarttime, fEndSeconds = 0;

	while (g_bProcessRunning) {
		try {
			if (HamamatsuCamera_C11440.hdcam) {
				if (!HamamatsuCamera_C11440.b_recording) {
					// not recording
					if (HamamatsuCamera_C11440.b_recstart) {
						// initialize recording
						if (m_ui.checkBox_rec_nirimage_pretrigger->isChecked()) {
							int pretrigtime = m_ui.spinBox_rec_nirimage_pretriggertime_sec->value();
							if (HamamatsuCamera_C11440.FrmNo > (int32)HamamatsuCamera_C11440.freq * pretrigtime)
								HamamatsuCamera_C11440.recFrms_Start = HamamatsuCamera_C11440.FrmNo - (int32)HamamatsuCamera_C11440.freq * pretrigtime;
							else
								HamamatsuCamera_C11440.recFrms_Start = HamamatsuCamera_C11440.FrmNo;
						}
						else
							HamamatsuCamera_C11440.recFrms_Start = HamamatsuCamera_C11440.FrmNo;
						RecBuffer.FLRSize.height = HamamatsuCamera_C11440.img_height;
						RecBuffer.FLRSize.width = HamamatsuCamera_C11440.img_width;
						HamamatsuCamera_C11440.recFrms = HamamatsuCamera_C11440.recFrms_Start;
						HamamatsuCamera_C11440.recFrms_End = 0;
						HamamatsuCamera_C11440.b_recstart = false;
						HamamatsuCamera_C11440.b_recording = true;
						HamamatsuCamera_C11440.b_recstop = false;

						RecBuffer.FLRRecEnabled = m_ui.checkBox_rec_fluorescent_enable->isChecked();
					}
					else
						Sleep(1); // no recording
				}
				else {
					// while recording
					if ((HamamatsuCamera_C11440.recFrms < HamamatsuCamera_C11440.FrmNo - 2) // recording available (current frame)
						&& (HamamatsuCamera_C11440.recFrms <= HamamatsuCamera_C11440.recFrms_End || HamamatsuCamera_C11440.recFrms_End == 0))
					{ // recording
						if (RecBuffer.FLRRecEnabled) {
							fstarttime = GetCycleCountSeconds();
							if (RecBuffer.pFLR_writer_binary[0]) {
								//HamamatsuCamera_C11440.HCAM_REC_binary(RecBuffer.pFLR_writer_binary, HamamatsuCamera_C11440.recFrms);	// Recording data && update recFrm number
								FLRRecord_frameNo[(HamamatsuCamera_C11440.recFrms - HamamatsuCamera_C11440.recFrms_Start) % harddrivecount_FLRsave].push_back(HamamatsuCamera_C11440.recFrms);
								if (RecBuffer.DataRecEnabled) {
									FLRImageData * temp = zaxis_controller.getimgdatafromIdx((int)HamamatsuCamera_C11440.recFrms);
									//if (temp) {
										std::lock_guard<std::mutex> lock(Rec_mutex);
										RecBuffer.pData_writer->SingleData[R1_FLR_FrmNo]->write(&(HamamatsuCamera_C11440.recFrms));
										int _id = ControllHardWare_2P.Piezo_scanSingal.frmNo_to_Index(HamamatsuCamera_C11440.recFrms);
										double _dz_temp = (double)ControllHardWare_2P.Piezo_scanSingal.dz_result_archive[_id];
										RecBuffer.pData_writer->SingleData[R1_BraindZ_um]->write(&_dz_temp);
										_dz_temp = (double)ControllHardWare_2P.Piezo_scanSingal.id_result_archive[_id];
										RecBuffer.pData_writer->SingleData[R1_BrainZ_um]->write(&_dz_temp);
										_dz_temp = (double)ControllHardWare_2P.Piezo_scanSingal.getValue(HamamatsuCamera_C11440.recFrms);
										RecBuffer.pData_writer->SingleData[R1_PiezoPos_um]->write(&_dz_temp);
										//RecBuffer.pData_writer->SingleData[R1_PiezoCenter]->write(&);
										//RecBuffer.pData_writer->SingleData[R1_FLR_intmean]->write(&(temp->intensity_mean));
										//RecBuffer.pData_writer->SingleData[R1_FLR_intmax]->write(&(temp->intensity_max));
										//RecBuffer.pData_writer->SingleData[R1_FLR_intfilled]->write(&(temp->filled));
										//RecBuffer.pData_writer->PointData[R2_Piezo_Z]->write(temp->piezo_z);//
									//}
								}
							}
							if (RecBuffer.pFLR_writer) {
								std::lock_guard<std::mutex> lock(Rec_mutex);
								HamamatsuCamera_C11440.HCAM_HDF5REC(RecBuffer.pFLR_writer, HamamatsuCamera_C11440.recFrms);	// Recording data && update recFrm number
							}
							fEndSeconds = GetCycleCountSeconds();
							m_ThreadProcessTimeMills_FluorescentImgRecProc.update((fEndSeconds - fstarttime) * 1000.0);
						}

						int maxframe_left = FLRRecord_frameNo[0].size();
						for (int _i = 1; _i < harddrivecount_FLRsave; _i++) {
							if (FLRRecord_frameNo[_i].size() > maxframe_left)
								maxframe_left = FLRRecord_frameNo[_i].size();
						}
						HamamatsuCamera_C11440.recFrms_left = maxframe_left;

							//HamamatsuCamera_C11440.recFrms_left = HamamatsuCamera_C11440.FrmNo - HamamatsuCamera_C11440.recFrms;
						//else
							//HamamatsuCamera_C11440.recFrms_left = HamamatsuCamera_C11440.recFrms_End - HamamatsuCamera_C11440.recFrms;
						// checking end frame
						HamamatsuCamera_C11440.recFrms++;
					}
					if (HamamatsuCamera_C11440.b_recstop && HamamatsuCamera_C11440.recFrms_End == 0) {// checking the stop signal
						HamamatsuCamera_C11440.recFrms_End = HamamatsuCamera_C11440.FrmNo; // endframe setting
					}
					if (HamamatsuCamera_C11440.recFrms > HamamatsuCamera_C11440.recFrms_End
						&& HamamatsuCamera_C11440.recFrms_End > 0) { // checking the stop signal
							// terminate the recording
							// file close
						for (int drivecount = 0; drivecount < harddrivecount_FLRsave; drivecount++)
							while (FLRRecord_frameNo[drivecount].size() > 0)
								Sleep(100);
						{
							RecBuffer.ReleaseRecording_FLR();
							HamamatsuCamera_C11440.recFrms_left = HamamatsuCamera_C11440.recFrms_End - HamamatsuCamera_C11440.recFrms + 1;
							HamamatsuCamera_C11440.b_recstart = false;
							HamamatsuCamera_C11440.b_recstop = false;
							HamamatsuCamera_C11440.b_recording = false;
						}
					}

				}
			}
			else {
				// no camera detected
				Sleep(1);
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "Recording (Fluorescent) thread: %s\n", e.what());
			fclose(ofp);
		}
	}
}

void UI::MainWindow::HamamatsuCamera_FLRRecord_binary(int n) {
	char NAME[128];
	sprintf(NAME, "No %d FLR Record Hamamatus Cam Record parallel", n);
	std::string threadname = std::string(NAME);
	SetThreadName(threadname);
	while (g_bProcessRunning) {
		if (FLRRecord_frameNo[n].size() > 0) {
			HamamatsuCamera_C11440.HCAM_REC_binary_win32(&RecBuffer.pFLR_writer_binary[n], FLRRecord_frameNo[n][0]);	// Recording data && update recFrm number
			FLRRecord_frameNo[n].erase(FLRRecord_frameNo[n].begin());
		}
		else
			Sleep(1); // no recording
	}
	//
}

void UI::MainWindow::MPCrecordingThread() {
	// TODO: MODIFY "make one file and save in a different disc"
	SetThreadName("Recording (NIR and tracking) thread");
	double fstarttime, fEndSeconds = 0;
	while (g_bProcessRunning) {
		try {
			if (!RecBuffer.b_recording) {
				// not recording
				if (RecBuffer.b_recstart) {
					// initialize recording
					if (m_ui.checkBox_rec_nirimage_pretrigger->isChecked()) {
						int pretrigtime = m_ui.spinBox_rec_nirimage_pretriggertime_sec->value();
						int32 freq = (int32)(10000.0 / (double)m_ui.spinBox_DO_ExtTrigger_NIR->value());
						if (RecBuffer.FrmNo > freq * pretrigtime)
							RecBuffer.recFrms_Start = RecBuffer.FrmNo - freq * pretrigtime;
						else
							RecBuffer.recFrms_Start = RecBuffer.FrmNo;
					}
					else
						RecBuffer.recFrms_Start = RecBuffer.FrmNo;
					RecBuffer.ImageSizeByte_NIR = RecBuffer.NIRSize.height * RecBuffer.NIRSize.width * sizeof(uint16);

					RecBuffer.recFrms = RecBuffer.recFrms_Start;
					RecBuffer.recFrms_End = 0;
					RecBuffer.b_recstart = false;
					RecBuffer.b_recording = true;
					RecBuffer.b_recstop = false;
					m_ThreadProcessTimeMills_Framethread_nirImgRecProc.clear();
					RecBuffer.NIRRecEnabled = m_ui.checkBox_rec_nirimage_enable->isChecked(); // for original image
					RecBuffer.DataRecEnabled = m_ui.checkBox_rec_data_enable->isChecked(); // for original image
				}
				else
					Sleep(1); // no recording
			}
			else {
				// while recording
				if ((RecBuffer.recFrms < RecBuffer.FrmNo - 1) // recording available (current frame)
					&& (RecBuffer.recFrms <= RecBuffer.recFrms_End || RecBuffer.recFrms_End == 0))
				{ // recording
					std::lock_guard<std::mutex> lock(Rec_mutex);
					fstarttime = GetCycleCountSeconds();
					RecBuffer.RecordFrame(RecBuffer.recFrms);	// Recording data
					if (RecBuffer.recFrms_End == 0)
						RecBuffer.recFrms_left = RecBuffer.FrmNo - RecBuffer.recFrms;
					else
						RecBuffer.recFrms_left = RecBuffer.recFrms_End - RecBuffer.recFrms;
					// checking end frame
					RecBuffer.recFrms++;
					fEndSeconds = GetCycleCountSeconds();
					m_ThreadProcessTimeMills_Framethread_nirImgRecProc.update((fEndSeconds - fstarttime) * 1000.0);
				}
				if (RecBuffer.b_recstop && RecBuffer.recFrms_End == 0) {// checking the stop signal
					RecBuffer.recFrms_End = RecBuffer.FrmNo; // endframe setting
				}
				if (RecBuffer.recFrms > RecBuffer.recFrms_End
					&& RecBuffer.recFrms_End) { // checking the stop signal
					// terminate the recording
					// file close
					if (HamamatsuCamera_C11440.hdcam){
						while (RecBuffer.pFLR_writer_binary[0] != NULL)
							Sleep(1);
					}
					{
						std::lock_guard<std::mutex> lock(Rec_mutex);
						RecBuffer.ReleaseRecording_NIR();
						RecBuffer.recFrms_left = RecBuffer.recFrms_End - RecBuffer.recFrms + 1;
						RecBuffer.b_recstart = false;
						RecBuffer.b_recstop = false;
						RecBuffer.b_recording = false;
					}
				}
			}
		}
		catch (exception& e) {
			char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\threadError.txt";
			FILE  * ofp = fopen(outputFilename, "w");
			fprintf(ofp, "MPCrecordingThread: %s\n", e.what());
			fclose(ofp);
		}
	}
}
// ----------------------------- other global functions   ----------------------------- //
void UI::MainWindow::reset_failurecount(void) {
	NIRFrameDropFailureCount = 0;
	PositionGatheringInFailureCount = 0;
	ReadyOnTimeMPCInputFailureCount = 0;
	FishDetectionFailureCount = 0;
	TeensyFrmMissedCount = 0;
	TeensyOnTimeErrCount = 0;
}

// ----------------------------- QT functions   ----------------------------- //
void UI::MainWindow::GlobalMap_imgprocessenable_QCheckBoxtoggled(bool flag) {
	GlobalMap_NIR.subEnabled = flag;
	GlobalMap_NIR.UpdateEnabled = flag;
	if (m_ui.checkBox_nir_imgproc_enable->isChecked() != flag)
		m_ui.checkBox_nir_imgproc_enable->setChecked(flag);
};
void UI::MainWindow::GlobalMap_globalmapbuildingenable_QCheckBoxtoggled(bool flag) {
	GlobalMap_NIR.UpdateAllEnabled = flag;
	if (m_ui.checkBox_nir_globalmapbuilding_enable->isChecked() != flag)
		m_ui.checkBox_nir_globalmapbuilding_enable->setChecked(flag);
};
void UI::MainWindow::tracking_enable_QCheckBoxtoggled(bool flag) {
	MPC_X.controlmanager.enabletracking(flag);
	MPC_Y.controlmanager.enabletracking(flag);
	if (m_ui.checkBox_nir_tracking_enable->isChecked() != flag)
		m_ui.checkBox_nir_tracking_enable->setChecked(flag);
};
void UI::MainWindow::tracking_MPCenable_QCheckBoxtoggled(bool flag) {
	MPC_X.controlmanager.enableMPC(flag);
	MPC_Y.controlmanager.enableMPC(flag);
	if (m_ui.checkBox_nir_mpc_enable->isChecked() != flag)
		m_ui.checkBox_nir_mpc_enable->setChecked(flag);
};
void UI::MainWindow::tracking_fishpredictionenable_QCheckBoxtoggled(bool flag) {
	m_trk_bfishpospredictionenable = flag;
	if (m_ui.checkBox_tracking_fishposestimation_enable->isChecked() != flag)
		m_ui.checkBox_tracking_fishposestimation_enable->setChecked(flag);
};
void UI::MainWindow::followingerror_enable_QCheckBoxtoggled(bool flag) {
	m_trk_bfollowingerrorEnable = flag;
	if (m_ui.checkBox_stage_followingerror_enable->isChecked() != flag)
		m_ui.checkBox_stage_followingerror_enable->setChecked(flag);
};
void UI::MainWindow::bounderror_enable_QCheckBoxtoggled(bool flag) {
	m_trk_bboundaryerrorEnable = flag;
	if (m_ui.checkBox_stage_stageboundaryerror_enable->isChecked() != flag)
		m_ui.checkBox_stage_stageboundaryerror_enable->setChecked(flag);
};

void UI::MainWindow::responseEstimation_RandomoffsetEnable_QCheckBoxtoggled(bool flag) {
	m_trk_brespoffsetenable = flag;
	m_trk_brespoffsetrange = (double)(m_ui.doubleSpinBox_resp_offsetrange->value());
	m_trk_brespoffsetrange_steps = (double)(m_ui.spinBox_resp_offsetrange->value());
	m_trk_respoffsetfishpos_ref[0] = globalFishPosAdjustor.CurrentValue.x;
	m_trk_respoffsetfishpos_ref[1] = globalFishPosAdjustor.CurrentValue.y;
	if (m_ui.checkBox_resp_offsetEnable->isChecked() != flag)
		m_ui.checkBox_resp_offsetEnable->setChecked(flag);
};
void UI::MainWindow::jitterfilter_Enable_QCheckBoxtoggled(bool flag) {
	m_trk_bjitterenable = flag;
	if (m_ui.checkBox_tracking_fishposeJitter_enable->isChecked() != flag)
		m_ui.checkBox_tracking_fishposeJitter_enable->setChecked(flag);
};
void UI::MainWindow::constanctdistantfromyolk_brainpos_Enable_QCheckBoxtoggled(bool flag) {
	m_bconstanctdistantfromyolk_brainpos_enable = flag;
	m_constanctdistantfromyolk_brainpos = (double)(m_ui.spinBox_tracking_fishpose_distancefromyolk_px->value());
	if (m_ui.checkBox_tracking_fishpose_distancefromyolk->isChecked() != flag)
		m_ui.checkBox_tracking_fishpose_distancefromyolk->setChecked(flag);
};
void UI::MainWindow::Piezo_trk_Enable_QCheckBoxtoggled(bool flag) {
	m_Piezo_trk_enable = flag;
	piezoparameterupdate_buttonclicked();
	if (m_ui.checkBox_Piezo_trk_enable->isChecked() != flag)
		m_ui.checkBox_Piezo_trk_enable->setChecked(flag);
};

void UI::MainWindow::piezoparameterupdate_buttonclicked(void) {


};
void UI::MainWindow::display_Enable_QCheckBoxtoggled(bool flag) {
	disp_data1_enable[0] = m_ui.checkBox_display_show_error->isChecked();
	disp_data1_enable[1] = m_ui.checkBox_display_collect_error->isChecked();
	//disp_data4_enable[0] = m_ui.checkBox_display_show_fish->isChecked();
	//disp_data4_enable[1] = m_ui.checkBox_display_collect_fish->isChecked();
	//disp_data5_enable[0] = m_ui.checkBox_display_show_stage->isChecked();
	//disp_data5_enable[1] = m_ui.checkBox_display_collect_stage->isChecked();
	disp_data2_enable[0] = m_ui.checkBox_display_show_piezo->isChecked();
	disp_data2_enable[1] = m_ui.checkBox_display_collect_piezo->isChecked();
	disp_data3_enable[0] = false;// m_ui.checkBox_Piezo_trk_enable->isChecked();
	disp_data3_enable[1] = false;//m_ui.checkBox_Piezo_trk_enable->isChecked();

	qtgraph0.init(5, m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx, Qt::green);
	qtgraph0.enabledraw = true;
	qtgraph0.enableupdate = true;
	qtgraph0.update(15.0, 15.0, 0);
	qtgraph0.update(15.0, -15.0, 1);
	qtgraph0.update(-15.0, -15.0, 2);
	qtgraph0.update(-15.0, 15.0, 3);
	qtgraph0.update(15.0, 15.0, 4);

	qtgraph0_thermal.init(13, m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx, Qt::red);
	qtgraph0_thermal.enabledraw = true;
	qtgraph0_thermal.enableupdate = true;

	qtgraph0_gmap.init(250 * 2, m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / insetmap_pxdist, 1 / insetmap_pxdist, Qt::darkMagenta);
	qtgraph0_gmap.enabledraw = true;
	qtgraph0_gmap.enableupdate = true;
	double grid_gmap[7];
	for (int i = 0; i < 7; i++)
		grid_gmap[i] = i * 10 - 30;
	qtgraph0_gmap.setgrid(7, grid_gmap, 7, grid_gmap, QColor(100, 100, 100));

	qtgraph1.init(250 * 2, m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx, Qt::yellow);
	qtgraph1.enabledraw = m_ui.checkBox_display_show_stage->isChecked();
	qtgraph1.enableupdate = m_ui.checkBox_display_collect_stage->isChecked();

	qtgraph2.init(250 * 2, m_ImageTargetPosPx.x, m_ImageTargetPosPx.y, 1 / GlobalMap_NIR.PxDist_mmppx, 1 / GlobalMap_NIR.PxDist_mmppx, Qt::magenta);
	qtgraph2.enabledraw = m_ui.checkBox_display_show_fish->isChecked();
	qtgraph2.enableupdate = m_ui.checkBox_display_collect_fish->isChecked();

	qtgraph3.init1D(250 * 2, 120, 1.0, 100.0, Qt::cyan);
	qtgraph3.enabledraw = m_ui.checkBox_display_show_error->isChecked();
	qtgraph3.enableupdate = m_ui.checkBox_display_collect_error->isChecked();
	double gridy[] = { 0, 0.1, 0.2, 0.3, 1 };
	double gridx[21];
	for (int i = 0; i < 21; i++)
		gridx[i] = i * 25;
	qtgraph3.setgrid(21, gridx, 5, gridy, QColor(100,100,100));

	double FLR_scale = 0.4;
	qtgraph4.scale = FLR_scale;
	qtgraph4.init1D(250 * 2, 120, 1.0, 100.0 / (double)(m_ui.spinBox_Hamamatsu_IntMaxSet->value()), Qt::red);
	qtgraph4.enabledraw = m_ui.checkBox_display_show_piezo->isChecked();
	qtgraph4.enableupdate = m_ui.checkBox_display_collect_piezo->isChecked();

	qtgraph5_EPIintMean.scale = FLR_scale;
	qtgraph5_EPIintMean.init(100 * 2, 200, 1500, 8.0, 1.0, Qt::red);
	qtgraph5_EPIintMean.enabledraw = m_ui.checkBox_display_show_stage->isChecked();
	qtgraph5_EPIintMean.enableupdate = m_ui.checkBox_display_collect_stage->isChecked();

	qtgraph5_EPI_peizoZset.scale = FLR_scale;
	qtgraph5_EPI_peizoZset.init(100 * 2, 200, 1500, 8.0, 1, Qt::magenta);
	qtgraph5_EPI_peizoZset.enabledraw = m_ui.checkBox_display_show_stage->isChecked();
	qtgraph5_EPI_peizoZset.enableupdate = m_ui.checkBox_display_collect_stage->isChecked();
	double gridy2[] = { 0, 100, 200, 300, 400.0 };
	double gridx2[21];
	for (int i = 0; i < 21; i++)
		gridx2[i] = i * 10;
	qtgraph5_EPI_peizoZset.setgrid(21, gridx2, 5, gridy2, QColor(100, 100, 100));

	qtgraph5_EPI_peizoZreal.scale = FLR_scale;
	qtgraph5_EPI_peizoZreal.init(100 * 2, 200, 1500, 8.0, 1, Qt::green);
	qtgraph5_EPI_peizoZreal.enabledraw = m_ui.checkBox_display_show_stage->isChecked();
	qtgraph5_EPI_peizoZreal.enableupdate = m_ui.checkBox_display_collect_stage->isChecked();

};

void UI::MainWindow::fakefishtest_Enable_QCheckBoxtoggled(bool flag) {
	if (fakefishposition_x) free(fakefishposition_x);
	if (fakefishposition_y) free(fakefishposition_y);
	if (fakefishposition_th) free(fakefishposition_th);
	fakefish_idx = 0;
	fakefish_idxmax = 0;
	double * _Angle = NULL;
	double * _PXDIST = NULL;
	H5File* _file = new H5File("D:\\TrackingMicroscopeData\\_setting\\testfishmovement.h5", H5F_ACC_RDONLY);
	fakefishposition_x = importImpRestInd(_file, "x", &fakefish_idxmax);
	fakefishposition_y = importImpRestInd(_file, "y", &fakefish_idxmax);
	fakefishposition_th = importImpRestInd(_file, "th", &fakefish_idxmax);
	for (int i = 0; i < fakefish_idxmax; i++) {
		fakefishposition_x[i] += ILS200LM.CurrentStatus.CurrentPosition[0];
		fakefishposition_y[i] += ILS200LM.CurrentStatus.CurrentPosition[1];
	}
	m_trk_bfakefishposition = flag;
	if (m_ui.checkBox_resp_fakefishEnable->isChecked() != flag)
		m_ui.checkBox_resp_fakefishEnable->setChecked(flag);
}

void UI::MainWindow::ThermalControl_update(void) {
	// update parameters
	thermalcontroller.mode = false; // stop motion
	thermalcontroller.enable = m_ui.checkBox_ThermalControl_enable->isChecked();
	thermalcontroller.voltage_max = m_ui.doubleSpinBox_ThermalControl_inputmax->value();
	thermalcontroller.voltage_min = m_ui.doubleSpinBox_ThermalControl_inputmin->value();
	thermalcontroller.cycle_sec = m_ui.doubleSpinBox_ThermalControl_cycle_sec->value();
	thermalcontroller.on_sec = m_ui.doubleSpinBox_ThermalControl_ontime_sec->value();
	thermalcontroller.dx_mm = m_ui.doubleSpinBox_ThermalControl_dx_mm->value();
	thermalcontroller.dy_mm = m_ui.doubleSpinBox_ThermalControl_dy_mm->value();
	thermalcontroller.cx_mm = m_ui.doubleSpinBox_Map_cx_mm->value();
	thermalcontroller.cy_mm = m_ui.doubleSpinBox_Map_cy_mm->value();
	thermalcontroller.Osc_Amp = m_ui.doubleSpinBox_ThermalControl_OscAmp->value();
	thermalcontroller.Osc_Period = m_ui.doubleSpinBox_ThermalControl_OscPeriod->value();
	thermalcontroller.Osc_enable = m_ui.checkBox_ThermalControl_oscillation_enable->isChecked();
	thermalcontroller.circle_inv_enable = m_ui.checkBox_ThermalControl_circle_rev_enable->isChecked();
	thermalcontroller.circle_inv_counter_max = m_ui.doubleSpinBox_ThermalControl_circle_rev_period->value() * 250;
	thermalcontroller.circle_inv_counter_cooldown = m_ui.doubleSpinBox_ThermalControl_circle_rev_ON->value() * 250;
	thermalcontroller.circle_inv_rmin = m_ui.doubleSpinBox_ThermalControl_circle_rev_Rmin->value();


	// validate parameters
	thermalcontroller.updateparameters();
	// update values
	m_ui.doubleSpinBox_ThermalControl_inputmax->setValue(thermalcontroller.voltage_max);
	m_ui.doubleSpinBox_ThermalControl_inputmin->setValue(thermalcontroller.voltage_min);
	m_ui.doubleSpinBox_ThermalControl_ontime_sec->setValue(thermalcontroller.on_sec);
	m_ui.spinBox_ThermalControl_cycle_frm->setValue(thermalcontroller.frm_total);
	// check enable or diable
	thermalcontroller.mode = m_ui.comboBox_ThermalControl_mode->currentIndex();
	m_ui.progressBar_ThermalControl_currentVolts->setMaximum(int(thermalcontroller.voltage_max * 100));
	m_ui.progressBar_ThermalControl_currentVolts->setMinimum(int(thermalcontroller.voltage_min * 100));
	m_ui.progressBar_ThermalControl_circle_rev_on->setMaximum(int(thermalcontroller.circle_inv_counter_cooldown * 100));
	m_ui.progressBar_ThermalControl_circle_rev_off->setMaximum(int((thermalcontroller.circle_inv_counter_max - thermalcontroller.circle_inv_counter_cooldown) * 100));
	// map update

	double x_size = m_ui.doubleSpinBox_Map_imsize_x->value();
	double y_size = m_ui.doubleSpinBox_Map_imsize_y->value();
	double offset_x = -m_ui.doubleSpinBox_Map_cx_mm->value();
	double offset_y = -m_ui.doubleSpinBox_Map_cy_mm->value();
	qtgraph0.update(x_size / 2 - offset_x, y_size / 2 - offset_y, 0);
	qtgraph0.update(x_size / 2 - offset_x, -y_size / 2 - offset_y, 1);
	qtgraph0.update(-x_size / 2 - offset_x, -y_size / 2 - offset_y, 2);
	qtgraph0.update(-x_size / 2 - offset_x, y_size / 2 - offset_y, 3);
	qtgraph0.update(x_size / 2 - offset_x, (y_size / 2 - offset_y), 4);
	double box_out[2] = { x_size / 2, y_size / 2 };
	double box_in[2] = { thermalcontroller.dx_mm / 2, thermalcontroller.dy_mm / 2 };
	qtgraph0_thermal.update(box_out[0] - offset_x, box_in[0] - offset_y, 0);
	qtgraph0_thermal.update(box_out[0] - offset_x, -box_in[0] - offset_y, 1);
	qtgraph0_thermal.update(box_in[0] - offset_x, -box_in[0] - offset_y, 2);
	qtgraph0_thermal.update(box_in[0] - offset_x, -box_out[0] - offset_y, 3);
	qtgraph0_thermal.update(-box_in[0] - offset_x, -box_out[0] - offset_y, 4);
	qtgraph0_thermal.update(-box_in[0] - offset_x, -box_in[0] - offset_y, 5);
	qtgraph0_thermal.update(-box_out[0] - offset_x, -box_in[0] - offset_y, 6);
	qtgraph0_thermal.update(-box_out[0] - offset_x, box_in[0] - offset_y, 7);
	qtgraph0_thermal.update(-box_in[0] - offset_x, box_in[0] - offset_y, 8);
	qtgraph0_thermal.update(-box_in[0] - offset_x, box_out[0] - offset_y, 9);
	qtgraph0_thermal.update(box_in[0] - offset_x, box_out[0] - offset_y, 10);
	qtgraph0_thermal.update(box_in[0] - offset_x, box_in[0] - offset_y, 11);
	qtgraph0_thermal.update(box_out[0] - offset_x, box_in[0] - offset_y, 12);


	double grid_gmap_x[4];
	double grid_gmap_y[4];
	for (int i = 0; i < 4; i++) {
		grid_gmap_x[i] = i * 10 - 15 + offset_x;
		grid_gmap_y[i] = i * 10 - 15 - offset_y;
	}
	qtgraph0_gmap.setgrid(4, grid_gmap_x, 4, grid_gmap_y, QColor(100, 100, 100));

}

void UI::MainWindow::ThermalControl_loadmap(void) {
	// load hdf5 map
	QString mapfilename = QFileDialog::getOpenFileName(this, "Select image h5.file", "D:\\TrackingMicroscopeData\\_ExpControl\\*.h5", tr("Videos (*.h5)"));
	if (mapfilename.isEmpty()) return;

	HDF5ImageReader thermalmap(mapfilename.toUtf8().constData());
	if (thermalmap.n_images() > 0) {
		thermalcontroller.imgsize_x = thermalmap.image_width();
		thermalcontroller.imgsize_y = thermalmap.image_height();
		if (thermalcontroller.map) free(thermalcontroller.map);
		thermalcontroller.map = (uint8 *)malloc(thermalcontroller.imgsize_x * thermalcontroller.imgsize_y * sizeof(uint8));
		thermalmap.read(0, thermalcontroller.map);
		imSize displayimsize(thermalcontroller.imgsize_x, thermalcontroller.imgsize_y);
		img_uint8 displayimage(thermalcontroller.map, displayimsize);
		auto pImage = toQImage(&displayimage);
		m_pImageR1->SetImage(pImage);
		//m_ui.spinBox_ThermalControl_imsize_x->setValue(thermalcontroller.imgsize_x);
		//m_ui.spinBox_ThermalControl_imsize_y->setValue(thermalcontroller.imgsize_y);
		thermalcontroller.cx_px = thermalcontroller.imgsize_x / 2;
		thermalcontroller.cy_px = thermalcontroller.imgsize_y / 2;
		//m_ui.doubleSpinBox_Map_cx_px->setValue(thermalcontroller.cx_px);
		//m_ui.doubleSpinBox_Map_cy_px->setValue(thermalcontroller.cy_px);
	}
	ThermalControl_update();
}


void UI::MainWindow::replayer_loadfishpos(void) {
	QString fishposdata_name = QFileDialog::getOpenFileName(this, "Select data file", +"d:\\TrackingMicroscopeData\\_ExpControl\\S*.h5", tr("Data file(*.h5)"));
	if (fishposdata_name.isEmpty()) {
		return;
	}
	m_ui.lineEdit_replay_fishpos_filename->setText(fishposdata_name);

	int n = 0;
	if (replayer_fishpos) free(replayer_fishpos);

	H5File* _file = new H5File(fishposdata_name.toUtf8().constData(), H5F_ACC_RDONLY);
	replayer_fishpos = importImpRestInd(_file, "FishPos", &n);
	replayer_fishpos_OrientationDeg = importImpRestInd(_file, "FishPos_Orientation_Deg", &n);
	replayer_fishpos_totalfrm = n;
	replayer_fishpos_curfrm = 0;

	/*double * replayer_isfishdetected = importImpRestInd(_file, "isFishPosDetection", &n);
	double temp_x = replayer_fishpos[0];
	double temp_y = replayer_fishpos[1];
	double temp_th = replayer_fishpos_OrientationDeg[0];
	for (int i = 0; i < n; i++){
		if (replayer_isfishdetected[i] == 1) {
			temp_x = replayer_fishpos[2 * (i - 1)];
			temp_y = replayer_fishpos[2 * (i - 1) + 1];
			temp_th = replayer_fishpos_OrientationDeg[i];
		}
		else {
			replayer_fishpos[2 * (i - 1)] = temp_x;
			replayer_fishpos[2 * (i - 1) + 1] = temp_y;
			replayer_fishpos_OrientationDeg[i] = temp_th;
		}
	};
	free(replayer_fishpos_OrientationDeg); */

	m_ui.lcdNumber_replay_totalfrm->display(replayer_fishpos_totalfrm);
	m_ui.lcdNumber_replay_curfrm->display(replayer_fishpos_curfrm);
	m_ui.lcdNumber_replay_fishpos_x->display(replayer_fishpos[0]);
	m_ui.lcdNumber_replay_fishpos_y->display(replayer_fishpos[1]);
	delete _file;
}

void UI::MainWindow::replayer_fishpos_enable_QCheckBoxtoggled(bool flag) {
	replayer_fishpos_enable = flag;
	if (m_ui.checkBox_replay_fishpos_enable->isChecked() != flag)
		m_ui.checkBox_replay_fishpos_enable->setChecked(flag);
};

void UI::MainWindow::replayer_fishpos_perfectprediction_QCheckBoxtoggled(bool flag) {
	replayer_fishpos_enable_perfectprediction = flag;
	if (m_ui.checkBox_replay_fishpos_enableperfectpred->isChecked() != flag)
		m_ui.checkBox_replay_fishpos_enableperfectpred->setChecked(flag);
};

void UI::MainWindow::replayer_fishpos_start_QCheckBoxtoggled(bool flag) {
	replayer_fishpos_enable_start = flag;
	if (replayer_fishpos_enable_start)
		replayer_fishpos_curfrm = 0;
	if (m_ui.checkBox_replay_fishpos_enablefeed->isChecked() != flag)
		m_ui.checkBox_replay_fishpos_enablefeed->setChecked(flag);
};
void UI::MainWindow::connectDMD(void) {
	if (lc9000.handle == NULL) {
		// connect
		if (lc9000.connect()) {
			m_ui.pushButton_DMDconnect->setText(QApplication::translate("MainWindow", "DMD disconnect", 0));
			m_ui.pushButton_DMDstartstop->setEnabled(true);
		}
	}
	else {
		// disconnect
		if (lc9000.disconnect()) {
			m_ui.pushButton_DMDconnect->setText(QApplication::translate("MainWindow", "DMD connect", 0));
			m_ui.pushButton_DMDstartstop->setEnabled(false);
		}
	}
};

void UI::MainWindow::startstopDMD(void) {
	if (lc9000.handle) {
		if (lc9000.run == 2) {
			// stop
			lc9000.stop();
			m_ui.pushButton_DMDstartstop->setText(QApplication::translate("MainWindow", "DMD on", 0));
		}
		else {
			// start
			lc9000.start();
			m_ui.pushButton_DMDstartstop->setText(QApplication::translate("MainWindow", "DMD off", 0));
		}
	}
};

void UI::MainWindow::autoresetDMD_QCheckBoxtoggled(bool flag) {
	lc9000_autoreset = flag;
	if (m_ui.checkBox_DMDautoreset->isChecked() != flag)
		m_ui.checkBox_DMDautoreset->setChecked(flag);
};

void UI::MainWindow::intensity_ratio_set_QCheckBoxtoggled(bool flag) {
	if (flag)
		m_ui.doubleSpinBox_intensity_ratio_set->setValue(zaxis_controller.imgMeanIntensity_DIFF_ratio);
};


void UI::MainWindow::toggle_EPI_display_checkBox_epiDisplay(bool flag) {
	HamamatsuCamera_C11440.epiDisplay = flag;
};


void UI::MainWindow::setting_update(bool _updateNI, bool _updateCameraSetting) {
	int nx = 0;
	double * _freq = NULL;
	double * _exp_us = NULL;
	double * _brightness = NULL;
	double * _LED_offset_us = NULL;
	double * _LED_offset_us_diff = NULL;
	double * _databit = NULL;
	double * _img_offset = NULL;
	double * _img_size = NULL;
	double * _mode = NULL;
	double * _shutter_ms = NULL;
	double * _DMD_offset_us = NULL;
	double * _gain = NULL;
	double * _DMD_resetwindow_us = NULL;
	double * _onpixel_threshold = NULL;
	double * _binning = NULL;
	double * _shutterOfftime = NULL;

	//H5File* _file = new H5File("D:\\TrackingMicroscopeData\\_setting\\HamamatsuSetting_200Hz.h5", H5F_ACC_RDONLY);
	H5File* _file = new H5File(HamamatsuCamera_C11440.filesettingName, H5F_ACC_RDONLY);
	_freq = importImpRestInd(_file, "freq", &nx);
	_exp_us = importImpRestInd(_file, "exposure_us", &nx);
	_LED_offset_us = importImpRestInd(_file, "LED_offset_us", &nx);
	_databit = importImpRestInd(_file, "databit", &nx);
	_img_offset = importImpRestInd(_file, "img_offset", &nx);
	_img_size = importImpRestInd(_file, "img_size", &nx);
	_mode = importImpRestInd(_file, "mode", &nx);
	_shutter_ms = importImpRestInd(_file, "shutter_ms", &nx);
	_brightness = importImpRestInd(_file, "brightness", &nx);
	_DMD_offset_us = importImpRestInd(_file, "DMD_offset_us", &nx);
	_DMD_resetwindow_us = importImpRestInd(_file, "DMD_resetwindow_us", &nx);
	_gain = importImpRestInd(_file, "gain", &nx);
	_LED_offset_us_diff = importImpRestInd(_file, "exposure_us_dif", &nx);
	_onpixel_threshold = importImpRestInd(_file, "onpixel_threshold", &nx);
	_binning = importImpRestInd(_file, "binning", &nx);
	_shutterOfftime = importImpRestInd(_file, "shutter_offtime", &nx);


	zaxis_controller.threshold = (uint16)(_onpixel_threshold[0]);

	if (m_ui.checkBox_ExptimeManual->isChecked()) _exp_us[0] = (double)m_ui.spinBox_Hamamatsu_ExpTime_us->value();
	//if (m_ui.checkBox_GainManual->isChecked()) _gain[0] = (double)m_ui.spinBox_Hamamatsu_Gain_dB->value();
	HamamatsuCamera_C11440.reset_timeInterval = (int32)_DMD_resetwindow_us[0] * HamamatsuCamera_C11440.sample_per_us;

	if (m_ui.checkBox_intensity_ratio_set->isChecked()) {
		double _ratio = m_ui.doubleSpinBox_intensity_ratio_set->value();
		_LED_offset_us_diff[0] = 0;
		_LED_offset_us_diff[1] = _exp_us[0] * (_ratio) / 100;
		m_ui.lcdNumber_intensity_ratio_2->display(_LED_offset_us_diff[1]);
		//if (_ratio < 0) _LED_offset_us_diff[0] = _exp_us[0] * fabs(_ratio) / 100;
	}

	if (_updateNI)
		HamamatsuCamera_C11440.getSignal_PG(_freq[0], _exp_us[0], _LED_offset_us, _DMD_offset_us, _LED_offset_us_diff, _shutterOfftime);


	m_ui.doubleSpinBox_Hamamatsu_freq->setValue(_freq[0]);
	m_ui.spinBox_Hamamatsu_ExpTime_us->setValue((int)(_exp_us[0]));
	m_ui.spinBox_Piezo_trk_Intensitythreshold->setValue(_onpixel_threshold[0]);

	/*
	char NAME[128];
	std::lock_guard<std::mutex> lock(Rec_mutex);
	sprintf(NAME, "d:\\TrackingMicroscopeData\\_setting\\temp.h5");
	HDFWriter * settingwriter = new HDFWriter(NAME);
	settingwriter->write("all", H5_INT, HamamatsuCamera_C11440.signal.data, HamamatsuCamera_C11440.signal.n);
	settingwriter->write("cam", H5_INT, HamamatsuCamera_C11440.signal_CamTrigger.data, HamamatsuCamera_C11440.signal_CamTrigger.n);
	settingwriter->write("DMD", H5_INT, HamamatsuCamera_C11440.signal_DMD.data, HamamatsuCamera_C11440.signal_DMD.n);
	settingwriter->write("LED", H5_INT, HamamatsuCamera_C11440.signal_LEDExp.data, HamamatsuCamera_C11440.signal_LEDExp.n);
	delete settingwriter;
	*/

	delete _file;
	if (_freq != NULL) free(_freq);
	if (_exp_us != NULL) free(_exp_us);
	if (_LED_offset_us != NULL) free(_LED_offset_us);
	if (_databit != NULL) free(_databit);
	if (_img_offset != NULL) free(_img_offset);
	if (_img_size != NULL) free(_img_size);
	if (_mode != NULL) free(_mode);
	if (_shutter_ms != NULL) free(_shutter_ms);
	if (_DMD_offset_us != NULL) free(_DMD_offset_us);
	if (_gain != NULL) free(_gain);
}


void UI::MainWindow::HamamatsuCamera_setting(int index) {
	switch (index) {
	case 0:
		HamamatsuCamera_C11440.binning = DCAMPROP_BINNING__2;
		sprintf(HamamatsuCamera_C11440.filesettingName, "d:\\TrackingMicroscopeData\\_setting\\HamamatsuSetting_200Hz_2x2.h5");
		break;
	case 1:
		HamamatsuCamera_C11440.binning = DCAMPROP_BINNING__1;
		sprintf(HamamatsuCamera_C11440.filesettingName, "d:\\TrackingMicroscopeData\\_setting\\HamamatsuSetting_40Hz_1x1.h5");
		break;
	default:
		HamamatsuCamera_C11440.binning = DCAMPROP_BINNING__2;
		sprintf(HamamatsuCamera_C11440.filesettingName, "d:\\TrackingMicroscopeData\\_setting\\HamamatsuSetting_200Hz_2x2.h5");
	} // end switch
}



void UI::MainWindow::refStackCapture(void) {

	int devNum;
	cudaError_t cudaStatus;
	cudaStatus = cudaGetDeviceCount(&devNum);
	cudaStatus = cudaSetDevice(GPU1);


	int n = m_ui.spinBox_Piezo_trk_lockdown->value();
	if (m_ui.checkBox_piezoConcatenate_enable->isChecked())
		n *= m_ui.spinBox_piezoConcatenate_no->value();
	if (HamamatsuCamera_C11440.hdcam && (HamamatsuCamera_C11440.FrmNo > 3*n)) {
		if (HamamatsuCamera_C11440.buffer_stack == NULL) {
			// capture the images
			int offset_count = 200 + 2; // fly back on the first image (3: the fly back image will be the last one)
			int _EPI_frm = ((int)((HamamatsuCamera_C11440.FrmNo - (2 * n) - offset_count) / (2 * n))) * 2 * n + offset_count; // find the first image is the flying back
			int _NIR_frm = _EPI_frm * 5 / 4;
			TrackingMessage * temp = RecBuffer.getFrameData(_NIR_frm);
			Point2d fishPos = Point2d(temp->m_CntrData.data[Xfish], temp->m_CntrData.data[Xfish]);
			for (int i = 1; i < n * 5 / 2; i++) {
				temp = RecBuffer.getFrameData(_NIR_frm + i);
				Point2d fishPos_2 = Point2d(temp->m_CntrData.data[Xfish], temp->m_CntrData.data[Xfish]);
				if ((fishPos - fishPos_2).norm() > 0) {
					return; // fail
				}
			}

			HamamatsuCamera_C11440.buffer_stack = (uInt16 **)malloc(n * sizeof(uInt16*));
			HamamatsuCamera_C11440.buffer_stack_raw = (uInt16 **)malloc(n * 2 * sizeof(uInt16*));
			for (int i = 0; i < n; i++) {
				HamamatsuCamera_C11440.buffer_stack[i] = (uInt16 *)malloc(HamamatsuCamera_C11440.ImageSizeByte);
			}
			for (int i = 0; i < 2 * n; i++) {
				HamamatsuCamera_C11440.buffer_stack_raw[i] = (uInt16 *)malloc(HamamatsuCamera_C11440.ImageSizeByte);
			}

			for (int i = 0; i < n; i++) {
				int n1 = HamamatsuCamera_C11440.buffer_indexNo[(_EPI_frm + 2 * i) % HamamatsuCamera_C11440.buffer_nframe];
				uint16 * A = HamamatsuCamera_C11440.buffer[n1];
				int n2 = HamamatsuCamera_C11440.buffer_indexNo[(_EPI_frm + 2 * i + 1) % HamamatsuCamera_C11440.buffer_nframe];
				uint16 * B = HamamatsuCamera_C11440.buffer[n2];
				memcpy(HamamatsuCamera_C11440.buffer_stack_raw[2*i], A, HamamatsuCamera_C11440.img_px*sizeof(uint16));
				memcpy(HamamatsuCamera_C11440.buffer_stack_raw[2*i+1], B, HamamatsuCamera_C11440.img_px*sizeof(uint16));
				ippsAdd_16u_Sfs(A, B, HamamatsuCamera_C11440.buffer_stack[i], HamamatsuCamera_C11440.img_px, 1);
			}

			m_ui.spinBox_RefBuffeNo->setValue(n);

			double v1 = (double)ControllHardWare_2P.Piezo_scanSingal.getValue(_EPI_frm);
			double v2 = (double)ControllHardWare_2P.Piezo_scanSingal.getValue(_EPI_frm + 2);
			if (v1 > v2) {
				refStack_reverse();
			}

			ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.set_ref_sweep(HamamatsuCamera_C11440.buffer_stack);

			ControllHardWare_2P.Piezo_scanSingal.set_reference(n, ControllHardWare_2P.Piezo_scanSingal.step_um);
			ControllHardWare_2P.Piezo_scanSingal.set_reference_target(n / 2);

			m_ui.spinBox_RefBufferDisplayNo->setMaximum(n - 1);
			m_ui.spinBox_RefBufferDisplayNo->setValue(0);
			m_ui.spinBox_Ref_targetNo->setValue(ControllHardWare_2P.Piezo_scanSingal.id_ref_target);
			m_ui.spinBox_Ref_spacing_um->setValue(ControllHardWare_2P.Piezo_scanSingal.d_ref);

			refStackDisplay(0);
			ControllHardWare_2P.Piezo_scanSingal.isready_tracking = true;
		}
	}
	return;
}


void UI::MainWindow::refStack_reverse(void) {
	int n = m_ui.spinBox_RefBuffeNo->value();
	for (int i = 0; i < n / 2; i++) {
		uint16 * A = HamamatsuCamera_C11440.buffer_stack[i];
		HamamatsuCamera_C11440.buffer_stack[i] = HamamatsuCamera_C11440.buffer_stack[n - i - 1];
		HamamatsuCamera_C11440.buffer_stack[n - i - 1] = A;
	}
	for (int i = 0; i < 2*n; i++) {
		uint16 * A = HamamatsuCamera_C11440.buffer_stack_raw[i];
		HamamatsuCamera_C11440.buffer_stack_raw[i] = HamamatsuCamera_C11440.buffer_stack_raw[2*n - i - 1];
		HamamatsuCamera_C11440.buffer_stack_raw[2 * n - i - 1] = A;
	}
}

void UI::MainWindow::refStackDisplay(int index) {
	if (HamamatsuCamera_C11440.buffer_stack) {
		// variable update
		int _min = m_ui.spinBox_Hamamatsu_IntMinSet->value();
		int _max = m_ui.spinBox_Hamamatsu_IntMaxSet->value();
		if (_min >= _max) {
			_max = _min + 1;
			m_ui.spinBox_Hamamatsu_IntMaxSet->setValue(_max);
		}
		img_uint16 FLRimageDisp = img_uint16(HamamatsuCamera_C11440.buffer_stack[index], imSize(HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height));
		//img_uint16 FLRimageDisp = img_uint16(HamamatsuCamera_C11440.buffer_stack[index], imSize(1024, 1024));
		auto pImage_FLR = toQImage_tf(&FLRimageDisp, (uint16)_min, (uint16)_max);
		m_pImageR1->SetImage(pImage_FLR);
	}
}
void UI::MainWindow::refStackSave(void) {
	if (HamamatsuCamera_C11440.buffer_stack) {
		int n = m_ui.spinBox_RefBuffeNo->value();

		char NAME[256];
		time_t t;
		struct tm * now;
		t = time(0); // get time now
		now = localtime(&t);
		//sprintf(NAME, "D:\\TrackingMicroscopeData\\_Stack\\Stack_%04d%02d%02d_%02d%02d%02d_EPI.dat", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		//std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
		//for (int i = 0; i < n; i++)
		//	pwriter_binary->write((char *)(HamamatsuCamera_C11440.buffer_stack[i]), HamamatsuCamera_C11440.ImageSizeByte);
		//pwriter_binary->close();
		//delete(pwriter_binary);

		//sprintf(NAME, "D:\\TrackingMicroscopeData\\_Stack\\Stack_%04d%02d%02d_%02d%02d%02d_raw.dat", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		//pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
		//for (int i = 0; i < 2*n; i++)
		//	pwriter_binary->write((char *)(HamamatsuCamera_C11440.buffer_stack_raw[i]), HamamatsuCamera_C11440.ImageSizeByte);
		//pwriter_binary->close();
		//delete(pwriter_binary);

		sprintf(NAME, "D:\\TrackingMicroscopeData\\_Stack\\Stack_%04d%02d%02d_%02d%02d%02d_EPI.h5", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		HDF5ImageWriter * p_writer = new HDF5ImageWriter(NAME, HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height, IMAGE_UINT16, 1);
		for (int i = 0; i < n; i++) p_writer->write(HamamatsuCamera_C11440.buffer_stack[i]);
		p_writer->flush(); delete(p_writer);

		sprintf(NAME, "D:\\TrackingMicroscopeData\\_Stack\\Stack_%04d%02d%02d_%02d%02d%02d_raw.h5", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
		p_writer = new HDF5ImageWriter(NAME, HamamatsuCamera_C11440.img_width, HamamatsuCamera_C11440.img_height, IMAGE_UINT16, 1);
		for (int i = 0; i < 2 * n; i++) p_writer->write(HamamatsuCamera_C11440.buffer_stack_raw[i]);
		p_writer->flush(); delete(p_writer);
	}
}
void UI::MainWindow::refStackLoad(void) {
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.initialization(772);
	QString filename = QFileDialog::getOpenFileName(this, "Select stack file", "d:\\TrackingMicroscopeData\\_Stack\\*.h5", tr("Videos (*.h5)"));
	if (filename.isEmpty()) return;
	HDF5ImageReader * p_reader = new HDF5ImageReader(filename.toUtf8());
	int n = p_reader->n_images();

	refStackClear();

	HamamatsuCamera_C11440.buffer_stack = (uInt16 **)malloc(n * sizeof(uInt16*));
	for (int i = 0; i < n; i++) HamamatsuCamera_C11440.buffer_stack[i] = (uInt16 *)malloc(p_reader->image_szbyte());
	for (int i = 0; i < n; i++) p_reader->read16(i, HamamatsuCamera_C11440.buffer_stack[i]);
	delete(p_reader);

	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.set_ref_sweep(HamamatsuCamera_C11440.buffer_stack);

	ControllHardWare_2P.Piezo_scanSingal.set_reference(n, ControllHardWare_2P.Piezo_scanSingal.step_um);
	ControllHardWare_2P.Piezo_scanSingal.set_reference_target(n / 2);

	m_ui.spinBox_RefBufferDisplayNo->setMaximum(n - 1);
	m_ui.spinBox_RefBufferDisplayNo->setValue(0);
	m_ui.spinBox_RefBuffeNo->setValue(n);
	m_ui.spinBox_Ref_targetNo->setValue(ControllHardWare_2P.Piezo_scanSingal.id_ref_target);
	ControllHardWare_2P.Piezo_scanSingal.d_ref = 8;
	m_ui.spinBox_Ref_spacing_um->setValue(ControllHardWare_2P.Piezo_scanSingal.d_ref);

	refStackDisplay(0);
	ControllHardWare_2P.Piezo_scanSingal.isready_tracking = true;


}

void UI::MainWindow::refSetTargetLayer(void) {
	int _n_target = m_ui.spinBox_RefBufferDisplayNo->value();
	ControllHardWare_2P.Piezo_scanSingal.set_reference_target(_n_target);
	m_ui.spinBox_Ref_targetNo->setValue(ControllHardWare_2P.Piezo_scanSingal.id_ref_target);
}

void UI::MainWindow::set_zmatcherlimit(void) {
	ControllHardWare_2P.Piezo_scanSingal.id_lower_limit = m_ui.spinBox_zmatcher_id_lowerlimit->value();
	ControllHardWare_2P.Piezo_scanSingal.id_upper_limit = m_ui.spinBox_zmatcher_id_upperlimit->value();
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_ignore_lower = ControllHardWare_2P.Piezo_scanSingal.id_lower_limit;
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_ignore_upper = ControllHardWare_2P.Piezo_scanSingal.id_upper_limit;
}


void UI::MainWindow::refStackTEST(void) {
	int _n_target = m_ui.spinBox_RefBufferDisplayNo->value();
	//uint16 * testimg = HamamatsuCamera_C11440.buffer_stack[_n_target];
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.initialization(772);
	QString filename = QFileDialog::getOpenFileName(this, "Select stack file", "d:\\TrackingMicroscopeData\\_Stack\\*.h5", tr("Videos (*.h5)"));
	if (filename.isEmpty()) return;
	HDF5ImageReader * p_reader = new HDF5ImageReader(filename.toUtf8());
	int n = p_reader->n_images();


	if (ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation) {
		for (int i = 0; i < ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation; i++){
			free(ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation[i]);
		}
		free(ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation);
	}
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation = p_reader->n_images();

	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation = (uInt16 **)malloc(ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation* sizeof(uInt16*));
	for (int i = 0; i < ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation; i++){
		ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation[i] = (uInt16 *)malloc(p_reader->image_szbyte());
	}

	int * res_index = (int *)malloc(ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation * sizeof(int));
	for (int i = 0; i < n; i++) {
		p_reader->read16(i, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation[i]);
		ControllHardWare_2P.Piezo_scanSingal.get_adjust_z_um(1525, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.buffer_simulation[i]);
		res_index[i] = ControllHardWare_2P.Piezo_scanSingal.id_result;
	}

	//int len = n * sizeof(int);// ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.W2_H2_f32; // N_W2_H2_f32
	//float * a = (float *)malloc(len);
	time_t t = time(0); // get time now;
	struct tm * now = localtime(&t);
	char NAME[1024];
	sprintf(NAME, "d:\\TrackingMicroscopeData\\_Snapshot\\testimage_%04d%02d%02d_%02d%02d%02d_GMap.dat", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
	std::ofstream *pwriter_binary = new std::ofstream(NAME, std::ofstream::binary | ios::out);
	pwriter_binary->write((char*)res_index, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.n_simulation * sizeof(int));
	//cudaMemcpy(a, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.d_moving_LP, len, cudaMemcpyDeviceToHost);
	//pwriter_binary->write((char*)a, len);
	//cudaMemcpy(a, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.d_moving, len, cudaMemcpyDeviceToHost);
	//pwriter_binary->write((char*)a, len);
	//cudaMemcpy(a, ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.d_correlation, len, cudaMemcpyDeviceToHost);
	//pwriter_binary->write((char*)a, len);
	pwriter_binary->close();
	delete(pwriter_binary);

	delete(p_reader);


	m_ui.checkBox_piezoSimulation->setChecked(false);
	m_ui.checkBox_piezoSimulation->setEnabled(true);
}

void UI::MainWindow::refStackClear(void) {
	m_ui.spinBox_RefBufferDisplayNo->setValue(0);
	ControllHardWare_2P.Piezo_scanSingal.isready_tracking = false;
	int n = m_ui.spinBox_RefBuffeNo->value();
	if (HamamatsuCamera_C11440.buffer_stack) {

		if (HamamatsuCamera_C11440.buffer_stack) {
			uint16 ** temp = HamamatsuCamera_C11440.buffer_stack;
			HamamatsuCamera_C11440.buffer_stack = NULL;
			for (int i = 0; i < n; i++)
				free(temp[i]);
			free(temp);
		}

		if (HamamatsuCamera_C11440.buffer_stack_raw) {
			uint16 ** temp2 = HamamatsuCamera_C11440.buffer_stack_raw;
			HamamatsuCamera_C11440.buffer_stack_raw = NULL;
			for (int i = 0; i < 2 * n; i++)
				free(temp2[i]);
			free(temp2);
		}
		m_ui.spinBox_RefBufferDisplayNo->setValue(0);
		m_ui.spinBox_RefBuffeNo->setValue(0);
		m_ui.spinBox_Ref_spacing_um->setValue(0);
		m_ui.spinBox_Ref_targetNo->setValue(0);
	}
}
void UI::MainWindow::setPiezoConcatenate(bool flag) {
	if (flag) {
		int n = m_ui.spinBox_piezoConcatenate_no->value();
		bool bidir = m_ui.checkBox_piezoBidirection_enable->isChecked();
		if (!bidir) {
			double ds = m_ui.doubleSpinBox_Piezo_trk_dz->value() * m_ui.spinBox_Piezo_trk_lockdown->value();
			double global_offset = ds*(n-1) / 2;
			for (int i = 0; i < n; i++) {
				PiezoDirset[i] = 1;
				PiezoOffset[i] = ds*i - global_offset;;
			}
			PiezoOffset_concatenate = n;
		}
		else{
			double ds = m_ui.doubleSpinBox_Piezo_trk_dz->value() * m_ui.spinBox_Piezo_trk_lockdown->value();
			double global_offset = ds*(n - 1) / 2;
			for (int i = 0; i < n; i++) {
				PiezoDirset[i] = 1;
				PiezoOffset[i] = ds*i - global_offset;
			}
			for (int i = 0; i < n; i++) {
				PiezoDirset[n + i] = -1;
				PiezoOffset[n + i] = PiezoOffset[n - (i+1)];
			}
			PiezoOffset_concatenate = 2*n;
		}


	}
	else {
		PiezoOffset_concatenate = 0;
		for (int i = 0; i < 1000; i++) {
			PiezoDirset[i] = 1;
			PiezoOffset[i] = 0;
		}
	}
};

void UI::MainWindow::setPiezoMaxupdateEnable(bool flag) {
	ControllHardWare_2P.Piezo_scanSingal.cor_max_updateEnable = !flag;
	if (m_ui.checkBox_piezo_MaxCorEnable->isChecked() != flag)
		m_ui.checkBox_piezo_MaxCorEnable->setChecked(flag);
};


void UI::MainWindow::refStackSimulation(bool flag) {
	ControllHardWare_2P.Piezo_scanSingal.zmatcher_gpu.enable_simulation = flag;
	if (m_ui.checkBox_piezoSimulation->isChecked() != flag)
		m_ui.checkBox_piezoSimulation->setChecked(flag);
};


void UI::MainWindow::DIFFDisplay_enable(bool flag) {
	HamamatsuCamera_C11440.b_pImgSetting = flag;
	if (m_ui.checkBox_DIFFDisplay_enable->isChecked() != flag)
		m_ui.checkBox_DIFFDisplay_enable->setChecked(flag);
};

void UI::MainWindow::DIFFDisplayA_enable(bool flag) {
	if (flag)
		HamamatsuCamera_C11440.pImgSetting = 0;
	else
		HamamatsuCamera_C11440.pImgSetting = 1;
	if (m_ui.checkBox_DIFFDisplayA_enable->isChecked() != flag)
		m_ui.checkBox_DIFFDisplayA_enable->setChecked(flag);
};
