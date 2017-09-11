#include "Base/base.h"
#include "PointGreyCamera.h"

PointGreyCamera::PointGreyCamera(void)
{
	b_start = false;
	b_stop = false;
	setSerialNumber(17198042); // FL3-U3-13Y3M-C (12180445)
	setROI(616, 462, 608, 608); //816, 614
	setExpTime(0.1f);
	CaptureEnabled = false;
	FrameCount = 0;
	InitFrameCount = 0;
	droppedFrames = 0;
	bytePerPx = 2;
}

PointGreyCamera::~PointGreyCamera()
{
	releaseCamera();
}

void PointGreyCamera::setSerialNumber(unsigned int _serial) {
	SerialNoNIRCamera = _serial;
}

unsigned int PointGreyCamera::getSerialNumber(void) {
	return SerialNoNIRCamera;
}

void PointGreyCamera::setROI(unsigned int _OffsetX, unsigned int _OffsetY, unsigned int _Width, unsigned int _Height) {
	ROIoffsetX = _OffsetX;
	ROIoffsetY = _OffsetY;
	ROIWidth = _Width;
	ROIHeight = _Height;

	ROIsz = ROIWidth*ROIHeight;
	ROIsz_byte = ROIsz * bytePerPx;
}

void PointGreyCamera::setExpTime(float _ExpTime) {
	ExposureTime = _ExpTime;
}

bool PointGreyCamera::connectCamera(void) {
	error = busMgr.GetCameraFromSerialNumber(SerialNoNIRCamera, &guid);
    error = cam.Connect(&guid);
	return true;
}


bool PointGreyCamera::setConfigurations(void) {
	mFrameRateMode = FlyCapture2::FrameRate::FRAMERATE_FORMAT7;
	if (!setExtTriggerMode(true)) return false;

	if (!setShutterSpeed()) return false;
	if (!setOutputLine(true, ExposureTime)) return false;
	if (!setEmbeddedImagePropertyData()) return false;
	if (!setROICamera()) return false;
	if (!setFC2Config()) return false;
	if (!setGain(0.0f)) return false;
	return true;
}

bool PointGreyCamera::startCapture(void) {
    error = cam.StartCapture();
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	CaptureEnabled = true;
	FrameCount = 0;
	InitFrameCount = 0;
	droppedFrames = 0;
	return CaptureEnabled;
}

void PointGreyCamera::resetcount(void) {
	FrameCount = 0;
	InitFrameCount = 0;
	droppedFrames = 0;
}

bool PointGreyCamera::grabImage(void) {
	if (CaptureEnabled) {
		error = cam.RetrieveBuffer( &rawImage );

		if (error.GetType() == FlyCapture2::ErrorType::PGRERROR_OK) {
			metadata = rawImage.GetMetadata();
			if  (InitFrameCount == 0)
				InitFrameCount = metadata.embeddedFrameCounter;

			unsigned int FrameCount_New;
			if (InitFrameCount <= metadata.embeddedFrameCounter)
				FrameCount_New = metadata.embeddedFrameCounter - InitFrameCount;
			else {
				//FrameCount_New = metadata.embeddedFrameCounter - InitFrameCount;
				FrameCount_New = metadata.embeddedFrameCounter + (UINT_MAX - InitFrameCount);
			}

			if (!(FrameCount_New - FrameCount == 1 || FrameCount_New == 0))
				droppedFrames = 0;
			FrameCount= FrameCount_New;
		}
		else {
			if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
		}
	}
	else
		return false;
	return true;
}

void PointGreyCamera::errorMsgDisp(void) {
//	error.PrintErrorTrace();
	sprintf(errMsg,"%s",error.GetDescription());

}
void PointGreyCamera::getCameraInfo(void) {
	FlyCapture2::Utilities::GetLibraryVersion( &fc2Version );
}

bool PointGreyCamera::setExtTriggerMode(bool _enable) {
	// set GPIO1 : input,
	error = cam.SetGPIOPinDirection(0, 0); // Pin to get the direction for. | 0 for input, 1 for output
	mTriggerMode.mode = FlyCapture2::Mode::MODE_14; // mode 0
	mTriggerMode.polarity= 0; // 0:Falling Edge, 1:Rising Edge
	mTriggerMode.source = 0; // GPIO0
	mTriggerMode.parameter = 0; // no parameter need
	mTriggerMode.onOff = _enable;
	error = cam.SetTriggerMode(&mTriggerMode);
	return true;
}

bool PointGreyCamera::setGain(float _gaindB) {
	// 4-3. set shutter time
	//Define the property to adjust.
	propGain.type = FlyCapture2::GAIN;
	//Ensure the property is on.
	propGain.autoManualMode = false; //Ensure auto-adjust mode is off.
	propGain.absControl = true; //Ensure the property is set up to use absolute value control.
	propGain.absValue = (float)_gaindB; //Set the absolute value of shutter to 20 ms.
	error = cam.SetProperty(&propGain);
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) { errorMsgDisp(); return false; }
	return true;
}

bool PointGreyCamera::setShutterSpeed(void) {
	// 4-3. set shutter time
	//Define the property to adjust.
	propShutter.type = FlyCapture2::SHUTTER;
	//Ensure the property is on.
	propShutter.onOff = true;
	propShutter.autoManualMode = false; //Ensure auto-adjust mode is off.
	propShutter.absControl = true; //Ensure the property is set up to use absolute value control.
	propShutter.absValue = ExposureTime; //Set the absolute value of shutter to 20 ms.
	error = cam.SetProperty( &propShutter );
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}

bool PointGreyCamera::setFrameRate(float fps) {
	FlyCapture2::Property propFrameRate;
    propFrameRate.type = FlyCapture2::FRAME_RATE;
	propFrameRate.onOff = true;
	propFrameRate.autoManualMode = false; //Ensure auto-adjust mode is off.
	propFrameRate.absControl = true; //Ensure the property is set up to use absolute value control.
	propFrameRate.absValue = fps; //Set the absolute value of shutter to 20 ms.
	error = cam.SetProperty( &propFrameRate );
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}

bool PointGreyCamera::setOutputLine(bool _enable, float _Pulsewidth) {
	// set output (maybe replace to the exposureActive later)
	mStrobe.onOff = _enable;
	mStrobe.source = 1;// GPIO1
	mStrobe.polarity = 0; // set Low (Activate LED: on-time)
	mStrobe.delay = 0.0f;
	mStrobe.duration = _Pulsewidth;
	error = cam.SetStrobe(&mStrobe);
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}
bool PointGreyCamera::setEmbeddedImagePropertyData(void) {
	// output image information setting
	FlyCapture2::EmbeddedImageInfoProperty OnProperty; OnProperty.onOff = true; OnProperty.available = true;
	FlyCapture2::EmbeddedImageInfoProperty DisabledProperty; DisabledProperty.onOff = false; DisabledProperty.available = false;
	FlyCapture2::EmbeddedImageInfo mEmbeddedImageInfo;
	mEmbeddedImageInfo.brightness = DisabledProperty;
	mEmbeddedImageInfo.exposure = DisabledProperty;
	mEmbeddedImageInfo.gain = DisabledProperty;
	mEmbeddedImageInfo.GPIOPinState= DisabledProperty;
	mEmbeddedImageInfo.shutter = DisabledProperty;
	mEmbeddedImageInfo.strobePattern= DisabledProperty;
	mEmbeddedImageInfo.whiteBalance = DisabledProperty;
	mEmbeddedImageInfo.ROIPosition = DisabledProperty;
	mEmbeddedImageInfo.frameCounter = OnProperty;
	mEmbeddedImageInfo.timestamp = OnProperty;
	error = cam.SetEmbeddedImageInfo(&mEmbeddedImageInfo);
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}
bool PointGreyCamera::setROICamera(void) {
	mPixelFormat = FlyCapture2::PixelFormat::PIXEL_FORMAT_RAW16;
	cam.SetVideoModeAndFrameRate(FlyCapture2::VIDEOMODE_FORMAT7, mFrameRateMode);
	mImageSetting.mode = FlyCapture2::Mode::MODE_7;
	mImageSetting.offsetX = ROIoffsetX;
	mImageSetting.offsetY = ROIoffsetY;
	mImageSetting.width= ROIWidth;
	mImageSetting.height = ROIHeight;
	mImageSetting.pixelFormat = mPixelFormat;
	error = cam.SetFormat7Configuration(&mImageSetting, 100.0f); // 100% packet size
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}


bool PointGreyCamera::setFC2Config(void) {
	cam.GetConfiguration(&mCamearConfig);
	//mCamearConfig.registerTimeout = 10;DROP_FRAMES
	mCamearConfig.numBuffers = 162;
	mCamearConfig.grabMode = FlyCapture2::GrabMode::BUFFER_FRAMES;// FlyCapture2::GrabMode::BUFFER_FRAMES/DROP_FRAMES
	mCamearConfig.highPerformanceRetrieveBuffer = true;
	mCamearConfig.grabTimeout = 2000;
	error = cam.SetConfiguration(&mCamearConfig);
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}


bool PointGreyCamera::releaseCamera(void) {
	if (cam.IsConnected()) {
		if (!setTriggerOff()) return false;
		if (!stopCapturing()) return false;
		if (!disconnectCamera()) return false;
	}
	return true;
}

bool PointGreyCamera::setTriggerOff(void) {
	error = cam.GetTriggerMode( &mTriggerMode );
	mTriggerMode.onOff = false;
	error = cam.SetTriggerMode( &mTriggerMode );
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	CaptureEnabled = false;
	return true;
}

bool PointGreyCamera::stopCapturing(void) {
	error = cam.StopCapture();
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}

bool PointGreyCamera::disconnectCamera(void) {
	error = cam.Disconnect();
	if (error.GetType() != FlyCapture2::ErrorType::PGRERROR_OK) {errorMsgDisp(); return false;}
	return true;
}


uint16 * PointGreyCamera::getDataPointer(void) {
	return (uint16*)rawImage.GetData();
}

unsigned int PointGreyCamera::getrowBytes(void) {
	return (double)rawImage.GetReceivedDataSize()/(double)rawImage.GetRows();
}

unsigned int PointGreyCamera::getrows(void) {
	return rawImage.GetRows();
}

unsigned int PointGreyCamera::getcols(void) {
	return rawImage.GetCols();
}

unsigned int PointGreyCamera::getFrameCount(void) {
	return FrameCount;
}


FlyCapture2::Image PointGreyCamera::getImageCopy(void) {
	FlyCapture2::Image imgCopy;
	error = rawImage.DeepCopy(&imgCopy);
	return imgCopy;
}
